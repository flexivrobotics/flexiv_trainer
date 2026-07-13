# Copyright 2026 Flexiv Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-Time Chunking (RTC) for LeRobot's DDIM/DDPM diffusion policy.

Consecutive rollout chunks are denoised from independent noise, so their action
values disagree at the replan seam and the robot jerks. RTC removes the seam *at
the source*: when a new chunk is sampled, its head is conditioned on the still-
executing tail of the previous chunk, so the two join smoothly.

LeRobot 0.5.1 ships an RTC helper (``lerobot/policies/rtc/modeling_rtc.py``) but
it is (a) wired only into ``pi0_fast`` and (b) built for flow-matching models
(velocity field, continuous time) -- it does not apply to a DDIM/DDPM epsilon or
sample predictor. So we assemble RTC from two velocity-free pieces:

1. The *fade-weight curve* -- imported from that helper's
   ``RTCProcessor.get_prefix_weights`` (pure schedule math, no velocity).
2. A *RePaint-style inpaint loop* adapted for diffusers schedulers, mirroring
   Psi0's ``predict_action_with_rtc_flow_naive_inpaint``: each denoising step,
   re-noise the known prefix to the step's noise level via
   ``scheduler.add_noise`` (backed by ``alphas_cumprod``, present on both DDIM and
   DDPM), blend it into the sample under the fade weights, denoise one step,
   repeat.

The functions here take the ``DiffusionModel`` instance as an explicit ``model``
argument so they can be bound onto an already-loaded policy at rollout time
(see ``policies/diffusion.py::apply_rollout_overrides``) without reinstantiating
the class or reloading weights.
"""

from __future__ import annotations

from typing import Any

import torch

# Reuse LeRobot's fade-weight schedule; it is velocity-free and applies as-is.
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
)
from torch import Tensor

# Marker attribute stamped on a policy instance whose sampler has been RTC-wrapped.
RTC_ATTACHED_FLAG = "_rtc_attached"


def _build_prefix_weights(
    inference_delay: int,
    execution_horizon: int,
    horizon: int,
    schedule: str,
    device: torch.device,
) -> Tensor:
    """Fade curve over the horizon: 1.0 for the first ``inference_delay`` steps,
    then decaying to 0.0 by ``execution_horizon``. Shape ``(1, horizon, 1)``.

    ``RTCProcessor`` only needs its ``prefix_attention_schedule`` for this call,
    so we build a throwaway config rather than requiring one from the caller.
    """
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    schedule_enum = (
        RTCAttentionSchedule.EXP
        if schedule == "exp"
        else RTCAttentionSchedule.LINEAR
    )
    proc = RTCProcessor(RTCConfig(prefix_attention_schedule=schedule_enum))
    weights = proc.get_prefix_weights(inference_delay, execution_horizon, horizon)
    return weights.to(device).view(1, horizon, 1)


def _pad_prev_actions(prev_actions: Tensor, horizon: int, action_dim: int) -> Tensor:
    """Right-pad the leftover previous-chunk tail to full ``horizon`` length.

    Positions past the real tail receive weight 0 from the fade curve, so the
    zero padding never reaches the output.
    """
    b, t, a = prev_actions.shape
    if t >= horizon and a >= action_dim:
        return prev_actions[:, :horizon, :action_dim]
    padded = torch.zeros(
        b, horizon, action_dim,
        device=prev_actions.device, dtype=prev_actions.dtype,
    )
    padded[:, :t, :a] = prev_actions[:, : min(t, horizon), : min(a, action_dim)]
    return padded


def rtc_conditional_sample(
    model: Any,
    batch_size: int,
    global_cond: Tensor | None = None,
    generator: torch.Generator | None = None,
    noise: Tensor | None = None,
    *,
    prev_actions: Tensor | None = None,
    inference_delay: int | None = None,
    execution_horizon: int | None = None,
    prefix_schedule: str = "exp",
) -> Tensor:
    """RePaint-style RTC denoising loop for a diffusers DDIM/DDPM ``DiffusionModel``.

    Drop-in replacement for ``DiffusionModel.conditional_sample``. When
    ``prev_actions`` is ``None`` the behaviour is bit-for-bit the base sampler.
    """
    device = get_device_from_parameters(model)
    dtype = get_dtype_from_parameters(model)
    horizon = model.config.horizon
    action_dim = model.config.action_feature.shape[0]

    sample = (
        noise
        if noise is not None
        else torch.randn(
            size=(batch_size, horizon, action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
    )

    model.noise_scheduler.set_timesteps(model.num_inference_steps)
    model._rtc_last_clamped_s = None

    if prev_actions is None:
        # RTC off / first chunk: identical to the stock sampler.
        for t in model.noise_scheduler.timesteps:
            model_output = model.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = model.noise_scheduler.step(
                model_output, t, sample, generator=generator
            ).prev_sample
        return sample

    # --- RTC path -------------------------------------------------------------
    # 0/None means "auto": freeze one step, fade over half the horizon. Clamp d/s
    # to the real prefix length so zero-padding never falls in the frozen region.
    prefix_len = prev_actions.shape[1]
    d = int(inference_delay) if inference_delay else 1
    s = int(execution_horizon) if execution_horizon else horizon // 2
    d = max(1, min(d, prefix_len, horizon - 1))
    s = max(d, min(s, prefix_len, horizon - d))
    model._rtc_last_clamped_s = s

    prev_actions = _pad_prev_actions(
        prev_actions.to(device=device, dtype=dtype), horizon, action_dim
    )
    weights = _build_prefix_weights(d, s, horizon, prefix_schedule, device)
    # Fixed target noise so the re-noised prefix stays on the same latent path as
    # the free-sampled region (Psi0 uses the initial sample as the noise anchor).
    target_noise = sample.clone()

    timesteps = model.noise_scheduler.timesteps
    for i, t in enumerate(timesteps):
        # Re-noise the clean prefix to this step's noise level. add_noise uses
        # alphas_cumprod[t] on both DDIM and DDPM, so this is scheduler-agnostic.
        batched_t = torch.full(
            sample.shape[:1], t, dtype=torch.long, device=sample.device
        )
        noisy_prev = model.noise_scheduler.add_noise(
            prev_actions, target_noise, batched_t
        )
        # Blend: hard where weight==1 (first d steps), fading to free-sampled by s.
        sample = weights * noisy_prev + (1.0 - weights) * sample

        model_output = model.unet(sample, batched_t, global_cond=global_cond)
        sample = model.noise_scheduler.step(
            model_output, t, sample, generator=generator
        ).prev_sample

        # Final step: pin the hard-frozen head exactly to the clean prefix.
        if i == len(timesteps) - 1:
            sample[:, :d, :] = prev_actions[:, :d, :]

    return sample


def rtc_generate_actions(
    model: Any, batch: dict[str, Tensor], noise: Tensor | None = None
) -> Tensor:
    """RTC-aware replacement for ``DiffusionModel.generate_actions``.

    Reads the RTC inputs stashed on the model instance (``_rtc_prev_actions`` /
    ``_rtc_inference_delay``) by the rollout service, threads them into
    ``rtc_conditional_sample``, then slices the executed window exactly as the
    stock method does.

    Alignment contract: ``_rtc_prev_actions`` MUST be laid out on the *full
    horizon* grid (index 0 == the first horizon step the U-Net denoises), so the
    frozen head (indices ``0:d``) and the ``[start:end]`` slice below stay
    consistent. The service builds it that way from the previous chunk's still-
    unexecuted suffix; an off-by-one here would reintroduce the seam.
    """
    batch_size, n_obs_steps = batch["observation.state"].shape[:2]
    assert n_obs_steps == model.config.n_obs_steps

    global_cond = model._prepare_global_conditioning(batch)

    actions = rtc_conditional_sample(
        model,
        batch_size,
        global_cond=global_cond,
        noise=noise,
        prev_actions=getattr(model, "_rtc_prev_actions", None),
        inference_delay=getattr(model, "_rtc_inference_delay", None),
        execution_horizon=getattr(model, "_rtc_execution_horizon", None),
        prefix_schedule=getattr(model, "_rtc_prefix_schedule", "exp"),
    )

    # Stash the full normalized horizon so the service can slice the RTC prefix
    # from it (index 0 == first horizon step) without a normalize round-trip.
    model._rtc_last_full_horizon = actions.detach()

    start = n_obs_steps - 1
    end = start + model.config.n_action_steps
    return actions[:, start:end]
