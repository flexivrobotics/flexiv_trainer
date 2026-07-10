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

"""Diffusion-policy configuration: training knobs + rollout overrides.

``TrainingConfig`` holds the train-time knobs, emitted as ``lerobot-train``
flags and rendered as the Web UI form (defaults track the reference
diffusion_policy recipe -- DDIM sampler, resize+crop augmentation, compact
U-Net -- rather than LeRobot's larger defaults). The denoising sampler is a
policy property baked into the checkpoint at training time (LeRobot's
``--policy.noise_scheduler_type``); training with DDIM lets rollout sample the
same weights in far fewer steps without a DDPM->DDIM swap at load.

``RolloutConfig`` holds the inference-time overrides applied when a checkpoint
is loaded for rollout (``apply_rollout_overrides``), letting the sampler / step
count be retuned without retraining.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.policies._shared import SharedRolloutConfig, SharedTrainingConfig


class TrainingConfig(SharedTrainingConfig):
    # Diffusion policy knobs (mapped to --policy.<name>). The sampler + step
    # count are baked into the checkpoint at train time; both are UI-selectable
    # here (defaults pre-filled) and emitted as --policy.* flags.
    noise_scheduler_type: Literal["DDIM", "DDPM"] = Field(
        "DDIM", description="DDIM or DDPM"
    )
    num_inference_steps: int = Field(8, ge=1, le=100, description="baked default; 5-16")
    horizon: int = Field(16, ge=2, le=64, description="action-chunk length")
    n_obs_steps: int = Field(2, ge=1, le=8, description="observation frames fed in")
    n_action_steps: int = Field(
        8, ge=1, le=48, description="<= horizon - n_obs_steps + 1"
    )
    resize_shape: tuple[int, int] = Field(
        (240, 320), description="H,W; ~1/2 camera dims"
    )
    crop_shape: tuple[int, int] = Field((216, 288), description="H,W; ~90% of resize")
    down_dims: tuple[int, int, int] = Field(
        (256, 512, 1024), description="U-Net widths; smaller = less overfit"
    )


class RolloutConfig(SharedRolloutConfig):
    """Diffusion-policy rollout overrides applied at load."""

    # "" = leave the checkpoint's own sampler/steps untouched.
    noise_scheduler_type: Literal["", "DDPM", "DDIM"] = "DDIM"
    num_denoise_steps: int = Field(default=16, ge=1, le=1000)

    # Real-Time Chunking (RTC): condition each fresh chunk's head on the still-
    # executing tail of the previous chunk so consecutive chunks join smoothly
    # (no replan-seam jerk). Inference-only; no retraining. See
    # policies/rtc_diffusion.py.
    rtc_enabled: bool = Field(
        default=True, description="smooth replan seams via prefix inpainting"
    )
    # d: head steps hard-frozen to the previous chunk. Should track measured
    # inference latency in steps (infer_ms / dt); 0 = auto (derived in planner).
    rtc_inference_delay: int = Field(default=0, ge=0, le=64)
    # s: fade-window end; blend old->new over steps d..s. 0 = auto (half horizon).
    # Constrained to d <= s <= horizon - d at sample time.
    rtc_execution_horizon: int = Field(default=0, ge=0, le=64)
    rtc_prefix_schedule: Literal["linear", "exp"] = "exp"


def apply_rollout_overrides(policy: Any, rollout_cfg: RolloutConfig) -> bool:
    """Apply diffusion-specific rollout overrides to a freshly loaded policy.

    Returns ``True`` if any override (scheduler swap and/or RTC) was applied, so
    the service logs it.
    """
    diffusion = getattr(policy, "diffusion", None)
    if diffusion is None:
        return False

    overridden = False
    scheduler = getattr(rollout_cfg, "noise_scheduler_type", "")
    existing = getattr(diffusion, "noise_scheduler", None)
    if scheduler and existing is not None:
        steps = getattr(rollout_cfg, "num_denoise_steps", 0)
        try:
            from lerobot.policies.diffusion.modeling_diffusion import (  # noqa: PLC0415
                _make_noise_scheduler,
            )

            # Reuse the trained schedule so only the sampler family changes. A
            # scheduler's full config carries family-specific keys the other
            # family rejects, so pass the same kwargs LeRobot uses when building
            # one.
            cfg = existing.config
            kwargs = dict(
                num_train_timesteps=cfg.num_train_timesteps,
                beta_start=cfg.beta_start,
                beta_end=cfg.beta_end,
                beta_schedule=cfg.beta_schedule,
                clip_sample=cfg.clip_sample,
                clip_sample_range=cfg.clip_sample_range,
                prediction_type=cfg.prediction_type,
            )
            diffusion.noise_scheduler = _make_noise_scheduler(scheduler, **kwargs)
            diffusion.num_inference_steps = steps
            overridden = True
        except Exception as exc:
            warn("Failed to override diffusion scheduler", describe_exception(exc))

    if getattr(rollout_cfg, "rtc_enabled", False):
        overridden = _attach_rtc(diffusion, rollout_cfg) or overridden

    return overridden


def _attach_rtc(diffusion: Any, rollout_cfg: RolloutConfig) -> bool:
    """Swap the diffusion model's sampler for the RTC (prefix-inpainting) version.

    Binds the standalone RTC functions onto this loaded instance and stashes the
    RTC knobs on it, so no weights are reloaded and the base policy loader is
    untouched. The service later stashes ``_rtc_prev_actions`` /
    ``_rtc_inference_delay`` on this same instance before each fresh inference.
    """
    try:
        import types  # noqa: PLC0415

        from flexivtrainer.policies import rtc_diffusion  # noqa: PLC0415

        if getattr(diffusion, rtc_diffusion.RTC_ATTACHED_FLAG, False):
            return True
        diffusion.generate_actions = types.MethodType(
            rtc_diffusion.rtc_generate_actions, diffusion
        )
        # Static config knobs; per-inference prev_actions/delay are set by the
        # service. Defaults left as None/"exp" so a first chunk (no prev_actions)
        # samples exactly like the stock policy.
        diffusion._rtc_prev_actions = None
        diffusion._rtc_inference_delay = rollout_cfg.rtc_inference_delay or None
        diffusion._rtc_execution_horizon = rollout_cfg.rtc_execution_horizon or None
        diffusion._rtc_prefix_schedule = rollout_cfg.rtc_prefix_schedule
        setattr(diffusion, rtc_diffusion.RTC_ATTACHED_FLAG, True)
    except Exception as exc:
        warn("Failed to enable RTC", describe_exception(exc))
        return False
    return True