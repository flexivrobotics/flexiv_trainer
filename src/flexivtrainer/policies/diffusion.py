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

The denoising sampler is a policy property baked in at training time (LeRobot's
``--policy.noise_scheduler_type``). Training with DDIM lets rollout sample the
same weights in far fewer steps without the DDPM->DDIM swap at load; the step
count is tunable at inference without retraining.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.policies._shared import SharedRolloutConfig


class TrainingConfig(BaseModel):
    # Sampler baked into the checkpoint at train time. DDIM samples in far fewer
    # steps than DDPM for the same weights -- fast enough to rollout in real time.
    noise_scheduler_type: Literal["DDPM", "DDIM"] = "DDIM"
    # Reverse-diffusion steps at inference; tunable without retraining.
    num_denoise_steps: int = Field(default=8, ge=1, le=1000)


class RolloutConfig(SharedRolloutConfig):
    """Diffusion-policy rollout overrides applied at load."""

    # "" = leave the checkpoint's own sampler/steps untouched.
    noise_scheduler_type: Literal["", "DDPM", "DDIM"] = "DDIM"
    num_denoise_steps: int = Field(default=16, ge=1, le=1000)


def apply_rollout_overrides(policy: Any, rollout_cfg: RolloutConfig) -> bool:
    """Apply diffusion-specific rollout overrides to a freshly loaded policy."""
    scheduler = getattr(rollout_cfg, "noise_scheduler_type", "")
    if not scheduler:
        return False
    diffusion = getattr(policy, "diffusion", None)
    existing = getattr(diffusion, "noise_scheduler", None)
    if diffusion is None or existing is None:
        return False
    steps = getattr(rollout_cfg, "num_denoise_steps", 0)
    try:
        from lerobot.policies.diffusion.modeling_diffusion import (  # noqa: PLC0415
            _make_noise_scheduler,
        )

        # Reuse the trained schedule so only the sampler family changes. A
        # scheduler's full config carries family-specific keys the other family
        # rejects, so pass the same kwargs LeRobot uses when building one.
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
    except Exception as exc:
        warn("Failed to override diffusion scheduler", describe_exception(exc))
        return False
    return True
