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

"""B-spline diffusion training configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from flexivtrainer.policies import diffusion
from flexivtrainer.policies._shared import SharedTrainingConfig


class TrainingConfig(SharedTrainingConfig):
    noise_scheduler_type: Literal["DDIM", "DDPM"] = Field(
        "DDIM", description="DDIM or DDPM"
    )
    num_inference_steps: int = Field(8, ge=1, le=100)
    horizon: int = Field(16, ge=8, le=64, description="spline parameter rows")
    n_obs_steps: int = Field(2, ge=1, le=8)
    resize_shape: tuple[int, int] = Field((240, 320))
    # See diffusion.TrainingConfig: crop_shape is derived from this, not read.
    crop_ratio: float = Field(0.9, gt=0.0, le=1.0)
    crop_shape: tuple[int, int] = Field((216, 288))
    down_dims: tuple[int, int, int] = Field((256, 512, 1024))


class RolloutConfig(diffusion.RolloutConfig):
    control_hz: int = Field(200, ge=1, le=1000)
    speed_scale: float = Field(1.0, gt=0)
    predict_before_end_s: float = Field(0.06, ge=0)
    time_align_error_threshold: float = Field(0.1, ge=0)
    time_align_max_fraction: float = Field(0.2, gt=0, le=1)
    # Alignment lands close, never exact. The leftover gap is decayed to zero over
    # this window instead of being applied as a single-tick step in the commanded
    # pose, which was the dominant source of rollout jitter. 0 restores the old
    # stepping behaviour.
    handoff_blend_s: float = Field(0.15, ge=0, le=1.0)
    # Stretches the blend for a large gap so peak added acceleration stays bounded.
    handoff_max_accel: float = Field(2.0, gt=0)


class PolicyFamilyConfig(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)


def apply_rollout_overrides(policy: Any, rollout_cfg: RolloutConfig) -> bool:
    return diffusion.apply_rollout_overrides(policy, rollout_cfg)
