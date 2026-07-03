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

"""Diffusion-policy configuration.

Defaults track the reference diffusion_policy recipe (DDIM sampler, resize+crop
augmentation, compact U-Net) rather than LeRobot's larger defaults.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flexivtrainer.policies._shared import SharedTrainingConfig


class RolloutConfig(BaseModel):
    # Sampler baked into the checkpoint at train time. DDIM samples in far fewer
    # steps than DDPM for the same weights -- fast enough to rollout in real time.
    noise_scheduler_type: Literal["DDPM", "DDIM"] = "DDIM"
    # Reverse-diffusion steps at inference; tunable without retraining.
    num_inference_steps: int = Field(default=8, ge=1, le=1000)


class TrainingConfig(SharedTrainingConfig):
    # Diffusion policy knobs (mapped to --policy.<name>).
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
