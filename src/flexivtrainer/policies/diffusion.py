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

from typing import Literal

from pydantic import BaseModel, Field

from flexivtrainer.policies._shared import SharedRolloutConfig


class TrainingConfig(BaseModel):
    # Sampler baked into the checkpoint at train time. DDIM samples in far fewer
    # steps than DDPM for the same weights -- fast enough to rollout in real time.
    noise_scheduler_type: Literal["DDPM", "DDIM"] = "DDIM"
    # Reverse-diffusion steps at inference; tunable without retraining.
    num_denoise_steps: int = Field(default=8, ge=1, le=1000)


class RolloutConfig(SharedRolloutConfig):
    """Diffusion-policy rollout overrides applied at load.

    Swaps the denoising sampler at rollout load. Old checkpoints train with
    DDPM/100 (~100 U-Net forwards per refill, stalling the loop); DDIM reuses the
    same weights but reaches the target in far fewer steps. "" leaves the
    checkpoint's own scheduler/steps untouched.

    Transitional: a bridge for pre-existing DDPM checkpoints. New checkpoints bake
    in DDIM via ``TrainingConfig`` at train time, so once all live checkpoints are
    DDIM-native, delete these fields + the scheduler swap in
    ``_apply_rollout_overrides``.
    """

    # "" = leave the checkpoint's own sampler/steps untouched.
    noise_scheduler_type: Literal["", "DDPM", "DDIM"] = "DDIM"
    num_denoise_steps: int = Field(default=16, ge=1, le=1000)
