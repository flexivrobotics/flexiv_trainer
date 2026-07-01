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

The denoising sampler is a policy property baked in at training time (LeRobot's
``--policy.noise_scheduler_type``). Training with DDIM lets rollout sample the
same weights in far fewer steps without the DDPM->DDIM swap at load; the step
count is tunable at inference without retraining.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DiffusionPolicyConfig(BaseModel):
    # Sampler baked into the checkpoint at train time. DDIM samples in far fewer
    # steps than DDPM for the same weights -- fast enough to rollout in real time.
    noise_scheduler_type: Literal["DDPM", "DDIM"] = "DDIM"
    # Reverse-diffusion steps at inference; tunable without retraining.
    num_inference_steps: int = Field(default=8, ge=1, le=1000)
