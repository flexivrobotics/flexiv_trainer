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

"""ACT policy training configuration (defaults track LeRobot's ACTConfig)."""

from __future__ import annotations

from pydantic import Field

from flexivtrainer.policies._shared import SharedTrainingConfig


class TrainingConfig(SharedTrainingConfig):
    chunk_size: int = Field(100, ge=1, le=1000, description="action-chunk length")
    n_action_steps: int = Field(
        100, ge=1, le=1000,
        description="actions executed per chunk; must be <= chunk_size",
    )
    n_encoder_layers: int = Field(
        4, ge=1, le=12, description="transformer encoder layers (ACT paper: 4)"
    )
    n_decoder_layers: int = Field(
        7, ge=1, le=12, description="transformer decoder layers (ACT paper: 7)"
    )
    dim_model: int = Field(512, ge=64, le=2048, description="transformer width")
    optimizer_lr: float = Field(1e-5, gt=0, le=1.0, description="learning rate")
    # Checkbox-gated in the Web UI; plain float (not float | None) so the schema
    # builder types it "float". Last field so its checkbox has no cell to overlap.
    temporal_ensemble_coeff: float = Field(
        0.1,
        gt=0,
        le=1.0,
        description="temporal-ensemble weight; forces n_action_steps=1 when enabled",
    )
