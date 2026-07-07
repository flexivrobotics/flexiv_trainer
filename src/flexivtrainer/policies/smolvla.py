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

"""SmolVLA policy training configuration (defaults track LeRobot's SmolVLAConfig)."""

from __future__ import annotations

from pydantic import Field

from flexivtrainer.policies._shared import SharedTrainingConfig


class TrainingConfig(SharedTrainingConfig):
    chunk_size: int = Field(50, ge=1, le=1000, description="action-chunk length")
    freeze_vision_encoder: bool = Field(True, description="freeze the VLM vision tower")
    optimizer_lr: float = Field(1e-4, gt=0, le=1.0, description="learning rate")
