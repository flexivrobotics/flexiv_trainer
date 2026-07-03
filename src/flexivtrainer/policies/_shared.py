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

"""Training-loop knobs shared by every policy (top-level ``lerobot-train`` flags)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SharedTrainingConfig(BaseModel):
    batch_size: int = Field(
        64, ge=1, le=512, description="8-128; GPU-memory bound",
        json_schema_extra={"flag": "--batch_size"},
    )
    epochs: int = Field(
        100, ge=1, le=1000, description="50-300 typical; converted to --steps",
        json_schema_extra={"flag": "--steps"},
    )
    save_freq: int = Field(
        5000, ge=100, le=100000, description="checkpoint every N steps",
        json_schema_extra={"flag": "--save_freq"},
    )
    log_freq: int = Field(
        200, ge=1, le=10000, description="log every N steps",
        json_schema_extra={"flag": "--log_freq"},
    )
    num_workers: int = Field(
        4, ge=0, le=32, description="dataloader procs; lower if /dev/shm errors",
        json_schema_extra={"flag": "--num_workers"},
    )
