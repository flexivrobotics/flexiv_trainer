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

"""Rollout knobs common to every policy family.

Each family's ``RolloutConfig`` subclasses ``SharedRolloutConfig`` and adds its
own sampler knobs; these three are applied at load / in the planner loop and
apply to any family (a bare ``SharedRolloutConfig`` is the default for families
without their own rollout overrides).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SharedRolloutConfig(BaseModel):
    # Override the checkpoint's action-chunk length at load; 0 = keep the
    # checkpoint default. Clamped best-effort to the family's valid range.
    n_action_steps: int = Field(default=0, ge=0, le=64)
    # Force a fresh inference every N planner ticks so a committed path always
    # remains while the next chunk computes (overlapped replanning, as in the
    # original diffusion_policy runner). 0 = auto (half the effective chunk,
    # min 1).
    replan_steps: int = Field(default=0, ge=0, le=64)
    # Waypoint k targets loop_start + (k + offset) * dt; offset >= 1 keeps
    # waypoint 0 ahead of the past-filter (inference latency would drop it).
    action_anchor_offset_steps: int = Field(default=1, ge=0, le=8)
