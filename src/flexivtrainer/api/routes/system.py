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

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from flexivtrainer.config import EndEffectorSideConfig, RobotSerialConfig
from flexivtrainer.data.hub import has_session_token, hub_token, set_session_token
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/system", tags=["system"])


class RobotConfigRequest(BaseModel):
    arm_mode: Literal["single", "dual"] = "dual"
    leader_robot_serials: list[str] = Field(default_factory=list)
    follower_robot_serials: list[str] = Field(default_factory=list)
    end_effector_config: dict[str, EndEffectorSideConfig] = Field(default_factory=dict)
    home_posture_deg: list[float] = Field(default_factory=list)
    gripper_default_width_m: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    gripper_velocity_m_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    gripper_force_limit_n: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    recording_entries: list[str] = Field(default_factory=list)
    record_resolution: str = ""
    camera_names: dict[str, list[str]] = Field(default_factory=dict)
    camera_counts: dict[str, int] = Field(default_factory=dict)


class HubTokenRequest(BaseModel):
    # Empty string clears the stored token.
    token: str = ""


@router.get("/summary")
def get_system_summary(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.system_summary()


def _hub_token_state(runtime: RuntimeManager) -> dict:
    """Whether a token is available, never the token itself."""
    return {
        "session_token_set": has_session_token(),
        # True when a configured or environment token already covers the request.
        "token_available": hub_token(runtime.settings) is not None,
    }


@router.get("/hub-token")
def get_hub_token_state(
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    return _hub_token_state(runtime)


@router.put("/hub-token")
def set_hub_token(
    request: HubTokenRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    """Hold an operator-supplied Hub token for this server process only."""
    set_session_token(request.token)
    return _hub_token_state(runtime)


@router.put("/robot-config")
def update_robot_config(
    request: RobotConfigRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    try:
        return runtime.update_robot_config(RobotSerialConfig(**request.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/services/{service_name}/{action}")
def control_service(
    service_name: str,
    action: str,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    try:
        return runtime.control_service(service_name, action)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
