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
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/system", tags=["system"])


class RobotConfigRequest(BaseModel):
    arm_mode: Literal["single", "dual"] = "dual"
    leader_robot_serials: list[str] = Field(default_factory=list)
    follower_robot_serials: list[str] = Field(default_factory=list)
    end_effector_config: dict[str, EndEffectorSideConfig] = Field(default_factory=dict)


@router.get("/summary")
def get_system_summary(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.system_summary()


@router.put("/robot-config")
def update_robot_config(
    request: RobotConfigRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    return runtime.update_robot_config(RobotSerialConfig(**request.model_dump()))


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
