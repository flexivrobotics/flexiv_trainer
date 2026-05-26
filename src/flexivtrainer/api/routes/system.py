from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from flexivtrainer.config import RobotSerialConfig
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/system", tags=["system"])


class RobotConfigRequest(BaseModel):
    local_robot_serials: list[str] = Field(default_factory=list)
    remote_robot_serials: list[str] = Field(default_factory=list)


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
