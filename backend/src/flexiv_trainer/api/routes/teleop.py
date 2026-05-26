from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from flexiv_trainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/teleop", tags=["teleop"])


class StartRecordingRequest(BaseModel):
    task: str = "Dual-arm Flexiv teleoperation demonstration"
    fps: int | None = Field(default=None, ge=1, le=120)


@router.post("/bootstrap")
def bootstrap(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.bootstrap_teleop_module()


@router.get("/status")
def status(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return {
        "teleop": runtime.teleop.snapshot().__dict__,
        "ddk": runtime.ddk.status(),
        "cameras": runtime.cameras.status(),
        "recording": runtime.recording.status(),
    }


@router.post("/start")
def start_teleop(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.teleop.start().__dict__


@router.post("/stop")
def stop_teleop(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.teleop.stop().__dict__


@router.post("/home")
def reset_home(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.teleop.reset_home()


@router.post("/recording/start")
def start_recording(
    request: StartRecordingRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    return runtime.recording.start(task=request.task, fps=request.fps)


@router.post("/recording/stop")
def stop_recording(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.recording.stop()


@router.post("/recording/save")
def save_recording(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.recording.save()


@router.post("/recording/discard")
def discard_recording(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.recording.discard()
