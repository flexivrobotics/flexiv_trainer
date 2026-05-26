from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from flexivtrainer.observability import error, info, ok, warn
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/teleop", tags=["teleop"])


class StartRecordingRequest(BaseModel):
    task: str = "Dual-arm Flexiv teleoperation demonstration"
    fps: int | None = Field(default=None, ge=1, le=120)


def _bootstrap_issue_detail(result: dict) -> str | None:
    issues: list[str] = []

    for stage in result.get("stages", []):
        stage_name = stage.get("stage", "unknown")
        detail = stage.get("detail") or {}
        if stage_name == "teleop":
            if detail.get("error"):
                issues.append(f"teleop={detail['error']}")
            if detail.get("fault"):
                issues.append(f"fault={detail['fault']}")
            continue

        for key, value in (detail.get("errors") or {}).items():
            issues.append(f"{stage_name}.{key}={value}")

    recording = result.get("recording") or {}
    if recording.get("error"):
        issues.append(f"recording={recording['error']}")

    return " | ".join(issues[:8]) or None


@router.post("/bootstrap")
def bootstrap(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.bootstrap_teleop_module()
    if result.get("ready"):
        ok("Teleoperation module bootstrapped")
    else:
        issue_detail = _bootstrap_issue_detail(result)
        stage_names = ", ".join(
            stage.get("stage", "unknown") for stage in result.get("stages", [])
        )
        warn(
            "Teleoperation bootstrap completed with issues",
            issue_detail or stage_names,
        )
    return result


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
    result = runtime.teleop.start().__dict__
    if result.get("error"):
        error("Teleoperation start failed", str(result.get("error")))
    elif result.get("started"):
        ok("Teleoperation started")
    else:
        warn("Teleoperation start request finished without entering started state")
    return result


@router.post("/stop")
def stop_teleop(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.teleop.stop().__dict__
    if result.get("error"):
        error("Teleoperation stop failed", str(result.get("error")))
    else:
        info("Teleoperation stopped")
    return result


@router.post("/home")
def reset_home(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.teleop.reset_home()
    if result.get("error"):
        error("Home reset failed", str(result.get("error")))
    elif result.get("warnings"):
        warn(
            "Home reset completed with warnings",
            "; ".join(str(item) for item in result.get("warnings", [])),
        )
    else:
        ok("Home reset command sent")
    return result


@router.post("/recording/start")
def start_recording(
    request: StartRecordingRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    result = runtime.recording.start(task=request.task, fps=request.fps)
    ok(
        "Recording started",
        " ".join(
            [
                f"episode={result.get('episode_name', 'unknown')}",
                f"fps={result.get('fps', 'unknown')}",
                f"task={request.task}",
            ]
        ),
    )
    return result


@router.post("/recording/stop")
def stop_recording(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.recording.stop()
    info(
        "Recording stopped",
        " ".join(
            [
                f"episode={result.get('episode_name', 'unknown')}",
                f"frames={result.get('frames_captured', 'unknown')}",
            ]
        ),
    )
    return result


@router.post("/recording/save")
def save_recording(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.recording.save()
    ok("Recording saved", f"episode={result.get('episode_name', 'unknown')}")
    return result


@router.post("/recording/discard")
def discard_recording(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.recording.discard()
    warn("Recording discarded", f"episode={result.get('episode_name', 'unknown')}")
    return result
