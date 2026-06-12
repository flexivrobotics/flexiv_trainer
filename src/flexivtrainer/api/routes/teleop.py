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

import binascii
import struct
import zlib

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from flexivtrainer.data.lerobot_io import (
    DEFAULT_RECORDING_ENTRY_KEYS,
    resolve_recording_entries,
)
from flexivtrainer.observability import error, info, ok, warn
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/teleop", tags=["teleop"])


class CameraConfigRequest(BaseModel):
    serials: dict[str, str] = Field(default_factory=dict)


class StartRecordingRequest(BaseModel):
    task: str = "Dual-arm Flexiv teleoperation demonstration"
    fps: int | None = Field(default=None, ge=1, le=120)
    recording_entries: list[str] = Field(
        default_factory=lambda: list(DEFAULT_RECORDING_ENTRY_KEYS)
    )


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


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _encode_png(image: np.ndarray) -> bytes:
    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR image array with shape (H, W, 3)")

    height, width, _ = frame.shape
    rgb = frame[:, :, ::-1]
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgb)

    return b"\x89PNG\r\n\x1a\n" + b"".join(
        [
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=1)),
            _png_chunk(b"IEND", b""),
        ]
    )


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
        "ddk": runtime.ddk.snapshot(initialize=False),
        "cameras": runtime.cameras.status(),
        "recording": runtime.recording.status(),
        "services": runtime.service_summary(),
    }


@router.get("/cameras/config")
def get_camera_config(
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    return {
        **runtime.camera_config_snapshot(),
        "devices": runtime.cameras.discover().get("devices", []),
    }


@router.put("/cameras/config")
def update_camera_config(
    request: CameraConfigRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    result = runtime.update_camera_config(request.serials)
    ok("Camera assignment updated")
    return result


@router.get("/cameras/{camera_name}/frame")
def camera_frame(
    camera_name: str, runtime: RuntimeManager = Depends(get_runtime_manager)
) -> Response:
    try:
        frame_payload = runtime.cameras.capture_frame(camera_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        content = _encode_png(frame_payload["image"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=content,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


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
    try:
        recording_entries = resolve_recording_entries(request.recording_entries)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = runtime.recording.start(
            task=request.task,
            fps=request.fps,
            recording_entries=recording_entries,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok(
        "Recording started",
        " ".join(
            [
                f"episode={result.get('episode_name', 'unknown')}",
                f"fps={result.get('fps', 'unknown')}",
                f"task={request.task}",
                f"entries={len(recording_entries)}",
            ]
        ),
    )
    return result


@router.post("/recording/stop")
def stop_recording(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    try:
        result = runtime.recording.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
    try:
        result = runtime.recording.save()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Recording saved", f"episode={result.get('episode_name', 'unknown')}")
    return result


@router.post("/recording/discard")
def discard_recording(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    try:
        result = runtime.recording.discard()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    warn("Recording discarded", f"episode={result.get('episode_name', 'unknown')}")
    return result
