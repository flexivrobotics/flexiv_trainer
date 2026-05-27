from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from flexivtrainer.config import AppSettings, CameraConfig
from flexivtrainer.observability import describe_exception

try:
    import pyrealsense2 as rs
except (
    ImportError
):  # pragma: no cover - dependency availability is environment-specific
    rs = None


@dataclass
class CameraRuntime:
    config: CameraConfig
    pipeline: Any | None = None
    started: bool = False
    actual_serial: str | None = None
    frame_count: int = 0
    last_frame_time: float | None = None
    fps: float = 0.0


class RealSenseService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._runtimes = {
            camera.name: CameraRuntime(config=camera) for camera in settings.cameras
        }
        self._last_frames: dict[str, dict[str, Any]] = {}
        self._errors: dict[str, str] = {}

    def available(self) -> bool:
        return rs is not None

    def discover(self) -> dict[str, Any]:
        if rs is None:
            return {
                "available": False,
                "devices": [],
                "errors": {"import": "pyrealsense2 is not importable"},
            }

        context = rs.context()
        devices = []
        for device in context.devices:
            devices.append(
                {
                    "name": device.get_info(rs.camera_info.name),
                    "serial": device.get_info(rs.camera_info.serial_number),
                }
            )
        return {"available": True, "devices": devices, "errors": dict(self._errors)}

    def _resolve_camera_names(self, camera_names: list[str] | None = None) -> list[str]:
        selected = list(self._runtimes) if camera_names is None else list(camera_names)
        unknown = [name for name in selected if name not in self._runtimes]
        if unknown:
            raise ValueError(f"Unsupported cameras: {', '.join(unknown)}")
        return selected

    def _available_serials(self) -> list[str]:
        available = [device["serial"] for device in self.discover()["devices"]]
        occupied = {
            runtime.actual_serial or runtime.config.device_serial
            for runtime in self._runtimes.values()
            if runtime.started
            and (runtime.actual_serial or runtime.config.device_serial)
        }
        return [serial for serial in available if serial not in occupied]

    def _stop_runtime(self, runtime: CameraRuntime) -> None:
        if runtime.pipeline is not None:
            try:
                runtime.pipeline.stop()
            except Exception as exc:  # pragma: no cover - hardware specific
                self._errors[runtime.config.name] = describe_exception(exc)
        runtime.pipeline = None
        runtime.started = False
        runtime.last_frame_time = None
        runtime.fps = 0.0

    def _resolve_runtime_serial(
        self, runtime: CameraRuntime, available_serials: list[str]
    ) -> str | None:
        serial = runtime.config.device_serial or runtime.actual_serial
        if serial:
            if serial in available_serials:
                available_serials.remove(serial)
                return serial
            runtime.actual_serial = serial
            self._errors[runtime.config.name] = (
                f"Camera serial {serial} is not detected"
            )
            return None

        if not available_serials:
            runtime.actual_serial = None
            self._errors[runtime.config.name] = (
                "No RealSense camera is available for this stream"
            )
            return None

        return available_serials.pop(0)

    def _start_runtime(
        self, runtime: CameraRuntime, available_serials: list[str]
    ) -> None:
        serial = self._resolve_runtime_serial(runtime, available_serials)
        if serial is None:
            runtime.pipeline = None
            runtime.started = False
            return

        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(
            rs.stream.color,
            runtime.config.width,
            runtime.config.height,
            rs.format.bgr8,
            runtime.config.fps,
        )
        try:
            profile = pipeline.start(config)
            runtime.pipeline = pipeline
            runtime.started = True
            runtime.last_frame_time = None
            runtime.fps = 0.0
            try:
                runtime.actual_serial = profile.get_device().get_info(
                    rs.camera_info.serial_number
                )
            except Exception:  # pragma: no cover - hardware specific
                runtime.actual_serial = serial
            self._errors.pop(runtime.config.name, None)
        except Exception as exc:  # pragma: no cover - hardware specific
            runtime.pipeline = None
            runtime.started = False
            self._errors[runtime.config.name] = describe_exception(exc)

    def start_streams(self, camera_names: list[str] | None = None) -> dict[str, Any]:
        if rs is None:
            return {
                "available": False,
                "started": False,
                "errors": {"import": "pyrealsense2 is not importable"},
            }

        available_serials = self._available_serials()
        for name in self._resolve_camera_names(camera_names):
            runtime = self._runtimes[name]
            if runtime.started:
                continue
            self._start_runtime(runtime, available_serials)

        return self.status()

    def stop_streams(self, camera_names: list[str] | None = None) -> dict[str, Any]:
        for name in self._resolve_camera_names(camera_names):
            self._stop_runtime(self._runtimes[name])
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "available": rs is not None,
            "cameras": {
                name: {
                    "configured_serial": runtime.config.device_serial,
                    "actual_serial": runtime.actual_serial,
                    "started": runtime.started,
                    "fps": runtime.fps,
                    "resolution": [runtime.config.width, runtime.config.height],
                    "error": self._errors.get(name),
                }
                for name, runtime in self._runtimes.items()
            },
            "errors": dict(self._errors),
        }

    def read_frames(
        self, block: bool = False, timeout_ms: int = 1_000
    ) -> dict[str, dict[str, Any]]:
        if rs is None:
            raise RuntimeError("pyrealsense2 is not available")

        frames: dict[str, dict[str, Any]] = {}
        for name, runtime in self._runtimes.items():
            if runtime.pipeline is None:
                continue

            try:
                raw_frames = (
                    runtime.pipeline.wait_for_frames(timeout_ms)
                    if block
                    else runtime.pipeline.poll_for_frames()
                )
                if not raw_frames:
                    continue
                color_frame = raw_frames.get_color_frame()
                if color_frame is None:
                    continue

                image = np.asanyarray(color_frame.get_data())
                now = time.monotonic()
                if runtime.last_frame_time is not None:
                    delta = max(now - runtime.last_frame_time, 1e-6)
                    runtime.fps = 1.0 / delta
                runtime.last_frame_time = now
                runtime.frame_count += 1

                frames[name] = {
                    "image": image,
                    "timestamp_ms": color_frame.get_timestamp(),
                    "fps": runtime.fps,
                    "width": image.shape[1],
                    "height": image.shape[0],
                }
            except Exception as exc:  # pragma: no cover - hardware specific
                self._errors[name] = describe_exception(exc)

        self._last_frames.update(frames)
        return frames

    def latest_frame_metadata(self) -> dict[str, Any]:
        metadata = {}
        for name, frame in self._last_frames.items():
            metadata[name] = {
                "timestamp_ms": frame["timestamp_ms"],
                "fps": frame["fps"],
                "width": frame["width"],
                "height": frame["height"],
            }
        return metadata
