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

import threading
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
        self._lock = threading.Lock()

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

    def configured_serials(self) -> dict[str, str | None]:
        return {
            name: runtime.config.device_serial
            for name, runtime in self._runtimes.items()
        }

    def set_device_serials(self, serials: dict[str, str | None]) -> None:
        """Assign device serials to camera locations.

        A serial may back only one slot. Any duplicate is dropped (the first
        slot in declaration order keeps it, later ones become N/A), which also
        cleans up stale persisted configs. When any assignment changes, every
        streaming camera is stopped and restarted so the new mapping resolves
        cleanly even when serials are swapped between locations.
        """
        with self._lock:
            before = {
                name: runtime.config.device_serial
                for name, runtime in self._runtimes.items()
            }

            for name, serial in serials.items():
                runtime = self._runtimes.get(name)
                if runtime is None:
                    continue
                runtime.config.device_serial = (
                    (str(serial).strip() or None) if serial else None
                )

            # Enforce uniqueness: keep the first slot holding each serial.
            seen: set[str] = set()
            for runtime in self._runtimes.values():
                serial = runtime.config.device_serial
                if not serial:
                    continue
                if serial in seen:
                    runtime.config.device_serial = None
                else:
                    seen.add(serial)

            changed = False
            for name, runtime in self._runtimes.items():
                if runtime.config.device_serial != before[name]:
                    changed = True
                    self._errors.pop(name, None)

            if changed:
                self._restart_started_cameras()

    def _restart_started_cameras(self) -> None:
        # Only re-resolve while the camera service is active (at least one slot
        # streaming). If nothing is running, leave it to an explicit connect.
        if not any(rt.started for rt in self._runtimes.values()):
            return

        for runtime in self._runtimes.values():
            if runtime.started:
                self._stop_runtime(runtime)
            # Drop sticky auto-assignments so each slot is re-resolved freshly.
            runtime.actual_serial = None

        if rs is None:
            return

        # Re-resolve every slot, not just the ones that happened to be running:
        # freeing a device from one slot may now let another slot acquire it.
        available_serials = self._available_serials()
        for runtime in self._runtimes.values():
            self._start_runtime(runtime, available_serials)

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
        serial = runtime.config.device_serial
        if not serial:
            # Slot is set to N/A: intentionally unassigned, not an error.
            runtime.actual_serial = None
            self._errors.pop(runtime.config.name, None)
            return None

        if serial in available_serials:
            available_serials.remove(serial)
            return serial

        runtime.actual_serial = serial
        self._errors[runtime.config.name] = f"Camera serial {serial} is not detected"
        return None

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

    def ensure_default_assignment(self) -> bool:
        """Assign detected cameras to the first slots when none are configured.

        The N detected serials are assigned to the first N camera slots (in
        declaration order); remaining slots stay unassigned (N/A). Only applied
        when no slot has an explicit serial yet, so user choices are preserved.
        Returns True when the configuration changed (so callers can persist it).
        """
        if rs is None:
            return False

        with self._lock:
            if any(rt.config.device_serial for rt in self._runtimes.values()):
                return False
            serials = [device["serial"] for device in self.discover()["devices"]]
            if not serials:
                return False
            changed = False
            for runtime, serial in zip(self._runtimes.values(), serials):
                if runtime.config.device_serial != serial:
                    runtime.config.device_serial = serial
                    changed = True
            return changed

    def start_streams(self, camera_names: list[str] | None = None) -> dict[str, Any]:
        if rs is None:
            return {
                "available": False,
                "started": False,
                "errors": {"import": "pyrealsense2 is not importable"},
            }

        self.ensure_default_assignment()
        with self._lock:
            available_serials = self._available_serials()
            for name in self._resolve_camera_names(camera_names):
                runtime = self._runtimes[name]
                if runtime.started:
                    continue
                self._start_runtime(runtime, available_serials)

        return self.status()

    def stop_streams(self, camera_names: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
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
        self,
        block: bool = False,
        timeout_ms: int = 1_000,
        camera_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if rs is None:
            raise RuntimeError("pyrealsense2 is not available")

        frames: dict[str, dict[str, Any]] = {}
        for name in self._resolve_camera_names(camera_names):
            runtime = self._runtimes[name]
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
                with self._lock:
                    if runtime.last_frame_time is not None:
                        delta = max(now - runtime.last_frame_time, 1e-6)
                        alpha = 0.3
                        runtime.fps = alpha * (1.0 / delta) + (1 - alpha) * runtime.fps
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

        with self._lock:
            self._last_frames.update(frames)
        return frames

    def capture_frame(
        self,
        camera_name: str,
        *,
        block: bool = True,
        timeout_ms: int = 350,
        allow_cached: bool = True,
    ) -> dict[str, Any]:
        selected_name = self._resolve_camera_names([camera_name])[0]
        runtime = self._runtimes[selected_name]
        if runtime.pipeline is None or not runtime.started:
            raise RuntimeError(f"Camera '{selected_name}' is not started")

        frames = self.read_frames(
            block=block,
            timeout_ms=timeout_ms,
            camera_names=[selected_name],
        )
        if selected_name in frames:
            return frames[selected_name]

        with self._lock:
            cached = self._last_frames.get(selected_name)
        if allow_cached and cached is not None:
            return cached

        raise RuntimeError(f"No frame is available for camera '{selected_name}'")

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
