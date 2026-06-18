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
from dataclasses import dataclass, field
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
    manual_assignment: bool = False
    frame_count: int = 0
    last_frame_time: float | None = None
    fps: float = 0.0
    capture_thread: Any | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)


class RealSenseService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._runtimes = {
            camera.name: CameraRuntime(
                config=camera,
                manual_assignment=bool(camera.device_serial),
            )
            for camera in settings.cameras
        }
        # Camera locations surfaced/streamed for the current arm mode. Runtimes
        # for inactive locations stay constructed (so their serials survive a
        # mode switch) but are excluded from status, default streaming, and
        # capture. Defaults to every configured location until the manager
        # narrows it via set_active_locations().
        self._active_locations: list[str] = list(self._runtimes)
        self._last_frames: dict[str, dict[str, Any]] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def set_active_locations(self, names: list[str]) -> None:
        self._active_locations = [name for name in names if name in self._runtimes]

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

    def set_device_serials(
        self, serials: dict[str, str | None], *, manual: bool = True
    ) -> None:
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
                runtime.manual_assignment = manual and bool(
                    runtime.config.device_serial
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
        selected = (
            list(self._active_locations) if camera_names is None else list(camera_names)
        )
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
        # Signal the acquisition thread to exit and let it drain its in-flight
        # wait_for_frames() before stopping the pipeline it owns. Release the
        # service lock while joining so the thread can grab it to store its
        # final frame, otherwise stop_streams would deadlock against it.
        runtime.stop_event.set()
        thread = runtime.capture_thread
        runtime.capture_thread = None
        if thread is not None and thread.is_alive():
            self._lock.release()
            try:
                thread.join(timeout=2.0)
            finally:
                self._lock.acquire()
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
            # A single background thread owns the pipeline and continuously
            # pulls frames into the cache. Consumers (live preview + recording)
            # read the cached frame instead of polling the pipeline themselves,
            # which previously made two readers contend for frames and made the
            # measured FPS swing wildly whenever recording started.
            runtime.stop_event = threading.Event()
            runtime.capture_thread = threading.Thread(
                target=self._acquire_loop,
                args=(runtime,),
                name=f"camera-acquire-{runtime.config.name}",
                daemon=True,
            )
            runtime.capture_thread.start()
        except Exception as exc:  # pragma: no cover - hardware specific
            runtime.pipeline = None
            runtime.started = False
            self._errors[runtime.config.name] = describe_exception(exc)

    def ensure_default_assignment(self) -> bool:
        """Reconcile configured slots with the currently detected cameras.

        Any configured serial that is still detected is preserved. Remaining
        detected devices are then assigned to the remaining slots in
        declaration order, which lets the service recover from stale persisted
        serials after cameras are unplugged and replaced. Explicit manual
        assignments remain pinned even when temporarily unavailable.
        Returns True when the configuration changed so callers can persist it.
        """
        if rs is None:
            return False

        with self._lock:
            serials = [device["serial"] for device in self.discover()["devices"]]
            if not serials:
                return False

            available = list(serials)
            desired: dict[str, str | None] = {}

            for name, runtime in self._runtimes.items():
                serial = runtime.config.device_serial
                if serial and serial in available:
                    desired[name] = serial
                    available.remove(serial)

            changed = False
            for name, runtime in self._runtimes.items():
                serial = desired.get(name)
                if (
                    serial is None
                    and runtime.manual_assignment
                    and runtime.config.device_serial
                ):
                    serial = runtime.config.device_serial
                elif serial is None and available:
                    serial = available.pop(0)
                if runtime.config.device_serial != serial:
                    runtime.config.device_serial = serial
                    runtime.manual_assignment = False
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
            detected_devices = self.discover()["devices"]
            available_serials = self._available_serials()
            if not detected_devices:
                for name in self._resolve_camera_names(camera_names):
                    self._errors[name] = "No RealSense camera is available"
                return self.status()
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
                    "configured_serial": self._runtimes[name].config.device_serial,
                    "actual_serial": self._runtimes[name].actual_serial,
                    "started": self._runtimes[name].started,
                    "fps": self._runtimes[name].fps,
                    "resolution": [
                        self._runtimes[name].config.width,
                        self._runtimes[name].config.height,
                    ],
                    "error": self._errors.get(name),
                }
                for name in self._active_locations
            },
            "errors": {
                name: error
                for name, error in self._errors.items()
                if name in self._active_locations
            },
        }

    def _acquire_loop(self, runtime: CameraRuntime) -> None:
        """Continuously pull frames for a single camera into the cache.

        This is the only place the pipeline is read, so frame delivery and the
        measured FPS reflect the camera's true production cadence regardless of
        how many consumers (live preview, recording) are reading concurrently.
        """
        pipeline = runtime.pipeline
        name = runtime.config.name
        while not runtime.stop_event.is_set():
            try:
                raw_frames = pipeline.wait_for_frames(1_000)
                if not raw_frames:
                    continue
                color_frame = raw_frames.get_color_frame()
                if color_frame is None:
                    continue

                image = np.asanyarray(color_frame.get_data())
                timestamp_ms = color_frame.get_timestamp()
                now = time.monotonic()
                with self._lock:
                    if runtime.last_frame_time is not None:
                        delta = max(now - runtime.last_frame_time, 1e-6)
                        alpha = 0.3
                        runtime.fps = alpha * (1.0 / delta) + (1 - alpha) * runtime.fps
                    runtime.last_frame_time = now
                    runtime.frame_count += 1
                    self._last_frames[name] = {
                        "image": image,
                        "timestamp_ms": timestamp_ms,
                        "fps": runtime.fps,
                        "width": image.shape[1],
                        "height": image.shape[0],
                    }
            except Exception as exc:  # pragma: no cover - hardware specific
                self._errors[name] = describe_exception(exc)
                runtime.stop_event.wait(timeout=0.05)

    def read_frames(
        self,
        block: bool = False,
        timeout_ms: int = 1_000,
        camera_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if rs is None:
            raise RuntimeError("pyrealsense2 is not available")

        names = self._resolve_camera_names(camera_names)

        # The acquisition threads own the pipelines; consumers just read the
        # latest cached frame. When blocking, wait briefly for the first frame
        # to land instead of polling the pipeline directly (which would race the
        # acquisition thread and reintroduce the FPS jitter this design avoids).
        deadline = time.monotonic() + max(0, timeout_ms) / 1_000.0
        while True:
            frames: dict[str, dict[str, Any]] = {}
            with self._lock:
                for name in names:
                    runtime = self._runtimes[name]
                    if runtime.pipeline is None:
                        continue
                    cached = self._last_frames.get(name)
                    if cached is not None:
                        frames[name] = dict(cached)
            if not block or frames or time.monotonic() >= deadline:
                return frames
            time.sleep(0.002)

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
