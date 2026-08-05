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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flexivtrainer.cameras.backends import (
    CameraBackend,
    CameraStream,
    OrbbecBackend,
    RealSenseBackend,
)
from flexivtrainer.config import AppSettings, CameraConfig
from flexivtrainer.observability import describe_exception

try:
    import pyrealsense2 as rs
except (
    ImportError
):  # pragma: no cover - dependency availability is environment-specific
    rs = None

# OrbbecSDK v2 (Gemini 305/330 series); `pyorbbecsdk` is the v1 package kept for
# older devices. Bound at module level to mirror `rs`, so tests can swap it.
try:  # pragma: no cover - dependency availability is environment-specific
    import pyorbbecsdk2 as ob
except ImportError:  # pragma: no cover
    try:
        import pyorbbecsdk as ob
    except ImportError:  # pragma: no cover
        ob = None

# Restart a started-but-silent pipeline after this long without a frame.
SILENT_RESTART_AFTER_S = 3.0
# Drop a depth-preview alignment lease this long after its last frame request.
_ALIGN_LEASE_S = 3.0


def _default_backends() -> list[CameraBackend]:
    """Backends in auto-assignment priority order.

    Each resolves its SDK through a getter rather than a captured import, so a
    test that swaps this module's ``rs`` attribute for a fake SDK still steers
    the RealSense backend.
    """
    return [
        RealSenseBackend(module_getter=lambda: rs),
        OrbbecBackend(module_getter=lambda: ob),
    ]


@dataclass
class CameraRuntime:
    config: CameraConfig
    stream: CameraStream | None = None
    backend: CameraBackend | None = None
    started: bool = False
    actual_serial: str | None = None
    manual_assignment: bool = False
    frame_count: int = 0
    last_frame_time: float | None = None
    fps: float = 0.0
    capture_thread: Any | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    # Depth is requested per config.use_depth; depth_started reflects whether the
    # pipeline actually came up with a depth stream (it can fall back to color).
    depth_started: bool = False
    # Alignment costs a per-pixel remap every frame, so it only runs while
    # something wants aligned depth. Recording holds a balanced reference;
    # survives restarts so the silent-camera watchdog cannot unalign it
    # mid-episode. The preview instead renews a deadline, because a polling
    # viewer has no close signal to release on.
    align_consumers: int = 0
    align_lease_until: float = 0.0
    # Depth values are raw device units. Recording stores uint16 millimeters so
    # LeRobot can infer and preserve the depth unit.
    depth_scale_m: float = 0.001


class CameraService:
    """Manages every configured camera slot across all supported vendors."""

    def __init__(
        self, settings: AppSettings, backends: list[CameraBackend] | None = None
    ) -> None:
        self._settings = settings
        # Declaration order is the priority order used when auto-assigning a
        # detected device to an empty slot.
        self._backends: list[CameraBackend] = (
            backends if backends is not None else _default_backends()
        )
        # Serial -> backend, refreshed on every discover() so a slot can be
        # started with the SDK that actually owns its device.
        self._serial_vendors: dict[str, str] = {}
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
        # Set by the runtime manager; depth alignment starves the policy loop of
        # the GIL, so a rollout forbids it outright rather than trusting the UI.
        self._alignment_blocked: Callable[[], bool] = lambda: False
        self._lock = threading.Lock()
        # Serializes whole start/stop/restart cycles (_lock is released mid-join).
        self._lifecycle_lock = threading.RLock()

    def set_active_locations(self, names: list[str]) -> None:
        with self._lifecycle_lock, self._lock:
            self._active_locations = [name for name in names if name in self._runtimes]
            # Release cameras held by slots that are no longer active so their
            # devices return to the pool for the new mode's active slots.
            for name, runtime in self._runtimes.items():
                if name not in self._active_locations and runtime.started:
                    self._stop_runtime(runtime)
                    runtime.actual_serial = None

    def available(self) -> bool:
        return any(backend.available() for backend in self._backends)

    def _available_backends(self) -> list[CameraBackend]:
        return [backend for backend in self._backends if backend.available()]

    def discover(self) -> dict[str, Any]:
        """Enumerate every camera visible to any installed vendor SDK."""
        usable = self._available_backends()
        if not usable:
            return {
                "available": False,
                "devices": [],
                "errors": {
                    backend.key: backend.unavailable_reason()
                    for backend in self._backends
                },
            }

        devices: list[dict[str, Any]] = []
        errors = dict(self._errors)
        vendors: dict[str, str] = {}
        for backend in usable:
            try:
                found = backend.discover()
            except Exception as exc:  # pragma: no cover - hardware specific
                # One vendor's SDK failing must not hide the other's cameras.
                errors[backend.key] = describe_exception(exc)
                continue
            for device in found:
                # A serial already claimed by an earlier backend wins; the same
                # physical camera must never appear twice in the picker.
                if device.serial in vendors:
                    continue
                vendors[device.serial] = device.vendor
                devices.append(
                    {
                        "name": device.name,
                        "serial": device.serial,
                        "vendor": device.vendor,
                    }
                )
        self._serial_vendors = vendors
        return {"available": True, "devices": devices, "errors": errors}

    def _backend_for_serial(self, serial: str) -> CameraBackend | None:
        vendor = self._serial_vendors.get(serial)
        for backend in self._backends:
            if backend.key == vendor:
                return backend
        return None

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
        with self._lifecycle_lock, self._lock:
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

        # Active slots only; an inactive slot could grab a device an active one needs.
        active = [self._runtimes[name] for name in self._active_locations]
        for runtime in active:
            if runtime.started:
                self._stop_runtime(runtime)
            # Drop sticky auto-assignments so each slot is re-resolved freshly.
            runtime.actual_serial = None

        if not self.available():
            return

        available_serials = self._available_serials()
        for runtime in active:
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
        if runtime.stream is not None:
            try:
                runtime.stream.stop()
            except Exception as exc:  # pragma: no cover - hardware specific
                self._errors[runtime.config.name] = describe_exception(exc)
        runtime.stream = None
        runtime.started = False
        runtime.depth_started = False
        runtime.depth_scale_m = 0.001
        runtime.last_frame_time = None
        runtime.fps = 0.0
        # Drop the cached frame so a stopped camera never serves a stale image.
        self._last_frames.pop(runtime.config.name, None)

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
            runtime.stream = None
            runtime.started = False
            return

        backend = self._backend_for_serial(serial)
        if backend is None:
            runtime.stream = None
            runtime.started = False
            self._errors[runtime.config.name] = (
                f"Camera serial {serial} is not detected"
            )
            return

        want_depth = bool(runtime.config.use_depth)
        try:
            stream, warning = backend.open(
                serial,
                width=runtime.config.width,
                height=runtime.config.height,
                fps=runtime.config.fps,
                want_depth=want_depth,
            )
            runtime.stream = stream
            runtime.backend = backend
            runtime.started = True
            runtime.depth_started = stream.depth_started
            runtime.depth_scale_m = stream.depth_scale_m
            runtime.last_frame_time = None
            runtime.fps = 0.0
            runtime.actual_serial = stream.actual_serial or serial
            if warning:
                # Surfaced as a status warning so recording still comes up.
                self._errors[runtime.config.name] = warning
            else:
                self._errors.pop(runtime.config.name, None)
            # A single background thread owns the pipeline and continuously
            # pulls frames into the cache. Consumers (live preview + recording)
            # read the cached frame instead of polling the pipeline themselves,
            # which previously made two readers contend for frames and made the
            # measured FPS swing wildly whenever recording started.
            # Pass stop_event/stream as args so a restart can't orphan this thread.
            runtime.stop_event = threading.Event()
            runtime.capture_thread = threading.Thread(
                target=self._acquire_loop,
                args=(runtime, stream, runtime.stop_event, serial),
                name=f"camera-acquire-{runtime.config.name}",
                daemon=True,
            )
            runtime.capture_thread.start()
        except Exception as exc:  # pragma: no cover - hardware specific
            runtime.stream = None
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
        if not self.available():
            return False

        with self._lifecycle_lock, self._lock:
            serials = [device["serial"] for device in self.discover()["devices"]]
            if not serials:
                return False

            # Only active-location slots may claim a detected camera; inactive
            # slots would otherwise starve an empty active slot of its device.
            active = [
                (name, self._runtimes[name])
                for name in self._active_locations
                if name in self._runtimes
            ]

            available = list(serials)
            desired: dict[str, str | None] = {}

            for name, runtime in active:
                serial = runtime.config.device_serial
                if serial and serial in available:
                    desired[name] = serial
                    available.remove(serial)

            changed = False
            for name, runtime in active:
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
        if not self.available():
            return {
                "available": False,
                "started": False,
                "errors": {
                    backend.key: backend.unavailable_reason()
                    for backend in self._backends
                },
            }

        with self._lifecycle_lock:
            self.ensure_default_assignment()
            with self._lock:
                detected_devices = self.discover()["devices"]
                available_serials = self._available_serials()
                if not detected_devices:
                    # Name the vendors actually searched, so an operator can tell
                    # "nothing plugged in" from "that vendor's SDK is missing".
                    labels = " or ".join(
                        backend.label for backend in self._available_backends()
                    )
                    message = f"No {labels} camera is available"
                    for name in self._resolve_camera_names(camera_names):
                        self._errors[name] = message
                    return self.status()
                for name in self._resolve_camera_names(camera_names):
                    runtime = self._runtimes[name]
                    if runtime.started:
                        continue
                    self._start_runtime(runtime, available_serials)

        return self.status()

    def stop_streams(self, camera_names: list[str] | None = None) -> dict[str, Any]:
        with self._lifecycle_lock, self._lock:
            for name in self._resolve_camera_names(camera_names):
                self._stop_runtime(self._runtimes[name])
        return self.status()

    def set_alignment_blocked_check(self, predicate: Callable[[], bool]) -> None:
        """Install the guard that forbids viewer-driven alignment."""
        self._alignment_blocked = predicate

    def acquire_depth_alignment(self, camera_names: list[str]) -> None:
        """Run depth->color alignment on these cameras until released."""
        with self._lock:
            for name in self._resolve_camera_names(camera_names):
                self._runtimes[name].align_consumers += 1

    def release_depth_alignment(self, camera_names: list[str]) -> None:
        with self._lock:
            for name in self._resolve_camera_names(camera_names):
                runtime = self._runtimes[name]
                runtime.align_consumers = max(0, runtime.align_consumers - 1)

    @staticmethod
    def _wants_alignment(runtime: CameraRuntime) -> bool:
        # A held reference or an unexpired viewer lease. Evaluated rather than
        # swept, so a lease lapses with nothing needing to call in again.
        return (
            runtime.align_consumers > 0
            or runtime.align_lease_until > time.monotonic()
        )

    def depth_alignment_active(self, camera_name: str) -> bool:
        with self._lock:
            runtime = self._runtimes.get(camera_name)
            return runtime is not None and self._wants_alignment(runtime)

    def renew_depth_alignment_lease(self, camera_name: str) -> None:
        """Keep alignment on for a poll-driven viewer with no close signal."""
        if self._alignment_blocked():
            return
        with self._lock:
            runtime = self._runtimes.get(camera_name)
            if runtime is not None:
                runtime.align_lease_until = time.monotonic() + _ALIGN_LEASE_S

    def clear_depth_alignment_leases(self) -> None:
        """Drop viewer leases now instead of waiting for them to lapse."""
        with self._lock:
            for runtime in self._runtimes.values():
                runtime.align_lease_until = 0.0

    def _is_streaming(self, runtime: CameraRuntime) -> bool:
        # Started AND delivered a frame recently (not just an open pipeline).
        return (
            runtime.started
            and runtime.last_frame_time is not None
            and time.monotonic() - runtime.last_frame_time < 2.0
        )

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available(),
            "cameras": {
                name: {
                    "configured_serial": self._runtimes[name].config.device_serial,
                    "actual_serial": self._runtimes[name].actual_serial,
                    "started": self._runtimes[name].started,
                    "streaming": self._is_streaming(self._runtimes[name]),
                    "fps": self._runtimes[name].fps,
                    "resolution": [
                        self._runtimes[name].config.width,
                        self._runtimes[name].config.height,
                    ],
                    "depth": {
                        "enabled": bool(self._runtimes[name].config.use_depth),
                        "started": self._runtimes[name].depth_started,
                        "aligned": self._wants_alignment(self._runtimes[name]),
                    },
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

    def _acquire_loop(
        self,
        runtime: CameraRuntime,
        stream: CameraStream,
        stop_event: threading.Event,
        serial: str,
    ) -> None:
        """Continuously pull frames for a single camera into the cache.

        This is the only place the stream is read, so frame delivery and the
        measured FPS reflect the camera's true production cadence regardless of
        how many consumers (live preview, recording) are reading concurrently.
        """
        name = runtime.config.name
        last_frame = time.monotonic()
        restarts = 0
        errored = False
        while not stop_event.is_set():
            try:
                # Alignment is read per frame so consumers can toggle it at
                # runtime; the backend skips the per-pixel remap when it is off.
                aligned = self._wants_alignment(runtime)
                pair = stream.read(1_000, aligned=aligned)
                if pair is None:
                    continue

                image = pair.color
                depth = None
                # Unaligned depth is never cached; it would pair the wrong
                # distance with each color pixel.
                if runtime.depth_started and aligned and pair.depth is not None:
                    raw_depth = pair.depth
                    # Convert raw device units to the uint16-mm unit LeRobot 0.6
                    # recognizes natively.
                    runtime.depth_scale_m = pair.depth_scale_m
                    scale_to_mm = pair.depth_scale_m * 1000.0
                    if abs(scale_to_mm - 1.0) < 1e-6:
                        depth = raw_depth.astype(np.uint16, copy=False)
                    else:
                        depth = np.rint(
                            raw_depth.astype(np.float32) * scale_to_mm
                        ).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
                timestamp_ms = pair.timestamp_ms
                now = time.monotonic()
                with self._lock:
                    if runtime.last_frame_time is not None:
                        delta = now - runtime.last_frame_time
                        # time.monotonic() can have coarse resolution (~16 ms on
                        # Windows), so two frames can land in the same tick with
                        # delta == 0 (or near it). Skip the FPS update for such
                        # sub-millisecond deltas and measure on the next frame
                        # rather than dividing by a tiny number.
                        if delta >= 1e-3:
                            instantaneous = 1.0 / delta
                            # Clamp the instantaneous reading to a physical ceiling
                            # before feeding the EMA. When the acquisition thread is
                            # briefly starved (e.g. while recording start builds the
                            # dataset/encoder), the RealSense SDK buffers frames and
                            # then delivers them in a burst with tiny inter-arrival
                            # deltas. Those bursts don't mean the camera sped up, so
                            # cap them at a small multiple of the configured rate to
                            # stop the displayed FPS spiking to hundreds/thousands.
                            ceiling = max(runtime.config.fps, 1) * 3.0
                            instantaneous = min(instantaneous, ceiling)
                            alpha = 0.3
                            runtime.fps = (
                                alpha * instantaneous + (1 - alpha) * runtime.fps
                            )
                            runtime.last_frame_time = now
                    else:
                        runtime.last_frame_time = now
                    runtime.frame_count += 1
                    payload: dict[str, Any] = {
                        "image": image,
                        "timestamp_ms": timestamp_ms,
                        "fps": runtime.fps,
                        "width": image.shape[1],
                        "height": image.shape[0],
                    }
                    if runtime.depth_started and aligned:
                        # Keep the previous depth map on a missing depth frame so
                        # a momentary drop never loses the tick's color frame.
                        if depth is None:
                            previous = self._last_frames.get(name)
                            if previous is not None and "depth" in previous:
                                depth = previous["depth"]
                        if depth is not None:
                            payload["depth"] = depth
                    self._last_frames[name] = payload
                    if errored:
                        errored = False
                        self._errors.pop(name, None)
                last_frame = time.monotonic()
                restarts = 0
            except Exception as exc:  # pragma: no cover - hardware specific
                # Stay silent while a camera is still connecting (never streamed);
                # only report a timeout once a streaming camera drops out.
                never_streamed = runtime.last_frame_time is None
                if (
                    not never_streamed
                    and time.monotonic() - last_frame >= SILENT_RESTART_AFTER_S
                ):
                    self._errors[name] = describe_exception(exc)
                    errored = True
                stream, last_frame, restarts = self._restart_if_silent(
                    runtime, stream, stop_event, serial, last_frame, restarts
                )
                stop_event.wait(timeout=0.05)

    def _restart_if_silent(
        self,
        runtime: CameraRuntime,
        stream: CameraStream,
        stop_event: threading.Event,
        serial: str,
        last_frame: float,
        restarts: int,
    ) -> tuple[CameraStream, float, int]:
        # Restart a silent stream, escalating to a hardware reset (a D405 on a
        # shared hub often re-wedges on a plain restart). Never gives up.
        if (
            time.monotonic() - last_frame < SILENT_RESTART_AFTER_S
            or stop_event.is_set()
        ):
            return stream, last_frame, restarts
        try:
            stream.stop()
        except Exception:  # pragma: no cover - hardware specific
            pass
        backend = runtime.backend
        if backend is None:  # pragma: no cover - a started runtime always has one
            return stream, time.monotonic(), restarts + 1
        # Every 3rd attempt escalate to a hardware reset.
        if restarts > 0 and restarts % 3 == 0:
            self._hardware_reset(runtime, stop_event)
        try:
            stream, _warning = backend.open(
                serial,
                width=runtime.config.width,
                height=runtime.config.height,
                fps=runtime.config.fps,
                want_depth=bool(runtime.config.use_depth),
            )
        except Exception as exc:  # pragma: no cover - hardware specific
            self._errors[runtime.config.name] = describe_exception(exc)
            return stream, time.monotonic(), restarts + 1
        with self._lock:
            if not stop_event.is_set():
                runtime.stream = stream
                runtime.depth_started = stream.depth_started
                runtime.depth_scale_m = stream.depth_scale_m
        return stream, time.monotonic(), restarts + 1

    def _hardware_reset(
        self, runtime: CameraRuntime, stop_event: threading.Event
    ) -> None:
        # A hardware reset drops the USB device; wait for it to re-enumerate.
        serial = runtime.actual_serial or runtime.config.device_serial
        backend = runtime.backend
        if not serial or backend is None:
            return
        if not backend.reset(serial):
            return
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not stop_event.is_set():
            if backend.is_present(serial):
                return
            stop_event.wait(0.3)

    def read_frames(
        self,
        block: bool = False,
        timeout_ms: int = 1_000,
        camera_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not self.available():
            raise RuntimeError("No camera SDK is available")

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
                    if runtime.stream is None:
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
        if runtime.stream is None or not runtime.started:
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


# The service handled only RealSense until Orbbec support landed. Kept so any
# out-of-tree caller importing the old name keeps working.
RealSenseService = CameraService
