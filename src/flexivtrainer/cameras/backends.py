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

"""Vendor abstraction over the depth-camera SDKs.

The camera service owns discovery, slot assignment, the acquisition threads and
the silent-stream watchdog. Everything that differs between vendor SDKs is
confined to a backend:

* enumerating devices (serial + display name),
* opening a stream for one device at a requested resolution/format,
* pulling one (color, depth) pair off an open stream,
* resetting a wedged device.

Backends deliberately expose the *same* units as the RealSense path always has:
color arrives as an HxWx3 BGR ``uint8`` array and depth as an HxW array in raw
device units, paired with the scale that converts those units to meters. The
service does the uint16-millimeter conversion once, for every vendor.

A ``CameraStream`` is owned by exactly one acquisition thread. Backends do not
need their own locking; the service never reads a stream from two threads.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class DeviceInfo:
    """One camera the backend can see right now."""

    serial: str
    name: str
    # Backend key ("realsense"/"orbbec"), so a serial can be routed back to the
    # SDK that produced it when several vendors are connected at once.
    vendor: str


@dataclass
class FramePair:
    """A color frame and, when the stream carries depth, its depth map.

    ``depth`` is in raw device units; multiply by ``depth_scale_m`` for meters.
    It is None whenever the caller did not ask for aligned depth, or the device
    dropped that frame.
    """

    color: np.ndarray
    timestamp_ms: float
    depth: np.ndarray | None = None
    depth_scale_m: float = 0.001


class CameraStream(Protocol):
    """An open pipeline for a single camera."""

    @property
    def depth_started(self) -> bool:
        """Whether the stream actually came up with a depth stream."""

    @property
    def actual_serial(self) -> str | None:
        """Serial reported by the device the stream really opened."""

    @property
    def depth_scale_m(self) -> float:
        """Multiplier converting raw depth units to meters."""

    def read(self, timeout_ms: int, *, aligned: bool) -> FramePair | None:
        """Block for the next frame set.

        Returns None when the SDK produced an empty/incomplete frame set, which
        is not an error. Raises on a genuine timeout or transport failure so the
        service's watchdog can restart the stream.
        """

    def stop(self) -> None:
        """Release the device. Must tolerate being called on a dead stream."""


class CameraBackend(Protocol):
    """A vendor SDK."""

    key: str
    label: str

    def available(self) -> bool:
        """Whether the SDK imported successfully."""

    def unavailable_reason(self) -> str:
        """Human-readable reason shown when available() is False."""

    def discover(self) -> list[DeviceInfo]:
        """Enumerate connected devices. Returns [] when the SDK is missing."""

    def open(
        self,
        serial: str,
        *,
        width: int,
        height: int,
        fps: int,
        want_depth: bool,
    ) -> tuple[CameraStream, str | None]:
        """Open a stream for ``serial``.

        Returns the stream plus an optional warning (used when depth was asked
        for but only color could be started). Raises when the device cannot be
        opened at all.
        """

    def reset(self, serial: str) -> bool:
        """Hardware-reset a wedged device. False when it could not be issued."""

    def is_present(self, serial: str) -> bool:
        """Whether ``serial`` is enumerable right now (post-reset re-enumeration)."""


# --------------------------------------------------------------------------
# RealSense
# --------------------------------------------------------------------------


class _RealSenseStream:
    def __init__(
        self,
        rs: Any,
        pipeline: Any,
        config: Any,
        *,
        depth_started: bool,
        actual_serial: str | None,
        depth_scale_m: float,
    ) -> None:
        self._rs = rs
        self._pipeline = pipeline
        self._config = config
        self._depth_started = depth_started
        self._actual_serial = actual_serial
        self._depth_scale_m = depth_scale_m
        self._align = rs.align(rs.stream.color) if depth_started else None

    @property
    def depth_started(self) -> bool:
        return self._depth_started

    @property
    def actual_serial(self) -> str | None:
        return self._actual_serial

    @property
    def depth_scale_m(self) -> float:
        return self._depth_scale_m

    def read(self, timeout_ms: int, *, aligned: bool) -> FramePair | None:
        raw_frames = self._pipeline.wait_for_frames(timeout_ms)
        if not raw_frames:
            return None
        # align.process replaces the frame set; only pay for it while a consumer
        # wants aligned depth.
        align = self._align if aligned else None
        frames = align.process(raw_frames) if align is not None else raw_frames
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        color = np.asanyarray(color_frame.get_data())
        depth = None
        # Unaligned depth is never returned; it would pair the wrong distance
        # with each color pixel.
        if self._depth_started and align is not None:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                depth = np.asanyarray(depth_frame.get_data())
        return FramePair(
            color=color,
            timestamp_ms=color_frame.get_timestamp(),
            depth=depth,
            depth_scale_m=self._depth_scale_m,
        )

    def stop(self) -> None:
        self._pipeline.stop()


class RealSenseBackend:
    key = "realsense"
    label = "RealSense"

    def __init__(
        self,
        module: Any | None = None,
        *,
        module_getter: Callable[[], Any] | None = None,
    ) -> None:
        # `module_getter` lets the caller resolve the SDK lazily on every access.
        # The service uses it so tests can swap in a fake SDK after construction.
        if module_getter is not None:
            self._resolve = module_getter
        else:
            resolved = module if module is not None else _try_import("pyrealsense2")
            self._resolve = lambda: resolved

    @property
    def _rs(self) -> Any:
        return self._resolve()

    @property
    def rs(self) -> Any:
        return self._rs

    def available(self) -> bool:
        return self._rs is not None

    def unavailable_reason(self) -> str:
        return "pyrealsense2 is not importable"

    def discover(self) -> list[DeviceInfo]:
        rs = self._rs
        if rs is None:
            return []
        devices = []
        for device in rs.context().devices:
            devices.append(
                DeviceInfo(
                    serial=device.get_info(rs.camera_info.serial_number),
                    name=device.get_info(rs.camera_info.name),
                    vendor=self.key,
                )
            )
        return devices

    def open(
        self,
        serial: str,
        *,
        width: int,
        height: int,
        fps: int,
        want_depth: bool,
    ) -> tuple[CameraStream, str | None]:
        rs = self._rs
        if rs is None:
            raise RuntimeError(self.unavailable_reason())

        def _build_config(with_depth: bool) -> Any:
            config = rs.config()
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            if with_depth:
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            return config

        warning: str | None = None
        pipeline = rs.pipeline()
        depth_started = want_depth
        if want_depth:
            try:
                config = _build_config(True)
                profile = pipeline.start(config)
            except Exception as exc:  # pragma: no cover - hardware specific
                # Depth may be unsupported on this device; retry color-only so
                # recording still comes up (surfaced via the status warning).
                warning = (
                    f"Depth stream unavailable, using color only: {_describe(exc)}"
                )
                # A failed start can leave an SDK pipeline unusable; retry with a
                # fresh instance for the color-only fallback.
                pipeline = rs.pipeline()
                config = _build_config(False)
                profile = pipeline.start(config)
                depth_started = False
        else:
            config = _build_config(False)
            profile = pipeline.start(config)

        depth_scale_m = 0.001
        if depth_started:
            try:
                depth_scale_m = float(
                    profile.get_device().first_depth_sensor().get_depth_scale()
                )
            except Exception:  # pragma: no cover - device/API specific
                pass
        try:
            actual_serial = profile.get_device().get_info(rs.camera_info.serial_number)
        except Exception:  # pragma: no cover - hardware specific
            actual_serial = serial

        stream = _RealSenseStream(
            rs,
            pipeline,
            config,
            depth_started=depth_started,
            actual_serial=actual_serial,
            depth_scale_m=depth_scale_m,
        )
        return stream, warning

    def reset(self, serial: str) -> bool:
        rs = self._rs
        if rs is None:
            return False
        try:
            for device in rs.context().devices:
                if device.get_info(rs.camera_info.serial_number) == serial:
                    device.hardware_reset()
                    return True
        except Exception:  # pragma: no cover - hardware specific
            return False
        return False

    def is_present(self, serial: str) -> bool:
        try:
            return any(device.serial == serial for device in self.discover())
        except Exception:  # pragma: no cover - hardware specific
            return False


# --------------------------------------------------------------------------
# Orbbec
# --------------------------------------------------------------------------


class _OrbbecStream:
    def __init__(
        self,
        ob: Any,
        pipeline: Any,
        *,
        depth_started: bool,
        actual_serial: str | None,
        depth_scale_m: float,
    ) -> None:
        self._ob = ob
        self._pipeline = pipeline
        self._depth_started = depth_started
        self._actual_serial = actual_serial
        self._depth_scale_m = depth_scale_m
        self._align = (
            ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
            if depth_started
            else None
        )

    @property
    def depth_started(self) -> bool:
        return self._depth_started

    @property
    def actual_serial(self) -> str | None:
        return self._actual_serial

    @property
    def depth_scale_m(self) -> float:
        return self._depth_scale_m

    def read(self, timeout_ms: int, *, aligned: bool) -> FramePair | None:
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if not frames:
            return None
        align = self._align if aligned else None
        if align is not None:
            frames = align.process(frames)
            if not frames:
                return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        color = _orbbec_color_to_bgr(self._ob, color_frame)
        if color is None:
            return None

        depth = None
        depth_scale_m = self._depth_scale_m
        if self._depth_started and align is not None:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                depth = np.frombuffer(
                    depth_frame.get_data(), dtype=np.uint16
                ).reshape(depth_frame.get_height(), depth_frame.get_width())
                # The Orbbec scale is reported per frame and is in millimeters
                # per unit; the service wants meters per unit.
                try:
                    depth_scale_m = float(depth_frame.get_depth_scale()) / 1000.0
                except Exception:  # pragma: no cover - device/API specific
                    pass

        return FramePair(
            color=color,
            timestamp_ms=_orbbec_timestamp_ms(color_frame),
            depth=depth,
            depth_scale_m=depth_scale_m,
        )

    def stop(self) -> None:
        self._pipeline.stop()


def _orbbec_timestamp_ms(frame: Any) -> float:
    # Prefer the host-side timestamp; the device clock is not wall-clock aligned
    # and some firmware leaves it at 0.
    for getter in ("get_system_timestamp_us", "get_timestamp_us"):
        try:
            value = float(getattr(frame, getter)())
        except Exception:  # pragma: no cover - API varies by SDK build
            continue
        if value:
            return value / 1000.0
    try:
        return float(frame.get_timestamp())
    except Exception:  # pragma: no cover - API varies by SDK build
        return 0.0


def _orbbec_color_to_bgr(ob: Any, frame: Any) -> np.ndarray | None:
    """Decode an Orbbec color frame to a contiguous HxWx3 BGR uint8 array.

    Unlike RealSense, the Orbbec SDK will not transcode to BGR for us, so a
    device may hand back MJPG/RGB/YUY2 depending on what it supports. We request
    BGR first and only hit the other paths on devices that cannot provide it.
    """
    width = frame.get_width()
    height = frame.get_height()
    try:
        fmt = frame.get_format()
    except Exception:  # pragma: no cover - API varies by SDK build
        fmt = None
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    formats = getattr(ob, "OBFormat", None)

    def _is(name: str) -> bool:
        member = getattr(formats, name, None)
        return member is not None and fmt == member

    if _is("BGR"):
        return data.reshape(height, width, 3)
    if _is("RGB"):
        return data.reshape(height, width, 3)[:, :, ::-1].copy()
    if _is("BGRA"):
        return data.reshape(height, width, 4)[:, :, :3].copy()
    if _is("RGBA"):
        return data.reshape(height, width, 4)[:, :, 2::-1].copy()

    # Compressed and packed-YUV formats need a decoder. cv2 ships with lerobot,
    # but keep the import local so a missing OpenCV degrades to "no frame"
    # rather than breaking import of the whole camera stack.
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV is a lerobot dependency
        return None

    if _is("MJPG"):
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return decoded
    if _is("YUYV") or _is("YUY2"):
        return cv2.cvtColor(
            data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2
        )
    if _is("UYVY"):
        return cv2.cvtColor(
            data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY
        )
    if _is("I420"):
        return cv2.cvtColor(
            data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420
        )
    if _is("NV12"):
        return cv2.cvtColor(
            data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12
        )
    if _is("NV21"):
        return cv2.cvtColor(
            data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV21
        )

    # Unknown format: fall back to a plain 3-channel view when the buffer size
    # allows it, otherwise report no frame rather than reshaping garbage.
    if data.size == width * height * 3:
        return data.reshape(height, width, 3)
    return None


class OrbbecBackend:
    key = "orbbec"
    label = "Orbbec"

    def __init__(
        self,
        module: Any | None = None,
        *,
        module_getter: Callable[[], Any] | None = None,
    ) -> None:
        if module_getter is not None:
            self._resolve = module_getter
        else:
            # Gemini 305/330-series cameras need OrbbecSDK v2 (`pyorbbecsdk2`).
            # `pyorbbecsdk` is the v1 package for older devices and imports under
            # its own name, so accept either.
            resolved = (
                module
                if module is not None
                else (_try_import("pyorbbecsdk2") or _try_import("pyorbbecsdk"))
            )
            self._resolve = lambda: resolved
        self._context: Any | None = None

    @property
    def _ob(self) -> Any:
        return self._resolve()

    @property
    def ob(self) -> Any:
        return self._ob

    def available(self) -> bool:
        return self._ob is not None

    def unavailable_reason(self) -> str:
        return "pyorbbecsdk2 is not importable"

    def _get_context(self) -> Any:
        # Enumerating through one long-lived Context keeps device handles stable;
        # rebuilding it per discover() call makes the SDK re-scan USB every time.
        if self._context is None:
            self._context = self._ob.Context()
        return self._context

    def discover(self) -> list[DeviceInfo]:
        ob = self._ob
        if ob is None:
            return []
        device_list = self._get_context().query_devices()
        devices = []
        for index in range(_orbbec_device_count(device_list)):
            serial = _orbbec_list_serial(device_list, index)
            if not serial:
                continue
            devices.append(
                DeviceInfo(
                    serial=serial,
                    name=_orbbec_list_name(device_list, index) or "Orbbec camera",
                    vendor=self.key,
                )
            )
        return devices

    def _find_device(self, serial: str) -> Any | None:
        device_list = self._get_context().query_devices()
        for index in range(_orbbec_device_count(device_list)):
            if _orbbec_list_serial(device_list, index) == serial:
                return device_list.get_device_by_index(index)
        return None

    def open(
        self,
        serial: str,
        *,
        width: int,
        height: int,
        fps: int,
        want_depth: bool,
    ) -> tuple[CameraStream, str | None]:
        ob = self._ob
        if ob is None:
            raise RuntimeError(self.unavailable_reason())

        device = self._find_device(serial)
        if device is None:
            raise RuntimeError(f"Camera serial {serial} is not detected")

        pipeline = ob.Pipeline(device)
        config = ob.Config()

        color_profile = _orbbec_pick_profile(
            ob,
            pipeline,
            ob.OBSensorType.COLOR_SENSOR,
            width=width,
            height=height,
            fps=fps,
            preferred_formats=("BGR", "RGB", "MJPG"),
        )
        if color_profile is None:
            raise RuntimeError("No color stream profile is available")
        config.enable_stream(color_profile)

        warning: str | None = None
        depth_started = False
        if want_depth:
            try:
                depth_profile = _orbbec_pick_profile(
                    ob,
                    pipeline,
                    ob.OBSensorType.DEPTH_SENSOR,
                    width=width,
                    height=height,
                    fps=fps,
                    preferred_formats=("Y16",),
                )
                if depth_profile is None:
                    raise RuntimeError("no depth profile matches the requested mode")
                config.enable_stream(depth_profile)
                depth_started = True
            except Exception as exc:  # pragma: no cover - hardware specific
                warning = (
                    f"Depth stream unavailable, using color only: {_describe(exc)}"
                )
                config = ob.Config()
                config.enable_stream(color_profile)
                depth_started = False

        try:
            pipeline.start(config)
        except Exception as exc:
            if not depth_started:
                raise
            # Same fallback as RealSense: a device that advertises depth may
            # still refuse the combined mode, so retry color-only.
            warning = f"Depth stream unavailable, using color only: {_describe(exc)}"
            pipeline = ob.Pipeline(device)
            config = ob.Config()
            config.enable_stream(color_profile)
            pipeline.start(config)
            depth_started = False

        if depth_started:
            # Frame sync keeps the color/depth pair from drifting apart before
            # alignment. Not fatal when the device does not implement it.
            try:
                pipeline.enable_frame_sync()
            except Exception:  # pragma: no cover - device specific
                pass

        stream = _OrbbecStream(
            ob,
            pipeline,
            depth_started=depth_started,
            actual_serial=_orbbec_device_serial(device) or serial,
            # Gemini depth is millimeters per unit by default; the first frame
            # refines this from the device's own reported scale.
            depth_scale_m=0.001,
        )
        return stream, warning

    def reset(self, serial: str) -> bool:
        if self._ob is None:
            return False
        try:
            device = self._find_device(serial)
            if device is None:
                return False
            # `reboot` is the v2 spelling; older builds expose `hard_reset`.
            reset = getattr(device, "reboot", None) or getattr(
                device, "hard_reset", None
            )
            if reset is None:
                return False
            reset()
        except Exception:  # pragma: no cover - hardware specific
            return False
        # The rebooted device drops off the bus, so the cached Context is stale.
        self._context = None
        return True

    def is_present(self, serial: str) -> bool:
        try:
            return any(device.serial == serial for device in self.discover())
        except Exception:  # pragma: no cover - hardware specific
            return False


def _orbbec_device_count(device_list: Any) -> int:
    try:
        return int(device_list.get_count())
    except Exception:  # pragma: no cover - API varies by SDK build
        try:
            return len(device_list)
        except Exception:
            return 0


def _orbbec_list_serial(device_list: Any, index: int) -> str | None:
    # The device list can answer directly (cheap, no device handle); fall back to
    # opening the device only when it cannot.
    try:
        return str(device_list.get_device_serial_number_by_index(index))
    except Exception:  # pragma: no cover - API varies by SDK build
        pass
    try:
        return _orbbec_device_serial(device_list.get_device_by_index(index))
    except Exception:  # pragma: no cover - API varies by SDK build
        return None


def _orbbec_list_name(device_list: Any, index: int) -> str | None:
    for getter in ("get_device_name_by_index", "get_device_by_index_name"):
        try:
            return str(getattr(device_list, getter)(index))
        except Exception:  # pragma: no cover - API varies by SDK build
            continue
    try:
        info = device_list.get_device_by_index(index).get_device_info()
        return str(info.get_name())
    except Exception:  # pragma: no cover - API varies by SDK build
        return None


def _orbbec_device_serial(device: Any) -> str | None:
    try:
        return str(device.get_device_info().get_serial_number())
    except Exception:  # pragma: no cover - API varies by SDK build
        pass
    try:
        return str(device.get_serial_number())
    except Exception:  # pragma: no cover - API varies by SDK build
        return None


def _orbbec_pick_profile(
    ob: Any,
    pipeline: Any,
    sensor_type: Any,
    *,
    width: int,
    height: int,
    fps: int,
    preferred_formats: tuple[str, ...],
) -> Any:
    """Pick the closest stream profile the sensor actually supports.

    The Orbbec SDK raises rather than negotiating when an exact
    width/height/format/fps combination is unavailable, so try the preferred
    formats in order and fall back to the sensor's default profile. Requesting
    height 0 lets the SDK match on width alone, as the vendor examples do.
    """
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    formats = getattr(ob, "OBFormat", None)
    for format_name in preferred_formats:
        fmt = getattr(formats, format_name, None)
        if fmt is None:
            continue
        for requested_height in (height, 0):
            try:
                profile = profile_list.get_video_stream_profile(
                    width, requested_height, fmt, fps
                )
            except Exception:
                continue
            if profile is not None:
                return profile
    try:
        return profile_list.get_default_video_stream_profile()
    except Exception:  # pragma: no cover - device specific
        return None


# --------------------------------------------------------------------------


def _try_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        # Dependency availability is environment-specific.
        return None
    except Exception:  # pragma: no cover - a broken install must not crash boot
        return None


def _describe(exc: Exception) -> str:
    from flexivtrainer.observability import describe_exception

    return describe_exception(exc)
