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

"""CameraService behaviour when several vendor SDKs are present at once."""

import time

import numpy as np

from flexivtrainer.cameras.backends import DeviceInfo, FramePair
from flexivtrainer.cameras.service import CameraService
from flexivtrainer.config import AppSettings, CameraConfig, StorageConfig


class FakeStream:
    def __init__(self, backend, serial: str, want_depth: bool) -> None:
        self._backend = backend
        self._serial = serial
        self.depth_started = want_depth
        self.actual_serial = serial
        self.depth_scale_m = 0.001
        self.stopped = False

    def read(self, timeout_ms: int, *, aligned: bool) -> FramePair:
        time.sleep(0.005)
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = (
            np.full((480, 640), 1000, dtype=np.uint16)
            if (self.depth_started and aligned)
            else None
        )
        return FramePair(
            color=color,
            timestamp_ms=time.time() * 1000,
            depth=depth,
            depth_scale_m=self.depth_scale_m,
        )

    def stop(self) -> None:
        self.stopped = True


class FakeBackend:
    def __init__(self, key: str, label: str, serials: list[str]) -> None:
        self.key = key
        self.label = label
        self.serials = list(serials)
        self.opened: list[str] = []
        self.resets: list[str] = []

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return f"{self.label} SDK is not importable"

    def discover(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(serial=serial, name=f"{self.label} cam", vendor=self.key)
            for serial in self.serials
        ]

    def open(self, serial, *, width, height, fps, want_depth):
        if serial not in self.serials:
            raise RuntimeError(f"Camera serial {serial} is not detected")
        self.opened.append(serial)
        return FakeStream(self, serial, want_depth), None

    def reset(self, serial: str) -> bool:
        self.resets.append(serial)
        return True

    def is_present(self, serial: str) -> bool:
        return serial in self.serials


def _service(tmp_path, backends, cameras=None) -> CameraService:
    return CameraService(
        AppSettings(
            storage=StorageConfig(root=tmp_path),
            cameras=cameras
            or [CameraConfig(name="ego"), CameraConfig(name="left_wrist")],
        ),
        backends=backends,
    )


def test_discover_merges_devices_from_every_vendor(tmp_path) -> None:
    service = _service(
        tmp_path,
        [
            FakeBackend("realsense", "RealSense", ["RS_1"]),
            FakeBackend("orbbec", "Orbbec", ["ORB_1", "ORB_2"]),
        ],
    )

    result = service.discover()

    assert result["available"] is True
    assert [device["serial"] for device in result["devices"]] == [
        "RS_1",
        "ORB_1",
        "ORB_2",
    ]
    assert [device["vendor"] for device in result["devices"]] == [
        "realsense",
        "orbbec",
        "orbbec",
    ]


def test_a_failing_vendor_sdk_does_not_hide_the_other(tmp_path) -> None:
    class BrokenBackend(FakeBackend):
        def discover(self):
            raise RuntimeError("USB enumeration failed")

    service = _service(
        tmp_path,
        [
            BrokenBackend("realsense", "RealSense", ["RS_1"]),
            FakeBackend("orbbec", "Orbbec", ["ORB_1"]),
        ],
    )

    result = service.discover()

    assert [device["serial"] for device in result["devices"]] == ["ORB_1"]
    assert "USB enumeration failed" in result["errors"]["realsense"]


def test_service_is_available_when_only_orbbec_is_installed(tmp_path) -> None:
    class MissingBackend(FakeBackend):
        def available(self) -> bool:
            return False

    service = _service(
        tmp_path,
        [
            MissingBackend("realsense", "RealSense", []),
            FakeBackend("orbbec", "Orbbec", ["ORB_1"]),
        ],
    )

    assert service.available() is True
    assert service.status()["available"] is True
    assert [d["serial"] for d in service.discover()["devices"]] == ["ORB_1"]


def test_no_sdk_installed_reports_every_reason(tmp_path) -> None:
    class MissingBackend(FakeBackend):
        def available(self) -> bool:
            return False

    service = _service(
        tmp_path,
        [
            MissingBackend("realsense", "RealSense", []),
            MissingBackend("orbbec", "Orbbec", []),
        ],
    )

    result = service.start_streams()

    assert result["available"] is False
    assert set(result["errors"]) == {"realsense", "orbbec"}


def test_missing_cameras_message_names_the_searched_vendors(tmp_path) -> None:
    service = _service(
        tmp_path,
        [
            FakeBackend("realsense", "RealSense", []),
            FakeBackend("orbbec", "Orbbec", []),
        ],
    )

    status = service.start_streams()

    assert "No RealSense or Orbbec camera is available" in status["errors"]["ego"]


def test_each_slot_starts_on_the_backend_that_owns_its_serial(tmp_path) -> None:
    realsense = FakeBackend("realsense", "RealSense", ["RS_1"])
    orbbec = FakeBackend("orbbec", "Orbbec", ["ORB_1"])
    service = _service(
        tmp_path,
        [realsense, orbbec],
        cameras=[
            CameraConfig(name="ego", device_serial="ORB_1"),
            CameraConfig(name="left_wrist", device_serial="RS_1"),
        ],
    )

    service.start_streams()
    try:
        # Each serial is routed to the SDK that enumerated it, not to the first
        # backend in the list.
        assert orbbec.opened == ["ORB_1"]
        assert realsense.opened == ["RS_1"]
        cameras = service.status()["cameras"]
        assert cameras["ego"]["actual_serial"] == "ORB_1"
        assert cameras["left_wrist"]["actual_serial"] == "RS_1"
    finally:
        service.stop_streams()


def test_mixed_vendor_cameras_stream_together(tmp_path) -> None:
    service = _service(
        tmp_path,
        [
            FakeBackend("realsense", "RealSense", ["RS_1"]),
            FakeBackend("orbbec", "Orbbec", ["ORB_1"]),
        ],
        cameras=[
            CameraConfig(name="ego", device_serial="ORB_1"),
            CameraConfig(name="left_wrist", device_serial="RS_1"),
        ],
    )
    service.start_streams()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            cameras = service.status()["cameras"]
            if all(camera["streaming"] for camera in cameras.values()):
                break
            time.sleep(0.01)

        cameras = service.status()["cameras"]
        assert cameras["ego"]["streaming"] is True
        assert cameras["left_wrist"]["streaming"] is True
        assert not service.status()["errors"]

        frames = service.read_frames(block=True, timeout_ms=1_000)
        assert set(frames) == {"ego", "left_wrist"}
        assert frames["ego"]["image"].shape == (480, 640, 3)
    finally:
        service.stop_streams()


def test_auto_assignment_spans_vendors(tmp_path) -> None:
    """Unassigned slots take detected devices regardless of vendor."""
    service = _service(
        tmp_path,
        [
            FakeBackend("realsense", "RealSense", ["RS_1"]),
            FakeBackend("orbbec", "Orbbec", ["ORB_1"]),
        ],
    )

    assert service.ensure_default_assignment() is True

    assert service.configured_serials() == {"ego": "RS_1", "left_wrist": "ORB_1"}


def test_orbbec_depth_is_converted_to_uint16_millimeters(tmp_path) -> None:
    service = _service(
        tmp_path,
        [FakeBackend("orbbec", "Orbbec", ["ORB_1"])],
        cameras=[CameraConfig(name="ego", device_serial="ORB_1", use_depth=True)],
    )
    service.acquire_depth_alignment(["ego"])
    service.start_streams()
    try:
        deadline = time.monotonic() + 2.0
        frame = None
        while time.monotonic() < deadline:
            frames = service.read_frames(block=True, timeout_ms=500)
            if "ego" in frames and "depth" in frames["ego"]:
                frame = frames["ego"]
                break
            time.sleep(0.01)

        assert frame is not None, "no aligned depth frame was cached"
        depth = frame["depth"]
        assert depth.dtype == np.uint16
        # 1000 raw units at 0.001 m/unit == 1000 mm.
        assert int(depth[0][0]) == 1000
    finally:
        service.stop_streams()


def test_watchdog_resets_through_the_owning_backend(tmp_path) -> None:
    class SilentBackend(FakeBackend):
        def open(self, serial, *, width, height, fps, want_depth):
            self.opened.append(serial)

            class SilentStream(FakeStream):
                def read(self, timeout_ms: int, *, aligned: bool):
                    raise RuntimeError("Frame didn't arrive")

            return SilentStream(self, serial, want_depth), None

    backend = SilentBackend("orbbec", "Orbbec", ["ORB_1"])
    import flexivtrainer.cameras.service as camera_module

    original = camera_module.SILENT_RESTART_AFTER_S
    camera_module.SILENT_RESTART_AFTER_S = 0.02
    service = _service(
        tmp_path,
        [backend],
        cameras=[CameraConfig(name="ego", device_serial="ORB_1")],
    )
    try:
        service.start_streams()
        deadline = time.monotonic() + 3.0
        while len(backend.opened) < 6 and time.monotonic() < deadline:
            time.sleep(0.02)

        # Never gives up, and escalates to the vendor's own reset call.
        assert len(backend.opened) >= 6
        assert backend.resets and backend.resets[0] == "ORB_1"
    finally:
        service.stop_streams()
        camera_module.SILENT_RESTART_AFTER_S = original
