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

from types import SimpleNamespace

import numpy as np

from flexivtrainer.cameras import service as camera_module
from flexivtrainer.cameras.service import CameraService
from flexivtrainer.config import AppSettings, CameraConfig, StorageConfig


def make_fake_rs(devices, start_calls: list[str | None]):
    class FakeConfig:
        def __init__(self) -> None:
            self.device_serial: str | None = None

        def enable_device(self, serial: str) -> None:
            self.device_serial = serial

        def enable_stream(self, *args) -> None:
            return None

    class FakePipeline:
        def start(self, config: FakeConfig) -> None:
            start_calls.append(config.device_serial)
            raise AssertionError("pipeline.start should not be called")

        def stop(self) -> None:
            return None

    return SimpleNamespace(
        camera_info=SimpleNamespace(name="name", serial_number="serial_number"),
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
        context=lambda: SimpleNamespace(devices=devices),
        pipeline=lambda: FakePipeline(),
        config=lambda: FakeConfig(),
    )


class FakeDevice:
    def __init__(self, name: str, serial: str) -> None:
        self._info = {
            "name": name,
            "serial_number": serial,
        }

    def get_info(self, key: str) -> str:
        return self._info[key]


def test_start_streams_fast_fails_when_no_cameras_are_detected(
    monkeypatch, tmp_path
) -> None:
    start_calls: list[str | None] = []
    fake_rs = make_fake_rs([], start_calls)
    monkeypatch.setattr(camera_module, "rs", fake_rs)

    service = CameraService(AppSettings(storage=StorageConfig(root=tmp_path)))

    status = service.start_streams()

    assert start_calls == []
    assert set(status["errors"]) == {"ego", "left_wrist", "right_wrist", "wrist"}
    # The message names whichever vendors were actually searched, so it depends
    # on which SDKs are installed in this environment ("No RealSense camera is
    # available" vs "No RealSense or Orbbec camera is available").
    assert all(
        message.startswith("No ") and message.endswith("camera is available")
        for message in status["errors"].values()
    )
    assert all(not camera["started"] for camera in status["cameras"].values())


def test_start_streams_fast_fails_when_configured_serial_is_missing(
    monkeypatch, tmp_path
) -> None:
    start_calls: list[str | None] = []
    fake_rs = make_fake_rs([FakeDevice("D435", "AVAILABLE")], start_calls)
    monkeypatch.setattr(camera_module, "rs", fake_rs)

    service = CameraService(
        AppSettings(
            storage=StorageConfig(root=tmp_path),
            cameras=[CameraConfig(name="ego", device_serial="MISSING")],
        )
    )

    status = service.start_streams()

    assert start_calls == []
    assert "MISSING" in status["errors"]["ego"]
    assert status["cameras"]["ego"]["started"] is False


def test_ensure_default_assignment_fills_unassigned_slots(
    monkeypatch, tmp_path
) -> None:
    start_calls: list[str | None] = []
    fake_rs = make_fake_rs(
        [
            FakeDevice("D435", "SERIAL_A"),
            FakeDevice("D435", "SERIAL_B"),
            FakeDevice("D435", "SERIAL_C"),
        ],
        start_calls,
    )
    monkeypatch.setattr(camera_module, "rs", fake_rs)

    service = CameraService(
        AppSettings(
            storage=StorageConfig(root=tmp_path),
            cameras=[
                CameraConfig(name="ego", device_serial="SERIAL_A"),
                CameraConfig(name="left_wrist", device_serial="SERIAL_B"),
                CameraConfig(name="right_wrist"),
            ],
        )
    )

    changed = service.ensure_default_assignment()

    assert changed is True
    assert service.configured_serials() == {
        "ego": "SERIAL_A",
        "left_wrist": "SERIAL_B",
        "right_wrist": "SERIAL_C",
    }


def test_ensure_default_assignment_replaces_stale_serials(
    monkeypatch, tmp_path
) -> None:
    start_calls: list[str | None] = []
    fake_rs = make_fake_rs(
        [
            FakeDevice("D435", "SERIAL_B"),
            FakeDevice("D435", "SERIAL_C"),
        ],
        start_calls,
    )
    monkeypatch.setattr(camera_module, "rs", fake_rs)

    service = CameraService(
        AppSettings(
            storage=StorageConfig(root=tmp_path),
            cameras=[
                CameraConfig(name="ego"),
                CameraConfig(name="left_wrist"),
                CameraConfig(name="right_wrist"),
            ],
        )
    )
    service.set_device_serials({"ego": "SERIAL_A"}, manual=False)

    changed = service.ensure_default_assignment()

    assert changed is True
    assert service.configured_serials() == {
        "ego": "SERIAL_B",
        "left_wrist": "SERIAL_C",
        "right_wrist": None,
    }


def test_capture_frame_reads_only_requested_camera(tmp_path) -> None:
    service = CameraService(AppSettings(storage=StorageConfig(root=tmp_path)))
    payload = {
        "image": np.zeros((2, 3, 3), dtype=np.uint8),
        "timestamp_ms": 12.3,
        "fps": 30.0,
        "width": 3,
        "height": 2,
    }
    service._runtimes["ego"].started = True
    service._runtimes["ego"].stream = object()

    def fake_read_frames(*, block, timeout_ms, camera_names):
        assert block is True
        assert timeout_ms == 350
        assert camera_names == ["ego"]
        return {"ego": payload}

    service.read_frames = fake_read_frames

    frame = service.capture_frame("ego")

    assert frame is payload


def test_capture_frame_falls_back_to_cached_frame(tmp_path) -> None:
    service = CameraService(AppSettings(storage=StorageConfig(root=tmp_path)))
    cached_payload = {
        "image": np.zeros((2, 2, 3), dtype=np.uint8),
        "timestamp_ms": 1.0,
        "fps": 0.0,
        "width": 2,
        "height": 2,
    }
    service._runtimes["ego"].started = True
    service._runtimes["ego"].stream = object()
    service._last_frames["ego"] = cached_payload
    service.read_frames = lambda *, block, timeout_ms, camera_names: {}

    frame = service.capture_frame("ego")

    assert frame is cached_payload


def test_set_active_locations_releases_now_inactive_cameras(tmp_path) -> None:
    # A camera started for a slot that is no longer active (e.g. the single-arm
    # "wrist" slot after switching to dual) must be released so its device
    # returns to the pool for the new mode's active slots.
    service = CameraService(
        AppSettings(
            storage=StorageConfig(root=tmp_path),
            cameras=[
                CameraConfig(name="ego"),
                CameraConfig(name="left_wrist"),
                CameraConfig(name="wrist"),
            ],
        )
    )
    service._runtimes["wrist"].started = True
    service._runtimes["wrist"].stream = SimpleNamespace(stop=lambda: None)

    service.set_active_locations(["ego", "left_wrist"])

    assert service._runtimes["wrist"].started is False
    assert service._runtimes["wrist"].stream is None


def _alignment_service(tmp_path) -> CameraService:
    return CameraService(
        AppSettings(
            storage=StorageConfig(root=tmp_path),
            cameras=[CameraConfig(name="ego"), CameraConfig(name="left_wrist")],
        )
    )


def test_depth_alignment_is_reference_counted(tmp_path) -> None:
    service = _alignment_service(tmp_path)
    assert service.depth_alignment_active("ego") is False

    # Recording and the depth preview can overlap, so the second release is
    # what must turn alignment off -- not the first.
    service.acquire_depth_alignment(["ego"])
    service.acquire_depth_alignment(["ego"])
    service.release_depth_alignment(["ego"])
    assert service.depth_alignment_active("ego") is True

    service.release_depth_alignment(["ego"])
    assert service.depth_alignment_active("ego") is False


def test_release_depth_alignment_never_goes_negative(tmp_path) -> None:
    service = _alignment_service(tmp_path)
    service.release_depth_alignment(["ego"])
    service.acquire_depth_alignment(["ego"])

    assert service.depth_alignment_active("ego") is True


def test_depth_alignment_only_affects_named_cameras(tmp_path) -> None:
    service = _alignment_service(tmp_path)
    service.acquire_depth_alignment(["ego"])

    assert service.depth_alignment_active("left_wrist") is False


def test_depth_alignment_survives_a_camera_restart(tmp_path) -> None:
    # The silent-camera watchdog can restart a pipeline mid-recording; losing
    # the reference there would silently record unaligned depth.
    service = _alignment_service(tmp_path)
    service.acquire_depth_alignment(["ego"])
    runtime = service._runtimes["ego"]
    runtime.started = True
    runtime.stream = SimpleNamespace(stop=lambda: None)

    with service._lock:
        service._stop_runtime(runtime)

    assert service.depth_alignment_active("ego") is True


def test_depth_alignment_lease_expires_with_no_further_calls(
    tmp_path, monkeypatch
) -> None:
    # Unchecking "Visualize depth" stops all polling, so nothing calls back in.
    # The lease must lapse on evaluation, not on the next renew.
    service = _alignment_service(tmp_path)
    now = [1000.0]
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: now[0])

    service.renew_depth_alignment_lease("ego")
    assert service.depth_alignment_active("ego") is True

    now[0] += camera_module._ALIGN_LEASE_S + 0.1

    assert service.depth_alignment_active("ego") is False


def test_depth_alignment_lease_does_not_consume_a_reference(
    tmp_path, monkeypatch
) -> None:
    service = _alignment_service(tmp_path)
    now = [1000.0]
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: now[0])

    for _ in range(5):
        now[0] += 0.1
        service.renew_depth_alignment_lease("ego")

    # The viewer lease is a deadline, so it never touches the refcount that
    # recording relies on.
    assert service._runtimes["ego"].align_consumers == 0
    assert service.depth_alignment_active("ego") is True


def test_expiring_lease_does_not_cancel_a_recording_hold(
    tmp_path, monkeypatch
) -> None:
    service = _alignment_service(tmp_path)
    now = [1000.0]
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: now[0])

    service.renew_depth_alignment_lease("ego")
    service.acquire_depth_alignment(["ego"])
    now[0] += camera_module._ALIGN_LEASE_S + 1.0

    # A long recording gets no preview polls; its hold must outlive the lease.
    assert service.depth_alignment_active("ego") is True
    service.release_depth_alignment(["ego"])
    assert service.depth_alignment_active("ego") is False


def test_clear_depth_alignment_leases_drops_viewers_but_not_holders(
    tmp_path, monkeypatch
) -> None:
    service = _alignment_service(tmp_path)
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: 1000.0)

    service.renew_depth_alignment_lease("ego")
    service.renew_depth_alignment_lease("left_wrist")
    service.acquire_depth_alignment(["left_wrist"])

    service.clear_depth_alignment_leases()

    assert service.depth_alignment_active("ego") is False
    assert service.depth_alignment_active("left_wrist") is True


def test_alignment_blocked_check_refuses_viewer_leases(tmp_path) -> None:
    # A rollout must not be starved by a stale browser tab or a direct curl, so
    # the refusal lives in the service rather than the frontend.
    service = _alignment_service(tmp_path)
    service.set_alignment_blocked_check(lambda: True)

    service.renew_depth_alignment_lease("ego")
    assert service.depth_alignment_active("ego") is False

    # Recording stays allowed: it cannot overlap a rollout, and failing it
    # silently would be worse than the preflight error that already exists.
    service.acquire_depth_alignment(["ego"])
    assert service.depth_alignment_active("ego") is True

