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

"""Tests for RealSenseService thread safety and FPS smoothing."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from flexivtrainer.cameras import realsense as realsense_module
from flexivtrainer.cameras.realsense import CameraRuntime, RealSenseService
from flexivtrainer.config import AppSettings, CameraConfig, StorageConfig


def _make_service(
    tmp_path,
    monkeypatch,
    cameras=None,
    device_serials=None,
    pipeline_factory=None,
    start=True,
):
    """Create a RealSenseService with fake cameras that produce frames."""

    class FakeColorFrame:
        def get_data(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def get_timestamp(self):
            return time.time() * 1000

    class FakeDepthFrame:
        def get_data(self):
            return np.zeros((480, 640), dtype=np.uint16)

    class FakeFrameset:
        def __bool__(self):
            return True

        def get_color_frame(self):
            return FakeColorFrame()

        def get_depth_frame(self):
            return FakeDepthFrame()

    class FakePipeline:
        def wait_for_frames(self, timeout_ms):
            time.sleep(0.005)  # simulate ~5ms frame delivery
            return FakeFrameset()

        def poll_for_frames(self):
            return FakeFrameset()

        def start(self, config):
            return _fake_profile()

        def stop(self):
            pass

    def _fake_profile():
        depth_sensor = SimpleNamespace(get_depth_scale=lambda: 0.001)
        device = SimpleNamespace(
            get_info=lambda key: "FAKE_SERIAL",
            first_depth_sensor=lambda: depth_sensor,
        )
        return SimpleNamespace(get_device=lambda: device)

    def fake_device(serial):
        return SimpleNamespace(
            get_info=lambda key: {"name": "Fake D435", "serial_number": serial}[key],
            hardware_reset=lambda: None,
        )

    device_serials = device_serials or ["FAKE_SERIAL"]
    # Stable device instances so tests can patch a device's hardware_reset and
    # have the service observe it across repeated context() calls.
    devices = [fake_device(serial) for serial in device_serials]
    fake_rs = SimpleNamespace(
        camera_info=SimpleNamespace(name="name", serial_number="serial_number"),
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        context=lambda: SimpleNamespace(devices=devices),
        pipeline=pipeline_factory or (lambda: FakePipeline()),
        config=lambda: SimpleNamespace(
            enable_device=lambda s: None, enable_stream=lambda *a: None
        ),
        align=lambda stream: SimpleNamespace(process=lambda frames: frames),
    )
    monkeypatch.setattr(realsense_module, "rs", fake_rs)

    service = RealSenseService(
        AppSettings(
            storage=StorageConfig(root=tmp_path),
            cameras=cameras
            or [CameraConfig(name="ego", device_serial="FAKE_SERIAL")],
        )
    )
    if start:
        service.start_streams()
    return service


def test_fps_uses_exponential_moving_average(tmp_path, monkeypatch) -> None:
    """FPS should use EMA smoothing, measured by the background acquisition thread."""
    service = _make_service(tmp_path, monkeypatch)

    runtime = service._runtimes["ego"]
    # The acquisition thread populates FPS on its own cadence; wait for it to
    # build up history rather than driving it from read_frames calls.
    deadline = time.monotonic() + 2.0
    while runtime.fps <= 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    # FPS should be positive and reasonable (not jumping wildly)
    assert runtime.fps > 0
    # The instantaneous reading is clamped to 3x the configured rate before the
    # EMA, so burst-delivered frames (or coarse-timer ticks on Windows) can never
    # push the displayed FPS past the ceiling. With a 30 fps config that's 90.
    ceiling = runtime.config.fps * 3.0
    assert runtime.fps <= ceiling

    # Consumers read the latest cached frame without touching the pipeline.
    frames = service.read_frames(block=True, timeout_ms=100, camera_names=["ego"])
    assert "ego" in frames


def test_concurrent_read_frames_does_not_crash(tmp_path, monkeypatch) -> None:
    """Multiple threads calling read_frames should not corrupt state."""
    service = _make_service(tmp_path, monkeypatch)
    errors = []

    def reader():
        try:
            for _ in range(20):
                frames = service.read_frames(
                    block=True, timeout_ms=100, camera_names=["ego"]
                )
                assert "ego" in frames
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent read_frames raised: {errors}"


def test_capture_frame_uses_cached_when_read_fails(tmp_path, monkeypatch) -> None:
    """capture_frame should fallback to cached frame when read_frames returns empty."""
    service = _make_service(tmp_path, monkeypatch)

    # First, read a frame to populate cache
    service.read_frames(block=True, timeout_ms=100, camera_names=["ego"])
    assert "ego" in service._last_frames

    # Now make read_frames return empty
    original_read = service.read_frames

    def empty_read(*, block, timeout_ms, camera_names):
        return {}

    service.read_frames = empty_read

    # capture_frame should use cache
    frame = service.capture_frame("ego")
    assert frame is not None
    assert "image" in frame


def test_stop_streams_clears_runtime_state(tmp_path, monkeypatch) -> None:
    """stop_streams should reset started flag and fps."""
    service = _make_service(tmp_path, monkeypatch)

    # Read a frame to get FPS
    service.read_frames(block=True, timeout_ms=100, camera_names=["ego"])
    assert service._runtimes["ego"].started is True

    service.stop_streams(["ego"])
    assert service._runtimes["ego"].started is False
    assert service._runtimes["ego"].fps == 0.0


def test_stop_streams_clears_cached_frame(tmp_path, monkeypatch) -> None:
    """A stopped camera must not serve a stale cached frame."""
    service = _make_service(tmp_path, monkeypatch)
    service.read_frames(block=True, timeout_ms=500, camera_names=["ego"])
    assert "ego" in service._last_frames

    service.stop_streams(["ego"])
    assert "ego" not in service._last_frames
    with pytest.raises(RuntimeError):
        service.capture_frame("ego")


def test_streaming_flag_requires_recent_frames(tmp_path, monkeypatch) -> None:
    service = _make_service(tmp_path, monkeypatch)
    deadline = time.monotonic() + 2.0
    while (
        not service.status()["cameras"]["ego"]["streaming"]
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert service.status()["cameras"]["ego"]["streaming"] is True

    service.stop_streams()
    camera = service.status()["cameras"]["ego"]
    assert camera["started"] is False
    assert camera["streaming"] is False

    # Started but silent (no recent frame) is not streaming.
    runtime = CameraRuntime(config=CameraConfig(name="x"))
    runtime.started = True
    runtime.last_frame_time = time.monotonic() - 10
    assert service._is_streaming(runtime) is False


def test_warming_up_timeout_is_not_reported_as_error(tmp_path, monkeypatch) -> None:
    """A wait_for_frames timeout during warm-up stays silent (no error shown)."""

    class FakeColorFrame:
        def get_data(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def get_timestamp(self):
            return time.time() * 1000

    class FakeFrameset:
        def __bool__(self):
            return True

        def get_color_frame(self):
            return FakeColorFrame()

    class WarmingPipeline:
        """Times out a few times (still connecting) then delivers frames."""

        def __init__(self):
            self._calls = 0

        def start(self, config):
            return SimpleNamespace(
                get_device=lambda: SimpleNamespace(get_info=lambda key: "FAKE_SERIAL")
            )

        def stop(self):
            pass

        def wait_for_frames(self, timeout_ms):
            self._calls += 1
            if self._calls <= 3:
                raise RuntimeError("Frame didn't arrive within 1000")
            time.sleep(0.005)
            return FakeFrameset()

    service = _make_service(
        tmp_path,
        monkeypatch,
        cameras=[
            CameraConfig(name="ego", device_serial="FAKE_SERIAL", use_depth=False)
        ],
        pipeline_factory=lambda: WarmingPipeline(),
    )
    # While warming (timeouts inside the grace window) no error is surfaced.
    deadline = time.monotonic() + 2.0
    while (
        not service.status()["cameras"]["ego"]["streaming"]
        and time.monotonic() < deadline
    ):
        assert service.status()["cameras"]["ego"]["error"] is None
        time.sleep(0.01)
    camera = service.status()["cameras"]["ego"]
    assert camera["streaming"] is True
    assert camera["error"] is None
    service.stop_streams()


def test_mid_session_dropout_surfaces_error(tmp_path, monkeypatch) -> None:
    """A camera that streamed then stops delivering DOES surface an error."""
    monkeypatch.setattr(realsense_module, "SILENT_RESTART_AFTER_S", 0.05)

    class FakeColorFrame:
        def get_data(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def get_timestamp(self):
            return time.time() * 1000

    class FakeFrameset:
        def __bool__(self):
            return True

        def get_color_frame(self):
            return FakeColorFrame()

    class DropoutPipeline:
        """Delivers a few frames, then times out forever (mid-session failure)."""

        def __init__(self):
            self._calls = 0

        def start(self, config):
            return SimpleNamespace(
                get_device=lambda: SimpleNamespace(get_info=lambda key: "FAKE_SERIAL")
            )

        def stop(self):
            pass

        def wait_for_frames(self, timeout_ms):
            self._calls += 1
            if self._calls <= 3:
                time.sleep(0.005)
                return FakeFrameset()
            raise RuntimeError("Frame didn't arrive within 1000")

    service = _make_service(
        tmp_path,
        monkeypatch,
        cameras=[
            CameraConfig(name="ego", device_serial="FAKE_SERIAL", use_depth=False)
        ],
        pipeline_factory=lambda: DropoutPipeline(),
    )
    deadline = time.monotonic() + 2.0
    while (
        service.status()["cameras"]["ego"]["error"] is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert service.status()["cameras"]["ego"]["error"]
    service.stop_streams()


def test_silent_pipeline_watchdog_retries_and_resets(tmp_path, monkeypatch) -> None:
    """A silent pipeline is retried indefinitely and escalates to hardware_reset."""
    monkeypatch.setattr(realsense_module, "SILENT_RESTART_AFTER_S", 0.02)
    starts: list[float] = []
    resets: list[str] = []

    class BrokenPipeline:
        def start(self, config):
            starts.append(time.monotonic())
            return SimpleNamespace(
                get_device=lambda: SimpleNamespace(get_info=lambda key: "FAKE_SERIAL")
            )

        def stop(self):
            pass

        def wait_for_frames(self, timeout_ms):
            raise RuntimeError("Frame didn't arrive within 1000")

    service = _make_service(
        tmp_path, monkeypatch, pipeline_factory=lambda: BrokenPipeline()
    )
    # hardware_reset lives on the device object from context(); record calls.
    fake_device = realsense_module.rs.context().devices[0]
    fake_device.hardware_reset = lambda: resets.append("reset")

    deadline = time.monotonic() + 3.0
    while len(starts) < 6 and time.monotonic() < deadline:
        time.sleep(0.02)
    # Never gives up: keeps restarting well past the old cap of 3.
    assert len(starts) >= 6
    # Escalated to at least one hardware reset (every 3rd attempt).
    assert len(resets) >= 1

    camera = service.status()["cameras"]["ego"]
    assert camera["started"] is True
    assert camera["streaming"] is False
    # A camera that has never delivered a frame is treated as "still connecting"
    # -- no timeout error is surfaced even though the watchdog keeps retrying.
    assert camera["error"] is None
    service.stop_streams()


def test_serial_churn_leaves_single_capture_thread(tmp_path, monkeypatch) -> None:
    """Concurrent config changes and starts must never orphan a capture thread."""
    service = _make_service(
        tmp_path,
        monkeypatch,
        cameras=[CameraConfig(name="churn_ego", device_serial="FAKE_SERIAL")],
    )
    errors: list[Exception] = []

    def churn():
        try:
            for _ in range(10):
                service.set_device_serials({"churn_ego": ""})
                service.set_device_serials({"churn_ego": "FAKE_SERIAL"})
        except Exception as exc:
            errors.append(exc)

    def restart():
        try:
            for _ in range(10):
                service.start_streams()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=churn), threading.Thread(target=restart)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    service.start_streams()
    time.sleep(0.3)  # let any signalled old threads drain
    alive = [
        t for t in threading.enumerate() if t.name == "camera-acquire-churn_ego"
    ]
    assert len(alive) == 1
    frames = service.read_frames(block=True, timeout_ms=500, camera_names=["churn_ego"])
    assert "churn_ego" in frames
    service.stop_streams()


def test_restart_started_cameras_skips_inactive_slots(tmp_path, monkeypatch) -> None:
    """A restart must not start an inactive slot that shares an active slot's device."""
    service = _make_service(
        tmp_path,
        monkeypatch,
        cameras=[
            CameraConfig(name="right_wrist", device_serial="SER_A"),
            CameraConfig(name="wrist", device_serial="SER_A"),
        ],
        device_serials=["SER_A"],
        start=False,
    )
    service.set_active_locations(["right_wrist"])
    service.start_streams()
    assert service._runtimes["right_wrist"].started is True
    assert service._runtimes["wrist"].started is False

    # Serial change triggers a restart cycle; the inactive slot must stay off.
    service.set_device_serials({"right_wrist": ""})
    assert service._runtimes["wrist"].started is False
    service.set_device_serials({"right_wrist": "SER_A"})
    service.start_streams()
    assert service._runtimes["right_wrist"].started is True
    assert service._runtimes["wrist"].started is False
    service.stop_streams()
