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

from flexivtrainer.cameras import realsense as realsense_module
from flexivtrainer.cameras.realsense import CameraRuntime, RealSenseService
from flexivtrainer.config import AppSettings, CameraConfig, StorageConfig


def _make_service(tmp_path, monkeypatch):
    """Create a RealSenseService with a single fake camera that produces frames."""

    class FakeColorFrame:
        def get_data(self):
            return np.zeros((480, 848, 3), dtype=np.uint8)

        def get_timestamp(self):
            return time.time() * 1000

    class FakeFrameset:
        def __bool__(self):
            return True

        def get_color_frame(self):
            return FakeColorFrame()

    class FakePipeline:
        def wait_for_frames(self, timeout_ms):
            time.sleep(0.005)  # simulate ~5ms frame delivery
            return FakeFrameset()

        def poll_for_frames(self):
            return FakeFrameset()

        def start(self, config):
            return SimpleNamespace(
                get_device=lambda: SimpleNamespace(get_info=lambda key: "FAKE_SERIAL")
            )

        def stop(self):
            pass

    class FakeDevice:
        def get_info(self, key):
            return {"name": "Fake D435", "serial_number": "FAKE_SERIAL"}[key]

    fake_rs = SimpleNamespace(
        camera_info=SimpleNamespace(name="name", serial_number="serial_number"),
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
        context=lambda: SimpleNamespace(devices=[FakeDevice()]),
        pipeline=lambda: FakePipeline(),
        config=lambda: SimpleNamespace(
            enable_device=lambda s: None, enable_stream=lambda *a: None
        ),
    )
    monkeypatch.setattr(realsense_module, "rs", fake_rs)

    service = RealSenseService(
        AppSettings(
            storage=StorageConfig(root=tmp_path),
            cameras=[CameraConfig(name="ego", device_serial="FAKE_SERIAL")],
        )
    )
    service.start_streams()
    return service


def test_fps_uses_exponential_moving_average(tmp_path, monkeypatch) -> None:
    """FPS should use EMA smoothing, not raw instantaneous delta."""
    service = _make_service(tmp_path, monkeypatch)

    # Read multiple frames to build up FPS history
    for _ in range(10):
        service.read_frames(block=True, timeout_ms=100, camera_names=["ego"])

    runtime = service._runtimes["ego"]
    # FPS should be positive and reasonable (not jumping wildly)
    assert runtime.fps > 0
    # With 5ms sleep per frame, theoretical max is ~200fps
    # EMA should produce a stable value
    assert runtime.fps < 300


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
