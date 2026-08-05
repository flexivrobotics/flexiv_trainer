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

"""Orbbec backend tests, driven by a fake pyorbbecsdk2 module.

The fakes mirror the shapes the real SDK returns (notably: depth as a flat
uint16 buffer, and a per-frame depth scale expressed in millimeters per unit).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from flexivtrainer.cameras.backends import OrbbecBackend

WIDTH = 640
HEIGHT = 480


class FakeFormats:
    BGR = "bgr"
    RGB = "rgb"
    MJPG = "mjpg"
    Y16 = "y16"


class FakeColorFrame:
    def __init__(self, data: np.ndarray, fmt: str = FakeFormats.BGR) -> None:
        self._data = data
        self._fmt = fmt

    def get_data(self):
        return self._data.tobytes()

    def get_format(self):
        return self._fmt

    def get_width(self):
        return WIDTH

    def get_height(self):
        return HEIGHT

    def get_system_timestamp_us(self):
        return 1_234_000


class FakeDepthFrame:
    def __init__(self, data: np.ndarray, scale_mm: float = 1.0) -> None:
        self._data = data
        self._scale_mm = scale_mm

    def get_data(self):
        return self._data.tobytes()

    def get_width(self):
        return WIDTH

    def get_height(self):
        return HEIGHT

    def get_depth_scale(self):
        return self._scale_mm


class FakeFrameSet:
    def __init__(self, color, depth=None) -> None:
        self._color = color
        self._depth = depth

    def __bool__(self):
        return True

    def get_color_frame(self):
        return self._color

    def get_depth_frame(self):
        return self._depth


class FakeProfileList:
    def __init__(self, supported: set[tuple[int, int, str, int]] | None = None) -> None:
        self._supported = supported
        self.requests: list[tuple] = []

    def get_video_stream_profile(self, width, height, fmt, fps):
        self.requests.append((width, height, fmt, fps))
        if self._supported is not None and (width, height, fmt, fps) not in (
            self._supported
        ):
            raise RuntimeError("unsupported profile")
        return SimpleNamespace(width=width, height=height, fmt=fmt, fps=fps)

    def get_default_video_stream_profile(self):
        return SimpleNamespace(default=True)


class FakePipeline:
    def __init__(self, device, *, frames=None, depth_start_fails=False) -> None:
        self.device = device
        self.started_config = None
        self.stopped = False
        self.frame_sync = False
        self._frames = frames
        self._depth_start_fails = depth_start_fails
        self.color_profiles = FakeProfileList()
        self.depth_profiles = FakeProfileList()

    def get_stream_profile_list(self, sensor_type):
        return (
            self.color_profiles
            if sensor_type == "color_sensor"
            else self.depth_profiles
        )

    def enable_frame_sync(self):
        self.frame_sync = True

    def start(self, config):
        if self._depth_start_fails and len(config.streams) > 1:
            raise RuntimeError("device refuses the combined depth mode")
        self.started_config = config

    def stop(self):
        self.stopped = True

    def wait_for_frames(self, timeout_ms):
        return self._frames


class FakeConfig:
    def __init__(self) -> None:
        self.streams: list = []

    def enable_stream(self, profile):
        self.streams.append(profile)


class FakeDeviceList:
    def __init__(self, devices) -> None:
        self._devices = devices

    def get_count(self):
        return len(self._devices)

    def get_device_serial_number_by_index(self, index):
        return self._devices[index].serial

    def get_device_name_by_index(self, index):
        return self._devices[index].name

    def get_device_by_index(self, index):
        return self._devices[index]


class FakeDevice:
    def __init__(self, serial: str, name: str = "Orbbec Gemini 305") -> None:
        self.serial = serial
        self.name = name
        self.reboots = 0

    def get_device_info(self):
        return SimpleNamespace(
            get_serial_number=lambda: self.serial,
            get_name=lambda: self.name,
        )

    def reboot(self):
        self.reboots += 1


def make_fake_ob(devices, *, frames=None, depth_start_fails=False):
    pipelines: list[FakePipeline] = []

    def _pipeline(device):
        pipeline = FakePipeline(
            device, frames=frames, depth_start_fails=depth_start_fails
        )
        pipelines.append(pipeline)
        return pipeline

    module = SimpleNamespace(
        Context=lambda: SimpleNamespace(
            query_devices=lambda: FakeDeviceList(devices)
        ),
        Pipeline=_pipeline,
        Config=FakeConfig,
        OBFormat=FakeFormats,
        OBSensorType=SimpleNamespace(
            COLOR_SENSOR="color_sensor", DEPTH_SENSOR="depth_sensor"
        ),
        OBStreamType=SimpleNamespace(COLOR_STREAM="color_stream"),
        AlignFilter=lambda align_to_stream: SimpleNamespace(
            process=lambda frames: frames
        ),
    )
    module.pipelines = pipelines
    return module


def test_discover_reports_serials_and_vendor() -> None:
    ob = make_fake_ob([FakeDevice("ORB_A"), FakeDevice("ORB_B", "Gemini 335")])
    backend = OrbbecBackend(ob)

    devices = backend.discover()

    assert [device.serial for device in devices] == ["ORB_A", "ORB_B"]
    assert [device.vendor for device in devices] == ["orbbec", "orbbec"]
    assert devices[1].name == "Gemini 335"


def test_discover_is_empty_without_the_sdk() -> None:
    backend = OrbbecBackend(module_getter=lambda: None)

    assert backend.available() is False
    assert backend.discover() == []


def test_open_enables_color_and_depth_and_reports_serial() -> None:
    ob = make_fake_ob([FakeDevice("ORB_A")])
    backend = OrbbecBackend(ob)

    stream, warning = backend.open(
        "ORB_A", width=WIDTH, height=HEIGHT, fps=30, want_depth=True
    )

    assert warning is None
    assert stream.depth_started is True
    assert stream.actual_serial == "ORB_A"
    pipeline = ob.pipelines[0]
    assert len(pipeline.started_config.streams) == 2
    # Frame sync keeps the color/depth pair aligned before the align filter runs.
    assert pipeline.frame_sync is True
    # BGR is requested first so no color conversion is needed on the hot path.
    assert pipeline.color_profiles.requests[0][2] == FakeFormats.BGR


def test_open_without_depth_enables_color_only() -> None:
    ob = make_fake_ob([FakeDevice("ORB_A")])
    backend = OrbbecBackend(ob)

    stream, warning = backend.open(
        "ORB_A", width=WIDTH, height=HEIGHT, fps=30, want_depth=False
    )

    assert warning is None
    assert stream.depth_started is False
    assert len(ob.pipelines[0].started_config.streams) == 1


def test_open_falls_back_to_color_when_depth_start_fails() -> None:
    ob = make_fake_ob([FakeDevice("ORB_A")], depth_start_fails=True)
    backend = OrbbecBackend(ob)

    stream, warning = backend.open(
        "ORB_A", width=WIDTH, height=HEIGHT, fps=30, want_depth=True
    )

    # Recording must still come up; the operator sees the reason as a warning.
    assert stream.depth_started is False
    assert warning is not None
    assert "using color only" in warning
    assert len(ob.pipelines[-1].started_config.streams) == 1


def test_open_raises_for_an_undetected_serial() -> None:
    backend = OrbbecBackend(make_fake_ob([FakeDevice("ORB_A")]))

    with pytest.raises(RuntimeError, match="MISSING"):
        backend.open("MISSING", width=WIDTH, height=HEIGHT, fps=30, want_depth=True)


def test_read_returns_bgr_color_and_raw_depth() -> None:
    color = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth = np.full((HEIGHT, WIDTH), 1500, dtype=np.uint16)
    frames = FakeFrameSet(FakeColorFrame(color), FakeDepthFrame(depth))
    backend = OrbbecBackend(make_fake_ob([FakeDevice("ORB_A")], frames=frames))

    stream, _ = backend.open(
        "ORB_A", width=WIDTH, height=HEIGHT, fps=30, want_depth=True
    )
    pair = stream.read(1_000, aligned=True)

    assert np.array_equal(pair.color, color)
    assert pair.depth is not None
    assert np.array_equal(pair.depth, depth)
    # Millimeters per unit on the wire becomes meters per unit for the service.
    assert pair.depth_scale_m == pytest.approx(0.001)
    assert pair.timestamp_ms == pytest.approx(1234.0)


def test_read_skips_depth_when_alignment_is_off() -> None:
    color = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth = np.full((HEIGHT, WIDTH), 900, dtype=np.uint16)
    frames = FakeFrameSet(FakeColorFrame(color), FakeDepthFrame(depth))
    backend = OrbbecBackend(make_fake_ob([FakeDevice("ORB_A")], frames=frames))

    stream, _ = backend.open(
        "ORB_A", width=WIDTH, height=HEIGHT, fps=30, want_depth=True
    )

    # Unaligned depth would pair the wrong distance with each color pixel.
    assert stream.read(1_000, aligned=False).depth is None


def test_read_converts_rgb_frames_to_bgr() -> None:
    rgb = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    frames = FakeFrameSet(FakeColorFrame(rgb, FakeFormats.RGB))
    backend = OrbbecBackend(make_fake_ob([FakeDevice("ORB_A")], frames=frames))

    stream, _ = backend.open(
        "ORB_A", width=WIDTH, height=HEIGHT, fps=30, want_depth=False
    )
    pair = stream.read(1_000, aligned=False)

    assert np.array_equal(pair.color, rgb[:, :, ::-1])


def test_read_returns_none_on_an_empty_frame_set() -> None:
    backend = OrbbecBackend(make_fake_ob([FakeDevice("ORB_A")], frames=None))

    stream, _ = backend.open(
        "ORB_A", width=WIDTH, height=HEIGHT, fps=30, want_depth=False
    )

    # An empty frame set is a skip, not an error, so the watchdog stays quiet.
    assert stream.read(1_000, aligned=False) is None


def test_reset_reboots_the_matching_device() -> None:
    device = FakeDevice("ORB_A")
    backend = OrbbecBackend(make_fake_ob([device, FakeDevice("ORB_B")]))

    assert backend.reset("ORB_A") is True
    assert device.reboots == 1
    assert backend.reset("UNKNOWN") is False


def test_reset_drops_the_cached_context() -> None:
    """A rebooted device re-enumerates, so the cached Context must be rebuilt."""
    device = FakeDevice("ORB_A")
    ob = make_fake_ob([device])
    contexts: list[object] = []
    original_context = ob.Context

    def _counting_context():
        context = original_context()
        contexts.append(context)
        return context

    ob.Context = _counting_context
    backend = OrbbecBackend(ob)

    backend.discover()
    backend.discover()
    # The Context is cached across calls rather than re-scanning USB each time.
    assert len(contexts) == 1

    backend.reset("ORB_A")
    backend.discover()
    assert len(contexts) == 2


def test_is_present_tracks_enumeration() -> None:
    backend = OrbbecBackend(make_fake_ob([FakeDevice("ORB_A")]))

    assert backend.is_present("ORB_A") is True
    assert backend.is_present("ORB_B") is False


def test_profile_selection_falls_back_to_the_sensor_default() -> None:
    """An unsupported resolution must not fail the whole camera."""
    ob = make_fake_ob([FakeDevice("ORB_A")])
    backend = OrbbecBackend(ob)
    # Nothing matches, so every explicit request raises.
    original = ob.Pipeline

    def _pipeline(device):
        pipeline = original(device)
        pipeline.color_profiles = FakeProfileList(supported=set())
        pipeline.depth_profiles = FakeProfileList(supported=set())
        return pipeline

    ob.Pipeline = _pipeline

    stream, _ = backend.open(
        "ORB_A", width=1234, height=567, fps=99, want_depth=False
    )

    assert stream.actual_serial == "ORB_A"
