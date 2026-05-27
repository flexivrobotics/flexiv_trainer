from types import SimpleNamespace

from flexivtrainer.cameras import realsense as realsense_module
from flexivtrainer.cameras.realsense import RealSenseService
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
    monkeypatch.setattr(realsense_module, "rs", fake_rs)

    service = RealSenseService(AppSettings(storage=StorageConfig(root=tmp_path)))

    status = service.start_streams()

    assert start_calls == []
    assert set(status["errors"]) == {"ego", "left_wrist", "right_wrist"}
    assert all(
        "No RealSense camera is available" in message
        for message in status["errors"].values()
    )
    assert all(not camera["started"] for camera in status["cameras"].values())


def test_start_streams_fast_fails_when_configured_serial_is_missing(
    monkeypatch, tmp_path
) -> None:
    start_calls: list[str | None] = []
    fake_rs = make_fake_rs([FakeDevice("D435", "AVAILABLE")], start_calls)
    monkeypatch.setattr(realsense_module, "rs", fake_rs)

    service = RealSenseService(
        AppSettings(
            storage=StorageConfig(root=tmp_path),
            cameras=[CameraConfig(name="ego", device_serial="MISSING")],
        )
    )

    status = service.start_streams()

    assert start_calls == []
    assert "MISSING" in status["errors"]["ego"]
    assert status["cameras"]["ego"]["started"] is False
