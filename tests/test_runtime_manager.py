from types import SimpleNamespace

import pytest

from flexivtrainer.config import AppSettings, RobotSerialConfig, StorageConfig
from flexivtrainer.runtime.manager import RuntimeManager


def make_manager(tmp_path, started_camera_count: int) -> RuntimeManager:
    manager = RuntimeManager.__new__(RuntimeManager)
    manager.settings = AppSettings(storage=StorageConfig(root=tmp_path))
    manager._robot_config = RobotSerialConfig(
        local_robot_serials=["LOCAL_A", "LOCAL_B"],
        remote_robot_serials=["REMOTE_A", "REMOTE_B"],
    ).normalized()
    manager.teleop = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            available=True,
            initialized=False,
            started=False,
            error=None,
            fault=None,
        )
    )
    manager.ddk = SimpleNamespace(
        status=lambda: {
            "available": True,
            "robots": {},
            "errors": {},
        }
    )

    camera_names = ["ego", "left_wrist", "right_wrist"]
    camera_status = {
        name: {"started": index < started_camera_count}
        for index, name in enumerate(camera_names)
    }
    manager.cameras = SimpleNamespace(
        status=lambda: {
            "available": True,
            "cameras": camera_status,
            "errors": {},
        }
    )
    return manager


@pytest.mark.parametrize(
    ("started_camera_count", "expected_state", "expected_tone"),
    [
        (0, "0/3 connected", "error"),
        (1, "1/3 connected", "working"),
        (3, "3/3 connected", "ok"),
    ],
)
def test_service_summary_reports_camera_count_and_tone(
    tmp_path, started_camera_count: int, expected_state: str, expected_tone: str
) -> None:
    manager = make_manager(tmp_path, started_camera_count)

    summary = manager.service_summary()

    assert summary["cameras"]["state"] == expected_state
    assert summary["cameras"]["tone"] == expected_tone


def test_bootstrap_teleop_module_is_not_ready_when_camera_start_fails(tmp_path) -> None:
    manager = RuntimeManager.__new__(RuntimeManager)
    manager.settings = AppSettings(storage=StorageConfig(root=tmp_path))
    manager._robot_config = RobotSerialConfig(
        local_robot_serials=["LOCAL_A", "LOCAL_B"],
        remote_robot_serials=["REMOTE_A", "REMOTE_B"],
    ).normalized()
    manager.teleop = SimpleNamespace(
        initialize=lambda: SimpleNamespace(
            configured=True,
            available=True,
            initialized=True,
            started=False,
            stopped=True,
            fault=None,
            error=None,
        )
    )
    manager.ddk = SimpleNamespace(
        initialize=lambda: {
            "available": True,
            "configured": True,
            "errors": {},
            "robots": {
                "REMOTE_A": {"connected": True},
                "REMOTE_B": {"connected": True},
            },
        }
    )
    manager.cameras = SimpleNamespace(
        start_streams=lambda: {
            "available": True,
            "errors": {"ego": "No RealSense camera is available for this stream"},
            "cameras": {
                "ego": {"started": False},
                "left_wrist": {"started": False},
                "right_wrist": {"started": False},
            },
        }
    )
    manager.recording = SimpleNamespace(status=lambda: {"active": False})

    result = manager.bootstrap_teleop_module()

    assert result["ready"] is False
