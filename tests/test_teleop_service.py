from types import SimpleNamespace

from flexivtrainer.config import AppSettings, StorageConfig
from flexivtrainer.teleop.service import TeleopService


class FakeRobot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    def ExecutePrimitive(self, primitive: str, params: dict | None = None) -> None:
        self.calls.append((primitive, params))

    def primitive_states(self) -> dict[str, bool]:
        return {"reachedTarget": True}


class FakeController:
    def __init__(self, robots: tuple[FakeRobot, ...]) -> None:
        self._robots = robots

    def instances(self) -> tuple[FakeRobot, ...]:
        return self._robots


def test_reset_home_executes_home_primitive_for_all_robot_instances(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    robots = (FakeRobot(), FakeRobot(), FakeRobot(), FakeRobot())
    service._controller = FakeController(robots)

    result = service.reset_home()

    assert result == {"ok": True, "warnings": []}
    for robot in robots:
        assert robot.calls == [("Home", {})]


def test_reset_home_returns_error_when_no_robot_instances_are_exposed(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    service._controller = SimpleNamespace(instances=lambda: ())

    result = service.reset_home()

    assert result["ok"] is False
    assert "TransparentCartesianTeleopLAN.instances()" in str(result["error"])
