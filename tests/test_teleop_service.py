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


class FakeTeleopController:
    """Mimics the relevant TDK ``TransparentCartesianTeleopLAN`` surface.

    ``stopped`` and ``fault``/``any_fault`` are *methods* on the real
    controller, so reading them as attributes returns a (truthy) bound method.
    """

    def __init__(self) -> None:
        self._faulted = False

    def Start(self) -> None:
        pass

    def Stop(self) -> None:
        pass

    def stopped(self, index: int = 0) -> bool:
        return True

    def fault(self, index: int = 0) -> bool:
        return self._faulted

    def any_fault(self) -> bool:
        return self._faulted


def test_snapshot_reports_not_started_after_initialize_only(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    # Simulate a connected-but-not-started controller.
    service._controller = FakeTeleopController()
    service._initialized = True

    snapshot = service.snapshot()

    assert snapshot.initialized is True
    assert snapshot.started is False
    assert snapshot.stopped is True


def test_snapshot_does_not_report_spurious_fault_from_method(tmp_path) -> None:
    # Regression: the controller exposes ``fault``/``any_fault`` as methods.
    # Reading ``fault`` as an attribute used to yield a truthy bound method and
    # report a permanent fault, which kept the Start button disabled.
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    service._controller = FakeTeleopController()
    service._initialized = True

    assert service.snapshot().fault is None


def test_snapshot_reports_fault_when_any_fault_is_true(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    controller = FakeTeleopController()
    controller._faulted = True
    service._controller = controller
    service._initialized = True

    assert service.snapshot().fault is not None


def test_start_then_stop_tracks_started_flag(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    service._controller = FakeTeleopController()
    service._initialized = True

    started = service.start()
    assert started.started is True
    assert started.stopped is False

    stopped = service.stop()
    assert stopped.started is False
    assert stopped.stopped is True


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
