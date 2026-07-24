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

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from flexivtrainer.rollout.gripper_executor import GripperExecutor


class FakeRobot:
    def __init__(self, mode: str = "IDLE") -> None:
        self.current_mode = mode

    def mode(self) -> str:
        return self.current_mode


class FakeGripper:
    def __init__(
        self,
        robot: FakeRobot,
        events: list[tuple[str, str]],
        move_started: threading.Event | None = None,
        move_release: threading.Event | None = None,
    ) -> None:
        self.robot = robot
        self.events = events
        self.moves: list[tuple[float, float, float]] = []
        self.move_started = move_started
        self.move_release = move_release

    def Enable(self, model: str) -> None:
        self.events.append(("enable", model))

    def Init(self) -> None:
        self.events.append(("init", ""))

    def params(self) -> SimpleNamespace:
        self.events.append(("params", ""))
        return SimpleNamespace(
            min_width=0.01,
            max_width=0.09,
            min_vel=0.01,
            max_vel=0.4,
            min_force=5.0,
            max_force=12.0,
        )

    def states(self) -> SimpleNamespace:
        return SimpleNamespace(width=0.04, force=-2.5)

    def Move(self, width: float, velocity: float, force: float) -> None:
        self.moves.append((width, velocity, force))
        if self.move_started is not None:
            self.move_started.set()
        if self.move_release is not None:
            assert self.move_release.wait(timeout=1.0)


class FakeTool:
    def __init__(self, robot: FakeRobot, events: list[tuple[str, str]]) -> None:
        self.robot = robot
        self.events = events

    def Switch(self, model: str) -> None:
        self.events.append(("switch", model))


def _controller(
    *,
    robot: FakeRobot | None = None,
    sides: list[str] | None = None,
    configs: dict | None = None,
    controlled_sides: list[str] | None = None,
    gripper_factory=None,
    wait=None,
    clock=None,
    target_source=None,
    failure_event=None,
) -> tuple[GripperExecutor, list[FakeGripper], list[tuple[str, str]]]:
    robot = robot or FakeRobot()
    sides = sides or ["left_arm"]
    controlled_sides = controlled_sides or ["left_arm"]
    configs = configs or {
        "left_arm": {"follower": "gripper", "gripper_model": "Flexiv-GN01"}
    }
    events: list[tuple[str, str]] = []
    grippers: list[FakeGripper] = []

    def make_gripper(value):
        gripper = (
            gripper_factory(value, events)
            if gripper_factory is not None
            else FakeGripper(value, events)
        )
        grippers.append(gripper)
        return gripper

    kwargs = {}
    if wait is not None:
        kwargs["wait"] = wait
    if clock is not None:
        kwargs["clock"] = clock
    controller = GripperExecutor(
        [robot],
        sides,
        configs,
        controlled_sides,
        gripper_factory=make_gripper,
        tool_factory=lambda value: FakeTool(value, events),
        idle_mode="IDLE",
        target_source=target_source,
        failure_event=failure_event,
        **kwargs,
    )
    return controller, grippers, events


def test_initializes_in_order_and_reports_measured_state() -> None:
    controller, grippers, events = _controller()

    controller.initialize()

    assert events == [
        ("enable", "Flexiv-GN01"),
        ("switch", "Flexiv-GN01"),
        ("init", ""),
        ("params", ""),
    ]
    assert grippers[0].robot.current_mode == "IDLE"
    assert controller.measured_states() == {
        "left_arm": {"width": 0.04, "force": -2.5}
    }


def test_rejects_controlled_side_without_configured_follower_gripper() -> None:
    with pytest.raises(
        ValueError,
        match="Controlled gripper side has no configured follower gripper: left_arm",
    ):
        _controller(
            configs={
                "left_arm": {"follower": "none", "gripper_model": "Flexiv-GN01"}
            }
        )


def test_rejects_initialization_when_robot_is_not_idle() -> None:
    controller, _, _ = _controller(robot=FakeRobot("NRT_CARTESIAN_MOTION_FORCE"))

    with pytest.raises(RuntimeError, match="must be IDLE.*left_arm"):
        controller.initialize()


def test_clamps_width_uses_device_limits_and_suppresses_near_duplicate() -> None:
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    def stop_after_tick(event: threading.Event, timeout: float) -> bool:
        event.set()
        return True

    controller, grippers, _ = _controller(wait=stop_after_tick, clock=clock)
    controller.initialize()
    controller.submit({"left_arm": 0.5})
    controller.start()
    controller.stop()

    assert grippers[0].moves == [(0.09, 0.4, 5.0)]

    controller.submit({"left_arm": 0.0896})
    controller._send_pending()
    assert grippers[0].moves == [(0.09, 0.4, 5.0)]


def test_submit_is_latest_only_while_move_is_blocked() -> None:
    move_started = threading.Event()
    move_release = threading.Event()

    def make_gripper(robot, events):
        return FakeGripper(robot, events, move_started, move_release)

    controller, grippers, _ = _controller(gripper_factory=make_gripper)
    controller.initialize()
    controller.submit({"left_arm": 0.02})
    controller.start()
    assert move_started.wait(timeout=1.0)
    assert controller.measured_states()["left_arm"]["width"] == 0.04

    controller.submit({"left_arm": 0.04})
    controller.submit({"left_arm": 0.08})
    move_release.set()

    for _ in range(20):
        if len(grippers[0].moves) == 2:
            break
        threading.Event().wait(0.01)
    controller.stop()

    assert [move[0] for move in grippers[0].moves] == [0.02, 0.08]


def test_worker_samples_nonblocking_spline_target_source() -> None:
    controller, grippers, _ = _controller(
        target_source=lambda: {"left_arm": 0.06}
    )
    controller.initialize()

    controller._send_pending()

    assert grippers[0].moves == [(0.06, 0.4, 5.0)]


def test_worker_failure_signals_rollout_stop() -> None:
    failure_event = threading.Event()

    class FailingGripper(FakeGripper):
        def Move(self, width: float, velocity: float, force: float) -> None:
            raise RuntimeError("move failed")

    controller, _, _ = _controller(
        gripper_factory=FailingGripper,
        failure_event=failure_event,
    )
    controller.initialize()
    controller.submit({"left_arm": 0.04})
    controller.start()

    assert failure_event.wait(timeout=1.0)
    controller.stop()
    assert isinstance(controller.error, RuntimeError)


def test_start_requires_initialization_and_stop_joins_worker() -> None:
    controller, _, _ = _controller()
    with pytest.raises(RuntimeError, match="Grippers are not initialized"):
        controller.start()

    controller.initialize()
    controller.start()
    controller.stop()
    assert controller._thread is None


@pytest.mark.parametrize("command_hz", [0.0, 30.1, float("inf")])
def test_command_rate_is_positive_and_bounded(command_hz: float) -> None:
    with pytest.raises(ValueError, match="command_hz"):
        GripperExecutor(
            [FakeRobot()],
            ["left_arm"],
            {
                "left_arm": {
                    "follower": "gripper",
                    "gripper_model": "Flexiv-GN01",
                }
            },
            ["left_arm"],
            command_hz=command_hz,
            gripper_factory=lambda robot: object(),
            tool_factory=lambda robot: object(),
            idle_mode="IDLE",
        )
