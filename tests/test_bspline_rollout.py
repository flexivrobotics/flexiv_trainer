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

import threading

import numpy as np
import pytest

from flexivtrainer.rollout.bspline_executor import (
    BSplineExecutor,
    _repair_knots,
    parse_bspline_action_layout,
)

_ROTATION_NAMES = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
_ROWS = 16
_DEGREE = 3
_KNOTS = np.asarray([0.0] * 4 + list(range(1, 9)) + [9.0] * 4)
_LIMITS = (0.25, 0.6, 1.0, 2.5)


class _FakeRobot:
    def __init__(self, stop_event: threading.Event | None = None) -> None:
        self.commands: list[tuple] = []
        self._stop_event = stop_event

    def SendCartesianMotionForce(self, *args) -> None:  # noqa: N802
        self.commands.append(args)
        if self._stop_event is not None and len(self.commands) == 3:
            self._stop_event.set()


def _channels(sides: tuple[str, ...], gripper: bool = False) -> list[str]:
    result = ["knot"]
    for side in sides:
        result.extend(f"{side}.tcp_pose.{axis}" for axis in ("x", "y", "z"))
        result.extend(
            f"{side}.tcp_rotation_6d.{axis}" for axis in _ROTATION_NAMES
        )
        if gripper:
            result.append(f"{side}.gripper.width")
    return result


def _feature_names(
    sides: tuple[str, ...] = ("arm",),
    *,
    gripper: bool = False,
) -> list[str]:
    channels = _channels(sides, gripper)
    return [
        f"bspline.row_{row:02d}.{channel}"
        for row in range(_ROWS)
        for channel in channels
    ]


def _action(
    sides: tuple[str, ...] = ("arm",),
    *,
    gripper: bool = False,
    position_offset: float = 0.0,
    gripper_offset: float = 0.0,
    knots: np.ndarray = _KNOTS,
) -> np.ndarray:
    channels = _channels(sides, gripper)
    matrix = np.zeros((_ROWS, len(channels)), dtype=np.float64)
    matrix[:, 0] = knots
    active_controls = _ROWS - (_DEGREE + 1)
    greville = np.asarray(
        [
            np.mean(knots[index + 1 : index + _DEGREE + 1])
            for index in range(active_controls)
        ]
    )
    for side_index, side in enumerate(sides):
        x_index = channels.index(f"{side}.tcp_pose.x")
        matrix[:active_controls, x_index] = (
            greville + position_offset + side_index
        )
        rotation_start = channels.index(f"{side}.tcp_rotation_6d.r1_x")
        matrix[:, rotation_start : rotation_start + 6] = [1, 0, 0, 0, 1, 0]
        if gripper:
            gripper_index = channels.index(f"{side}.gripper.width")
            matrix[:active_controls, gripper_index] = greville + gripper_offset
    matrix[active_controls:, 1:] = matrix[active_controls - 1, 1:]
    return matrix.reshape(-1)


def _executor(
    *,
    sides: tuple[str, ...] = ("arm",),
    gripper: bool = False,
    robots: list[_FakeRobot] | None = None,
    stop_event: threading.Event | None = None,
    clock=None,
    threshold: float = 0.1,
) -> BSplineExecutor:
    kwargs = {} if clock is None else {"clock": clock}
    return BSplineExecutor(
        robots or [_FakeRobot() for _ in sides],
        _feature_names(sides, gripper=gripper),
        stop_event or threading.Event(),
        _LIMITS,
        checkpoint_fps=10.0,
        time_align_error_threshold=threshold,
        **kwargs,
    )


def test_preflight_parses_authoritative_layout_without_robots() -> None:
    layout = parse_bspline_action_layout(
        _feature_names(("left", "right"), gripper=True)
    )

    assert layout.rows == 16
    assert layout.flat_action_dim == 16 * 21
    assert layout.sides == ("left", "right")
    assert layout.gripper_sides == ("left", "right")


@pytest.mark.parametrize(
    "names, message",
    [
        (["action.0"], "Malformed"),
        (
            [
                "bspline.row_00.knot",
                "bspline.row_00.arm.tcp_pose.x",
                "bspline.row_02.knot",
            ],
            "contiguous",
        ),
        (
            [
                "bspline.row_00.knot",
                "bspline.row_00.arm.tcp_pose.x",
            ],
            "Incomplete",
        ),
    ],
)
def test_preflight_rejects_malformed_layouts(
    names: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_bspline_action_layout(names)


def test_knot_repair_matches_companion_rule() -> None:
    repaired = _repair_knots(np.asarray([0.0, 0.5, 0.25, 0.1, 1.0]))

    np.testing.assert_allclose(repaired, [0.0, 0.5, 0.500001, 0.500002, 1.0])


def test_sampling_maps_wall_time_and_sends_cartesian_command() -> None:
    robot = _FakeRobot()
    executor = _executor(robots=[robot], gripper=True)
    result = executor.install(_action(gripper=True), inference_latency_s=0.0, now=10.0)

    assert result.start_time == 0.0
    assert executor.execute_once(now=10.5)
    pose, wrench, twist, *limits = robot.commands[-1]
    assert pose == pytest.approx([5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert wrench == [0.0] * 6
    assert twist == [0.0] * 6
    assert limits == list(_LIMITS)
    assert executor.last_gripper_widths == {"arm": pytest.approx(5.0)}
    assert executor.last_raw_command is not None


def test_sampling_decodes_dual_arm_layout() -> None:
    robots = [_FakeRobot(), _FakeRobot()]
    executor = _executor(
        sides=("left", "right"),
        gripper=True,
        robots=robots,
    )
    executor.install(
        _action(("left", "right"), gripper=True),
        inference_latency_s=0.0,
        now=0.0,
    )

    executor.execute_once(now=0.2)

    assert robots[0].commands[-1][0][0] == pytest.approx(2.0)
    assert robots[1].commands[-1][0][0] == pytest.approx(3.0)
    assert np.linalg.norm(robots[0].commands[-1][0][3:]) == pytest.approx(1.0)
    assert executor.last_gripper_widths == {
        "left": pytest.approx(2.0),
        "right": pytest.approx(2.0),
    }


def test_first_plan_clamps_zero_to_valid_domain() -> None:
    shifted_knots = _KNOTS + 2.0
    executor = _executor()

    result = executor.install(
        _action(knots=shifted_knots),
        inference_latency_s=0.0,
        now=0.0,
    )

    assert result.start_time == 2.0


def test_replacement_alignment_uses_l1_and_excludes_gripper() -> None:
    executor = _executor(gripper=True, threshold=1e-4)
    executor.install(_action(gripper=True), inference_latency_s=0.0, now=0.0)
    executor.execute_once(now=0.15)

    result = executor.install(
        _action(gripper=True, gripper_offset=100.0),
        inference_latency_s=0.05,
        now=0.2,
    )

    assert result.start_time == pytest.approx(1.5, abs=1e-3)
    assert result.alignment_error < 1e-4
    assert result.warning is None


def test_alignment_is_capped_to_first_twenty_percent_and_warns() -> None:
    executor = _executor(threshold=1e-4)
    executor.install(_action(), inference_latency_s=0.0, now=0.0)
    executor.execute_once(now=0.4)

    result = executor.install(_action(), inference_latency_s=0.01, now=0.5)

    assert result.start_time <= 1.8
    assert result.alignment_error == pytest.approx(4.0 - result.start_time)
    assert result.warning is not None
    assert executor.handoff_warnings == 1


def test_invalid_replacement_does_not_replace_active_plan() -> None:
    executor = _executor()
    executor.install(_action(), inference_latency_s=0.0, now=0.0)
    invalid = _action()
    invalid[0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        executor.install(invalid, inference_latency_s=0.1, now=0.1)

    assert executor.execute_once(now=0.2)
    assert executor.last_raw_command is not None
    assert executor.last_raw_command[0] == pytest.approx(2.0)


def test_rejects_empty_spline_domain() -> None:
    executor = _executor()

    with pytest.raises(ValueError, match="non-empty"):
        executor.install(
            _action(knots=np.zeros(_ROWS)),
            inference_latency_s=0.0,
            now=0.0,
        )


def test_remaining_time_and_replan_state() -> None:
    executor = _executor()
    assert executor.replan_needed(now=0.0)
    assert executor.remaining_s(now=0.0) is None

    executor.install(_action(), inference_latency_s=0.0, now=0.0)
    executor.execute_once(now=0.0)
    executor.execute_once(now=0.5)

    assert executor.remaining_s(now=0.5) == pytest.approx(0.4)
    assert not executor.replan_needed(now=0.5)
    assert executor.replan_needed(now=0.895)
    status = executor.status(now=0.5)
    assert status.achieved_send_hz == pytest.approx(2.0)
    assert status.sent_count == 2
    assert status.remaining_s == pytest.approx(0.4)


def test_loop_uses_absolute_deadlines_and_counts_skipped_ticks() -> None:
    stop_event = threading.Event()
    robot = _FakeRobot(stop_event)
    current = 0.0

    def advancing_clock() -> float:
        nonlocal current
        current += 0.02
        return current

    executor = _executor(
        robots=[robot],
        stop_event=stop_event,
        clock=advancing_clock,
    )
    executor.install(_action(), inference_latency_s=0.0, now=0.0)

    executor.start()
    executor.join()

    assert len(robot.commands) == 3
    assert executor.missed_deadlines >= 3
    assert executor.error is None
