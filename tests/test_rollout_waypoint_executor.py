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

import pytest

from flexivtrainer.rollout.executors.waypoint import (
    WaypointExecutor,
    build_action_layout,
    canonical_action_names,
)


class _FakeRobot:
    def __init__(self) -> None:
        self.commands: list[tuple] = []

    def SendCartesianMotionForce(self, *args) -> None:  # noqa: N802
        self.commands.append(args)


def _pose_layout() -> list[dict]:
    return [
        {
            "side": "single_arm",
            "pose": slice(0, 7),
            "twist": None,
            "wrench": None,
        }
    ]


def _unit_pose(x: float) -> list[float]:
    return [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def _executor(stop_event: threading.Event | None = None) -> WaypointExecutor:
    return WaypointExecutor(
        [_FakeRobot()],
        _pose_layout(),
        stop_event or threading.Event(),
        (0.25, 0.6, 1.0, 2.5),
    )


def test_build_action_layout_locates_command_runs() -> None:
    names = canonical_action_names(19, ["single_arm"])

    layout = build_action_layout(names, ["single_arm"])

    assert len(layout) == 1
    assert layout[0]["pose"] == slice(0, 7)
    assert layout[0]["twist"] == slice(7, 13)
    assert layout[0]["wrench"] == slice(13, 19)
    assert layout[0]["gripper_width"] is None


def test_named_layout_maps_gripper_width_and_requires_it_before_force() -> None:
    names = [
        *canonical_action_names(19, ["single_arm"]),
        "single_arm.gripper.width",
        "single_arm.gripper.force",
    ]

    layout = build_action_layout(names, ["single_arm"], len(names))

    assert layout[0]["gripper_width"] == 19

    force_only = [*names[:19], "single_arm.gripper.force"]
    with pytest.raises(ValueError, match="force.*without required.*width"):
        build_action_layout(force_only, ["single_arm"], len(force_only))


def test_named_layout_maps_only_the_configured_gripper_side() -> None:
    names = canonical_action_names(38, ["left_arm", "right_arm"])
    names[19:19] = [
        "left_arm.gripper.width",
        "left_arm.gripper.force",
    ]

    layout = build_action_layout(names, ["left_arm", "right_arm"], len(names))

    assert layout[0]["gripper_width"] == 19
    assert layout[1]["gripper_width"] is None


def test_named_layout_accepts_target_width_and_rejects_ambiguous_modes() -> None:
    base = canonical_action_names(19, ["single_arm"])
    layout = build_action_layout(
        [*base, "single_arm.gripper.target_width"], ["single_arm"]
    )

    assert layout[0]["gripper_width"] == 19
    assert layout[0]["gripper_target_mode"] == "target_width"
    with pytest.raises(ValueError, match="both"):
        build_action_layout(
            [
                *base,
                "single_arm.gripper.width",
                "single_arm.gripper.target_width",
            ],
            ["single_arm"],
        )
    with pytest.raises(ValueError, match="unsupported"):
        build_action_layout([*base, "single_arm.gripper.close"], ["single_arm"])


@pytest.mark.parametrize(
    ("sides", "action_dim", "expected_twist", "expected_wrench"),
    [
        (["single_arm"], 13, slice(7, 13), None),
        (["single_arm"], 19, slice(7, 13), slice(13, 19)),
        (["left_arm", "right_arm"], 26, slice(20, 26), None),
        (["left_arm", "right_arm"], 38, slice(26, 32), slice(32, 38)),
    ],
)
def test_canonical_layouts_cover_full_and_no_wrench_actions(
    sides, action_dim, expected_twist, expected_wrench
) -> None:
    names = canonical_action_names(action_dim, sides)
    layout = build_action_layout(names, sides, action_dim)

    assert len(names) == action_dim
    assert layout[-1]["twist"] == expected_twist
    assert layout[-1]["wrench"] == expected_wrench


@pytest.mark.parametrize("include_wrench", [False, True])
def test_dual_arm_dispatch_maps_each_arm_and_zero_fills_missing_wrench(
    include_wrench,
) -> None:
    sides = ["left_arm", "right_arm"]
    action_dim = 38 if include_wrench else 26
    layout = build_action_layout(
        canonical_action_names(action_dim, sides), sides, action_dim
    )
    robots = [_FakeRobot(), _FakeRobot()]
    executor = WaypointExecutor(
        robots,
        layout,
        threading.Event(),
        (0.25, 0.6, 1.0, 2.5),
        action_dim=action_dim,
    )
    left_pose = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    right_pose = [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]
    left_twist = [1.0] * 6
    right_twist = [2.0] * 6
    left_wrench = [3.0] * 6
    right_wrench = [4.0] * 6
    action = [*left_pose, *left_twist]
    if include_wrench:
        action.extend(left_wrench)
    action.extend([*right_pose, *right_twist])
    if include_wrench:
        action.extend(right_wrench)

    executor.replace_waypoints([action], [101.0], now=100.0)
    executor._send_waypoint(executor._waypoints[0])

    assert robots[0].commands[0][0] == left_pose
    assert robots[0].commands[0][1] == (left_wrench if include_wrench else [0.0] * 6)
    assert robots[0].commands[0][2] == left_twist
    assert robots[1].commands[0][0] == right_pose
    assert robots[1].commands[0][1] == (right_wrench if include_wrench else [0.0] * 6)
    assert robots[1].commands[0][2] == right_twist


def test_malformed_action_width_is_rejected_before_dispatch() -> None:
    sides = ["left_arm", "right_arm"]
    layout = build_action_layout(canonical_action_names(26, sides), sides, 26)
    robots = [_FakeRobot(), _FakeRobot()]
    executor = WaypointExecutor(
        robots,
        layout,
        threading.Event(),
        (0.25, 0.6, 1.0, 2.5),
        action_dim=26,
    )

    with pytest.raises(ValueError, match="width 25, expected 26"):
        executor.replace_waypoints([[0.0] * 25], [101.0], now=100.0)

    assert executor._waypoints == []
    assert all(not robot.commands for robot in robots)


def test_gripper_width_is_submitted_only_when_timed_waypoint_fires() -> None:
    names = [
        *canonical_action_names(19, ["single_arm"]),
        "single_arm.gripper.width",
        "single_arm.gripper.force",
    ]
    layout = build_action_layout(names, ["single_arm"], len(names))
    robot = _FakeRobot()
    submissions: list[dict[str, float]] = []
    executor = WaypointExecutor(
        [robot],
        layout,
        threading.Event(),
        (0.25, 0.6, 1.0, 2.5),
        action_dim=len(names),
        submit_gripper=lambda targets: submissions.append(dict(targets)),
    )
    action = [*_unit_pose(0.1), *([0.0] * 12), 0.042, -3.0]

    executor.replace_waypoints([action], [101.0], now=100.0)

    assert submissions == []
    executor._send_waypoint(executor._waypoints[0])
    assert submissions == [{"single_arm": pytest.approx(0.042)}]
    # Recorded force is intentionally not forwarded to Gripper.Move().
    assert set(submissions[0]) == {"single_arm"}


def test_nonfinite_gripper_width_is_rejected_before_any_dispatch() -> None:
    names = [
        *canonical_action_names(19, ["single_arm"]),
        "single_arm.gripper.width",
        "single_arm.gripper.force",
    ]
    layout = build_action_layout(names, ["single_arm"], len(names))
    robot = _FakeRobot()
    submissions: list[dict[str, float]] = []
    executor = WaypointExecutor(
        [robot],
        layout,
        threading.Event(),
        (0.25, 0.6, 1.0, 2.5),
        action_dim=len(names),
        submit_gripper=lambda targets: submissions.append(dict(targets)),
    )
    action = [*_unit_pose(0.1), *([0.0] * 12), float("nan"), -3.0]

    with pytest.raises(ValueError, match="non-finite gripper width"):
        executor.replace_waypoints([action], [101.0], now=100.0)

    assert robot.commands == []
    assert submissions == []


def test_layout_rejects_unknown_width_and_partial_named_group() -> None:
    with pytest.raises(ValueError, match="action width is 27"):
        canonical_action_names(27, ["left_arm", "right_arm"])

    # A width valid for a different arm count names that mode as the fix.
    with pytest.raises(ValueError, match="Arm mode mismatch: set dual-arm mode"):
        canonical_action_names(26, ["single_arm"])
    with pytest.raises(ValueError, match="for 1 arm are 13 or 19"):
        canonical_action_names(26, ["single_arm"])

    names = canonical_action_names(13, ["single_arm"])
    names.pop()
    with pytest.raises(ValueError, match="must contain 6 contiguous axes"):
        build_action_layout(names, ["single_arm"])


def test_layout_reports_recorded_side_mismatch() -> None:
    # Names recorded: a differing arm count is still the arm-mode case.
    dual = canonical_action_names(26, ["left_arm", "right_arm"])
    with pytest.raises(ValueError, match="Arm mode mismatch: set dual-arm mode"):
        build_action_layout(dual, ["single_arm"])
    recorded = r"Policy records 2 arms \(left_arm, right_arm\)"
    with pytest.raises(ValueError, match=recorded):
        build_action_layout(dual, ["single_arm"])

    # This recorder emits only "single_arm" or "left_arm"+"right_arm", so a
    # different name means a foreign checkpoint -- arm mode cannot fix it.
    with pytest.raises(ValueError, match="Arm layout mismatch"):
        build_action_layout(canonical_action_names(13, ["arm"]), ["single_arm"])


def test_replace_waypoints_replaces_pending_waypoints() -> None:
    executor = _executor()
    dt = 0.05
    now = 100.0
    actions_a = [_unit_pose(float(index)) for index in range(8)]
    times_a = [now + (index + 1) * dt for index in range(8)]
    executor.replace_waypoints(actions_a, times_a, now=now)

    assert len(executor._waypoints) == 8
    assert executor._waypoints[-1].target_time == pytest.approx(now + 8 * dt)

    now_b = now + 2 * dt
    actions_b = [_unit_pose(100.0 + index) for index in range(8)]
    times_b = [now_b + (index + 1) * dt for index in range(8)]
    executor.replace_waypoints(actions_b, times_b, now=now_b)

    assert len(executor._waypoints) == 8
    assert executor._waypoints[0].target_time == pytest.approx(now_b + dt)
    assert executor._waypoints[-1].target_time == pytest.approx(now_b + 8 * dt)
    command = executor._waypoints[0].commands[0]
    assert command is not None
    assert command.pose[0] == pytest.approx(100.0)


def test_anchor_offset_keeps_first_waypoint_ahead_of_filter() -> None:
    dt = 0.05
    latency = dt / 2
    for anchor, expected in ((1, 8), (0, 7)):
        executor = _executor()
        loop_start = 100.0
        actions = [_unit_pose(float(index)) for index in range(8)]
        target_times = [loop_start + (index + anchor) * dt for index in range(8)]

        executor.replace_waypoints(actions, target_times, now=loop_start + latency)

        assert executor.scheduled_count == expected


def test_executor_thread_stops_after_stop_event() -> None:
    stop_event = threading.Event()
    executor = _executor(stop_event)
    executor.start()

    stop_event.set()
    executor.join()

    assert not any(
        thread.name == "rollout-waypoint-executor" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_replan_that_schedules_nothing_is_reported(monkeypatch) -> None:
    # A silent drop froze the arm at its last pose twice before; the count is the
    # only signal that inference latency has outrun the anchor lead.
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "flexivtrainer.rollout.executors.waypoint.warn",
        lambda message, detail="": warnings.append((message, detail)),
    )
    executor = _executor()
    dt = 1.0 / 60.0
    loop_start = 100.0
    actions = [_unit_pose(0.0)]

    # Single-waypoint chunk (temporal ensembling) with a 22ms tick at 60Hz: the
    # only waypoint is already in the past.
    executor.replace_waypoints(actions, [loop_start + dt], now=loop_start + 0.022)

    assert executor.scheduled_count == 0
    assert len(warnings) == 1
    assert "scheduled no waypoints" in warnings[0][0]


def test_single_waypoint_chunk_survives_only_when_inference_beats_dt() -> None:
    # Replan wipes pending waypoints every tick, so a single-waypoint chunk fires
    # only while inference is faster than dt. This is why 60Hz needs a faster
    # forward pass rather than a larger anchor.
    dt = 1.0 / 60.0
    loop_start = 100.0
    actions = [_unit_pose(0.0)]

    for latency, expected in ((0.012, 1), (0.022, 0)):
        executor = _executor()
        executor.replace_waypoints(actions, [loop_start + dt], now=loop_start + latency)
        assert executor.scheduled_count == expected
