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

from flexivtrainer.rollout.waypoint_executor import (
    WaypointExecutor,
    build_action_layout,
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
    names = (
        [f"single_arm.tcp_pose.{index}" for index in range(7)]
        + [f"single_arm.tcp_twist.{index}" for index in range(6)]
        + [f"single_arm.tcp_wrench.{index}" for index in range(6)]
    )

    layout = build_action_layout(names, ["single_arm"])

    assert len(layout) == 1
    assert layout[0]["pose"] == slice(0, 7)
    assert layout[0]["twist"] == slice(7, 13)
    assert layout[0]["wrench"] == slice(13, 19)


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


def test_last_dispatched_poses_populated_after_send() -> None:
    executor = _executor()
    dt = 1.0
    now = 100.0
    actions = [_unit_pose(float(index)) for index in range(3)]
    times = [now + (index + 1) * dt for index in range(3)]
    executor.replace_waypoints(actions, times, now=now)

    assert executor.last_dispatched_poses is None
    waypoint = executor._waypoints.pop(0)
    executor._send_waypoint(waypoint)

    poses = executor.last_dispatched_poses
    assert poses is not None
    assert poses[0] is not None
    assert poses[0][0] == pytest.approx(0.0)


def test_last_boundary_gap_measures_known_kink() -> None:
    executor = _executor()
    dt = 1.0
    now = 100.0
    # Dispatch old-last at x=0 (t0), leave old-next pending at x=1 (t0+1):
    # old slope = 1 unit/s.
    executor.replace_waypoints(
        [_unit_pose(0.0), _unit_pose(1.0)],
        [now + dt, now + 2 * dt],
        now=now,
    )
    executor._send_waypoint(executor._waypoints.pop(0))

    # New chunk's first waypoint at x=3 at the same time as old-next
    # (t0 + 1): boundary velocity = 3 unit/s -> gap = |3 - 1| = 2.
    executor.replace_waypoints(
        [_unit_pose(3.0), _unit_pose(4.0)],
        [now + 2 * dt, now + 3 * dt],
        now=now + dt,
    )

    assert executor.last_boundary_gap == pytest.approx(2.0)


def test_last_dropped_counts_past_filtered_waypoints() -> None:
    executor = _executor()
    dt = 0.05
    now = 100.0
    actions = [_unit_pose(float(index)) for index in range(8)]
    # First two target times are at/before `now` and must be dropped.
    target_times = [now + (index - 1) * dt for index in range(8)]

    executor.replace_waypoints(actions, target_times, now=now)

    assert executor.last_dropped == 2
    assert executor.scheduled_count == 6


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
