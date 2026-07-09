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
import time
from dataclasses import dataclass
from typing import Any

from flexivtrainer.observability import describe_exception

_POSE_DIM = 7
_TWIST_DIM = 6
_WRENCH_DIM = 6


def _find_run(names: list[str], prefix: str) -> int | None:
    for index, name in enumerate(names):
        if name.startswith(prefix):
            return index
    return None


def build_action_layout(
    action_names: list[str], sides: list[str]
) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    for side in sides:
        pose_start = _find_run(action_names, f"{side}.tcp_pose.")
        twist_start = _find_run(action_names, f"{side}.tcp_twist.")
        wrench_start = _find_run(action_names, f"{side}.tcp_wrench.")
        layout.append(
            {
                "side": side,
                "pose": (
                    None
                    if pose_start is None
                    else slice(pose_start, pose_start + _POSE_DIM)
                ),
                "twist": (
                    None
                    if twist_start is None
                    else slice(twist_start, twist_start + _TWIST_DIM)
                ),
                "wrench": (
                    None
                    if wrench_start is None
                    else slice(wrench_start, wrench_start + _WRENCH_DIM)
                ),
            }
        )
    return layout


def normalize_pose_quaternion(pose: list[float]) -> list[float]:
    pose = list(pose)
    if len(pose) < _POSE_DIM:
        return pose
    quat = pose[3:7]
    norm = sum(component * component for component in quat) ** 0.5
    if norm > 1e-6:
        pose[3:7] = [component / norm for component in quat]
    return pose


@dataclass
class _RobotCommand:
    pose: list[float]
    wrench: list[float]
    twist: list[float]


@dataclass
class _TimedWaypoint:
    target_time: float
    commands: list[_RobotCommand | None]


class WaypointExecutor:
    """Execute rollout waypoints at their target times."""

    def __init__(
        self,
        robots: list[Any],
        layout: list[dict[str, Any]],
        stop_event: threading.Event,
        motion_limits: tuple[float, float, float, float],
    ) -> None:
        self._robots = robots
        self._layout = layout
        self._stop_event = stop_event
        self._motion_limits = motion_limits
        self._condition = threading.Condition()
        self._waypoints: list[_TimedWaypoint] = []
        self._error: str | None = None
        self._scheduled_count = 0
        self._thread: threading.Thread | None = None

    def replace_waypoints(
        self,
        actions: list[list[float]],
        target_times: list[float],
        now: float,
    ) -> None:
        waypoints: list[_TimedWaypoint] = []
        for action, target_time in zip(actions, target_times):
            if target_time <= now:
                continue
            commands: list[_RobotCommand | None] = []
            for index, arm_plan in enumerate(self._layout):
                if index >= len(self._robots):
                    break
                pose_slice = arm_plan["pose"]
                if pose_slice is None:
                    commands.append(None)
                    continue
                twist_slice = arm_plan["twist"]
                wrench_slice = arm_plan["wrench"]
                commands.append(
                    _RobotCommand(
                        pose=normalize_pose_quaternion(list(action[pose_slice])),
                        wrench=(
                            list(action[wrench_slice])
                            if wrench_slice is not None
                            else [0.0] * _WRENCH_DIM
                        ),
                        twist=(
                            list(action[twist_slice])
                            if twist_slice is not None
                            else [0.0] * _TWIST_DIM
                        ),
                    )
                )
            waypoints.append(_TimedWaypoint(float(target_time), commands))
        with self._condition:
            self._waypoints = waypoints
            self._scheduled_count = len(waypoints)
            self._condition.notify()

    def _send_waypoint(self, waypoint: _TimedWaypoint) -> None:
        max_lin_vel, max_ang_vel, max_lin_acc, max_ang_acc = self._motion_limits
        for index, command in enumerate(waypoint.commands):
            if command is None or index >= len(self._robots):
                continue
            self._robots[index].SendCartesianMotionForce(
                command.pose,
                command.wrench,
                command.twist,
                max_lin_vel,
                max_ang_vel,
                max_lin_acc,
                max_ang_acc,
            )

    def _execute_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                with self._condition:
                    if not self._waypoints:
                        self._condition.wait(0.1)
                        continue
                    delay = self._waypoints[0].target_time - time.monotonic()
                    if delay > 0:
                        self._condition.wait(min(delay, 0.1))
                        continue
                    waypoint = self._waypoints.pop(0)
                self._send_waypoint(waypoint)
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
            self._stop_event.set()
            with self._condition:
                self._condition.notify()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._execute_loop,
            daemon=True,
            name="rollout-waypoint-executor",
        )
        self._thread.start()

    def join(self, timeout: float = 2.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def scheduled_count(self) -> int:
        return self._scheduled_count
