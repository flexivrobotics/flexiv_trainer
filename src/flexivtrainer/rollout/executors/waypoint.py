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

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from flexivtrainer.observability import describe_exception, warn

_POSE_DIM = 7
_TWIST_DIM = 6
_WRENCH_DIM = 6
_POSE_AXES = ("x", "y", "z", "q_w", "q_x", "q_y", "q_z")
_TWIST_AXES = ("vx", "vy", "vz", "wx", "wy", "wz")
_WRENCH_AXES = ("fx", "fy", "fz", "mx", "my", "mz")


def _group_slice(
    names: list[str], prefix: str, axes: tuple[str, ...], *, required: bool
) -> slice | None:
    indices = [index for index, name in enumerate(names) if name.startswith(prefix)]
    if not indices:
        if required:
            raise ValueError(f"Action schema is missing required group '{prefix}*'")
        return None
    start = indices[0]
    expected = [f"{prefix}{axis}" for axis in axes]
    actual = [names[index] for index in indices]
    if (
        len(indices) != len(axes)
        or indices != list(range(start, start + len(axes)))
        or actual != expected
    ):
        raise ValueError(
            f"Action group '{prefix}*' must contain {len(axes)} contiguous "
            "axes in canonical order"
        )
    return slice(start, start + len(axes))


def canonical_action_names(action_dim: int, sides: list[str]) -> list[str]:
    """Infer only the recorder's unambiguous legacy waypoint layouts."""

    if not sides:
        raise ValueError("At least one active arm side is required")
    if action_dim == len(sides) * (_POSE_DIM + _TWIST_DIM + _WRENCH_DIM):
        groups = (
            ("tcp_pose", _POSE_AXES),
            ("tcp_twist", _TWIST_AXES),
            ("tcp_wrench", _WRENCH_AXES),
        )
    elif action_dim == len(sides) * (_POSE_DIM + _TWIST_DIM):
        groups = (("tcp_pose", _POSE_AXES), ("tcp_twist", _TWIST_AXES))
    else:
        supported = sorted(
            {
                len(sides) * (_POSE_DIM + _TWIST_DIM),
                len(sides) * (_POSE_DIM + _TWIST_DIM + _WRENCH_DIM),
            }
        )
        raise ValueError(
            "Cannot infer waypoint action layout from "
            f"width {action_dim}; canonical widths for {len(sides)} arm(s) "
            f"are {supported}"
        )
    return [
        f"{side}.{group}.{axis}"
        for side in sides
        for group, axes in groups
        for axis in axes
    ]


def build_action_layout(
    action_names: list[str], sides: list[str], action_dim: int | None = None
) -> list[dict[str, Any]]:
    if not action_names:
        raise ValueError("Waypoint action feature names are required")
    if len(set(action_names)) != len(action_names):
        raise ValueError("Waypoint action feature names must be unique")
    expected_dim = len(action_names) if action_dim is None else action_dim
    if len(action_names) != expected_dim:
        raise ValueError(
            "Checkpoint action width does not match its named schema: "
            f"output={expected_dim} names={len(action_names)}"
        )

    pose_sides: list[str] = []
    marker = ".tcp_pose."
    for name in action_names:
        if marker not in name:
            continue
        side = name.split(marker, 1)[0]
        if side not in pose_sides:
            pose_sides.append(side)
    if pose_sides != sides:
        raise ValueError(
            "Checkpoint arm layout does not match active sides: "
            f"checkpoint={pose_sides} active={sides}"
        )

    layout: list[dict[str, Any]] = []
    for side in sides:
        gripper_width_name = f"{side}.gripper.width"
        gripper_close_name = f"{side}.gripper.close"
        gripper_force_name = f"{side}.gripper.force"
        gripper_width = (
            action_names.index(gripper_width_name)
            if gripper_width_name in action_names
            else None
        )
        gripper_close = (
            action_names.index(gripper_close_name)
            if gripper_close_name in action_names
            else None
        )
        if gripper_width is not None and gripper_close is not None:
            raise ValueError(
                f"Action schema cannot contain both '{gripper_width_name}' and "
                f"'{gripper_close_name}'"
            )
        if gripper_force_name in action_names and gripper_width is None:
            raise ValueError(
                f"Action schema has '{gripper_force_name}' without required "
                f"'{gripper_width_name}'"
            )
        layout.append(
            {
                "side": side,
                "pose": _group_slice(
                    action_names,
                    f"{side}.tcp_pose.",
                    _POSE_AXES,
                    required=True,
                ),
                "twist": _group_slice(
                    action_names,
                    f"{side}.tcp_twist.",
                    _TWIST_AXES,
                    required=False,
                ),
                "wrench": _group_slice(
                    action_names,
                    f"{side}.tcp_wrench.",
                    _WRENCH_AXES,
                    required=False,
                ),
                # Recorded waypoint datasets also contain gripper.force. It is
                # measured feedback, not a hardware command: GripperExecutor
                # derives a conservative Move() force from device limits.
                "gripper_width": gripper_width,
                "gripper_close": gripper_close,
            }
        )
    modes = {
        "width" if arm["gripper_width"] is not None else "close"
        for arm in layout
        if arm["gripper_width"] is not None or arm["gripper_close"] is not None
    }
    if len(modes) > 1:
        raise ValueError("Mixed gripper width/close action schemas are unsupported")
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
    gripper_targets: dict[str, float]


class WaypointExecutor:
    """Execute rollout waypoints at their target times."""

    def __init__(
        self,
        robots: list[Any],
        layout: list[dict[str, Any]],
        stop_event: threading.Event,
        motion_limits: tuple[float, float, float, float],
        *,
        action_dim: int | None = None,
        submit_gripper: Callable[[Mapping[str, float]], None] | None = None,
    ) -> None:
        self._robots = robots
        self._layout = layout
        self._stop_event = stop_event
        self._motion_limits = motion_limits
        self._submit_gripper = submit_gripper
        required_dim = max(
            (
                section.stop if isinstance(section, slice) else section + 1
                for arm in layout
                for section in (
                    arm.get("pose"),
                    arm.get("twist"),
                    arm.get("wrench"),
                    arm.get("gripper_width"),
                    arm.get("gripper_close"),
                )
                if (
                    isinstance(section, int)
                    or (isinstance(section, slice) and section.stop is not None)
                )
            ),
            default=0,
        )
        self._action_dim = required_dim if action_dim is None else action_dim
        if self._action_dim < required_dim:
            raise ValueError(
                "Waypoint action layout exceeds checkpoint output width: "
                f"layout={required_dim} output={self._action_dim}"
            )
        self._condition = threading.Condition()
        self._waypoints: list[_TimedWaypoint] = []
        self._error: str | None = None
        self._scheduled_count = 0
        self._starved_replans = 0
        self._thread: threading.Thread | None = None

    def replace_waypoints(
        self,
        actions: list[list[float]],
        target_times: list[float],
        now: float,
    ) -> None:
        self.validate_actions(actions)
        if len(actions) != len(target_times):
            raise ValueError(
                "Waypoint action and target-time counts must match: "
                f"actions={len(actions)} target_times={len(target_times)}"
            )

        waypoints: list[_TimedWaypoint] = []
        for action, target_time in zip(actions, target_times):
            if target_time <= now:
                continue
            commands: list[_RobotCommand | None] = []
            gripper_targets: dict[str, float] = {}
            for index, arm_plan in enumerate(self._layout):
                if index >= len(self._robots):
                    break
                gripper_width = arm_plan.get("gripper_width")
                gripper_close = arm_plan.get("gripper_close")
                target_index = (
                    gripper_width
                    if isinstance(gripper_width, int)
                    else gripper_close
                )
                if isinstance(target_index, int):
                    gripper_targets[str(arm_plan["side"])] = float(
                        action[target_index]
                    )
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
            waypoints.append(
                _TimedWaypoint(float(target_time), commands, gripper_targets)
            )
        if actions and not waypoints:
            # Every waypoint was already in the past, so the arm keeps holding its
            # last pose. Silent before, which hid the same bug twice.
            self._starved_replans += 1
            if self._starved_replans in (1, 10, 100, 1000):
                warn(
                    "Rollout replan scheduled no waypoints",
                    f"all {len(actions)} were already stale "
                    f"(count={self._starved_replans}); raise "
                    "action_anchor_offset_steps or speed up inference",
                )
        with self._condition:
            self._waypoints = waypoints
            self._scheduled_count = len(waypoints)
            self._condition.notify()

    def validate_actions(self, actions: list[list[float]]) -> None:
        """Reject malformed chunks before constructing any per-arm commands."""

        if not actions:
            raise ValueError("Waypoint policy returned an empty action chunk")
        for index, action in enumerate(actions):
            if len(action) != self._action_dim:
                raise ValueError(
                    f"Waypoint action {index} has width {len(action)}, "
                    f"expected {self._action_dim}"
                )
            for arm in self._layout:
                gripper_width = arm.get("gripper_width")
                gripper_close = arm.get("gripper_close")
                target_index = (
                    gripper_width
                    if isinstance(gripper_width, int)
                    else gripper_close
                )
                if (
                    isinstance(target_index, int)
                    and not math.isfinite(float(action[target_index]))
                ):
                    target_name = (
                        "gripper width"
                        if isinstance(gripper_width, int)
                        else "gripper close"
                    )
                    raise ValueError(
                        "Waypoint action "
                        f"{index} has non-finite {target_name} for {arm['side']}"
                    )

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
        if waypoint.gripper_targets and self._submit_gripper is not None:
            self._submit_gripper(waypoint.gripper_targets)

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
