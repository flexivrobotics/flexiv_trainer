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

import numpy as np

from flexivtrainer.data.lerobot_io import (
    ROTATION_6D_AXES,
    rotation_6d_to_quaternion_wxyz,
)
from flexivtrainer.observability import describe_exception, warn

_POSE_DIM = 9
# Checkpoints recorded before poses moved to rotation-6D.
_LEGACY_POSE_DIM = 7
_TWIST_DIM = 6
_WRENCH_DIM = 6
_POSITION_AXES = ("x", "y", "z")
_LEGACY_POSE_AXES = ("x", "y", "z", "q_w", "q_x", "q_y", "q_z")
_POSE_GROUPS_6D = (
    ("tcp_pose", _POSITION_AXES),
    ("tcp_rotation_6d", ROTATION_6D_AXES),
)
_POSE_GROUPS_LEGACY = (("tcp_pose", _LEGACY_POSE_AXES),)
_TWIST_AXES = ("vx", "vy", "vz", "wx", "wy", "wz")
_WRENCH_AXES = ("fx", "fy", "fz", "mx", "my", "mz")
# Keyed by active arm count; values match the ``arm_mode`` setting operators see.
_ARM_MODES = {1: "single-arm", 2: "dual-arm"}


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


def _layout_hint(action_dim: int, arm_count: int) -> str:
    """Explain a width mismatch, naming the fix when there is one.

    A width that divides cleanly by 13 or 19 points at a different arm count --
    the common case, and the only one with a one-step fix.
    """
    for per_arm in (
        _POSE_DIM + _TWIST_DIM,
        _POSE_DIM + _TWIST_DIM + _WRENCH_DIM,
        _LEGACY_POSE_DIM + _TWIST_DIM,
        _LEGACY_POSE_DIM + _TWIST_DIM + _WRENCH_DIM,
    ):
        if action_dim % per_arm:
            continue
        needed = action_dim // per_arm
        if needed == arm_count or needed not in _ARM_MODES:
            continue
        return (
            f"{_width_preamble(action_dim, arm_count)}. Width {action_dim} "
            f"corresponds to {_ARM_MODES[needed]} ({needed} x {per_arm}). "
            f"Arm mode mismatch: set {_ARM_MODES[needed]} mode and restart "
            "the rollout"
        )
    return (
        f"{_width_preamble(action_dim, arm_count)}. The checkpoint does not "
        "record action-axis names and the width is not inferable. Supply "
        "action_names when starting the rollout"
    )


def layout_confirmation(action_dim: int, arm_count: int) -> str:
    """Positive counterpart to :func:`_layout_hint`, same sentence structure."""
    mode = _ARM_MODES.get(arm_count, f"{arm_count}-arm")
    return (
        f"{_width_preamble(action_dim, arm_count)}. The checkpoint does not "
        "record action-axis names, so the layout was inferred from its width. "
        f"Arm mode matches: {mode}"
    )


def recorded_layout_confirmation(action_dim: int, arm_count: int) -> str:
    """Confirmation for a checkpoint that records its own action-axis names.

    Omits the canonical widths: a named checkpoint is not bound to them (a
    gripper axis makes 14 valid), so quoting "13 or 19" would contradict it.
    """
    mode = _ARM_MODES.get(arm_count, f"{arm_count}-arm")
    return (
        f"Policy action width is {action_dim}. The checkpoint records its "
        "action-axis names, so the layout was read from the checkpoint. "
        f"Arm mode matches: {mode}. Policy loaded successfully"
    )


def _width_preamble(action_dim: int, arm_count: int) -> str:
    """Shared opening clause: the observed width against the canonical ones."""
    return (
        f"Policy action width is {action_dim}; canonical widths for "
        f"{_arm_count_text(arm_count)} are "
        f"{_canonical_widths_text(arm_count, action_dim)}"
    )


def _arm_count_text(arm_count: int) -> str:
    return f"{arm_count} arm" if arm_count == 1 else f"{arm_count} arms"


def _canonical_widths_text(arm_count: int, action_dim: int | None = None) -> str:
    """The two inferable widths for an arm count, as ``13 or 19``."""
    widths = _canonical_widths(arm_count)
    # Quote the family the width actually belongs to, so a legacy checkpoint is
    # not told its own working width is non-canonical.
    if action_dim is not None and action_dim in _legacy_widths(arm_count):
        widths = _legacy_widths(arm_count)
    return f"{widths[0]} or {widths[1]}"


def _sides_hint(checkpoint_sides: list[str], active_sides: list[str]) -> str:
    """Counterpart to :func:`_layout_hint` for checkpoints with recorded names.

    A differing arm *count* has the same one-step fix. A matching count with
    different side names cannot come from this recorder (it emits only
    ``single_arm`` or ``left_arm``+``right_arm``), so it is a foreign checkpoint
    that switching arm mode would not resolve.
    """
    checkpoint_text = ", ".join(checkpoint_sides) or "none"
    active_text = ", ".join(active_sides) or "none"
    preamble = (
        f"Policy records {_arm_count_text(len(checkpoint_sides))} "
        f"({checkpoint_text}); {_arm_count_text(len(active_sides))} "
        f"configured ({active_text})"
    )
    needed = len(checkpoint_sides)
    if needed != len(active_sides) and needed in _ARM_MODES:
        return (
            f"{preamble}. Arm mode mismatch: set {_ARM_MODES[needed]} mode "
            "and restart the rollout"
        )
    return (
        f"{preamble}. Arm layout mismatch: the checkpoint was trained for "
        "different arms than the ones configured"
    )


def _canonical_widths(arm_count: int) -> tuple[int, int]:
    """The two inferable action widths for an arm count."""
    return (
        arm_count * (_POSE_DIM + _TWIST_DIM),
        arm_count * (_POSE_DIM + _TWIST_DIM + _WRENCH_DIM),
    )


def _legacy_widths(arm_count: int) -> tuple[int, int]:
    """The same two widths for checkpoints predating rotation-6D poses."""
    return (
        arm_count * (_LEGACY_POSE_DIM + _TWIST_DIM),
        arm_count * (_LEGACY_POSE_DIM + _TWIST_DIM + _WRENCH_DIM),
    )


def _pose_sides(action_names: list[str]) -> list[str]:
    """Arm sides named by a flat action schema, in first-seen order."""
    marker = ".tcp_pose."
    sides: list[str] = []
    for name in action_names:
        if marker not in name:
            continue
        side = name.split(marker, 1)[0]
        if side not in sides:
            sides.append(side)
    return sides


def canonical_action_names(action_dim: int, sides: list[str]) -> list[str]:
    """Infer only the recorder's unambiguous waypoint layouts."""

    if not sides:
        raise ValueError("At least one active arm side is required")
    for pose_groups in (_POSE_GROUPS_6D, _POSE_GROUPS_LEGACY):
        for tail in (
            (("tcp_twist", _TWIST_AXES), ("tcp_wrench", _WRENCH_AXES)),
            (("tcp_twist", _TWIST_AXES),),
        ):
            groups = (*pose_groups, *tail)
            width = sum(len(axes) for _, axes in groups)
            if action_dim == len(sides) * width:
                return [
                    f"{side}.{group}.{axis}"
                    for side in sides
                    for group, axes in groups
                    for axis in axes
                ]
    raise ValueError(_layout_hint(action_dim, len(sides)))


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

    pose_sides = _pose_sides(action_names)
    if pose_sides != sides:
        raise ValueError(_sides_hint(pose_sides, sides))

    layout: list[dict[str, Any]] = []
    for side in sides:
        legacy_width_name = f"{side}.gripper.width"
        target_width_name = f"{side}.gripper.target_width"
        gripper_close_name = f"{side}.gripper.close"
        gripper_force_name = f"{side}.gripper.force"
        if gripper_close_name in action_names:
            raise ValueError(
                "Boolean gripper.close actions are unsupported; convert the "
                "checkpoint dataset to gripper.target_width"
            )
        legacy_width = (
            action_names.index(legacy_width_name)
            if legacy_width_name in action_names
            else None
        )
        target_width = (
            action_names.index(target_width_name)
            if target_width_name in action_names
            else None
        )
        if legacy_width is not None and target_width is not None:
            raise ValueError(
                f"Action schema cannot contain both '{legacy_width_name}' and "
                f"'{target_width_name}'"
            )
        rotation_slice = _group_slice(
            action_names,
            f"{side}.tcp_rotation_6d.",
            ROTATION_6D_AXES,
            required=False,
        )
        gripper_width = target_width if target_width is not None else legacy_width
        if gripper_force_name in action_names and legacy_width is None:
            raise ValueError(
                f"Action schema has '{gripper_force_name}' without required "
                f"legacy '{legacy_width_name}'"
            )
        layout.append(
            {
                "side": side,
                "pose": _group_slice(
                    action_names,
                    f"{side}.tcp_pose.",
                    _POSITION_AXES if rotation_slice else _LEGACY_POSE_AXES,
                    required=True,
                ),
                "rotation": rotation_slice,
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
                "gripper_width": gripper_width,
                "gripper_target_mode": (
                    "target_width"
                    if target_width is not None
                    else "width"
                    if legacy_width is not None
                    else None
                ),
            }
        )
    modes = {
        arm["gripper_target_mode"]
        for arm in layout
        if arm["gripper_target_mode"] is not None
    }
    if len(modes) > 1:
        raise ValueError(
            "Mixed legacy gripper.width and gripper.target_width schemas are "
            "unsupported"
        )
    return layout


def layout_problem(
    action_names: list[str] | None, action_dim: int, sides: list[str]
) -> str | None:
    """Why a layout is unusable for the configured arms, or None when it is fine.

    Negative counterpart to :func:`layout_confirmation`. Every message is built
    from the caller's own values rather than from a caught exception, so an API
    can report the cause without echoing internal error text back to a client.
    """
    if not sides:
        return None
    if action_names is None:
        if action_dim not in (
            *_canonical_widths(len(sides)),
            *_legacy_widths(len(sides)),
        ):
            return _layout_hint(action_dim, len(sides))
        names = canonical_action_names(action_dim, sides)
    else:
        names = action_names

    checkpoint_sides = _pose_sides(names)
    if checkpoint_sides != sides:
        return _sides_hint(checkpoint_sides, sides)

    try:
        build_action_layout(names, sides, action_dim)
    except ValueError as exc:
        # Whatever remains is a schema fault in a foreign checkpoint. Keep the
        # specifics in the server log instead of the response body.
        warn("Rollout action schema is unusable", describe_exception(exc))
        return (
            "The checkpoint action schema is not usable with the configured "
            "arms; see the server log for details"
        )
    return None


def pose_command(plan: Mapping[str, Any], action: Any) -> list[float]:
    """Driver pose ``[x, y, z, q_w..q_z]`` for one arm's planned action.

    ``SendCartesianMotionForce`` takes a quaternion, so a rotation-6D schema is
    converted here and everything downstream stays on 7-element poses.
    """
    pose = [float(value) for value in action[plan["pose"]]]
    rotation = plan.get("rotation")
    if rotation is None:
        return normalize_pose_quaternion(pose)
    quaternion = rotation_6d_to_quaternion_wxyz(
        np.asarray(action[rotation], dtype=np.float64)
    )
    return pose + [float(value) for value in quaternion]


def normalize_pose_quaternion(pose: list[float]) -> list[float]:
    pose = list(pose)
    if len(pose) < _LEGACY_POSE_DIM:
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
                if isinstance(gripper_width, int):
                    gripper_targets[str(arm_plan["side"])] = float(
                        action[gripper_width]
                    )
                pose_slice = arm_plan["pose"]
                if pose_slice is None:
                    commands.append(None)
                    continue
                twist_slice = arm_plan["twist"]
                wrench_slice = arm_plan["wrench"]
                commands.append(
                    _RobotCommand(
                        pose=pose_command(arm_plan, action),
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
                if isinstance(gripper_width, int) and not math.isfinite(
                    float(action[gripper_width])
                ):
                    target_label = (
                        "gripper target width"
                        if arm.get("gripper_target_mode") == "target_width"
                        else "gripper width"
                    )
                    raise ValueError(
                        f"Waypoint action {index} has non-finite {target_label} "
                        f"for {arm['side']}"
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
