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
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

from flexivtrainer.data.bspline import rotation_6d_to_quaternion_wxyz
from flexivtrainer.observability import describe_exception

_FEATURE_PATTERN = re.compile(r"^bspline\.row_(\d+)\.(.+)$")
_POSITION_AXES = ("x", "y", "z")
_ROTATION_AXES = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
_ZERO_VECTOR = [0.0] * 6


@dataclass(frozen=True, slots=True)
class BSplineInstallResult:
    start_time: float
    alignment_error: float
    warning: str | None


@dataclass(frozen=True, slots=True)
class BSplineExecutorStatus:
    remaining_s: float | None
    replan_needed: bool
    achieved_send_hz: float
    sent_count: int
    missed_deadlines: int
    handoff_warnings: int
    error: str | None


@dataclass(frozen=True, slots=True)
class BSplineActionLayout:
    rows: int
    channels: tuple[str, ...]
    sides: tuple[str, ...]
    gripper_sides: tuple[str, ...]

    @property
    def flat_action_dim(self) -> int:
        return self.rows * len(self.channels)


@dataclass(frozen=True, slots=True)
class _ArmLayout:
    side: str
    position_indices: tuple[int, ...]
    rotation_indices: tuple[int, ...]
    gripper_index: int | None

    @property
    def alignment_indices(self) -> tuple[int, ...]:
        return self.position_indices + self.rotation_indices


@dataclass(frozen=True, slots=True)
class _Plan:
    spline: BSpline
    min_time: float
    max_time: float
    start_time: float
    installed_at: float


def _repair_knots(knots: np.ndarray) -> np.ndarray:
    repaired = np.asarray(knots, dtype=np.float64).copy()
    for index in range(1, len(repaired)):
        if repaired[index] < repaired[index - 1]:
            repaired[index] = repaired[index - 1] + 1e-6
    return repaired


def _parse_layout(
    feature_names: Sequence[str],
) -> tuple[BSplineActionLayout, tuple[_ArmLayout, ...]]:
    rows: list[list[str]] = []
    for feature_name in feature_names:
        match = _FEATURE_PATTERN.fullmatch(str(feature_name))
        if match is None:
            raise ValueError(
                f"Malformed B-spline action feature name: {feature_name!r}"
            )
        row = int(match.group(1))
        if row == len(rows):
            rows.append([])
        if row != len(rows) - 1:
            raise ValueError("B-spline action rows must be contiguous and row-major")
        rows[row].append(match.group(2))

    if not rows:
        raise ValueError("B-spline action feature names are required")
    channels = tuple(rows[0])
    if len(channels) < 2 or channels[0] != "knot":
        raise ValueError("Each B-spline row must start with a knot channel")
    if len(set(channels)) != len(channels):
        raise ValueError("B-spline channels must be unique within each row")
    if any(tuple(row) != channels for row in rows[1:]):
        raise ValueError("B-spline action rows must have identical channel layouts")

    control_names = channels[1:]
    name_to_index = {name: index for index, name in enumerate(control_names)}
    side_suffix = ".tcp_pose.x"
    sides = [
        name[: -len(side_suffix)]
        for name in control_names
        if name.endswith(side_suffix)
    ]
    if not sides or len(set(sides)) != len(sides):
        raise ValueError("B-spline controls must contain one complete pose per arm")

    expected_names: set[str] = set()
    layouts: list[_ArmLayout] = []
    for side in sides:
        position = tuple(f"{side}.tcp_pose.{axis}" for axis in _POSITION_AXES)
        rotation = tuple(
            f"{side}.tcp_rotation_6d.{axis}" for axis in _ROTATION_AXES
        )
        gripper = f"{side}.gripper.width"
        missing = [
            name for name in (*position, *rotation) if name not in name_to_index
        ]
        if missing:
            raise ValueError(
                f"Incomplete B-spline controls for side '{side}'; missing {missing}"
            )
        expected_names.update((*position, *rotation))
        gripper_index = name_to_index.get(gripper)
        if gripper_index is not None:
            expected_names.add(gripper)
        layouts.append(
            _ArmLayout(
                side=side,
                position_indices=tuple(name_to_index[name] for name in position),
                rotation_indices=tuple(name_to_index[name] for name in rotation),
                gripper_index=gripper_index,
            )
        )

    unexpected = set(control_names) - expected_names
    if unexpected:
        raise ValueError(f"Unsupported B-spline control channels: {sorted(unexpected)}")
    public_layout = BSplineActionLayout(
        rows=len(rows),
        channels=channels,
        sides=tuple(layout.side for layout in layouts),
        gripper_sides=tuple(
            layout.side for layout in layouts if layout.gripper_index is not None
        ),
    )
    return public_layout, tuple(layouts)


def parse_bspline_action_layout(
    feature_names: Sequence[str],
) -> BSplineActionLayout:
    layout, _ = _parse_layout(feature_names)
    return layout


class BSplineExecutor:
    def __init__(
        self,
        robots: Sequence[Any],
        feature_names: Sequence[str],
        stop_event: threading.Event,
        motion_limits: tuple[float, float, float, float],
        *,
        checkpoint_fps: float,
        degree: int = 3,
        control_hz: float = 200.0,
        speed_scale: float = 1.0,
        predict_before_end_s: float = 0.06,
        time_align_error_threshold: float = 0.1,
        time_align_max_fraction: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        public_layout, layouts = _parse_layout(feature_names)
        if len(robots) != len(layouts):
            raise ValueError(
                f"Received {len(robots)} robots for {len(layouts)} B-spline arms"
            )
        if degree < 1 or public_layout.rows <= degree + 1:
            raise ValueError("B-spline rows must exceed degree + 1")
        if not 0 < control_hz <= 1000:
            raise ValueError("control_hz must be in (0, 1000]")
        if checkpoint_fps <= 0 or speed_scale <= 0:
            raise ValueError("checkpoint_fps and speed_scale must be positive")
        if predict_before_end_s < 0:
            raise ValueError("predict_before_end_s must be nonnegative")
        if time_align_error_threshold < 0:
            raise ValueError("time_align_error_threshold must be nonnegative")
        if not 0 < time_align_max_fraction <= 1:
            raise ValueError("time_align_max_fraction must be in (0, 1]")

        self._robots = list(robots)
        self._layout = public_layout
        self._rows = public_layout.rows
        self._channels = public_layout.channels
        self._layouts = layouts
        self._stop_event = stop_event
        self._motion_limits = motion_limits
        self._degree = degree
        self._control_hz = float(control_hz)
        self._source_rate = float(checkpoint_fps) * float(speed_scale)
        self._predict_before_end_s = float(predict_before_end_s)
        self._alignment_threshold = float(time_align_error_threshold)
        self._alignment_max_fraction = float(time_align_max_fraction)
        self._clock = clock

        self._condition = threading.Condition(threading.RLock())
        self._plan: _Plan | None = None
        self._last_raw_command: np.ndarray | None = None
        self._last_gripper_widths: dict[str, float] = {}
        self._error: str | None = None
        self._sent_count = 0
        self._missed_deadlines = 0
        self._handoff_warnings = 0
        self._first_sent_at: float | None = None
        self._last_sent_at: float | None = None
        self._thread: threading.Thread | None = None

    def _decode(self, flat_action: Sequence[float] | np.ndarray) -> BSpline:
        action = np.asarray(flat_action, dtype=np.float64)
        expected = self._rows * len(self._channels)
        if action.ndim != 1 or action.size != expected:
            raise ValueError(
                f"Expected flat B-spline action [{expected}], got {action.shape}"
            )
        matrix = action.reshape(self._rows, len(self._channels))
        if not np.all(np.isfinite(matrix)):
            raise ValueError("B-spline action contains non-finite values")
        knots = _repair_knots(matrix[:, 0])
        controls = matrix[: -(self._degree + 1), 1:]
        min_time = float(knots[self._degree])
        max_time = float(knots[-self._degree - 1])
        if not math.isfinite(min_time) or not math.isfinite(max_time):
            raise ValueError("B-spline domain contains non-finite values")
        if max_time <= min_time:
            raise ValueError(
                f"B-spline domain must be non-empty, got [{min_time}, {max_time}]"
            )
        return BSpline(knots, controls, self._degree, extrapolate=False)

    def _alignment_error(self, spline: BSpline, target: np.ndarray, t: float) -> float:
        current = np.asarray(spline(t), dtype=np.float64)
        indices = [
            index for layout in self._layouts for index in layout.alignment_indices
        ]
        return float(np.max(np.abs(current[indices] - target[indices])))

    def _align(
        self,
        spline: BSpline,
        target: np.ndarray,
        inference_latency_s: float,
    ) -> tuple[float, float]:
        min_time = float(spline.t[self._degree])
        max_time = float(spline.t[-self._degree - 1])
        max_allowed = min_time + (
            max_time - min_time
        ) * self._alignment_max_fraction
        initial_max = float(
            np.clip(
                min_time + max(0.0, inference_latency_s) * self._source_rate,
                min_time,
                max_allowed,
            )
        )
        indices = np.asarray(
            [
                index
                for layout in self._layouts
                for index in layout.alignment_indices
            ],
            dtype=np.intp,
        )

        def objective(t: float) -> float:
            return float(np.abs(np.asarray(spline(t))[indices] - target[indices]).sum())

        best_time = min_time
        best_error = self._alignment_error(spline, target, best_time)
        scale = 1.0
        while best_error > self._alignment_threshold and scale <= 20:
            upper = min(
                min_time + (initial_max - min_time) * scale,
                max_allowed,
            )
            if upper <= min_time:
                break
            result = minimize_scalar(
                objective,
                bounds=(min_time, upper),
                method="bounded",
            )
            best_time = float(result.x)
            best_error = self._alignment_error(
                spline,
                target,
                best_time,
            )
            if upper >= max_allowed:
                break
            scale *= 1.5
        return best_time, best_error

    def install(
        self,
        flat_action: Sequence[float] | np.ndarray,
        *,
        inference_latency_s: float,
        now: float | None = None,
    ) -> BSplineInstallResult:
        spline = self._decode(flat_action)
        install_time = self._clock() if now is None else float(now)
        min_time = float(spline.t[self._degree])
        max_time = float(spline.t[-self._degree - 1])

        with self._condition:
            if self._plan is None or self._last_raw_command is None:
                start_time = float(np.clip(0.0, min_time, max_time))
                alignment_error = 0.0
            else:
                start_time, alignment_error = self._align(
                    spline,
                    self._last_raw_command,
                    inference_latency_s,
                )
            warning = None
            if alignment_error > self._alignment_threshold:
                warning = (
                    "B-spline time-align error exceeds threshold: "
                    f"{alignment_error:.6f} > {self._alignment_threshold:.6f}"
                )
                self._handoff_warnings += 1
            self._plan = _Plan(
                spline=spline,
                min_time=min_time,
                max_time=max_time,
                start_time=start_time,
                installed_at=install_time,
            )
            self._condition.notify_all()
        return BSplineInstallResult(start_time, alignment_error, warning)

    def _spline_time(self, plan: _Plan, now: float) -> float:
        return float(
            np.clip(
                plan.start_time + (now - plan.installed_at) * self._source_rate,
                plan.min_time,
                plan.max_time,
            )
        )

    def execute_once(self, now: float | None = None) -> bool:
        current_time = self._clock() if now is None else float(now)
        with self._condition:
            plan = self._plan
            if plan is None:
                return False
            raw = np.asarray(
                plan.spline(self._spline_time(plan, current_time)),
                dtype=np.float64,
            )
            if not np.all(np.isfinite(raw)):
                raise ValueError("Sampled B-spline command contains non-finite values")

            max_lin_vel, max_ang_vel, max_lin_acc, max_ang_acc = self._motion_limits
            grippers: dict[str, float] = {}
            for robot, layout in zip(self._robots, self._layouts):
                position = raw[list(layout.position_indices)]
                quaternion = rotation_6d_to_quaternion_wxyz(
                    raw[list(layout.rotation_indices)]
                )
                pose = np.concatenate([position, quaternion]).tolist()
                robot.SendCartesianMotionForce(
                    pose,
                    _ZERO_VECTOR.copy(),
                    _ZERO_VECTOR.copy(),
                    max_lin_vel,
                    max_ang_vel,
                    max_lin_acc,
                    max_ang_acc,
                )
                if layout.gripper_index is not None:
                    grippers[layout.side] = float(raw[layout.gripper_index])
            self._last_raw_command = raw.copy()
            self._last_gripper_widths = grippers
            self._sent_count += 1
            if self._first_sent_at is None:
                self._first_sent_at = current_time
            self._last_sent_at = current_time
        return True

    def _execute_loop(self) -> None:
        period = 1.0 / self._control_hz
        deadline = self._clock()
        try:
            while not self._stop_event.is_set():
                now = self._clock()
                if now < deadline:
                    with self._condition:
                        self._condition.wait(min(deadline - now, 0.1))
                    continue
                missed = int((now - deadline) // period)
                if missed:
                    with self._condition:
                        self._missed_deadlines += missed
                    deadline += missed * period
                self.execute_once(now)
                deadline += period
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
            self._stop_event.set()
            with self._condition:
                self._condition.notify_all()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._execute_loop,
            daemon=True,
            name="rollout-bspline-executor",
        )
        self._thread.start()

    def join(self, timeout: float = 2.0) -> bool:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        return thread is None or not thread.is_alive()

    def remaining_s(self, now: float | None = None) -> float | None:
        current_time = self._clock() if now is None else float(now)
        with self._condition:
            if self._plan is None:
                return None
            spline_time = self._spline_time(self._plan, current_time)
            return max(0.0, (self._plan.max_time - spline_time) / self._source_rate)

    def replan_needed(self, now: float | None = None) -> bool:
        remaining = self.remaining_s(now)
        return remaining is None or remaining <= self._predict_before_end_s

    def status(self, now: float | None = None) -> BSplineExecutorStatus:
        remaining = self.remaining_s(now)
        with self._condition:
            duration = (
                0.0
                if self._first_sent_at is None or self._last_sent_at is None
                else self._last_sent_at - self._first_sent_at
            )
            achieved_send_hz = (
                (self._sent_count - 1) / duration
                if self._sent_count > 1 and duration > 0
                else 0.0
            )
            return BSplineExecutorStatus(
                remaining_s=remaining,
                replan_needed=(
                    remaining is None or remaining <= self._predict_before_end_s
                ),
                achieved_send_hz=achieved_send_hz,
                sent_count=self._sent_count,
                missed_deadlines=self._missed_deadlines,
                handoff_warnings=self._handoff_warnings,
                error=self._error,
            )

    @property
    def sides(self) -> tuple[str, ...]:
        return self._layout.sides

    @property
    def gripper_sides(self) -> tuple[str, ...]:
        return self._layout.gripper_sides

    @property
    def last_raw_command(self) -> np.ndarray | None:
        with self._condition:
            return (
                None
                if self._last_raw_command is None
                else self._last_raw_command.copy()
            )

    @property
    def last_gripper_widths(self) -> dict[str, float]:
        with self._condition:
            return dict(self._last_gripper_widths)

    @property
    def sent_count(self) -> int:
        with self._condition:
            return self._sent_count

    @property
    def missed_deadlines(self) -> int:
        with self._condition:
            return self._missed_deadlines

    @property
    def handoff_warnings(self) -> int:
        with self._condition:
            return self._handoff_warnings

    @property
    def error(self) -> str | None:
        return self._error
