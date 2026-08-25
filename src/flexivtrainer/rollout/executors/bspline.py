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

from flexivtrainer.data.lerobot_io import (
    ROTATION_6D_AXES,
    rotation_6d_to_quaternion_wxyz,
)
from flexivtrainer.observability import describe_exception

_FEATURE_PATTERN = re.compile(r"^bspline\.row_(\d+)\.(.+)$")
_POSITION_AXES = ("x", "y", "z")
_ZERO_VECTOR = [0.0] * 6


# Peak |w''| of the quintic handoff decay below, at u = (3±sqrt(3))/6. The blend
# length is derived from it so a larger gap is closed over a longer window
# instead of a more violent one.
_QUINTIC_PEAK_CURVATURE = 10.0 / math.sqrt(3.0)


_HERMITE_VELOCITY_PEAK_CURVATURE = 6.0


def _handoff_decay(u: float) -> float:
    """Quintic ease from 1 to 0, flat at both ends. Carries the position gap.

    A linear fade would hold a constant offset velocity for the whole window and
    then drop it, trading one position step for two velocity steps -- the
    position-fade kink the waypoint path used to suffer. This profile enters and
    leaves at zero velocity and zero acceleration, so nothing switches on or off.
    """
    if u <= 0.0:
        return 1.0
    if u >= 1.0:
        return 0.0
    return 1.0 - u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def _handoff_decay_rate(u: float) -> float:
    """d/du of _handoff_decay."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return -30.0 * u * u * (1.0 - u) * (1.0 - u)


def _handoff_velocity_decay(u: float) -> float:
    """Hermite companion: 0 at both ends, unit slope at u=0.

    _handoff_decay alone only makes position continuous; the two curves still
    leave the splice at different velocities, which is what reads as a direction
    snap. Scaling this term by the velocity mismatch cancels that too, making the
    handoff C1. Zero value and slope at u=1 so it lands cleanly on the new plan.
    """
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return u * (1.0 - u) ** 3


def _handoff_velocity_decay_rate(u: float) -> float:
    """d/du of _handoff_velocity_decay."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return (1.0 - u) ** 2 * (1.0 - 4.0 * u)


@dataclass(frozen=True, slots=True)
class BSplineInstallResult:
    start_time: float
    alignment_error: float
    warning: str | None
    blend_s: float = 0.0
    align_searched: bool = False
    # Search stopped at the time_align_max_fraction rail, not by converging.
    align_capped: bool = False
    # Error at min_time; minus alignment_error, this is what the search bought.
    align_endpoint_error: float = 0.0


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
    gripper_target_mode: str | None

    @property
    def flat_action_dim(self) -> int:
        return self.rows * len(self.channels)


@dataclass(frozen=True, slots=True)
class _ArmLayout:
    side: str
    position_indices: tuple[int, ...]
    rotation_indices: tuple[int, ...]
    gripper_index: int | None
    gripper_target_mode: str | None

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
    # Position and velocity mismatch against the plan this one replaced, decayed to
    # zero over blend_s so the handoff steps neither the commanded pose nor its
    # rate. deriv is the spline's first derivative, cached for the next handoff.
    deriv: BSpline | None = None
    offset: np.ndarray | None = None
    velocity_offset: np.ndarray | None = None
    blend_s: float = 0.0


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
        rotation = tuple(f"{side}.tcp_rotation_6d.{axis}" for axis in ROTATION_6D_AXES)
        gripper = f"{side}.gripper.width"
        gripper_target_width = f"{side}.gripper.target_width"
        gripper_close = f"{side}.gripper.close"
        missing = [name for name in (*position, *rotation) if name not in name_to_index]
        if missing:
            raise ValueError(
                f"Incomplete B-spline controls for side '{side}'; missing {missing}"
            )
        expected_names.update((*position, *rotation))
        gripper_index = name_to_index.get(gripper)
        target_width_index = name_to_index.get(gripper_target_width)
        if gripper_close in name_to_index:
            raise ValueError(
                "Boolean gripper.close B-spline controls are unsupported; "
                "convert them to gripper.target_width"
            )
        if gripper_index is not None and target_width_index is not None:
            raise ValueError(
                f"B-spline controls cannot contain both width modes for {side}"
            )
        if gripper_index is not None:
            expected_names.add(gripper)
            target_mode = "width"
        elif target_width_index is not None:
            gripper_index = target_width_index
            expected_names.add(gripper_target_width)
            target_mode = "target_width"
        else:
            target_mode = None
        layouts.append(
            _ArmLayout(
                side=side,
                position_indices=tuple(name_to_index[name] for name in position),
                rotation_indices=tuple(name_to_index[name] for name in rotation),
                gripper_index=gripper_index,
                gripper_target_mode=target_mode,
            )
        )

    unexpected = set(control_names) - expected_names
    if unexpected:
        raise ValueError(f"Unsupported B-spline control channels: {sorted(unexpected)}")
    target_modes = {
        layout.gripper_target_mode
        for layout in layouts
        if layout.gripper_target_mode is not None
    }
    if len(target_modes) > 1:
        raise ValueError(
            "Mixed legacy gripper.width and gripper.target_width B-spline "
            "controls are unsupported"
        )
    public_layout = BSplineActionLayout(
        rows=len(rows),
        channels=channels,
        sides=tuple(layout.side for layout in layouts),
        gripper_sides=tuple(
            layout.side for layout in layouts if layout.gripper_index is not None
        ),
        gripper_target_mode=next(iter(target_modes), None),
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
        handoff_blend_s: float = 0.15,
        handoff_max_accel: float = 2.0,
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
        if handoff_blend_s < 0:
            raise ValueError("handoff_blend_s must be nonnegative")
        if handoff_max_accel <= 0:
            raise ValueError("handoff_max_accel must be positive")

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
        self._handoff_blend_s = float(handoff_blend_s)
        self._handoff_max_accel = float(handoff_max_accel)
        self._clock = clock
        # Positions + rotation-6D for every arm; the gripper target is excluded from
        # both alignment and the handoff offset.
        self._aligned_indices = np.asarray(
            [index for layout in layouts for index in layout.alignment_indices],
            dtype=np.intp,
        )

        self._condition = threading.Condition(threading.RLock())
        self._plan: _Plan | None = None
        self._last_raw_command: np.ndarray | None = None
        self._last_gripper_targets: dict[str, float] = {}
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
        indices = self._aligned_indices
        return float(np.max(np.abs(current[indices] - target[indices])))

    def _blend_duration(
        self, position_gap: float, velocity_gap: float, remaining_s: float
    ) -> float:
        """Pick a blend window that bounds the acceleration the correction adds.

        The position term peaks at ``_QUINTIC_PEAK_CURVATURE * gap / T**2`` and the
        velocity term at ``_HERMITE_VELOCITY_PEAK_CURVATURE * gap / T``; solving
        each for ``T`` at ``handoff_max_accel`` keeps a large mismatch gentle
        instead of violent. Never shorter than the configured base, and never long
        enough to outlive the plan it is correcting.
        """
        if self._handoff_blend_s <= 0.0:
            return 0.0
        if position_gap <= 0.0 and velocity_gap <= 0.0:
            return 0.0
        blend = self._handoff_blend_s
        if position_gap > 0.0:
            blend = max(
                blend,
                math.sqrt(
                    _QUINTIC_PEAK_CURVATURE * position_gap / self._handoff_max_accel
                ),
            )
        if velocity_gap > 0.0:
            blend = max(
                blend,
                _HERMITE_VELOCITY_PEAK_CURVATURE
                * velocity_gap
                / self._handoff_max_accel,
            )
        if remaining_s > 0.0:
            blend = min(blend, 0.5 * remaining_s)
        return max(blend, 0.0)

    def _sample(self, plan: _Plan, now: float) -> np.ndarray:
        """Commanded pose for a plan, including any active handoff correction."""
        out = np.asarray(plan.spline(self._spline_time(plan, now)), dtype=np.float64)
        if plan.blend_s <= 0.0:
            return out
        u = (now - plan.installed_at) / plan.blend_s
        if u >= 1.0:
            return out
        if plan.offset is not None:
            out = out + plan.offset * _handoff_decay(u)
        if plan.velocity_offset is not None:
            out = out + plan.velocity_offset * (
                plan.blend_s * _handoff_velocity_decay(u)
            )
        return out

    def _sample_velocity(self, plan: _Plan, now: float) -> np.ndarray:
        """Commanded velocity in real time, matching what _sample produces."""
        raw_time = plan.start_time + (now - plan.installed_at) * self._source_rate
        if plan.deriv is None:
            velocity = np.zeros(len(self._channels) - 1, dtype=np.float64)
        else:
            velocity = (
                np.asarray(plan.deriv(self._spline_time(plan, now)), dtype=np.float64)
                * self._source_rate
            )
            # Outside the domain the pose is held, so it is not moving at all.
            if not plan.min_time < raw_time < plan.max_time:
                velocity = np.zeros_like(velocity)
        if plan.blend_s <= 0.0:
            return velocity
        u = (now - plan.installed_at) / plan.blend_s
        if u >= 1.0:
            return velocity
        if plan.offset is not None:
            velocity = velocity + plan.offset * (_handoff_decay_rate(u) / plan.blend_s)
        if plan.velocity_offset is not None:
            velocity = velocity + plan.velocity_offset * (
                _handoff_velocity_decay_rate(u)
            )
        return velocity

    def _align(
        self,
        spline: BSpline,
        target: np.ndarray,
        observation_age_s: float,
    ) -> tuple[float, float, bool, bool, float]:
        min_time = float(spline.t[self._degree])
        max_time = float(spline.t[-self._degree - 1])
        max_allowed = min_time + (max_time - min_time) * self._alignment_max_fraction
        initial_max = float(
            np.clip(
                min_time + max(0.0, observation_age_s) * self._source_rate,
                min_time,
                max_allowed,
            )
        )
        indices = self._aligned_indices

        def objective(t: float) -> float:
            return float(np.abs(np.asarray(spline(t))[indices] - target[indices]).sum())

        # The threshold governs widening only. Gating the loop on the error at
        # min_time skipped the search on every real replan; see BSPLINE_ROLLOUT.md.
        best_time = min_time
        endpoint_error = self._alignment_error(spline, target, min_time)
        best_error = endpoint_error
        searched = False
        capped = False
        scale = 1.0
        while True:
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
            candidate = float(result.x)
            error = self._alignment_error(spline, target, candidate)
            searched = True
            # Bounded minimize_scalar can land worse than an endpoint on a
            # non-unimodal objective, so min_time stays a candidate.
            if error < best_error:
                best_time, best_error = candidate, error
            if upper >= max_allowed:
                capped = True
                break
            if best_error <= self._alignment_threshold or scale > 20:
                break
            scale *= 1.5
        return best_time, best_error, searched, capped, endpoint_error

    def install(
        self,
        flat_action: Sequence[float] | np.ndarray,
        *,
        observation_age_s: float,
        now: float | None = None,
    ) -> BSplineInstallResult:
        spline = self._decode(flat_action)
        install_time = self._clock() if now is None else float(now)
        min_time = float(spline.t[self._degree])
        max_time = float(spline.t[-self._degree - 1])

        deriv = spline.derivative(1)
        with self._condition:
            offset: np.ndarray | None = None
            velocity_offset: np.ndarray | None = None
            blend_s = 0.0
            align_searched = False
            align_capped = False
            align_endpoint_error = 0.0
            previous = self._plan
            if previous is None or self._last_raw_command is None:
                start_time = float(np.clip(0.0, min_time, max_time))
                alignment_error = 0.0
            else:
                # Align and blend against where the outgoing plan is *at this
                # instant*, not the pose sent on the last tick. Using the stale
                # value makes the correction replay a pose up to one control period
                # old, which commands a momentary dead stop at every handoff.
                target = self._sample(previous, install_time)
                (
                    start_time,
                    alignment_error,
                    align_searched,
                    align_capped,
                    align_endpoint_error,
                ) = self._align(spline, target, observation_age_s)
                indices = self._aligned_indices
                # Alignment returns the closest approach, not an exact match, and it
                # never considers velocity. Carry both mismatches as decaying terms.
                residual = np.zeros_like(target)
                residual[indices] = (
                    target[indices]
                    - np.asarray(spline(start_time), dtype=np.float64)[indices]
                )
                rate_residual = np.zeros_like(target)
                rate_residual[indices] = (
                    self._sample_velocity(previous, install_time)[indices]
                    - np.asarray(deriv(start_time), dtype=np.float64)[indices]
                    * self._source_rate
                )
                remaining_s = (max_time - start_time) / self._source_rate
                blend_s = self._blend_duration(
                    float(np.max(np.abs(residual[indices]))),
                    float(np.max(np.abs(rate_residual[indices]))),
                    remaining_s,
                )
                if blend_s > 0.0:
                    if np.any(residual):
                        offset = residual
                    if np.any(rate_residual):
                        velocity_offset = rate_residual
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
                deriv=deriv,
                offset=offset,
                velocity_offset=velocity_offset,
                blend_s=blend_s,
            )
            self._condition.notify_all()
        return BSplineInstallResult(
            start_time,
            alignment_error,
            warning,
            blend_s,
            align_searched=align_searched,
            align_capped=align_capped,
            align_endpoint_error=align_endpoint_error,
        )

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
            # The blend runs on real seconds, not spline time: the smoothness that
            # matters is the robot's, so speed_scale must not change how fast the
            # handoff correction is retired.
            raw = self._sample(plan, current_time)
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
            # Must be the blended pose that was actually sent, not the bare spline
            # sample: the next handoff aligns against this, and matching a pose the
            # robot never received would silently reintroduce the step.
            self._last_raw_command = raw.copy()
            self._last_gripper_targets = grippers
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
            return dict(self._last_gripper_targets)

    @property
    def last_gripper_targets(self) -> dict[str, float]:
        with self._condition:
            return dict(self._last_gripper_targets)

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
