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

"""State columns computed from the arm's recent poses, as UMI observes them.

A UMI export (flexiv_gdk ``03_convert_to_lerobot.py``) can add two pose inputs
the robot does not report:

- ``<side>.tcp_motion_<ms>ms.{x,y,z,r1_x..r2_z}``: the TCP pose ``<ms>`` ms
  earlier in the current TCP frame, ``T_cur⁻¹ · T_past`` (rows of R), with
  ``<side>.gripper.width_change_<ms>ms`` the width change since then;
- ``<side>.tcp_rotation_wrt_start.{r1_x..r2_z}``: the rows of R of
  ``T_start⁻¹ · T_cur``, the rotation since the episode (rollout) began.

``PoseHistory`` keeps the recent TCP poses and widths each tick and computes
these columns the way the export does: past values are interpolated in time
(lerp, slerp) and clamp to the oldest sample, so a fresh history reads no
motion.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__ = ["HistorySpec", "PoseHistory", "history_columns"]

_MOTION = re.compile(r"^(?P<side>[^.]+)\.tcp_motion_(?P<ms>\d+)ms\.(?P<axis>[^.]+)$")
_WIDTH_CHANGE = re.compile(r"^(?P<side>[^.]+)\.gripper\.width_change_(?P<ms>\d+)ms$")
_WRT_START = re.compile(r"^(?P<side>[^.]+)\.tcp_rotation_wrt_start\.(?P<axis>[^.]+)$")
# Index of each axis name in the [xyz, first row of R, second row of R] vector.
_POSE_AXES = ("x", "y", "z", "r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
_ROTATION_AXES = _POSE_AXES[3:]
# History kept beyond the delay, so a slow tick still finds its past sample.
_HISTORY_MARGIN_S = 1.0


@dataclass(frozen=True)
class HistorySpec:
    """The history columns a checkpoint was trained on, in state order."""

    names: tuple[str, ...]
    delay_s: float | None


def history_columns(state_names: list[str] | None) -> HistorySpec | None:
    """The trained state names this module computes, or None when there are none."""
    names: list[str] = []
    delays: set[int] = set()
    for name in state_names or []:
        motion = _MOTION.match(name)
        width = _WIDTH_CHANGE.match(name)
        start = _WRT_START.match(name)
        if motion is not None and motion["axis"] in _POSE_AXES:
            delays.add(int(motion["ms"]))
        elif width is not None:
            delays.add(int(width["ms"]))
        elif start is None or start["axis"] not in _ROTATION_AXES:
            continue
        names.append(name)
    if not names:
        return None
    if len(delays) > 1:
        raise ValueError(f"State names mix motion delays: {sorted(delays)} ms")
    delay_s = delays.pop() / 1000.0 if delays else None
    return HistorySpec(names=tuple(names), delay_s=delay_s)


@dataclass
class _Sample:
    t: float
    position: np.ndarray
    rotation: Rotation
    width: float | None


def _pose_vector(position: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([position, matrix[0], matrix[1]])


class PoseHistory:
    """Recent TCP poses and gripper widths per arm, for the history columns."""

    def __init__(self, spec: HistorySpec, sides: list[str]) -> None:
        self.spec = spec
        self._sides = list(sides)
        self._samples: dict[str, deque[_Sample]] = {side: deque() for side in sides}
        self._start: dict[str, _Sample] = {}

    def reset(self) -> None:
        """Forget the motion, e.g. after a takeover; the start pose stays."""
        for samples in self._samples.values():
            samples.clear()

    def update(self, t: float, snapshot: dict[str, Any] | None) -> None:
        """Add this tick's measured pose and width of every arm.

        ``read_robot_snapshot`` lists ``robot_0..robot_N`` in ``sides`` order.
        """
        payloads = list(((snapshot or {}).get("robots") or {}).values())
        for side, payload in zip(self._sides, payloads, strict=False):
            states = payload.get("states") if isinstance(payload, dict) else None
            pose = states.get("tcp_pose") if isinstance(states, dict) else None
            if not isinstance(pose, list | tuple) or len(pose) != 7:
                continue
            w, x, y, z = (float(v) for v in pose[3:])
            gripper = payload.get("gripper")
            width = gripper.get("width") if isinstance(gripper, dict) else None
            sample = _Sample(
                t=float(t),
                position=np.asarray(pose[:3], dtype=np.float64),
                rotation=Rotation.from_quat([x, y, z, w]),
                width=float(width) if isinstance(width, int | float) else None,
            )
            self._start.setdefault(side, sample)
            samples = self._samples[side]
            samples.append(sample)
            keep_from = sample.t - (self.spec.delay_s or 0.0) - _HISTORY_MARGIN_S
            while len(samples) > 1 and samples[1].t <= keep_from:
                samples.popleft()

    def values(self) -> np.ndarray:
        """The history columns in trained order, as of each arm's newest sample."""
        cache: dict[str, dict[str, float]] = {}
        out = []
        for name in self.spec.names:
            side = name.split(".", 1)[0]
            if side not in cache:
                cache[side] = self._side_values(side)
            out.append(cache[side][name])
        return np.asarray(out, dtype=np.float32)

    def _side_values(self, side: str) -> dict[str, float]:
        samples = self._samples.get(side)
        if not samples:
            raise RuntimeError(f"No measured TCP pose for {side} yet")
        current = samples[-1]
        current_matrix = current.rotation.as_matrix()
        values: dict[str, float] = {}
        if self.spec.delay_s is not None:
            ms = int(round(self.spec.delay_s * 1000))
            past = self._at(samples, current.t - self.spec.delay_s)
            # T_cur⁻¹ · T_past: the past pose in the current TCP frame.
            position = current_matrix.T @ (past.position - current.position)
            matrix = current_matrix.T @ past.rotation.as_matrix()
            for axis, value in zip(
                _POSE_AXES, _pose_vector(position, matrix), strict=True
            ):
                values[f"{side}.tcp_motion_{ms}ms.{axis}"] = float(value)
            if current.width is not None and past.width is not None:
                values[f"{side}.gripper.width_change_{ms}ms"] = (
                    current.width - past.width
                )
        start = self._start[side]
        relative = start.rotation.as_matrix().T @ current_matrix
        rows = (*relative[0], *relative[1])
        for axis, value in zip(_ROTATION_AXES, rows, strict=True):
            values[f"{side}.tcp_rotation_wrt_start.{axis}"] = float(value)
        missing = [
            n for n in self.spec.names if n.startswith(f"{side}.") and n not in values
        ]
        if missing:
            raise RuntimeError(
                f"Cannot compute {missing}: the gripper of {side} reports no width"
            )
        return values

    @staticmethod
    def _at(samples: deque[_Sample], t: float) -> _Sample:
        """The pose and width at time t, interpolated; clamped to the oldest."""
        times = [s.t for s in samples]
        i = bisect_right(times, t)
        if i == 0:
            return samples[0]
        if i == len(samples):
            return samples[-1]
        before, after = samples[i - 1], samples[i]
        span = after.t - before.t
        if span <= 0:
            return after
        a = (t - before.t) / span
        rotations = Rotation.concatenate([before.rotation, after.rotation])
        width = None
        if before.width is not None and after.width is not None:
            width = (1 - a) * before.width + a * after.width
        return _Sample(
            t=t,
            position=(1 - a) * before.position + a * after.position,
            rotation=Slerp([0.0, 1.0], rotations)([a])[0],
            width=width,
        )
