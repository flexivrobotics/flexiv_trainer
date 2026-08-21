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

"""Sign canonicalization for recorded TCP-pose quaternions.

``q`` and ``-q`` are the same rotation; the robot's reported sign flips
mid-trajectory on this rig (tool-down, |q_w| ~ 0.01, so ``q_w >= 0`` is
useless). Sign is chosen for continuity: state follows the previous canonical
state of the arm, action follows the state at the same frame. Only the sign is
touched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

_QUAT_AXES = ("q_w", "q_x", "q_y", "q_z")
_POSE_LABEL = "tcp_pose"


def _as_quaternion(values: Sequence[float]) -> np.ndarray | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        return None
    if float(np.dot(array, array)) < 1e-12:
        return None
    return array


def _same_hemisphere(quat: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    if reference is None or float(np.dot(quat, reference)) >= 0.0:
        return quat
    return -quat


class QuaternionSignCanonicalizer:
    """``reference`` maps a side label to a ``[q_w, q_x, q_y, q_z]`` orientation
    that seeds the first frame of each episode; unset sides continue the sign
    of the previous episode, or keep the reported sign."""

    def __init__(self, reference: Mapping[str, Sequence[float]] | None = None) -> None:
        self._reference: dict[str, np.ndarray] = {}
        for side, values in (reference or {}).items():
            quat = _as_quaternion(values)
            if quat is None:
                raise ValueError(
                    f"Quaternion reference for {side!r} must be 4 finite non-zero "
                    f"values [q_w, q_x, q_y, q_z], got {list(values)!r}"
                )
            self._reference[str(side)] = quat / np.linalg.norm(quat)
        self._last_state: dict[str, np.ndarray] = {}
        self._last_action: dict[str, np.ndarray] = {}
        self._episode_seed: dict[str, np.ndarray] = {}

    def begin_episode(self) -> None:
        for side, quat in self._last_state.items():
            self._episode_seed[side] = quat
        for side, quat in self._last_action.items():
            self._episode_seed.setdefault(side, quat)
        self._last_state.clear()
        self._last_action.clear()

    def _seed_for(self, side: str) -> np.ndarray | None:
        reference = self._reference.get(side)
        if reference is not None:
            return reference
        return self._episode_seed.get(side)

    def canonical_state(self, side: str, values: Sequence[float]) -> list[float]:
        quat = _as_quaternion(values)
        if quat is None:
            return [float(v) for v in values]
        anchor = self._last_state.get(side)
        if anchor is None:
            anchor = self._seed_for(side)
        quat = _same_hemisphere(quat, anchor)
        self._last_state[side] = quat
        return quat.tolist()

    def canonical_action(self, side: str, values: Sequence[float]) -> list[float]:
        quat = _as_quaternion(values)
        if quat is None:
            return [float(v) for v in values]
        anchor = self._last_state.get(side)
        if anchor is None:
            anchor = self._last_action.get(side)
        if anchor is None:
            anchor = self._seed_for(side)
        quat = _same_hemisphere(quat, anchor)
        self._last_action[side] = quat
        return quat.tolist()

    def apply(
        self, values: Sequence[float], names: Sequence[str], kind: str
    ) -> list[float]:
        """Canonicalize every complete ``<side>.tcp_pose.q_*`` group in ``values``."""
        if kind not in ("state", "action"):
            raise ValueError(f"kind must be 'state' or 'action', got {kind!r}")
        result = [float(v) for v in values]
        index_by_name = {name: index for index, name in enumerate(names)}
        suffix = f".{_POSE_LABEL}.{_QUAT_AXES[0]}"
        for name in names:
            if not name.endswith(suffix):
                continue
            side = name[: -len(suffix)]
            indices = [
                index_by_name.get(f"{side}.{_POSE_LABEL}.{axis}") for axis in _QUAT_AXES
            ]
            if any(index is None for index in indices):
                continue
            quat = [result[index] for index in indices]  # type: ignore[index]
            canonical = (
                self.canonical_state(side, quat)
                if kind == "state"
                else self.canonical_action(side, quat)
            )
            for index, value in zip(indices, canonical):
                result[index] = value  # type: ignore[index]
        return result
