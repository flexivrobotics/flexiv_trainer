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

"""Between practice episodes: go to a random start in a declared box.

A generic primitive, not a task script. It knows nothing of what the skill
does, only where its episodes may start: a box in the arm's base frame, the
tool's orientation there, and how far that may turn about the tool's own axis.

It first backs the tool off along its own axis (`retreat_m`), out of whatever
it was touching, then moves to a start drawn at random. Each move is sent once,
slowly, and the arm is read until it is there. Anything the primitive cannot
undo -- a part latched home, a jam -- is for the operator: the practice loop
can wait for them after a success.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from flexivtrainer.practice.space import Sample
from flexivtrainer.rollout.primitives import MoveInterrupted, move_tool
from flexivtrainer.rollout.safety import SafetyEnvelope

__all__ = ["ResetInterrupted", "StartBox", "go_to_start"]

#: Someone else took the arm while it was being reset.
ResetInterrupted = MoveInterrupted


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} is [x, y, z] in metres, got {value!r}")
    numbers = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numbers  # type: ignore[return-value]


@dataclass(frozen=True)
class StartBox:
    """Where practice episodes start, and how the arm gets there."""

    #: Opposite corners of the box, in the arm's base frame (m).
    min_m: tuple[float, float, float]
    max_m: tuple[float, float, float]
    #: The tool's orientation at a start, [qw, qx, qy, qz]; None keeps the one
    #: it has when practice begins.
    rotation: tuple[float, float, float, float] | None = None
    #: A start turns the tool by up to this about its own z, either way (deg).
    yaw_deg: float = 0.0
    #: How far the tool first backs off along its own axis (m).
    retreat_m: float = 0.03
    #: The speed of the moves (m/s).
    speed_m_s: float = 0.05
    #: When the arm counts as there.
    tolerance_m: float = 0.001
    tolerance_deg: float = 0.5
    #: How long a move may take before the reset fails (s).
    timeout_s: float = 20.0

    def __post_init__(self) -> None:
        low, high = _triple(self.min_m, "min_m"), _triple(self.max_m, "max_m")
        if any(a > b for a, b in zip(low, high, strict=True)):
            raise ValueError(f"the start box {list(low)}..{list(high)} is not a box")
        object.__setattr__(self, "min_m", low)
        object.__setattr__(self, "max_m", high)
        if self.rotation is not None:
            if len(self.rotation) != 4:
                raise ValueError("rotation is [qw, qx, qy, qz]")
            object.__setattr__(self, "rotation", tuple(float(v) for v in self.rotation))
        for name in ("yaw_deg", "retreat_m"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be 0 or more, got {value}")
        for name in ("speed_m_s", "tolerance_m", "tolerance_deg", "timeout_s"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive, got {value}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StartBox:
        if not isinstance(data, Mapping):
            raise ValueError("a start box is an object")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown start box settings {unknown}")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("min_m", "max_m", "rotation"):
            if values[key] is not None:
                values[key] = list(values[key])
        return values

    def draw(
        self, rng: np.random.Generator, rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """A start: a point in the box, and `rotation` turned about the tool's z."""
        position = rng.uniform(self.min_m, self.max_m)
        yaw = rng.uniform(-self.yaw_deg, self.yaw_deg) if self.yaw_deg else 0.0
        turn = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        return position, rotation @ turn

    def reference(self, sample: Sample) -> np.ndarray:
        """The tool's orientation at a start: the box's, or the sample's."""
        if self.rotation is None:
            return sample.rotation
        w, x, y, z = self.rotation
        return Rotation.from_quat([x, y, z, w]).as_matrix()


def go_to_start(
    box: StartBox,
    *,
    read: Callable[[], Sample],
    send: Callable[[Sequence[float], tuple[float, float, float, float]], bool],
    stop: threading.Event,
    rng: np.random.Generator,
    rotation: np.ndarray,
    safety: SafetyEnvelope | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[np.ndarray, np.ndarray]:
    """Back off along the tool's axis, then go to a random start; returns it.

    Raises ResetInterrupted when a send is refused (an operator has the arm),
    and RuntimeError when the arm does not get there in time.
    """

    def move(position: np.ndarray, target: np.ndarray, what: str) -> None:
        move_tool(
            position,
            target,
            read=read,
            send=send,
            stop=stop,
            speed_m_s=box.speed_m_s,
            tolerance_m=box.tolerance_m,
            tolerance_deg=box.tolerance_deg,
            timeout_s=box.timeout_s,
            what=what,
            goal="the start",
            safety=safety,
            clock=clock,
        )

    here = read()
    if box.retreat_m:
        back = here.position - here.rotation[:, 2] * box.retreat_m
        move(back, here.rotation, "backed off")
    position, target = box.draw(rng, rotation)
    move(position, target, "went to the start")
    return position, target
