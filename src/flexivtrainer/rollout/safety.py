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

"""A safety envelope around a policy's commands.

The controller has its own limits, and they stay the last line. These checks
sit before it and stop a rollout -- they never quietly clamp a command, because
a clamped policy is a different policy and its rollout would be evaluated as
the one that was trained.

ALWAYS ON: every commanded number must be finite. A NaN pose sent to the arm is
undefined behaviour in the one place that can hurt someone, and a policy
produces NaN without raising -- an input it cannot read is enough.

OPT-IN, because the right numbers depend on the cell and the task:

  workspace_min_m / workspace_max_m   every commanded position inside this box,
                                      in the arm's own base frame
  max_reference_offset_m              how far a fresh plan's first target may be
                                      from where the arm measurably is. Under a
                                      pose-target impedance controller distance
                                      IS force (2 cm at 3000 N/m is 60 N), so a
                                      far target is a hard push
  max_force_n                         measured external force on any arm
  stale_plan_s                        longest wait for a fresh plan; the arm holds
                                      the last target while the planner is stuck
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["SafetyEnvelope", "SafetyStop"]


class SafetyStop(RuntimeError):
    """A command or a measurement left the envelope. `code` names which check."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"Safety stop ({code}): {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class SafetyEnvelope:
    workspace_min_m: tuple[float, float, float] | None = None
    workspace_max_m: tuple[float, float, float] | None = None
    max_reference_offset_m: float | None = None
    max_force_n: float | None = None
    stale_plan_s: float | None = None

    def __post_init__(self) -> None:
        if (self.workspace_min_m is None) != (self.workspace_max_m is None):
            raise ValueError("give both workspace corners or neither")
        if self.workspace_min_m is not None:
            low, high = self.workspace_min_m, self.workspace_max_m
            if (
                len(low) != 3
                or len(high) != 3
                or any(a >= b for a, b in zip(low, high))
            ):
                raise ValueError(f"workspace box {low}..{high} is not a box")
        for name in ("max_reference_offset_m", "max_force_n", "stale_plan_s"):
            value = getattr(self, name)
            if value is not None and not (value > 0 and math.isfinite(value)):
                raise ValueError(f"{name} must be a positive number, got {value}")

    @classmethod
    def from_settings(cls, rollout: Any) -> SafetyEnvelope:
        def triple(value: Any) -> tuple[float, float, float] | None:
            return None if value is None else tuple(float(v) for v in value)

        return cls(
            workspace_min_m=triple(getattr(rollout, "workspace_min_m", None)),
            workspace_max_m=triple(getattr(rollout, "workspace_max_m", None)),
            max_reference_offset_m=getattr(rollout, "max_reference_offset_m", None),
            max_force_n=getattr(rollout, "max_force_n", None),
            stale_plan_s=getattr(rollout, "stale_plan_s", None),
        )

    def describe(self) -> str:
        parts = ["finite commands"]
        if self.workspace_min_m is not None:
            parts.append(
                f"workspace {list(self.workspace_min_m)}.."
                f"{list(self.workspace_max_m)} m"
            )
        if self.max_reference_offset_m is not None:
            parts.append(f"reference offset <= {self.max_reference_offset_m} m")
        if self.max_force_n is not None:
            parts.append(f"force <= {self.max_force_n} N")
        if self.stale_plan_s is not None:
            parts.append(f"fresh plan within {self.stale_plan_s} s")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_min_m": list(self.workspace_min_m)
            if self.workspace_min_m
            else None,
            "workspace_max_m": list(self.workspace_max_m)
            if self.workspace_max_m
            else None,
            "max_reference_offset_m": self.max_reference_offset_m,
            "max_force_n": self.max_force_n,
            "stale_plan_s": self.stale_plan_s,
        }

    # --- the checks ------------------------------------------------------

    def check_finite(self, actions: Sequence[Sequence[float]]) -> None:
        """Every commanded number finite. Run before anything reads a pose out
        of the actions: converting a NaN rotation is itself undefined."""
        for step, action in enumerate(actions):
            for index, value in enumerate(action):
                if not math.isfinite(float(value)):
                    raise SafetyStop(
                        "non_finite", f"action {step} has {value} at index {index}"
                    )

    def check_workspace(self, positions: Sequence[Sequence[Sequence[float]]]) -> None:
        """Every commanded position inside the box; `positions[k][arm]` is xyz."""
        if self.workspace_min_m is None:
            return
        low, high = self.workspace_min_m, self.workspace_max_m
        for step, arms in enumerate(positions):
            for arm, xyz in enumerate(arms):
                for axis, (value, a, b) in enumerate(zip(xyz, low, high)):
                    if not (a <= value <= b):
                        raise SafetyStop(
                            "workspace",
                            f"step {step} arm {arm} commands "
                            f"{'xyz'[axis]}={value:.4f} m "
                            f"outside [{a}, {b}]",
                        )

    def check_measured_workspace(self, measured: Sequence[Sequence[float]]) -> None:
        """Every arm's measured position inside the box: while an operator
        drives the arm there is no command to check, only where it is."""
        if self.workspace_min_m is None:
            return
        low, high = self.workspace_min_m, self.workspace_max_m
        for arm, pose in enumerate(measured):
            for axis, (value, a, b) in enumerate(zip(pose[:3], low, high)):
                if not (a <= value <= b):
                    raise SafetyStop(
                        "workspace",
                        f"arm {arm} measures {'xyz'[axis]}={value:.4f} m "
                        f"outside [{a}, {b}]",
                    )

    def check_reference(
        self, commanded: Sequence[Sequence[float]], measured: Sequence[Sequence[float]]
    ) -> None:
        """A fresh plan's first target near each arm's measured position."""
        if self.max_reference_offset_m is None:
            return
        for arm, (target, actual) in enumerate(zip(commanded, measured)):
            if len(target) < 3 or len(actual) < 3:
                continue
            offset = math.dist(target[:3], actual[:3])
            if offset > self.max_reference_offset_m:
                raise SafetyStop(
                    "reference_offset",
                    f"arm {arm} target is {offset * 1000:.1f} mm from its measured "
                    f"position (limit {self.max_reference_offset_m * 1000:.1f} mm)",
                )

    def check_force(self, wrenches: Sequence[Sequence[float]]) -> None:
        """Measured force magnitude on every arm under the limit."""
        if self.max_force_n is None:
            return
        for arm, wrench in enumerate(wrenches):
            if len(wrench) < 3:
                continue
            force = math.hypot(*(float(v) for v in wrench[:3]))
            if not math.isfinite(force) or force > self.max_force_n:
                raise SafetyStop(
                    "force",
                    f"arm {arm} measures {force:.1f} N (limit {self.max_force_n} N)",
                )

    def check_stale(self, now: float, last_fresh: float | None) -> None:
        if self.stale_plan_s is None or last_fresh is None:
            return
        waited = now - last_fresh
        if waited > self.stale_plan_s:
            raise SafetyStop(
                "stale_plan",
                f"no fresh plan for {waited:.2f} s (limit {self.stale_plan_s} s)",
            )
