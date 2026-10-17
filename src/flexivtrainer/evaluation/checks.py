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

"""Automatic checks of a recorded rollout, scored from its per-tick log.

A check is a yes/no question the robot's own measurements can settle, such as
"did the tool come to rest inside this box". Checks are declared in a session's
plan and scored when each trial's recording is saved:

    "checks": {
      "placed": {
        "kind": "settled_in_box",
        "box_min_m": [0.40, -0.05, 0.10],
        "box_max_m": [0.44, -0.01, 0.12],
        "hold_s": 0.5,
        "max_speed_m_s": 0.003
      }
    }

THE OPERATOR'S VERDICT STAYS THE MEASUREMENT. A check earns trust only by
agreeing with verdicts over many trials, which the session summary reports
(`agreement`). Until then it is a candidate, stored beside the verdict. The
interface never shows a check's result next to the verdict prompt: an operator
who sees the machine's answer first is no longer an independent judge, and the
agreement would be measured against itself.

Thresholds are tuned offline, on trials already recorded, with
`python -m flexivtrainer.evaluation.rescore`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stats import wilson_interval

__all__ = [
    "CHECK_KINDS",
    "SettledInBox",
    "agreement",
    "parse_checks",
    "read_ticks",
    "run_checks",
]

TICKS_FILE = Path("meta") / "rollout_ticks.jsonl"


@dataclass(frozen=True)
class SettledInBox:
    """Passes when the tool's measured position stays inside a box, nearly
    still, for at least `hold_s` -- at any time (`when` "any") or at the end of
    the rollout (`when` "end").

    Positions are metres in the arm's base frame. `arm` indexes the arms in
    the tick log, in the order the rollout lists them. The speed is the
    robot's reported tool velocity when the tick carries it, else the change
    in position between ticks.
    """

    box_min_m: tuple[float, float, float]
    box_max_m: tuple[float, float, float]
    hold_s: float = 0.5
    max_speed_m_s: float = 0.005
    when: str = "any"
    arm: int = 0

    kind = "settled_in_box"

    def __post_init__(self) -> None:
        if len(self.box_min_m) != 3 or len(self.box_max_m) != 3:
            raise ValueError("box_min_m and box_max_m need three values each")
        values = [*self.box_min_m, *self.box_max_m, self.hold_s, self.max_speed_m_s]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("a box check's numbers must be finite")
        if any(lo >= hi for lo, hi in zip(self.box_min_m, self.box_max_m)):
            raise ValueError("box_min_m must be below box_max_m on every axis")
        if self.hold_s <= 0 or self.max_speed_m_s <= 0:
            raise ValueError("hold_s and max_speed_m_s must be positive")
        if self.when not in ("any", "end"):
            raise ValueError(f"when must be 'any' or 'end', not {self.when!r}")
        if self.arm < 0:
            raise ValueError("arm must be 0 or more")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SettledInBox:
        known = {
            "kind",
            "box_min_m",
            "box_max_m",
            "hold_s",
            "max_speed_m_s",
            "when",
            "arm",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown keys {unknown}; known are {sorted(known)}")
        if "box_min_m" not in data or "box_max_m" not in data:
            raise ValueError("a box check needs box_min_m and box_max_m")
        return cls(
            box_min_m=tuple(float(v) for v in data["box_min_m"]),
            box_max_m=tuple(float(v) for v in data["box_max_m"]),
            hold_s=float(data.get("hold_s", cls.hold_s)),
            max_speed_m_s=float(data.get("max_speed_m_s", cls.max_speed_m_s)),
            when=str(data.get("when", cls.when)),
            arm=int(data.get("arm", cls.arm)),
        )

    def evaluate(self, ticks: list[dict[str, Any]]) -> dict[str, Any]:
        """`passed`, the longest qualifying stretch, and when the first began."""
        samples = [self._sample(tick) for tick in ticks]
        speeds = self._speeds(samples)
        runs: list[tuple[float, float, bool]] = []  # (start s, end s, reaches the end)
        start: float | None = None
        last: float | None = None
        for index, (sample, speed) in enumerate(zip(samples, speeds)):
            inside = (
                sample is not None
                and speed is not None
                and speed <= self.max_speed_m_s
                and all(
                    lo <= p <= hi
                    for p, lo, hi in zip(sample[1], self.box_min_m, self.box_max_m)
                )
            )
            if inside:
                if start is None:
                    start = sample[0]
                last = sample[0]
            elif start is not None:
                runs.append((start, last, False))
                start = None
            if index == len(samples) - 1 and start is not None:
                runs.append((start, last, True))
        qualifying = [r for r in runs if r[1] - r[0] >= self.hold_s]
        if self.when == "end":
            qualifying = [r for r in qualifying if r[2]]
        return {
            "kind": self.kind,
            "passed": bool(qualifying),
            "held_s": round(max((r[1] - r[0] for r in runs), default=0.0), 3),
            "first_at_s": round(qualifying[0][0], 3) if qualifying else None,
            "ticks": len(ticks),
        }

    def _sample(
        self, tick: dict[str, Any]
    ) -> tuple[float, list[float], list[float] | None] | None:
        try:
            arm = tick["measured"][self.arm]
            pose = [float(v) for v in arm["tcp_pose"][:3]]
            t = float(tick["t"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose + [t]):
            return None
        velocity = arm.get("tcp_vel")
        linear = None
        if isinstance(velocity, list) and len(velocity) >= 3:
            linear = [float(v) for v in velocity[:3]]
            if not all(math.isfinite(v) for v in linear):
                linear = None
        return t, pose, linear

    @staticmethod
    def _speeds(samples: list[Any]) -> list[float | None]:
        speeds: list[float | None] = []
        previous = None
        for sample in samples:
            speed = None
            if sample is not None:
                if sample[2] is not None:
                    speed = math.dist(sample[2], (0.0, 0.0, 0.0))
                elif previous is not None and sample[0] > previous[0]:
                    speed = math.dist(sample[1], previous[1]) / (
                        sample[0] - previous[0]
                    )
            speeds.append(speed)
            previous = sample
        return speeds


CHECK_KINDS = {SettledInBox.kind: SettledInBox}


def parse_checks(data: Any) -> dict[str, SettledInBox]:
    """A plan's `checks` object -> name -> check. Strict: an unknown kind or key,
    or a malformed number, is refused rather than skipped."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("'checks' must map a name to a check")
    checks = {}
    for name, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"check {name!r} must be an object")
        kind = spec.get("kind")
        if kind not in CHECK_KINDS:
            raise ValueError(
                f"check {name!r}: kind {kind!r} is not one of {sorted(CHECK_KINDS)}"
            )
        try:
            checks[str(name)] = CHECK_KINDS[kind].from_dict(spec)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"check {name!r}: {exc}") from None
    return checks


def read_ticks(episode_dir: Path) -> list[dict[str, Any]]:
    """The tick log of one recorded rollout; empty if it has none."""
    path = Path(episode_dir) / TICKS_FILE
    if not path.is_file():
        return []
    ticks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ticks.append(json.loads(line))
    return ticks


def run_checks(
    checks: dict[str, SettledInBox], ticks: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Every check's result. One that cannot be scored records why instead of
    a `passed`, so it is counted as unscored, never as a failure."""
    results = {}
    for name, check in checks.items():
        if not ticks:
            results[name] = {"kind": check.kind, "error": "no tick log"}
            continue
        try:
            results[name] = check.evaluate(ticks)
        except Exception as exc:  # a check must never lose a trial
            results[name] = {
                "kind": check.kind,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def agreement(pairs: list[tuple[dict[str, Any] | None, str]]) -> dict[str, Any]:
    """How one check's results line up with judged verdicts.

    `pairs` is (the check's result or None, verdict) for every trial judged
    "success" or "failure". A trial whose check was not scored is counted
    apart, not as a disagreement.
    """
    scored = [
        (bool(result["passed"]), verdict)
        for result, verdict in pairs
        if isinstance(result, dict) and "passed" in result
    ]
    n = len(scored)
    agree = sum(1 for passed, verdict in scored if passed == (verdict == "success"))
    low, high = wilson_interval(agree, n)
    return {
        "trials": n,
        "agree": agree,
        "rate": (agree / n) if n else None,
        "ci95": [low, high],
        "passed_and_success": sum(1 for p, v in scored if p and v == "success"),
        "passed_but_failure": sum(1 for p, v in scored if p and v == "failure"),
        "failed_but_success": sum(1 for p, v in scored if not p and v == "success"),
        "failed_and_failure": sum(1 for p, v in scored if not p and v == "failure"),
        "unscored": len(pairs) - n,
    }
