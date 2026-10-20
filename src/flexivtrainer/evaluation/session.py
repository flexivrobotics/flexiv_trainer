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

"""Planned evaluation sessions: which checkpoints, which start conditions, what order.

A session is a directory under the rollouts root holding a `session.json` plan
and one recorded rollout per trial:

    rollouts/<session>/session.json
    rollouts/<session>/<episode>/          a LeRobot dataset + meta/rollout_outcome.json

The plan is written by whoever prepares the comparison, before any trial runs:

    {
      "about": "new defaults against old",
      "arms": {
        "A": {"checkpoint": "training/run_a/checkpoints/100000/pretrained_model"},
        "B": {"checkpoint": "training/run_b/checkpoints/100000/pretrained_model"}
      },
      "conditions": [
        {"id": "home-1", "description": "start from HOME"},
        {"id": "rim-1", "description": "stick resting on the rim of the slot"}
      ],
      "trials_per_condition": 3,
      "seed": 7,
      "blind": true,
      "failure_modes": ["missed the target", "stalled", "dropped the part"],
      "checks": {"placed": {"kind": "settled_in_box", ...}}
    }

`checks` are optional automatic checks, scored from each trial's tick log and
reported beside the verdicts with how often they agree (`checks.py`).

WHY A PLAN AND NOT AD HOC STARTS. Deciding the trials before seeing results is
what stops a comparison from being steered by them -- "one more try" for the
policy one hopes wins. The order is shuffled by a fixed seed, in blocks that
cover every (arm, condition) pair once before any repeats, so a slow drift in
lighting or wear lands evenly on both arms.

WHY BLIND. The operator judges each trial, and an operator who knows which
checkpoint is running judges the favourite kindly. With `blind` the interface
shows only the arm's label; the plan file is the only place the mapping lives.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checks import Check, agreement, parse_checks
from .outcome import DEFAULT_FAILURE_MODES, OUTCOME_FILE, RolloutOutcome, read_outcome
from .stats import paired_exact_p, wilson_interval

__all__ = [
    "PLAN_FILE",
    "ADHOC_SESSION",
    "ArmSpec",
    "Condition",
    "EvaluationPlan",
    "Trial",
    "schedule",
    "SessionStore",
    "summarize",
]

PLAN_FILE = "session.json"
#: Where a recorded rollout that belongs to no session is filed.
ADHOC_SESSION = "adhoc"

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")


@dataclass(frozen=True)
class ArmSpec:
    """One checkpoint under test. `checkpoint` is a local path (absolute, or
    relative to the storage root) or, with `source` "hub", `repo_id`."""

    checkpoint: str | None = None
    source: str = "local"
    repo_id: str | None = None
    revision: str | None = None
    task: str | None = None
    send_twist: bool | None = None

    def __post_init__(self) -> None:
        if self.source == "local" and not self.checkpoint:
            raise ValueError("a local arm needs a checkpoint path")
        if self.source == "hub" and not self.repo_id:
            raise ValueError("a hub arm needs a repo_id")
        if self.source not in ("local", "hub"):
            raise ValueError(f"arm source {self.source!r} is not local or hub")

    @property
    def origin(self) -> str:
        return self.repo_id if self.source == "hub" else str(self.checkpoint)


@dataclass(frozen=True)
class Condition:
    """A start condition the operator sets up before a trial."""

    id: str
    description: str = ""


@dataclass(frozen=True)
class Trial:
    index: int
    arm: str
    condition: str
    repetition: int


@dataclass(frozen=True)
class EvaluationPlan:
    name: str
    arms: dict[str, ArmSpec]
    conditions: tuple[Condition, ...]
    trials_per_condition: int = 1
    seed: int = 0
    blind: bool = True
    failure_modes: tuple[str, ...] = DEFAULT_FAILURE_MODES
    about: str = ""
    checks: dict[str, Check] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("a session needs at least one arm")
        for label in self.arms:
            if not _LABEL.match(label):
                raise ValueError(
                    f"arm label {label!r} must be 1-32 letters, digits, '.', '_' or '-'"
                )
        if not self.conditions:
            raise ValueError("a session needs at least one start condition")
        ids = [c.id for c in self.conditions]
        bad = [i for i in ids if not _LABEL.match(i)]
        if bad:
            raise ValueError(
                f"condition ids {bad} must be 1-32 letters, digits, '.', '_' or '-'"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"condition ids repeat: {ids}")
        if self.trials_per_condition < 1:
            raise ValueError("trials_per_condition must be at least 1")
        modes = [m.strip() for m in self.failure_modes]
        if not modes or any(not m for m in modes) or len(set(modes)) != len(modes):
            raise ValueError("failure_modes must be non-empty and distinct")

    def condition(self, condition_id: str) -> Condition:
        for candidate in self.conditions:
            if candidate.id == condition_id:
                return candidate
        raise KeyError(condition_id)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> EvaluationPlan:
        known = {
            "about",
            "arms",
            "conditions",
            "trials_per_condition",
            "seed",
            "blind",
            "failure_modes",
            "checks",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown session keys {unknown}; known are {sorted(known)}"
            )
        arms_data = data.get("arms")
        if not isinstance(arms_data, dict):
            raise ValueError("'arms' must map a label to a checkpoint")
        arms = {str(label): ArmSpec(**spec) for label, spec in arms_data.items()}
        conditions_data = data.get("conditions")
        if not isinstance(conditions_data, list):
            raise ValueError("'conditions' must be a list")
        conditions = tuple(
            Condition(id=str(c["id"]), description=str(c.get("description", "")))
            if isinstance(c, dict)
            else Condition(id=str(c))
            for c in conditions_data
        )
        return cls(
            name=name,
            arms=arms,
            conditions=conditions,
            trials_per_condition=int(data.get("trials_per_condition", 1)),
            seed=int(data.get("seed", 0)),
            blind=bool(data.get("blind", True)),
            failure_modes=tuple(data.get("failure_modes") or DEFAULT_FAILURE_MODES),
            about=str(data.get("about", "")),
            checks=parse_checks(data.get("checks")),
        )


def schedule(plan: EvaluationPlan) -> list[Trial]:
    """Every (arm, condition) pair `trials_per_condition` times, shuffled in blocks.

    Deterministic: the same plan always yields the same order, so a session can
    be resumed after a restart and its trial numbers mean the same thing.
    """
    trials: list[Trial] = []
    for repetition in range(plan.trials_per_condition):
        block = [(arm, c.id) for arm in sorted(plan.arms) for c in plan.conditions]
        random.Random(f"{plan.seed}:{repetition}").shuffle(block)
        for arm, condition in block:
            trials.append(Trial(len(trials), arm, condition, repetition))
    return trials


def summarize(plan: EvaluationPlan, outcomes: list[RolloutOutcome]) -> dict[str, Any]:
    """Success rates per arm and per condition, and paired comparisons.

    Only "success" and "failure" verdicts count; "unknown" is reported but kept
    out of every rate, because a trial nobody could judge is not evidence. Each
    of the plan's checks is reported by how often it agrees with those verdicts.
    """
    by_trial = {t.index: t for t in schedule(plan)}
    judged: dict[int, str] = {}
    results: dict[int, dict[str, Any]] = {}
    unknown: dict[str, int] = {arm: 0 for arm in plan.arms}
    modes: dict[str, dict[str, int]] = {arm: {} for arm in plan.arms}
    for outcome in outcomes:
        if (
            outcome.trial is None
            or outcome.trial not in by_trial
            or outcome.verdict is None
        ):
            continue
        trial = by_trial[outcome.trial]
        if outcome.verdict.outcome == "unknown":
            unknown[trial.arm] += 1
            continue
        judged[trial.index] = outcome.verdict.outcome
        results[trial.index] = outcome.checks or {}
        if outcome.verdict.outcome == "failure":
            mode = outcome.verdict.failure_mode or "unspecified"
            modes[trial.arm][mode] = modes[trial.arm].get(mode, 0) + 1

    def rate(cells: list[int]) -> dict[str, Any]:
        n = len(cells)
        k = sum(1 for index in cells if judged[index] == "success")
        low, high = wilson_interval(k, n)
        return {
            "successes": k,
            "trials": n,
            "rate": (k / n) if n else None,
            "ci95": [low, high],
        }

    arms: dict[str, Any] = {}
    for arm in sorted(plan.arms):
        mine = [i for i in judged if by_trial[i].arm == arm]
        arms[arm] = {
            **rate(mine),
            "unknown": unknown[arm],
            "failure_modes": dict(sorted(modes[arm].items(), key=lambda kv: -kv[1])),
            "conditions": {
                c.id: rate([i for i in mine if by_trial[i].condition == c.id])
                for c in plan.conditions
            },
        }

    comparisons = []
    labels = sorted(plan.arms)
    for a_index, a in enumerate(labels):
        for b in labels[a_index + 1 :]:
            cells: dict[tuple[str, int], dict[str, str]] = {}
            for index, verdict in judged.items():
                trial = by_trial[index]
                if trial.arm in (a, b):
                    cells.setdefault((trial.condition, trial.repetition), {})[
                        trial.arm
                    ] = verdict
            pairs = [cell for cell in cells.values() if a in cell and b in cell]
            only_a = sum(1 for p in pairs if p[a] == "success" and p[b] == "failure")
            only_b = sum(1 for p in pairs if p[b] == "success" and p[a] == "failure")
            comparisons.append(
                {
                    "arms": [a, b],
                    "pairs": len(pairs),
                    "only_" + a: only_a,
                    "only_" + b: only_b,
                    "both": sum(1 for p in pairs if p[a] == p[b] == "success"),
                    "neither": sum(1 for p in pairs if p[a] == p[b] == "failure"),
                    "p_value": paired_exact_p(only_a, only_b),
                }
            )
    checks = {
        name: agreement(
            [(results[index].get(name), verdict) for index, verdict in judged.items()]
        )
        for name in plan.checks
    }
    return {"arms": arms, "comparisons": comparisons, "checks": checks}


@dataclass
class SessionStore:
    """The sessions under one rollouts root."""

    root: Path

    def sessions(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.parent.name for p in self.root.glob(f"*/{PLAN_FILE}"))

    def plan(self, name: str) -> EvaluationPlan:
        path = self._session_dir(name) / PLAN_FILE
        if not path.is_file():
            raise FileNotFoundError(f"no evaluation session named {name!r}")
        return EvaluationPlan.from_dict(
            name, json.loads(path.read_text(encoding="utf-8"))
        )

    def outcomes(self, name: str) -> list[RolloutOutcome]:
        return [outcome for _, outcome in self.episodes(name)]

    def episodes(self, name: str) -> list[tuple[Path, RolloutOutcome]]:
        """Each recorded trial's directory and outcome."""
        directory = self._session_dir(name)
        found = []
        for path in sorted(directory.glob(f"*/{OUTCOME_FILE.as_posix()}")):
            outcome = read_outcome(path.parent.parent)
            if outcome is not None and outcome.session == name:
                found.append((path.parent.parent, outcome))
        return found

    def progress(self, name: str) -> dict[str, Any]:
        """Where a session stands, and the next trial to run.

        The next trial is held back while any trial waits for its verdict: a
        verdict given after the next trial ran is a verdict given from memory.
        """
        plan = self.plan(name)
        trials = schedule(plan)
        outcomes = self.outcomes(name)
        done = {o.trial for o in outcomes if o.judged}
        awaiting = sorted(
            (
                {"trial": o.trial, "episode": o.episode}
                for o in outcomes
                if not o.judged and o.trial is not None
            ),
            key=lambda entry: entry["trial"],
        )
        waiting = {entry["trial"] for entry in awaiting}
        upcoming = next(
            (t for t in trials if t.index not in done and t.index not in waiting), None
        )
        return {
            "name": name,
            "about": plan.about,
            "blind": plan.blind,
            "total": len(trials),
            "judged": len(done),
            "awaiting_verdict": awaiting,
            "next": None
            if upcoming is None or awaiting
            else {
                "trial": upcoming.index,
                "arm": upcoming.arm,
                "condition": upcoming.condition,
                "description": plan.condition(upcoming.condition).description,
            },
            "failure_modes": list(plan.failure_modes),
            "summary": summarize(plan, outcomes),
        }

    def _session_dir(self, name: str) -> Path:
        if not _LABEL.match(name) or name == ADHOC_SESSION:
            raise ValueError(f"{name!r} is not a session name")
        return self.root / name
