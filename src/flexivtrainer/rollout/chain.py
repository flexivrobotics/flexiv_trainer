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

"""Run a task as a chain of short steps joined by checks, with a retry rule.

Each step is a neural skill (any checkpoint a rollout can run) or a primitive
(`primitives.py`). A step may have a start check, which must pass before it
runs, and an end check, which ends it when it passes:

    {
      "steps": [
        {"name": "approach", "skill": "skills/approach", "end": "ready",
         "max_s": 30},
        {"name": "insert", "skill": "skills/insert", "start": "ready",
         "end": "home", "max_s": 20, "tries": 3,
         "back_off": {"moves": [{"tool_m": [0, 0, -0.02]},
                                {"box_min_m": [...], "box_max_m": [...]}],
                      "max_force_n": 25}},
        {"name": "lift", "primitive": {"moves": [{"base_m": [0, 0, 0.05]}]}}
      ],
      "checks": {
        "ready": {"kind": "settled_in_box", ...},
        "home": {"kind": "learned", "model": "checks/home"}
      }
    }

`checks` are declared as in an evaluation session (evaluation/checks.py). A
skill step needs an end check: nothing else says it is done. A primitive step
is done when its moves are, and its end check, if any, has then passed.

THE RETRY RULE. A step fails when its end check has not passed within `max_s`
(or its primitive did not finish). Then:

1. back off: the step's `back_off` primitive takes the arm back towards where
   the step starts, and the step's start check must pass again;
2. run the step again.

After `tries` failed tries the chain stops and calls the operator. So does a
back-off that meets more force than its `max_force_n` (a part half latched:
more force is not the answer), a start check that does not pass, and a step
the safety envelope stopped.

THE OPERATOR, once called, does what the chain could not -- with the leader
arm through teleoperation, or by hand -- and then says how to go on: run the
step again from its start check ("retry"), count it as done ("skip"), or stop.

NOTHING HERE KNOWS THE TASK. The steps, checks, limits and tries are the
plan's; the chain only sequences them. Each step runs as a rollout of its own
(`run_skill`, `run_primitive`, given by the rollout service), so the leader arm
can take over a skill as in any rollout.

Every run is written as it goes to `<rollouts>/chains/<run>.json`: each step's
tries, how each ended and in what time, its back-offs and the operator's calls.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from flexivtrainer.evaluation.checks import Check, parse_checks
from flexivtrainer.rollout.primitives import Primitive

__all__ = ["CHAINS_DIR", "ChainPlan", "ChainRunner", "ChainStep", "Outcome"]

#: Where chain runs are written, under the rollouts root.
CHAINS_DIR = "chains"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")

RETRY = "retry"
SKIP = "skip"


@dataclass(frozen=True)
class ChainStep:
    """One step: a skill or a primitive, with its checks and its retry rule."""

    name: str
    #: A checkpoint (absolute, or relative to the storage root).
    skill: str | None = None
    task: str | None = None
    primitive: Primitive | None = None
    #: Checks, by name in the plan's `checks`.
    start: str | None = None
    end: str | None = None
    #: How long one try may run (s).
    max_s: float = 60.0
    #: Tries before the operator is called.
    tries: int = 1
    #: Back to where the step starts, between tries; None tries again in place.
    back_off: Primitive | None = None

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise ValueError(
                f"step name {self.name!r} must be 1-32 letters, digits, '.', '_' or '-'"
            )
        if (self.skill is None) == (self.primitive is None):
            raise ValueError(f"step {self.name!r} is one skill or one primitive")
        if self.skill is not None and self.end is None:
            raise ValueError(
                f"step {self.name!r} runs a skill, so it needs an end check: "
                "nothing else says it is done"
            )
        if not (math.isfinite(self.max_s) and self.max_s > 0):
            raise ValueError(f"step {self.name!r}: max_s must be positive")
        if isinstance(self.tries, bool) or not isinstance(self.tries, int):
            raise ValueError(f"step {self.name!r}: tries must be a whole number")
        if self.tries < 1:
            raise ValueError(f"step {self.name!r}: tries must be 1 or more")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChainStep:
        if not isinstance(data, Mapping):
            raise ValueError("a step is an object")
        values = dict(data)
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown step settings {unknown}")
        name = str(values.get("name", ""))
        try:
            for key in ("primitive", "back_off"):
                if values.get(key) is not None:
                    values[key] = Primitive.from_dict(values[key])
            if "max_s" in values:
                values["max_s"] = float(values["max_s"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"step {name!r}: {exc}") from None
        return cls(**values)


@dataclass(frozen=True)
class ChainPlan:
    """The steps of a task, in order, and the checks they name."""

    steps: tuple[ChainStep, ...]
    checks: dict[str, Check] = field(default_factory=dict)
    #: The Cartesian stiffness of primitive steps; a back-off moves at its
    #: skill's. None takes the rollout setting.
    stiffness: tuple[float, ...] | None = None
    #: How long a start check may take to pass (s), beyond its own hold.
    check_s: float = 1.0
    about: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise ValueError("a chain needs at least one step")
        names = [step.name for step in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"step names repeat: {names}")
        for step in self.steps:
            for role in ("start", "end"):
                name = getattr(step, role)
                if name is not None and name not in self.checks:
                    raise ValueError(
                        f"step {step.name!r}: its {role} check {name!r} is not "
                        f"one of the plan's checks {sorted(self.checks)}"
                    )
        if self.stiffness is not None:
            values = tuple(float(v) for v in self.stiffness)
            if len(values) != 6 or not all(math.isfinite(v) for v in values):
                raise ValueError("stiffness is six numbers")
            object.__setattr__(self, "stiffness", values)
        if not (math.isfinite(self.check_s) and self.check_s > 0):
            raise ValueError("check_s must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChainPlan:
        if not isinstance(data, Mapping):
            raise ValueError("a chain plan is an object")
        known = {"steps", "checks", "stiffness", "check_s", "about"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown chain settings {unknown}")
        steps = data.get("steps")
        if not isinstance(steps, list):
            raise ValueError("steps is a list")
        return cls(
            steps=tuple(ChainStep.from_dict(step) for step in steps),
            checks=parse_checks(data.get("checks")),
            stiffness=data.get("stiffness"),
            check_s=float(data.get("check_s", cls.check_s)),
            about=str(data.get("about", "")),
        )

    def live(self, name: str) -> dict[str, Check]:
        """Check `name`, scored on every tick and ending its rollout on a pass."""
        return {name: replace(self.checks[name], live=True, end_on_pass=True)}

    def watch_s(self, name: str) -> float:
        """How long to watch check `name` for it to pass."""
        return self.check_s + float(self.checks[name].hold_s)


@dataclass(frozen=True)
class Outcome:
    """How one rollout of a chain ended."""

    passed: bool
    #: Its stop reason: "check X passed", "time limit", "blocked", ...
    reason: str
    seconds: float = 0.0
    error: str | None = None
    safety_stop: dict[str, str] | None = None
    blocked: dict[str, float] | None = None
    #: How often, and for how long, an operator took the arm over.
    takeovers: int = 0
    operator_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "passed": self.passed,
            "reason": self.reason,
            "seconds": round(self.seconds, 2),
        }
        for key in ("error", "safety_stop", "blocked"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.takeovers:
            out["takeovers"] = self.takeovers
            out["operator_s"] = round(self.operator_s, 2)
        return out

    def why(self) -> str:
        """Why the chain cannot go on by itself after this."""
        if self.blocked is not None:
            return (
                f"the back-off met {self.blocked['force_n']:.1f} N, over its "
                f"{self.blocked['limit_n']:g} N limit"
            )
        if self.safety_stop is not None:
            return f"the safety envelope stopped it: {self.safety_stop['detail']}"
        if self.error is not None:
            return self.error
        return self.reason


#: Runs one try of a skill step: (step, its end check) -> how it ended.
RunSkill = Callable[[ChainStep, dict[str, Check]], Outcome]
#: Runs a primitive on behalf of a step, then watches `checks` for `watch_s`:
#: (primitive or None for checks only, checks, watch_s, step) -> how it ended.
RunPrimitive = Callable[[Primitive | None, dict[str, Check], float, ChainStep], Outcome]


class _Stopped(Exception):
    pass


class ChainRunner:
    """Own the thread that runs a chain, one rollout at a time."""

    def __init__(
        self,
        plan: ChainPlan,
        *,
        run_skill: RunSkill,
        run_primitive: RunPrimitive,
        record_path: Path,
        append_log: Callable[..., None],
        on_finished: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._plan = plan
        self._run_skill = run_skill
        self._run_primitive = run_primitive
        self._record_path = Path(record_path)
        self._append_log = append_log
        self._on_finished = on_finished
        self._clock = clock
        self.stop_event = threading.Event()
        self._go_on = threading.Event()
        self._action: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state = "starting"
        self._index = 0
        self._try = 0
        self._waiting: str | None = None
        self._record: dict[str, Any] = {
            "about": plan.about,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "steps": [step.name for step in plan.steps],
            "events": [],
            "result": None,
        }
        self._started = clock()

    # --- the service's side -------------------------------------------------

    def start(self) -> None:
        self._write()
        thread = threading.Thread(target=self._run, daemon=True, name="chain")
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._go_on.set()

    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def go_on(self, action: str) -> None:
        """The operator has done what the chain could not: `retry` the step
        from its start check, or `skip` it as done."""
        if action not in (RETRY, SKIP):
            raise ValueError(f"go on with {RETRY!r} or {SKIP!r}, not {action!r}")
        with self._lock:
            if self._waiting is None:
                raise RuntimeError("The chain is not waiting for the operator")
            self._action = action
        self._go_on.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            step = (
                self._plan.steps[self._index]
                if self._index < len(self._plan.steps)
                else None
            )
            return {
                "state": self._state,
                "step": step.name if step is not None else None,
                "step_index": self._index,
                "steps": [s.name for s in self._plan.steps],
                "try": self._try,
                "tries": step.tries if step is not None else None,
                "waiting_for_operator": self._waiting,
                "record": str(self._record_path),
                "result": self._record["result"],
            }

    # --- the chain ----------------------------------------------------------

    def _run(self) -> None:
        result = "stopped"
        try:
            steps = self._plan.steps
            while self._index < len(steps):
                step = steps[self._index]
                why = self._step(step)
                if why is None:
                    self._index += 1
                    continue
                action = self._wait_for_operator(step, why)
                if action == SKIP:
                    self._event(step, "skipped by the operator")
                    self._index += 1
            result = "done"
        except _Stopped:
            result = "stopped"
        except Exception as exc:
            result = "failed"
            self._event(None, "failed", error=f"{type(exc).__name__}: {exc}")
            self._append_log("ERROR", "CHAIN", "The chain failed", str(exc))
        finally:
            with self._lock:
                self._state = result
                self._waiting = None
                self._record["result"] = result
                self._record["seconds"] = round(self._clock() - self._started, 1)
            self._write()
            self._append_log("INFO", "CHAIN", f"Chain {result}", str(self._record_path))
            if self._on_finished is not None:
                self._on_finished(result)

    def _step(self, step: ChainStep) -> str | None:
        """Run `step` under the retry rule; None when it passed, else why the
        operator is needed."""
        self._check_stop()
        if step.start is not None:
            self._set(step, "checking", 0)
            outcome = self._checks_only(step, step.start)
            self._event(step, "start check", outcome=outcome)
            if not outcome.passed:
                return f"its start check {step.start!r} did not pass: {outcome.why()}"
        for k in range(1, step.tries + 1):
            self._set(step, "running", k)
            outcome = self._try_step(step)
            self._event(step, "try", attempt=k, outcome=outcome)
            if outcome.passed:
                self._append_log(
                    "OK", "CHAIN", f"{step.name} done", f"try {k}, {outcome.reason}"
                )
                return None
            self._check_stop()
            if outcome.safety_stop is not None or outcome.error is not None:
                return f"try {k} stopped: {outcome.why()}"
            if k == step.tries:
                return f"{step.tries} {'try' if step.tries == 1 else 'tries'} failed"
            self._append_log(
                "WARN", "CHAIN", f"{step.name} try {k} failed", outcome.reason
            )
            why = self._back_off(step)
            if why is not None:
                return why
        return None  # pragma: no cover - the loop returns

    def _try_step(self, step: ChainStep) -> Outcome:
        if step.skill is not None:
            assert step.end is not None
            return self._run_skill(step, self._plan.live(step.end))
        checks = self._plan.live(step.end) if step.end is not None else {}
        watch = self._plan.watch_s(step.end) if step.end is not None else 0.0
        return self._run_primitive(step.primitive, checks, watch, step)

    def _back_off(self, step: ChainStep) -> str | None:
        """Back to where `step` starts; None when it got there."""
        if step.back_off is None and step.start is None:
            return None
        self._set(step, "backing off", self._try)
        checks = self._plan.live(step.start) if step.start is not None else {}
        watch = self._plan.watch_s(step.start) if step.start is not None else 0.0
        outcome = self._run_primitive(step.back_off, checks, watch, step)
        self._event(step, "back-off", outcome=outcome)
        self._check_stop()
        if outcome.blocked is not None:
            return outcome.why()
        if not outcome.passed:
            if step.start is not None and outcome.reason == "check failed":
                return f"after the back-off its start check {step.start!r} did not pass"
            return f"the back-off did not finish: {outcome.why()}"
        return None

    def _checks_only(self, step: ChainStep, name: str) -> Outcome:
        return self._run_primitive(
            None, self._plan.live(name), self._plan.watch_s(name), step
        )

    def _wait_for_operator(self, step: ChainStep, why: str) -> str:
        # Cleared before go_on() can be accepted, so an early answer is kept.
        self._go_on.clear()
        with self._lock:
            self._state = "waiting for the operator"
            self._waiting = f"{step.name}: {why}"
            self._action = None
        self._event(step, "operator called", why=why)
        self._append_log("WARN", "CHAIN", f"Operator needed at {step.name}", why)
        while not self.stop_event.is_set():
            if self._go_on.wait(0.1):
                break
        self._check_stop()
        with self._lock:
            action = self._action or RETRY
            self._waiting = None
        self._event(step, "operator went on", action=action)
        return action

    # --- bookkeeping ----------------------------------------------------------

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise _Stopped

    def _set(self, step: ChainStep, state: str, attempt: int) -> None:
        with self._lock:
            self._state = state
            self._try = attempt
        self._append_log("INFO", "CHAIN", f"{step.name}: {state}", f"try {attempt}")

    def _event(self, step: ChainStep | None, what: str, **details: Any) -> None:
        outcome = details.pop("outcome", None)
        event = {
            "t": round(self._clock() - self._started, 2),
            "step": step.name if step is not None else None,
            "event": what,
            **details,
            **({"outcome": outcome.to_dict()} if outcome is not None else {}),
        }
        with self._lock:
            self._record["events"].append(event)
        self._write()

    def _write(self) -> None:
        with self._lock:
            text = json.dumps(self._record, indent=2) + "\n"
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._record_path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self._record_path)
