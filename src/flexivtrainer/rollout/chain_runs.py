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

"""The chain runs on disk, the operator's verdict on each, and how many in a
row were clean.

A chain's own result says whether it got to its last step, not whether the
task was done: a learned check can pass a near miss, and the operator may
have skipped a step. So each run is judged, as a single rollout is, and the
verdict is written into its record (`<rollouts>/chains/<name>.json`):

    "verdict": {"outcome": "success", "failure_mode": null, "note": "", "at": ...}

A run is **clean** when its verdict is a success and nobody helped: no
takeover in any try and no call for the operator. The chain's own retries are
allowed. Runs are counted per plan, newest first; the **streak** is how many
finished runs in a row, up to the newest, were clean. A finished run without a
verdict ends the streak: nobody has said it was clean.

A run whose record never got its result -- the trainer was stopped or died
mid-chain -- is **interrupted**: it counts, and can be judged, unless it is the
chain running now.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flexivtrainer.evaluation.outcome import DEFAULT_FAILURE_MODES, Verdict
from flexivtrainer.observability import warn

__all__ = ["judge_run", "read_runs", "summarize_runs"]

_WRITE_TRIES = 5
INTERRUPTED = "interrupted"
#: Verdicts are written one at a time: each rewrites the whole record.
_judging = threading.Lock()


def _records(directory: Path) -> dict[str, Path]:
    """Every run's record, by name. Names are chosen among the files on disk."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in sorted(directory.glob("*.json"))}


def _modes(record: dict[str, Any]) -> list[str]:
    """The run's failure modes; a run recorded before plans named them takes
    the defaults, as its plan did."""
    return list(record.get("failure_modes") or DEFAULT_FAILURE_MODES)


def _describe(name: str, record: dict[str, Any], live: str | None) -> dict[str, Any]:
    events = [event for event in record.get("events") or [] if isinstance(event, dict)]
    takeovers = sum(
        int((event.get("outcome") or {}).get("takeovers") or 0) for event in events
    )
    calls = sum(1 for event in events if event.get("event") == "operator called")
    verdict = record.get("verdict")
    if verdict is not None and not isinstance(verdict, dict):
        raise ValueError("its verdict is not an object")
    result = record.get("result")
    if result is None and name != live:
        result = INTERRUPTED
    return {
        "name": name,
        "plan": record.get("plan"),
        "about": record.get("about", ""),
        "started_at": record.get("started_at"),
        "result": result,
        "seconds": record.get("seconds"),
        "steps": record.get("steps") or [],
        "back_offs": sum(1 for event in events if event.get("event") == "back-off"),
        "takeovers": takeovers,
        "operator_calls": calls,
        "failure_modes": _modes(record),
        "verdict": verdict,
        "clean": bool(
            verdict
            and verdict.get("outcome") == "success"
            and takeovers == 0
            and calls == 0
        ),
    }


def read_runs(
    directory: Path, limit: int | None = None, *, live: str | None = None
) -> list[dict[str, Any]]:
    """The runs in `directory`, newest first; an unreadable one says why.
    `live` names the chain running now, whose record has no result yet."""
    runs = []
    names = sorted(_records(directory), reverse=True)
    for name in names[:limit] if limit else names:
        path = Path(directory) / f"{name}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("not a chain record")
            runs.append(_describe(name, record, live))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            runs.append({"name": name, "error": str(exc)})
    return runs


def _plan_key(run: dict[str, Any]) -> str:
    return run.get("plan") or run.get("about") or ""


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per plan (its file's name, or its `about`): runs, verdicts and streaks.

    `runs` is newest first, as `read_runs` gives them.
    """
    summary: dict[str, dict[str, Any]] = {}
    for key in dict.fromkeys(_plan_key(run) for run in runs if "error" not in run):
        mine = [
            run
            for run in runs
            if "error" not in run
            and _plan_key(run) == key
            and run.get("result") is not None
        ]
        verdicts = [run["verdict"] for run in mine if run.get("verdict")]
        streak = 0
        for run in mine:
            if not run["clean"]:
                break
            streak += 1
        longest = current = 0
        for run in reversed(mine):
            current = current + 1 if run["clean"] else 0
            longest = max(longest, current)
        summary[key] = {
            "runs": len(mine),
            "judged": len(verdicts),
            "successes": sum(1 for v in verdicts if v.get("outcome") == "success"),
            "clean": sum(1 for run in mine if run["clean"]),
            "streak": streak,
            "longest_streak": longest,
        }
    return summary


def judge_run(
    directory: Path,
    name: str,
    outcome: str,
    *,
    failure_mode: str | None = None,
    note: str = "",
    live: str | None = None,
) -> dict[str, Any]:
    """Write the operator's verdict into run `name`'s record; returns the run.

    A failure needs one of the run's failure modes. A new verdict replaces an
    earlier one. The chain running now (`live`) has nothing to judge yet.
    """
    path = _records(directory).get(name)
    if path is None:
        raise FileNotFoundError(f"no chain run named {name!r}")
    if name == live:
        raise RuntimeError(f"chain run {name!r} has not finished")
    with _judging:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"{path.name} is not a chain record")
        modes = _modes(record)
        mode = (failure_mode or "").strip() or None if outcome == "failure" else None
        if outcome == "failure":
            if mode is None:
                raise ValueError("a failure needs a failure mode")
            if mode not in modes:
                raise ValueError(f"failure mode {mode!r} is not one of {modes}")
        verdict = Verdict(outcome=outcome, failure_mode=mode, note=note.strip())
        record["verdict"] = asdict(verdict)
        handle, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{name}-", suffix=".tmp"
        )
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
        # Windows refuses the replace while someone has the file open.
        for attempt in range(_WRITE_TRIES):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                if attempt == _WRITE_TRIES - 1:
                    Path(temporary).unlink(missing_ok=True)
                    warn("The chain run's verdict was not saved", str(exc))
                    raise RuntimeError(f"{path.name} is in use; try again") from exc
                time.sleep(0.05)
    return _describe(name, record, live)
