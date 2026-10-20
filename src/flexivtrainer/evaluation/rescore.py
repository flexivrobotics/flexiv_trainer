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

"""Score a session's recorded trials again, with its checks or with new ones.

    python -m flexivtrainer.evaluation.rescore <session> [--checks FILE] [--write]

Tuning a check's thresholds means trying them on trials that are already
judged, not running the robot again. Without `--write` nothing changes on disk:
the agreement with the verdicts, and each trial's result, are printed as JSON.
With `--write` each trial's outcome gets the new results. Verdicts are never
touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .checks import Check, parse_checks, read_ticks, run_checks
from .outcome import write_outcome
from .session import SessionStore, summarize

__all__ = ["rescore", "main"]


def rescore(
    root: Path,
    session: str,
    checks: dict[str, Check] | None = None,
    *,
    write: bool = False,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    """Score every recorded trial of `session`; `checks` replaces the plan's.

    `storage_root` resolves learned checks' relative model paths.
    """
    store = SessionStore(Path(root))
    plan = store.plan(session)
    checks = plan.checks if checks is None else checks
    if not checks:
        raise ValueError(f"session {session!r} has no checks, and none were given")
    rescored = []
    trials = []
    for directory, outcome in store.episodes(session):
        results = run_checks(
            checks,
            read_ticks(directory),
            episode_dir=directory,
            storage_root=storage_root,
        )
        updated = replace(outcome, checks={**outcome.checks, **results})
        if write:
            write_outcome(directory, updated)
        rescored.append(updated)
        trials.append(
            {
                "trial": outcome.trial,
                "episode": outcome.episode,
                "verdict": outcome.verdict.outcome if outcome.verdict else None,
                "checks": results,
            }
        )
    summary = summarize(replace(plan, checks=checks), rescored)
    return {
        "session": session,
        "written": write,
        "agreement": summary["checks"],
        "trials": sorted(trials, key=lambda t: (t["trial"] is None, t["trial"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m flexivtrainer.evaluation.rescore",
        description="Score a session's recorded trials with automatic checks.",
    )
    parser.add_argument("session")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="the rollouts folder; default is the configured storage's",
    )
    parser.add_argument(
        "--checks",
        type=Path,
        default=None,
        help="a JSON file with a 'checks' object to use instead of the plan's",
    )
    parser.add_argument(
        "--write", action="store_true", help="store the results in each outcome"
    )
    args = parser.parse_args(argv)
    from flexivtrainer.config import AppSettings  # noqa: PLC0415

    storage = AppSettings().storage
    root = args.root if args.root is not None else storage.rollouts_root
    checks = None
    if args.checks is not None:
        data = json.loads(args.checks.read_text(encoding="utf-8"))
        checks = parse_checks(data.get("checks") if isinstance(data, dict) else data)
    try:
        report = rescore(
            root, args.session, checks, write=args.write, storage_root=storage.root
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
