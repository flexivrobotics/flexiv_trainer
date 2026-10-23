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

"""Chain and practice plans kept on disk, to start from the rollout page.

A plan is one file, `<storage>/rollouts/plans/<name>.json`. What is in it says
what it runs:

- a chain plan is the plan `POST /rollout/chain` takes: `{"steps": [...],
  "checks": {...}}`;
- a practice plan is the body `POST /rollout/practice` takes:
  `{"checkpoint_path": ..., "task": "", "plan": {...}}`. A relative checkpoint
  path is taken from the storage root, as a chain's skills are.

Every plan is read whole and checked when it is listed, so a mistake shows on
the page before anything moves.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = ["PLANS_DIR", "list_plans", "read_plan"]

PLANS_DIR = "plans"
CHAIN = "chain"
PRACTICE = "practice"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _parse(data: Any) -> tuple[str, Any, dict[str, Any]]:
    """(kind, the parsed plan, a short summary) of a plan file's contents."""
    if not isinstance(data, dict):
        raise ValueError("a plan must be a JSON object")
    if "steps" in data:
        from flexivtrainer.rollout.chain import ChainPlan  # noqa: PLC0415

        plan = ChainPlan.from_dict(data)
        return (
            CHAIN,
            plan,
            {"about": plan.about, "steps": [step.name for step in plan.steps]},
        )
    if "plan" in data:
        from flexivtrainer.practice.session import PracticePlan  # noqa: PLC0415

        checkpoint = data.get("checkpoint_path")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("a practice plan needs its checkpoint_path")
        task = data.get("task", "")
        if not isinstance(task, str):
            raise ValueError("a practice plan's task must be text")
        plan = PracticePlan.from_dict(data["plan"])
        return (
            PRACTICE,
            plan,
            {
                "checkpoint": checkpoint,
                "task": task,
                "reward": plan.check_name,
                "episodes": plan.episodes,
                "learner": f"{plan.learner_host}:{plan.learner_port}",
            },
        )
    raise ValueError("not a chain plan (no 'steps') nor a practice plan (no 'plan')")


def _files(rollouts_root: Path) -> dict[str, Path]:
    directory = Path(rollouts_root) / PLANS_DIR
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.glob("*.json"))
        # Hidden files are an editor's, not plans.
        if path.is_file() and not path.name.startswith(".")
    }


def _bad_name(name: str) -> str | None:
    """Why a plan file's name cannot be used, or None."""
    if not _NAME.match(name):
        return (
            "a plan's file name is 1-64 letters, digits, '.', '_' or '-', starting "
            "with a letter or digit; rename the file"
        )
    if name == "session":
        # Evaluation sessions are folders holding a session.json.
        return (
            "session.json makes the plans folder look like an evaluation session; "
            "rename it"
        )
    return None


def _read(path: Path) -> Any:
    # utf-8-sig: an editor may save the file with a byte-order mark.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def list_plans(rollouts_root: Path) -> list[dict[str, Any]]:
    """Every plan on disk: its name, kind and summary, or why it is unusable."""
    plans = []
    for name, path in _files(rollouts_root).items():
        why = _bad_name(name)
        if why is not None:
            plans.append({"name": name, "kind": None, "error": why})
            continue
        try:
            kind, _, summary = _parse(_read(path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            plans.append({"name": name, "kind": None, "error": str(exc)})
            continue
        plans.append({"name": name, "kind": kind, **summary})
    return plans


def read_plan(rollouts_root: Path, name: str) -> tuple[str, Any, dict[str, Any]]:
    """(kind, the parsed plan, the file's contents) of the plan `name`.

    The file is chosen among those on disk, never built from `name`.
    """
    path = _files(rollouts_root).get(name)
    if path is None:
        raise FileNotFoundError(f"no plan named {name!r}")
    why = _bad_name(name)
    if why is not None:
        raise ValueError(why)
    data = _read(path)
    kind, plan, _ = _parse(data)
    return kind, plan, data
