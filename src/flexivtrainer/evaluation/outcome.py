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

"""What happened in one rollout: `meta/rollout_outcome.json` beside its data.

A recorded rollout is a LeRobot dataset of exactly the shape teleoperation
records, so it can be reviewed, merged and trained on like any demonstration.
Everything the dataset format has no column for -- which checkpoint ran, from
which start condition, why it stopped, and whether it worked -- lives in this
one file, which LeRobot's readers ignore.

THE VERDICT IS THE OPERATOR'S, and it can be "unknown". A rollout nobody judged
is not a failure, and a check nobody has validated is not a verdict: automatic
checks are stored beside the verdict, never in place of it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "OUTCOME_FILE",
    "VERDICTS",
    "DEFAULT_FAILURE_MODES",
    "Verdict",
    "RolloutOutcome",
    "read_outcome",
    "write_outcome",
]

OUTCOME_FILE = Path("meta") / "rollout_outcome.json"
SCHEMA_VERSION = 1

VERDICTS = ("success", "failure", "unknown")

#: Offered by the verdict form when a session names none of its own. Generic on
#: purpose; a session lists the failure modes of its own task.
DEFAULT_FAILURE_MODES = (
    "did not reach the target",
    "stalled",
    "wrong contact",
    "released too early",
    "dropped the part",
    "safety stop",
    "controller fault",
    "other",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Verdict:
    """The operator's judgement of one rollout."""

    outcome: str
    failure_mode: str | None = None
    note: str = ""
    at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.outcome not in VERDICTS:
            raise ValueError(f"verdict {self.outcome!r} is not one of {VERDICTS}")
        if self.outcome == "success" and self.failure_mode:
            raise ValueError("a successful rollout has no failure mode")
        if len(self.note) > 2000:
            raise ValueError("a verdict note is limited to 2000 characters")


@dataclass
class RolloutOutcome:
    """One rollout's record. `session`, `trial`, `arm` and `condition` are set
    when the rollout was a trial of a planned evaluation session."""

    episode: str
    checkpoint: str
    task: str | None
    started_at: str
    duration_s: float = 0.0
    steps: int = 0
    frames: int = 0
    dropped_frames: int = 0
    stop_reason: str | None = None
    error: str | None = None
    session: str | None = None
    trial: int | None = None
    arm: str | None = None
    condition: str | None = None
    send_twist: bool = True
    safety: dict[str, Any] = field(default_factory=dict)
    verdict: Verdict | None = None
    checks: dict[str, Any] = field(default_factory=dict)
    #: Leader-arm takeovers, when one was possible: {count, ticks, seconds,
    #: switch_s}; switch_s holds every switch as {"to", "s"[, "error"]}.
    interventions: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def judged(self) -> bool:
        return self.verdict is not None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = asdict(self.verdict) if self.verdict is not None else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RolloutOutcome:
        if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(
                f"rollout outcome schema {data.get('schema_version')!r} is not "
                f"{SCHEMA_VERSION}"
            )
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in data.items() if key in known}
        verdict = values.get("verdict")
        values["verdict"] = Verdict(**verdict) if isinstance(verdict, dict) else None
        return cls(**values)


def read_outcome(episode_dir: Path) -> RolloutOutcome | None:
    """The outcome beside a recorded rollout, or None if it has none."""
    path = Path(episode_dir) / OUTCOME_FILE
    if not path.is_file():
        return None
    return RolloutOutcome.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_outcome(episode_dir: Path, outcome: RolloutOutcome) -> Path:
    """Write atomically: a verdict half-written by a crash would read as none."""
    path = Path(episode_dir) / OUTCOME_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(outcome.to_dict(), indent=2, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".outcome-", suffix=".json"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path
