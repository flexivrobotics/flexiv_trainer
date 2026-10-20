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

"""The follower control settings a recording was made under: `meta/control.json`.

A pose-target policy learns targets that sit F/K beyond where the arm ends up,
so the same targets at a different stiffness K press with a different force.
The stiffness the demonstrations were recorded under therefore travels with
the data: from each recording, through a merge, into the checkpoint
(`pretrained_model/control.json`), and from there into the rollout.

The file holds:

    format_version                1
    recorded_in                   "teleop" or "rollout": which controller held
                                  the follower
    follower_cartesian_stiffness  [kx, ky, kz] N/m then [kRx, kRy, kRz] N*m/rad,
                                  or "controller default" when the trainer
                                  set none
    follower_max_contact_wrench   [fx, fy, fz] N then [mx, my, mz] N*m, or
                                  "controller default"
    wrench_feedback_scale         the teleop force feedback scale, or
                                  "controller default"

"controller default" means the trainer did not set the value, so the controller
used its own. The teleop and rollout controllers have DIFFERENT default
stiffness, so a default from one never matches a default from the other.

A dataset without the file was recorded before it existed. A recorded rollout
then still has its stiffness in `meta/rollout_outcome.json`; any other dataset
was a teleop recording, made at the teleop controller's default.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTROL_FILENAME = "control.json"
CONTROL_RELATIVE_PATH = Path("meta") / CONTROL_FILENAME
CONTROLLER_DEFAULT = "controller default"
RECORDED_IN = ("teleop", "rollout")
_DECIMALS = 6
_OUTCOME_RELATIVE_PATH = Path("meta") / "rollout_outcome.json"


def _vector(value: Any, name: str) -> tuple[float, ...] | None:
    """Six finite, non-negative numbers, or None for the controller default."""
    if value is None or value == CONTROLLER_DEFAULT:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 6:
        raise ValueError(f"{name} must be six numbers or {CONTROLLER_DEFAULT!r}")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{name} must be six numbers or {CONTROLLER_DEFAULT!r}")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} values must be finite and not negative")
        numbers.append(round(number, _DECIMALS))
    return tuple(numbers)


def _scale(value: Any) -> float | None:
    if value is None or value == CONTROLLER_DEFAULT:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"wrench_feedback_scale must be a number or {CONTROLLER_DEFAULT!r}"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("wrench_feedback_scale must be finite and not negative")
    return round(number, _DECIMALS)


def _shown(value: tuple[float, ...] | float | None) -> Any:
    if value is None:
        return CONTROLLER_DEFAULT
    return list(value) if isinstance(value, tuple) else value


def format_stiffness(values: Sequence[float] | None) -> str:
    """Stiffness for a log line or a message."""
    if values is None:
        return CONTROLLER_DEFAULT
    return "[" + ", ".join(f"{float(v):g}" for v in values) + "]"


@dataclass(frozen=True)
class ControlSettings:
    """What the trainer set on the follower while a recording was made.

    None in a field means the trainer set nothing, so the controller default
    applied.
    """

    recorded_in: str = "teleop"
    follower_cartesian_stiffness: tuple[float, ...] | None = None
    follower_max_contact_wrench: tuple[float, ...] | None = None
    wrench_feedback_scale: float | None = None
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("Unsupported control settings format_version")
        if self.recorded_in not in RECORDED_IN:
            raise ValueError(f"recorded_in must be one of {RECORDED_IN}")
        object.__setattr__(
            self,
            "follower_cartesian_stiffness",
            _vector(self.follower_cartesian_stiffness, "follower_cartesian_stiffness"),
        )
        object.__setattr__(
            self,
            "follower_max_contact_wrench",
            _vector(self.follower_max_contact_wrench, "follower_max_contact_wrench"),
        )
        object.__setattr__(
            self, "wrench_feedback_scale", _scale(self.wrench_feedback_scale)
        )

    @property
    def stiffness(self) -> list[float] | None:
        values = self.follower_cartesian_stiffness
        return None if values is None else list(values)

    def matches(self, other: ControlSettings) -> bool:
        """Whether data recorded under both settings may be trained together.

        A default stiffness is the controller's own, and the teleop and rollout
        controllers have different defaults, so two defaults only match when
        the same controller recorded both.
        """
        if (
            self.follower_cartesian_stiffness != other.follower_cartesian_stiffness
            or self.follower_max_contact_wrench != other.follower_max_contact_wrench
            or self.wrench_feedback_scale != other.wrench_feedback_scale
        ):
            return False
        if self.follower_cartesian_stiffness is None:
            return self.recorded_in == other.recorded_in
        return True

    def describe(self) -> str:
        parts = [
            f"stiffness {format_stiffness(self.follower_cartesian_stiffness)}",
            f"max contact wrench {format_stiffness(self.follower_max_contact_wrench)}",
        ]
        if self.wrench_feedback_scale is not None:
            parts.append(f"wrench feedback scale {self.wrench_feedback_scale:g}")
        return f"{', '.join(parts)} ({self.recorded_in})"

    @classmethod
    def from_dict(cls, payload: Any) -> ControlSettings:
        if not isinstance(payload, dict):
            raise ValueError("Control settings must be a JSON object")
        return cls(
            recorded_in=payload.get("recorded_in", "teleop"),
            follower_cartesian_stiffness=payload.get("follower_cartesian_stiffness"),
            follower_max_contact_wrench=payload.get("follower_max_contact_wrench"),
            wrench_feedback_scale=payload.get("wrench_feedback_scale"),
            format_version=payload.get("format_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "recorded_in": self.recorded_in,
            "follower_cartesian_stiffness": _shown(self.follower_cartesian_stiffness),
            "follower_max_contact_wrench": _shown(self.follower_max_contact_wrench),
            "wrench_feedback_scale": _shown(self.wrench_feedback_scale),
        }


def read_control_settings(path: Path) -> ControlSettings:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid control settings: {path.name}") from exc
    try:
        return ControlSettings.from_dict(payload)
    except ValueError as exc:
        raise ValueError(f"Invalid control settings in {path.name}: {exc}") from exc


def write_control_settings(
    root: Path, settings: ControlSettings, *, checkpoint: bool = False
) -> Path:
    """Write `meta/control.json` of a dataset, or `control.json` of a checkpoint."""
    path = root / CONTROL_FILENAME if checkpoint else root / CONTROL_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def dataset_control_settings(root: Path) -> ControlSettings | None:
    """The dataset's stored settings, or None when it has no control.json."""
    path = root / CONTROL_RELATIVE_PATH
    return read_control_settings(path) if path.is_file() else None


def effective_control_settings(root: Path) -> ControlSettings:
    """The settings a dataset was recorded under, also for older datasets.

    A recorded rollout from before control.json existed has its stiffness and
    contact-wrench limit in its outcome file; any other dataset without the
    file was recorded by teleop, which never set either.
    """
    stored = dataset_control_settings(root)
    if stored is not None:
        return stored
    outcome_path = root / _OUTCOME_RELATIVE_PATH
    if outcome_path.is_file():
        try:
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            safety = outcome.get("safety") if isinstance(outcome, dict) else None
            safety = safety if isinstance(safety, dict) else {}
            return ControlSettings(
                recorded_in="rollout",
                follower_cartesian_stiffness=safety.get("cartesian_stiffness"),
                follower_max_contact_wrench=safety.get("max_contact_wrench"),
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Invalid rollout outcome, so its control settings are unknown: "
                f"{root.name}"
            ) from exc
    return ControlSettings(recorded_in="teleop")


def checkpoint_control_settings(model_dir: Path) -> ControlSettings | None:
    """A checkpoint's `control.json`, or None for one trained before it existed."""
    path = model_dir / CONTROL_FILENAME
    return read_control_settings(path) if path.is_file() else None
