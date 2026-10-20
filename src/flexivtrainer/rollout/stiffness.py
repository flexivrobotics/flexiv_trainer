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

"""Which Cartesian stiffness a rollout runs at, and where it came from.

In order:

1. `ROLLOUT__CARTESIAN_STIFFNESS`, when set. It wins, with a warning when the
   checkpoint was trained at another stiffness.
2. The stiffness the checkpoint stores (`control.json`, written by training
   from the dataset's `meta/control.json`).
3. Otherwise the rollout is refused: the controller's default may be several
   times stiffer than the demonstrations, so the same targets would press
   several times harder. `ROLLOUT__ALLOW_CONTROLLER_DEFAULT_STIFFNESS=true`
   lets it run at that default instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from flexivtrainer.data.control_settings import (
    CONTROLLER_DEFAULT,
    ControlSettings,
    format_stiffness,
)

SOURCE_SETTING = "setting"
SOURCE_CHECKPOINT = "checkpoint"
SOURCE_CONTROLLER_DEFAULT = "controller default"
NOT_RECORDED = "not recorded"


class StiffnessRefused(RuntimeError):
    """Neither the settings nor the checkpoint say which stiffness to use."""


@dataclass(frozen=True)
class StiffnessChoice:
    """The stiffness a rollout applies. `values` None is the controller default."""

    values: list[float] | None
    source: str
    # The checkpoint's stored stiffness, or why there is none.
    trained: list[float] | str
    warning: str | None = None

    def describe(self) -> str:
        return f"{format_stiffness(self.values)} from the {self.source}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": list(self.values) if self.values is not None else None,
            "source": self.source,
            "trained": (
                list(self.trained)
                if isinstance(self.trained, list)
                else str(self.trained)
            ),
            "warning": self.warning,
        }


def _same(a: Sequence[float], b: Sequence[float]) -> bool:
    return len(a) == len(b) and all(
        math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-9)
        for x, y in zip(a, b, strict=True)
    )


def choose_rollout_stiffness(
    setting: Sequence[float] | None,
    checkpoint: ControlSettings | None,
    *,
    allow_controller_default: bool,
) -> StiffnessChoice:
    """Apply the precedence above; raises StiffnessRefused for step 3."""
    stored = checkpoint.stiffness if checkpoint is not None else None
    if stored is not None:
        trained: list[float] | str = stored
    elif checkpoint is not None:
        trained = f"{CONTROLLER_DEFAULT} ({checkpoint.recorded_in})"
    else:
        trained = NOT_RECORDED

    if setting is not None:
        values = [float(v) for v in setting]
        warning = None
        if stored is not None and not _same(values, stored):
            warning = (
                f"Trained at stiffness {format_stiffness(stored)}, running at "
                f"{format_stiffness(values)} (ROLLOUT__CARTESIAN_STIFFNESS): the "
                "same targets press with a different force"
            )
        return StiffnessChoice(values, SOURCE_SETTING, trained, warning)

    if stored is not None:
        return StiffnessChoice(list(stored), SOURCE_CHECKPOINT, trained)

    why = (
        "the checkpoint stores no stiffness (it was trained before control "
        "settings were recorded)"
        if checkpoint is None
        else "the checkpoint was trained on data recorded at the "
        f"{checkpoint.recorded_in} controller's default stiffness, whose value "
        "is unknown"
    )
    if not allow_controller_default:
        raise StiffnessRefused(
            f"No Cartesian stiffness for this rollout: {why}. Set "
            "FLEXIV_TRAINER_ROLLOUT__CARTESIAN_STIFFNESS to the stiffness the "
            "demonstrations were recorded at, or set "
            "FLEXIV_TRAINER_ROLLOUT__ALLOW_CONTROLLER_DEFAULT_STIFFNESS=true to "
            "run at the controller's default, which may press much harder than "
            "the demonstrations."
        )
    return StiffnessChoice(
        None,
        SOURCE_CONTROLLER_DEFAULT,
        trained,
        f"Running at the controller's default stiffness: {why}",
    )
