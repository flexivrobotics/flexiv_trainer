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

"""Score learned checks while a rollout runs.

A learned check is trained and scored on recordings (evaluation/learned_check.py).
To end a skill when its state is reached, or to reward learning on the robot,
it has to answer during the rollout. Its image network takes milliseconds the
control loop does not have, so the planner hands each tick's frames and state
over and moves on; a worker thread scores the NEWEST tick it was given,
skipping any that arrived while it was busy.

Each check keeps the offline rule: it passes once its probability has stayed
at or above the threshold for `hold_s`, timed by the ticks it scored. A check
with `end_on_pass` ends the rollout when it passes, as the end of a skill.

The probabilities are written into each tick of the recording -- the newest
one scored, so about a tick behind the frame it is written beside. The verdict
a trial is judged by stays the offline score of the saved recording, which sees
every frame; the live value is what the robot acted on.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flexivtrainer.evaluation.checks import LearnedCheck
from flexivtrainer.observability import describe_exception, warn

__all__ = ["LiveChecks"]


@dataclass
class _Held:
    """One check's running state."""

    p: float | None = None
    t: float | None = None
    run_start: float | None = None
    first_at: float | None = None
    error: str | None = None


class LiveChecks:
    """The learned checks of one rollout, scored off the control loop."""

    def __init__(
        self,
        checks: dict[str, LearnedCheck],
        storage_root: Path | None,
        *,
        state_names: Any,
        loader: Any = None,
    ) -> None:
        """`state_names(snapshot, images, pose_format, width)` names the columns
        of the state vector the planner hands over, as the recording names them.
        `loader` loads a check's model (the default reads its folder)."""
        if loader is None:
            from flexivtrainer.evaluation.learned_check import (  # noqa: PLC0415
                load_model as loader,
            )
        self._checks = dict(checks)
        self._storage_root = storage_root
        self._state_names = state_names
        self._loader = loader
        self._models: dict[str, Any] = {}
        self._columns: dict[str, list[int]] | None = None
        self._held = {name: _Held() for name in self._checks}
        self._lock = threading.Condition()
        self._pending: dict[str, Any] | None = None
        # Bumped by reset(): a tick handed over before it is not scored after.
        self._generation = 0
        self._stopping = False
        self._thread: threading.Thread | None = None
        self.scored = 0
        self.skipped = 0

    def __bool__(self) -> bool:
        return bool(self._checks)

    # --- the service's side ----------------------------------------------------

    def start(self) -> None:
        """Load every model, then start scoring. Raises if one does not load,
        so a trial never starts with a check that cannot answer."""
        for name, check in self._checks.items():
            self._models[name] = self._loader(check.model_path(self._storage_root))
        self._thread = threading.Thread(
            target=self._run, name="rollout-live-checks", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def reset(self) -> None:
        """Start again, as for a new episode: no check has passed, and a tick
        from before is not scored. The models stay loaded."""
        with self._lock:
            self._generation += 1
            self._pending = None
            self._held = {name: _Held() for name in self._checks}

    # --- the planner's side ----------------------------------------------------

    def submit(
        self,
        *,
        t: float,
        state: Any,
        images: dict[str, np.ndarray],
        snapshot: dict[str, Any],
        pose_format: str,
    ) -> None:
        """Hand over one tick. Never blocks; replaces a tick not yet scored."""
        item = {
            "t": float(t),
            "state": np.asarray(state, dtype=np.float32).reshape(-1),
            "images": images,
            "snapshot": snapshot,
            "pose_format": pose_format,
        }
        with self._lock:
            item["generation"] = self._generation
            if self._pending is not None:
                self.skipped += 1
            self._pending = item
            self._lock.notify()

    def values(self) -> dict[str, float]:
        """Each check's newest probability, for the tick being recorded."""
        with self._lock:
            return {
                name: round(held.p, 4)
                for name, held in self._held.items()
                if held.p is not None
            }

    def ended(self) -> str | None:
        """The first check marked `end_on_pass` that has passed, if any."""
        with self._lock:
            for name, check in self._checks.items():
                if check.end_on_pass and self._held[name].first_at is not None:
                    return name
        return None

    def summary(self) -> dict[str, Any]:
        """Per check: whether and when it passed, its last probability, and how
        many ticks were scored and skipped -- for the trial's outcome."""
        with self._lock:
            checks = {
                name: {
                    "passed": held.first_at is not None,
                    "first_at_s": None
                    if held.first_at is None
                    else round(held.first_at, 3),
                    "last_p": None if held.p is None else round(held.p, 4),
                    **({"error": held.error} if held.error else {}),
                }
                for name, held in self._held.items()
            }
            return {"checks": checks, "scored": self.scored, "skipped": self.skipped}

    # --- the worker ------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._lock:
                while self._pending is None and not self._stopping:
                    self._lock.wait()
                if self._stopping:
                    return
                item, self._pending = self._pending, None
            try:
                self._score(item)
            except Exception as exc:  # a check must never stop the rollout
                detail = describe_exception(exc)
                with self._lock:
                    for held in self._held.values():
                        held.error = detail
                warn("A live check could not be scored", detail)

    def _score(self, item: dict[str, Any]) -> None:
        if self._columns is None:
            names = list(
                self._state_names(
                    item["snapshot"],
                    item["images"],
                    item["pose_format"],
                    len(item["state"]),
                )
            )
            columns = {}
            for name, model in self._models.items():
                absent = [s for s in model.inputs.state if s not in names]
                if absent:
                    raise ValueError(f"check {name!r}: no state column {absent}")
                columns[name] = [names.index(s) for s in model.inputs.state]
            self._columns = columns
        results = {}
        for name, model in self._models.items():
            images = {
                camera: item["images"][camera][None] for camera in model.inputs.cameras
            }
            state = item["state"][self._columns[name]][None]
            results[name] = float(model.probabilities(images, state)[0])
        t = item["t"]
        with self._lock:
            if item.get("generation", self._generation) != self._generation:
                return  # handed over before a reset
            self.scored += 1
            for name, p in results.items():
                check, held = self._checks[name], self._held[name]
                threshold = (
                    self._models[name].threshold
                    if check.threshold is None
                    else check.threshold
                )
                held.p, held.t = p, t
                if p >= threshold:
                    if held.run_start is None:
                        held.run_start = t
                    if held.first_at is None and t - held.run_start >= check.hold_s:
                        held.first_at = held.run_start
                else:
                    held.run_start = None
