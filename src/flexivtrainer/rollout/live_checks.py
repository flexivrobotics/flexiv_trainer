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

"""Score checks while a rollout runs.

A learned check is trained and scored on recordings (evaluation/learned_check.py).
To end a skill when its state is reached, or to reward learning on the robot,
it has to answer during the rollout. Its image network takes milliseconds the
control loop does not have, so the planner hands each tick's frames and state
over and moves on; a worker thread scores the NEWEST tick it was given,
skipping any that arrived while it was busy.

A box check (`settled_in_box`) needs only the measured pose and velocity, so
it is scored on the planner's thread, on every tick.

Each check keeps the offline rule: it passes once it has held for `hold_s`,
timed by the ticks it scored -- a learned check's probability at or above its
threshold, a box check's tool inside the box and nearly still. A check with
`end_on_pass` ends the rollout when it passes, as the end of a skill.

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

from flexivtrainer.evaluation.checks import Check, LearnedCheck, SettledInBox
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
    """The checks of one rollout, scored while it runs."""

    def __init__(
        self,
        checks: dict[str, Check],
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
        self._learned = {
            name: check
            for name, check in self._checks.items()
            if isinstance(check, LearnedCheck)
        }
        self._boxes = {
            name: check
            for name, check in self._checks.items()
            if isinstance(check, SettledInBox)
        }
        # Each box check's last sample, for a speed when the tick carries none.
        self._previous: dict[str, tuple[float, list[float]]] = {}
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

    @property
    def needs_frames(self) -> bool:
        """Whether a check reads camera frames and the state vector; box
        checks read only the measured pose."""
        return bool(self._learned)

    # --- the service's side ----------------------------------------------------

    def start(self) -> None:
        """Load every model, then start scoring. Raises if one does not load,
        so a trial never starts with a check that cannot answer."""
        for name, check in self._learned.items():
            self._models[name] = self._loader(check.model_path(self._storage_root))
        if not self._learned:
            return
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
            self._previous = {}
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
        if self._boxes:
            self._score_boxes(float(t), snapshot)
        if not self._learned:
            return
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

    def all_passed(self) -> bool:
        """Whether every check has passed."""
        with self._lock:
            return all(held.first_at is not None for held in self._held.values())

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

    def _score_boxes(self, t: float, snapshot: dict[str, Any]) -> None:
        robots = snapshot.get("robots") if isinstance(snapshot, dict) else None
        arms = list((robots or {}).values())
        with self._lock:
            for name, check in self._boxes.items():
                states = (
                    arms[check.arm].get("states") if check.arm < len(arms) else None
                )
                if not isinstance(states, dict) or "tcp_pose" not in states:
                    continue
                position = [float(v) for v in states["tcp_pose"][:3]]
                velocity = states.get("tcp_vel")
                speed = None
                if velocity is not None and len(velocity) >= 3:
                    speed = float(np.linalg.norm(np.asarray(velocity[:3], float)))
                elif name in self._previous and t > self._previous[name][0]:
                    t0, p0 = self._previous[name]
                    speed = float(np.linalg.norm(np.subtract(position, p0))) / (t - t0)
                self._previous[name] = (t, position)
                inside = check.holds(position, speed)
                held = self._held[name]
                held.p, held.t = float(inside), t
                _hold(held, check, t, inside)

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
                    for name in self._learned:
                        self._held[name].error = detail
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
                _hold(held, check, t, p >= threshold)


def _hold(held: _Held, check: Check, t: float, passing: bool) -> None:
    """The offline rule, one tick at a time: a check passes once it has held
    for `hold_s`, and is timed from the start of that stretch."""
    if passing:
        if held.run_start is None:
            held.run_start = t
        if held.first_at is None and t - held.run_start >= check.hold_s:
            held.first_at = held.run_start
    else:
        held.run_start = None
