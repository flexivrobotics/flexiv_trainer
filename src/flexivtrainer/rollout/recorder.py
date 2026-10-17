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

"""Record a rollout as a dataset, in the same shape teleoperation records.

Every policy rollout is evidence: of how often the policy succeeds, of where it
fails, and -- when it succeeds -- a demonstration the robot produced by itself.
None of that survives if the rollout is only watched. This writes each one as a
LeRobot dataset whose `observation.state` and `action` columns are named and
ordered exactly as a teleoperated episode's, so a rollout can be reviewed,
merged and trained on like any demonstration.

WHAT IS RECORDED IS WHAT THE POLICY SAW AND SAID: the state vector and camera
frames it was given (at the resolution it was given them), and the first action
of the plan it returned. While an operator has taken over with the leader arm,
the action is what a teleoperated recording stores instead (rollout/takeover.py),
and the tick is marked as an intervention. Everything else the dataset format
has no column for goes beside it:

    meta/rollout_ticks.jsonl    per tick: time, measured pose, force, joints,
                                whether the plan was fresh, inference time,
                                who had control and any switch that finished
    meta/rollout_outcome.json   which checkpoint, which condition, why it
                                stopped, and later the operator's verdict

THE PLANNER NEVER WAITS FOR THE DISK. Ticks go onto a bounded queue and a
background thread writes them; if the writer falls behind, ticks are dropped
and COUNTED, and the count is in the outcome -- a recording with gaps says so,
rather than presenting itself as the rollout that happened.
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from flexivtrainer.data.lerobot_io import (
    POSE_FORMAT_ROTATION_6D,
    build_features_from_sample,
    default_recording_entry_keys,
)
from flexivtrainer.evaluation.outcome import RolloutOutcome, write_outcome
from flexivtrainer.observability import describe_exception, warn

__all__ = ["RolloutRecorder", "RecordingPlan", "TICKS_FILE"]

TICKS_FILE = Path("meta") / "rollout_ticks.jsonl"

#: About three seconds of ticks at 30 Hz.
DEFAULT_QUEUE_TICKS = 90

_STOP = object()


@dataclass(frozen=True)
class RecordingPlan:
    """Where and how one rollout is recorded."""

    destination: Path
    staging: Path
    fps: int
    task: str | None
    robot_type: str
    video_codec: str
    action_names: list[str]
    sides: list[str]
    cameras: list[str] | None
    pose_format: str = POSE_FORMAT_ROTATION_6D


def _default_dataset_factory(plan: RecordingPlan, features: dict[str, Any]) -> Any:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from flexivtrainer.data.lerobot_io import resolve_recording_vcodec

    return LeRobotDataset.create(
        repo_id=f"local/{plan.staging.name}",
        fps=plan.fps,
        features=features,
        root=plan.staging,
        robot_type=plan.robot_type,
        use_videos=True,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec=resolve_recording_vcodec(plan.video_codec)),
    )


def _float_list(values: Any) -> list[float]:
    return [float(v) for v in values] if values is not None else []


def measured_from_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Per arm: what the robot reported this tick, for the ticks file."""
    robots = snapshot.get("robots") if isinstance(snapshot, dict) else None
    arms = []
    for payload in (robots or {}).values():
        states = payload.get("states") if isinstance(payload, dict) else None
        if not isinstance(states, dict):
            continue
        arm = {
            "tcp_pose": _float_list(states.get("tcp_pose")),
            "tcp_vel": _float_list(states.get("tcp_vel")),
            "wrench": _float_list(states.get("ext_wrench_in_world")),
        }
        for key in ("q", "tau_ext"):
            if states.get(key) is not None:
                arm[key] = _float_list(states[key])
        gripper = payload.get("gripper")
        if isinstance(gripper, dict):
            arm["gripper"] = {
                k: float(v) for k, v in gripper.items() if isinstance(v, int | float)
            }
        arms.append(arm)
    return arms


class RolloutRecorder:
    """Writes one rollout. `add_tick` from the planner; `finish` once, at the end."""

    def __init__(
        self,
        plan: RecordingPlan,
        outcome: RolloutOutcome,
        *,
        dataset_factory: Callable[[RecordingPlan, dict[str, Any]], Any] | None = None,
        max_queue: int = DEFAULT_QUEUE_TICKS,
    ) -> None:
        self._plan = plan
        self._outcome = outcome
        self._dataset_factory = dataset_factory or _default_dataset_factory
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._frames = 0
        self._dropped = 0
        #: Counted at the planner, so a dropped tick is still counted.
        self._intervention_ticks = 0
        self._error: str | None = None
        self._dataset: Any = None
        self._ticks_stream: Any = None
        self._started = time.monotonic()
        self._finished: Path | None = None
        plan.staging.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="rollout-recorder", daemon=True
        )
        self._thread.start()

    # --- the planner's side ---------------------------------------------

    def add_tick(
        self,
        *,
        state: Any,
        action: list[float],
        images: dict[str, np.ndarray],
        snapshot: dict[str, Any],
        step: int,
        fresh: bool,
        infer_ms: float,
        pose_format: str | None = None,
        control: str = "policy",
        switches: list[dict[str, Any]] | None = None,
    ) -> None:
        """Queue one tick. Never blocks; a full queue drops the tick and counts it.

        `pose_format` is the state layout the policy was fed; the first tick's
        decides the recording's column names. `control` is who had the arm
        (`policy`, `operator` or `switching`); `switches` are the takeover
        switches that finished since the last tick, with their durations.
        """
        intervention = control != "policy"
        tick: dict[str, Any] = {
            "t": round(time.monotonic() - self._started, 4),
            "step": int(step),
            "fresh": bool(fresh),
            "infer_ms": round(float(infer_ms), 2),
            "control": control,
            "intervention": intervention,
        }
        if switches:
            tick["switches"] = [dict(event) for event in switches]
        if intervention:
            with self._lock:
                self._intervention_ticks += 1
        item = {
            "pose_format": pose_format or self._plan.pose_format,
            "state": np.asarray(state, dtype=np.float32).reshape(-1),
            "action": np.asarray(action, dtype=np.float32).reshape(-1),
            "images": images,
            "snapshot": snapshot,
            "tick": tick,
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._dropped += 1

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "episode": self._outcome.episode,
                "frames": self._frames,
                "dropped_frames": self._dropped,
                "error": self._error,
                "path": str(self._finished) if self._finished else None,
            }

    # --- the end -----------------------------------------------------------

    def finish(
        self,
        *,
        stop_reason: str | None,
        error: str | None,
        steps: int,
        safety_stop: dict[str, str] | None = None,
        interventions: dict[str, Any] | None = None,
    ) -> tuple[Path, RolloutOutcome, bool]:
        """Drain, finalize, move into place and write the outcome (no verdict yet).

        `interventions` is the takeover summary (rollout/takeover.py), given
        when the leader arm could take over; the ticks it covered are added.
        Returns the episode directory, the outcome, and whether any data was
        saved. THE OUTCOME IS WRITTEN EVEN WHEN NO DATA IS: a rollout that faulted
        before its first tick, or whose writer failed, still happened, and an
        evaluation that silently lost it would count one trial fewer.
        """
        self._queue.put(_STOP)
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            with self._lock:
                self._error = (
                    self._error or "the recording writer did not finish within 60 s"
                )
        with self._lock:
            frames, dropped, writer_error = self._frames, self._dropped, self._error
            intervention_ticks = self._intervention_ticks
        safety = dict(self._outcome.safety)
        if safety_stop is not None:
            safety["stop"] = dict(safety_stop)
        outcome = replace(
            self._outcome,
            duration_s=round(time.monotonic() - self._started, 3),
            steps=int(steps),
            frames=frames,
            dropped_frames=dropped,
            stop_reason=stop_reason,
            error=error,
            safety=safety,
            interventions=(
                {**interventions, "ticks": intervention_ticks}
                if interventions is not None
                else dict(self._outcome.interventions)
            ),
        )
        destination = self._plan.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        saved = False
        if self._dataset is not None and frames > 0 and writer_error is None:
            try:
                self._dataset.save_episode()
                self._dataset.finalize()
                shutil.move(str(self._plan.staging), str(destination))
                saved = True
            except Exception as exc:
                with self._lock:
                    detail = describe_exception(exc)
                    self._error = f"Failed to save the rollout recording: {detail}"
        with self._lock:
            writer_error = self._error
        if not saved:
            self._discard_staging()
            if writer_error is not None:
                warn("Rollout recording failed", writer_error)
                outcome = replace(
                    outcome, checks={**outcome.checks, "recording_error": writer_error}
                )
        write_outcome(destination, outcome)
        self._outcome = outcome
        with self._lock:
            self._finished = destination
        return destination, outcome, saved

    def abort(self) -> None:
        """Stop and leave nothing behind: for a rollout that never started."""
        self._queue.put(_STOP)
        self._thread.join(timeout=10.0)
        self._discard_staging()

    # --- the writer thread -------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            with self._lock:
                failed = self._error is not None
            if failed:
                continue
            try:
                if self._dataset is None:
                    self._open(item)
                self._write(item)
            except Exception as exc:
                with self._lock:
                    self._error = (
                        f"Rollout recording stopped: {describe_exception(exc)}"
                    )
        if self._ticks_stream is not None:
            self._ticks_stream.close()
            self._ticks_stream = None

    def _features(self, item: dict[str, Any]) -> dict[str, Any]:
        plan = self._plan
        state_entries = [
            entry
            for entry in default_recording_entry_keys(plan.sides, plan.cameras)
            if entry.startswith("observation.state.")
        ]
        image_entries = [f"observation.images.{name}" for name in item["images"]]
        features, _, _ = build_features_from_sample(
            item["snapshot"],
            item["images"],
            state_entries + image_entries,
            plan.sides,
            cameras=plan.cameras,
            pose_format=item["pose_format"],
        )
        state = features.get("observation.state")
        state_width = int(item["state"].shape[0])
        if state is None or len(state["names"]) != state_width:
            # The policy's state is the ground truth of what it saw; names that
            # do not line up with it would mislabel every column.
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_width,),
                "names": [f"state_{i}" for i in range(state_width)],
            }
        action_width = int(item["action"].shape[0])
        names = list(plan.action_names)
        if len(names) != action_width:
            names = [f"action_{i}" for i in range(action_width)]
        features["action"] = {
            "dtype": "float32",
            "shape": (action_width,),
            "names": names,
        }
        return features

    def _open(self, item: dict[str, Any]) -> None:
        features = self._features(item)
        self._dataset = self._dataset_factory(self._plan, features)
        ticks_path = self._plan.staging / TICKS_FILE
        ticks_path.parent.mkdir(parents=True, exist_ok=True)
        self._ticks_stream = ticks_path.open("w", encoding="utf-8")

    def _write(self, item: dict[str, Any]) -> None:
        frame: dict[str, Any] = {
            "observation.state": item["state"],
            "action": item["action"],
            "task": self._plan.task or "rollout",
        }
        for name, image in item["images"].items():
            frame[f"observation.images.{name}"] = image
        self._dataset.add_frame(frame)
        with self._lock:
            index = self._frames
            self._frames += 1
        record = {
            "frame": index,
            **item["tick"],
            "command": [round(float(v), 6) for v in item["action"]],
            "measured": measured_from_snapshot(item["snapshot"]),
        }
        self._ticks_stream.write(json.dumps(record) + "\n")

    def _discard_staging(self) -> None:
        shutil.rmtree(self._plan.staging, ignore_errors=True)
