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

"""Turn teleop recordings into demonstrations in a practice space.

A teleop recording holds what the arm did, `sense_hz` times a second: the
tool's pose, velocity and measured wrench in `observation.state`, the pose the
operator commanded in `action`, and the cameras. A step skill needs, at each
of its steps, what it saw (its observation, `space.py`) and the step that heads
where the operator was heading (its action, in [-1, 1]).

THE LABEL. For a step that starts at sample t, the move is the pose the
operator commanded one step later (t + stride), relative to where the tool was
at t, in the tool frame; the force target is the force the tool applied one
step later. Both are clipped to the bounds. How often each axis was clipped is
reported: a bound the demonstrations keep outrunning is too small for them.

The observation comes from the `StepTracker` the practice environment uses on
the robot, fed the recorded samples in order, so the last commands it shows are
the clipped actions of the steps before.

THE END OF A DEMONSTRATION. Without a check, every episode counts as a success:
its last step earns the reward and ends it. With a check (`check_labels.json`),
an episode ends at the step that reaches the check's first labelled frame and
earns the reward there; an episode labelled "never", or not labelled, is left
out, because a demonstration shows the skill done.

The output is a LeRobot dataset at `rate_hz` with `next.reward` and
`next.done`, the form LeRobot's practice loop reads demonstrations in. Depth
streams and cameras the space does not list are left out. It carries the space
(`meta/practice_space.json`), the source's control settings
(`meta/control.json`) and a report (`meta/practice_demos.json`).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from flexivtrainer.data.control_settings import (
    effective_control_settings,
    write_control_settings,
)
from flexivtrainer.data.lerobot_io import (
    ROTATION_6D_AXES,
    first_dataset_task,
    resolve_recording_vcodec,
    rotation_6d_to_matrix,
)
from flexivtrainer.practice.space import (
    AXES,
    IMAGE_PREFIX,
    Command,
    PracticeSpace,
    Sample,
    StepTracker,
    write_space,
)

__all__ = ["REPORT_RELATIVE_PATH", "convert_demos", "required_columns"]

REPORT_RELATIVE_PATH = Path("meta") / "practice_demos.json"
REWARD = "next.reward"
DONE = "next.done"
#: Share of steps over which an axis may be clipped before the report warns.
CLIP_WARNING = 0.05

_POSE = [
    *(f"tcp_pose.{a}" for a in AXES),
    *(f"tcp_rotation_6d.{a}" for a in ROTATION_6D_AXES),
]
_TWIST = [f"tcp_twist.{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
_WRENCH = [f"tcp_wrench.{a}" for a in ("fx", "fy", "fz", "mx", "my", "mz")]


def required_columns(arm: str) -> dict[str, list[str]]:
    """The recorded columns a demonstration is built from, per feature."""
    return {
        "observation.state": [f"{arm}.{n}" for n in _POSE + _TWIST + _WRENCH],
        "action": [f"{arm}.{n}" for n in _POSE],
    }


def _arm(state_names: Sequence[str], arm: str | None) -> str:
    arms = [
        n.removesuffix(".tcp_pose.x") for n in state_names if n.endswith(".tcp_pose.x")
    ]
    if arm is not None:
        if arm not in arms:
            raise ValueError(f"the recording has no arm {arm!r}; it has {arms}")
        return arm
    if len(arms) != 1:
        raise ValueError(
            f"the recording has {len(arms)} arms {arms}; name the one that "
            "does the skill"
        )
    return arms[0]


def _indices(names: Sequence[str], wanted: Sequence[str], feature: str) -> list[int]:
    missing = [w for w in wanted if w not in names]
    if missing:
        raise ValueError(
            f"the recording's {feature} has no {missing}. A demonstration needs "
            "the arm's tcp_pose, tcp_twist and tcp_wrench state and its tcp_pose "
            "action: record with those entries selected"
        )
    return [list(names).index(w) for w in wanted]


def _one_at_a_time(chunks: Iterator[Any]) -> Iterator[np.ndarray]:
    """Frames from (n, C, H, W) chunks, one (H, W, C) frame at a time."""
    for chunk in chunks:
        for frame in chunk:
            yield np.ascontiguousarray(frame.permute(1, 2, 0).numpy())


def _starts(length: int, stride: int, success: int | None) -> list[int]:
    """The samples the steps start on. Without a labelled success, they run
    while a later sample exists; with one, until the step that reaches it."""
    end = length - 1 if success is None else success
    return list(range(0, end, stride))


def convert_demos(
    source_root: Path,
    output_root: Path,
    space: PracticeSpace,
    *,
    check: str | None = None,
    arm: str | None = None,
    episodes: Sequence[int] | None = None,
    video_codec: str = "h264",
) -> dict[str, Any]:
    """Write `output_root`: the demonstrations of `source_root` in `space`."""
    from lerobot.configs.video import RGBEncoderConfig  # noqa: PLC0415
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    from flexivtrainer.evaluation.check_labels import labels_for  # noqa: PLC0415
    from flexivtrainer.evaluation.learned_check import DatasetReader  # noqa: PLC0415

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    if output.is_relative_to(source):
        raise ValueError("The output cannot be placed inside the source dataset")

    reader = DatasetReader(source)
    if abs(reader.fps - space.sense_hz) > 1e-6:
        raise ValueError(
            f"{source.name} was recorded at {reader.fps:g} Hz but the space senses "
            f"at {space.sense_hz:g} Hz; they must match"
        )
    if abs(space.rate_hz - round(space.rate_hz)) > 1e-6:
        raise ValueError(
            f"a recording's rate is whole steps a second, not {space.rate_hz}"
        )
    missing_views = [v for v in space.views if v not in reader.cameras]
    if missing_views:
        raise ValueError(
            f"{source.name} has no RGB camera {missing_views}; it has {reader.cameras}"
        )
    side = _arm(reader.state_names, arm)
    columns = required_columns(side)
    action_names = list(reader.dataset.features["action"].get("names") or [])
    state_at = _indices(reader.state_names, columns["observation.state"], "state")
    action_at = _indices(action_names, columns["action"], "action")
    labels = labels_for(source, check) if check is not None else None
    chosen = reader.episodes if episodes is None else sorted(set(episodes))
    unknown = sorted(set(chosen) - set(reader.episodes))
    if unknown:
        raise ValueError(f"{source.name} has no episodes {unknown}")

    plan: list[tuple[int, list[int]]] = []
    left_out: list[dict[str, Any]] = []
    for episode in chosen:
        times = reader.columns(episode, ["timestamp"])["timestamp"]
        length = len(times)
        success: int | None = None
        if labels is not None:
            if episode not in labels:
                left_out.append({"episode": episode, "why": f"no {check} label"})
                continue
            success = labels[episode]
            if success is None:
                left_out.append({"episode": episode, "why": f"never reaches {check}"})
                continue
            if success >= length:
                raise ValueError(
                    f"episode {episode} has {length} frames, but its {check} label "
                    f"is frame {success}"
                )
        starts = _starts(length, space.stride, success)
        if not starts:
            left_out.append({"episode": episode, "why": "no step before its end"})
            continue
        plan.append((episode, starts))
    if not plan:
        raise ValueError(f"no episode of {source.name} is a demonstration: {left_out}")

    features: dict[str, dict[str, Any]] = {}
    for view in space.views:
        shape = reader.dataset.features[IMAGE_PREFIX + view]["shape"]
        features[IMAGE_PREFIX + view] = {
            "dtype": "video",
            "shape": tuple(int(v) for v in shape),
            "names": ["height", "width", "channels"],
        }
    for key, names in space.state_dims().items():
        features[key] = {"dtype": "float32", "shape": (len(names),), "names": names}
    features["action"] = {
        "dtype": "float32",
        "shape": (space.action_dim,),
        "names": space.action_names,
    }
    features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}
    features[DONE] = {"dtype": "bool", "shape": (1,), "names": None}

    task = first_dataset_task(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.practice-", dir=output.parent)
    )
    unclipped: list[np.ndarray] = []
    episodes_report: list[dict[str, Any]] = []
    try:
        dataset = LeRobotDataset.create(
            repo_id=f"local/{output.name}",
            fps=round(space.rate_hz),
            root=staging / output.name,
            features=features,
            use_videos=True,
            streaming_encoding=True,
            rgb_encoder=RGBEncoderConfig(vcodec=resolve_recording_vcodec(video_codec)),
        )
        tracker = StepTracker(space)
        for episode, starts in plan:
            rows = reader.columns(episode, ["timestamp", "observation.state", "action"])
            state = rows["observation.state"].astype(np.float64)
            action = rows["action"].astype(np.float64)[:, action_at]
            samples = [
                Sample.from_recorded(row[:9], row[9:15], row[15:21])
                for row in state[:, state_at]
            ]
            length = len(samples)
            step_times = rows["timestamp"].astype(np.float64).reshape(-1)[starts]
            frames = {
                v: _one_at_a_time(reader.frames(episode, v, step_times))
                for v in space.views
            }
            tracker.reset(samples[0])
            at = 0
            for n, start in enumerate(starts):
                while at < start:
                    at += 1
                    tracker.sense(samples[at])
                observation = tracker.observation(
                    {v: next(f) for v, f in frames.items()}
                )
                ahead = min(start + space.stride, length - 1)
                target = Command(
                    position=action[ahead, :3],
                    rotation=rotation_6d_to_matrix(action[ahead, 3:9]),
                    force=samples[ahead].tool_wrench()[:3],
                )
                step, units = tracker.action_toward(target)
                tracker.take(step)
                unclipped.append(units)
                last = n == len(starts) - 1
                dataset.add_frame(
                    {
                        **observation,
                        "action": step,
                        REWARD: np.array([1.0 if last else 0.0], dtype=np.float32),
                        DONE: np.array([last]),
                        "task": task,
                    }
                )
            dataset.save_episode()
            episodes_report.append(
                {"episode": episode, "steps": len(starts), "frames": length}
            )
        dataset.finalize()

        units = np.abs(np.stack(unclipped))
        clipped = {
            name: {
                "share": round(float(np.mean(units[:, i] > 1.0)), 4),
                "largest": round(float(np.max(units[:, i])), 3),
                # The bound under which 95% of the demonstrated steps fit.
                f"covers_95_{unit}": round(
                    float(np.percentile(units[:, i], 95)) * bound, 6
                ),
            }
            for i, (name, (bound, unit)) in enumerate(
                zip(space.action_names, space.action_bounds, strict=True)
            )
        }
        warnings = [
            f"{name}: the demonstrations outrun the bound on "
            f"{100 * c['share']:.0f}% of steps (up to {c['largest']:.1f}x)"
            for name, c in clipped.items()
            if c["share"] > CLIP_WARNING
        ]
        report = {
            "format_version": 1,
            "source_dataset": str(source),
            "arm": side,
            "check": check,
            "episodes": episodes_report,
            "left_out": left_out,
            "steps": int(len(units)),
            "clipped": clipped,
            "warnings": warnings,
        }
        root = staging / output.name
        write_space(root, space)
        write_control_settings(root, effective_control_settings(source))
        (root / REPORT_RELATIVE_PATH).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        root.replace(output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"output_root": str(output), **report}
