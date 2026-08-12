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

"""Convert measured legacy gripper actions to inferred target-width commands."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from flexivtrainer.data.gripper_command import (
    GripperCommandMetadata,
    write_gripper_command_metadata,
)
from flexivtrainer.jobs.convert_bspline_dataset import (
    _load_action_names,
    _load_recorded_frames,
    _read_json,
    _refresh_action_statistics,
    _replace_action_data,
    _validate_output,
)
from flexivtrainer.observability import Pulse, section

_INFO_PATH = Path("meta/info.json")
_AUDIT_PATH = Path("meta/gripper_action_conversion.json")


def _load_initial_overrides(path: Path | None) -> dict[int, dict[str, bool]]:
    if path is None:
        return {}
    payload = _read_json(path)
    raw_episodes = payload.get("episodes", payload)
    if not isinstance(raw_episodes, Mapping):
        raise ValueError("Initial-state manifest must map episode indices to sides")
    overrides: dict[int, dict[str, bool]] = {}
    for raw_episode, raw_sides in raw_episodes.items():
        try:
            episode = int(raw_episode)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid episode index in manifest: {raw_episode!r}"
            ) from exc
        if not isinstance(raw_sides, Mapping):
            raise ValueError(f"Episode {episode} manifest entry must be an object")
        side_states: dict[str, bool] = {}
        for raw_side, raw_state in raw_sides.items():
            state = str(raw_state).strip().lower()
            if state not in {"open", "close"}:
                raise ValueError(
                    f"Initial state for episode {episode} side {raw_side!r} "
                    "must be 'open' or 'close'"
                )
            side_states[str(raw_side)] = state == "close"
        overrides[episode] = side_states
    return overrides


def _infer_close_commands(
    widths: np.ndarray,
    forces: np.ndarray | None,
    frame_indices: Sequence[int],
    *,
    motion_threshold_m: float,
    force_threshold_n: float,
    initial_override: bool | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    warnings: list[str] = []
    if initial_override is not None:
        initial_close = initial_override
        initial_source = "override"
    elif forces is not None and math.isfinite(float(forces[0])):
        initial_close = float(forces[0]) >= force_threshold_n
        initial_source = "force"
    else:
        initial_close = False
        initial_source = "default_open"
        warnings.append("Initial force unavailable; defaulted to Open")

    commands = np.full(len(widths), initial_close, dtype=np.float32)
    current = initial_close
    run_sign = 0
    run_start = 0
    displacement = 0.0
    transitions: list[dict[str, Any]] = []
    for index in range(1, len(widths)):
        commands[index] = float(current)
        delta = float(widths[index] - widths[index - 1])
        if not math.isfinite(delta) or delta == 0.0:
            continue
        sign = 1 if delta > 0 else -1
        if sign != run_sign:
            run_sign = sign
            run_start = index
            displacement = abs(delta)
        else:
            displacement += abs(delta)
        if displacement < motion_threshold_m:
            continue
        inferred_close = sign < 0
        if inferred_close != current:
            commands[run_start : index + 1] = float(inferred_close)
            current = inferred_close
            transitions.append(
                {
                    "frame_index": int(frame_indices[run_start]),
                    "command": "close" if current else "open",
                }
            )
        run_sign = 0
        displacement = 0.0

    if not transitions:
        warnings.append("No Open/Close transition was detected")
    close_frames = int(np.count_nonzero(commands >= 0.5))
    return commands, {
        "initial_state": "close" if initial_close else "open",
        "initial_source": initial_source,
        "transitions": transitions,
        "open_frames": int(len(commands) - close_frames),
        "close_frames": close_frames,
        "warnings": warnings,
    }


def convert_legacy_gripper_actions(
    source_root: Path,
    output_root: Path,
    *,
    open_width_m: float,
    close_width_m: float,
    velocity_m_s: float,
    force_limit_n: float,
    motion_threshold_m: float = 0.0002,
    force_threshold_n: float = 5.0,
    initial_state_manifest: Path | None = None,
) -> dict[str, Any]:
    """Create a validated copy with inferred target widths for every side."""

    for name, value in {
        "open_width_m": open_width_m,
        "close_width_m": close_width_m,
    }.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    metadata = GripperCommandMetadata(
        velocity_m_s=velocity_m_s,
        force_limit_n=force_limit_n,
    )
    metadata = GripperCommandMetadata.from_dict(metadata.to_dict())
    if not math.isfinite(motion_threshold_m) or motion_threshold_m <= 0:
        raise ValueError("motion_threshold_m must be finite and positive")
    if not math.isfinite(force_threshold_n) or force_threshold_n < 0:
        raise ValueError("force_threshold_n must be finite and nonnegative")

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir() or not (source / _INFO_PATH).is_file():
        raise FileNotFoundError(f"LeRobot dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    if output.is_relative_to(source):
        raise ValueError("Output dataset cannot be placed inside the source dataset")

    info = _read_json(source / _INFO_PATH)
    action_names = _load_action_names(info)
    width_indices = {
        name.removesuffix(".gripper.width"): index
        for index, name in enumerate(action_names)
        if name.endswith(".gripper.width")
    }
    if not width_indices:
        raise ValueError("Source action schema has no legacy gripper.width axes")
    for side in width_indices:
        if f"{side}.gripper.close" in action_names:
            raise ValueError("Source already contains unsupported gripper.close axes")
        if f"{side}.gripper.target_width" in action_names:
            raise ValueError("Source already contains gripper.target_width axes")
    force_indices = {
        side: (
            action_names.index(f"{side}.gripper.force")
            if f"{side}.gripper.force" in action_names
            else None
        )
        for side in width_indices
    }

    frames = _load_recorded_frames(source, len(action_names))
    if not frames:
        raise ValueError("Source dataset contains no frames")
    grouped: dict[int, list[Any]] = defaultdict(list)
    for frame in frames:
        grouped[frame.episode_index].append(frame)
    overrides = _load_initial_overrides(initial_state_manifest)
    unknown_episodes = set(overrides) - set(grouped)
    if unknown_episodes:
        raise ValueError(
            f"Manifest references unknown episodes: {sorted(unknown_episodes)}"
        )
    known_sides = set(width_indices)
    for episode, side_states in overrides.items():
        unknown_sides = set(side_states) - known_sides
        if unknown_sides:
            raise ValueError(
                f"Manifest episode {episode} references unknown sides: "
                f"{sorted(unknown_sides)}"
            )

    inferred: dict[tuple[int, int, str], float] = {}
    reports: list[dict[str, Any]] = []
    for episode_index in sorted(grouped):
        episode_frames = sorted(
            grouped[episode_index], key=lambda item: item.frame_index
        )
        actions = np.stack([frame.action for frame in episode_frames])
        frame_indices = [frame.frame_index for frame in episode_frames]
        for side, width_index in width_indices.items():
            force_index = force_indices[side]
            forces = actions[:, force_index] if force_index is not None else None
            commands, report = _infer_close_commands(
                actions[:, width_index],
                forces,
                frame_indices,
                motion_threshold_m=motion_threshold_m,
                force_threshold_n=force_threshold_n,
                initial_override=overrides.get(episode_index, {}).get(side),
            )
            for frame, command in zip(episode_frames, commands, strict=True):
                key = (frame.episode_index, frame.frame_index, side)
                inferred[key] = float(command)
            reports.append({"episode_index": episode_index, "side": side, **report})

    output_names: list[str] = []
    output_axes: list[tuple[str, int | str]] = []
    for index, name in enumerate(action_names):
        side = name.removesuffix(".gripper.width")
        if side in width_indices and name.endswith(".gripper.width"):
            output_names.append(f"{side}.gripper.target_width")
            output_axes.append(("target_width", side))
        elif any(name == f"{item}.gripper.force" for item in width_indices):
            continue
        else:
            output_names.append(name)
            output_axes.append(("source", index))

    targets: dict[tuple[int, int], np.ndarray] = {}
    for frame in frames:
        values = [
            (
                frame.action[int(source)]
                if kind == "source"
                else (
                    close_width_m
                    if inferred[(frame.episode_index, frame.frame_index, str(source))]
                    >= 0.5
                    else open_width_m
                )
            )
            for kind, source in output_axes
        ]
        targets[(frame.episode_index, frame.frame_index)] = np.asarray(
            values, dtype=np.float32
        )

    converted_info = json.loads(json.dumps(info))
    action_feature = converted_info["features"]["action"]
    action_feature.update(
        {"dtype": "float32", "shape": [len(output_names)], "names": output_names}
    )
    audit = {
        "format_version": 1,
        "source_dataset": str(source),
        "source_action_names": action_names,
        "output_action_names": output_names,
        "sides": list(width_indices),
        "motion_threshold_m": motion_threshold_m,
        "force_threshold_n": force_threshold_n,
        "open_width_m": open_width_m,
        "close_width_m": close_width_m,
        "velocity_m_s": metadata.velocity_m_s,
        "force_limit_n": metadata.force_limit_n,
        "initial_state_manifest": (
            str(Path(initial_state_manifest).expanduser().resolve())
            if initial_state_manifest is not None
            else None
        ),
        "episodes": reports,
    }

    section("Gripper action conversion", f"{source.name} -> {output.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.gripper-", dir=output.parent)
    )
    try:
        pulse = Pulse("Copying source dataset").start()
        shutil.copytree(source, staging, dirs_exist_ok=True)
        pulse.stop("OK", "Copied source dataset")
        (staging / _INFO_PATH).write_text(
            json.dumps(converted_info, indent=4) + "\n", encoding="utf-8"
        )
        _replace_action_data(staging, targets, converted_info["features"])
        _refresh_action_statistics(staging, targets, action_feature, parameter_rows=1)
        (staging / _AUDIT_PATH).write_text(
            json.dumps(audit, indent=4) + "\n", encoding="utf-8"
        )
        write_gripper_command_metadata(staging, metadata)
        _validate_output(staging, len(frames), len(output_names))
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "source_root": str(source),
        "output_root": str(output),
        "frames": len(frames),
        "episodes": len(grouped),
        "sides": list(width_indices),
        "action_dim": len(output_names),
    }
