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

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RECORDING_ENTRY_KEYS: list[str] = [
    "observation.images.ego",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.state.tcp_pose",
    "observation.state.tcp_twist",
    "observation.state.tcp_wrench",
    "action.tcp_pose",
    "action.tcp_twist",
    "action.tcp_wrench",
]

_IMAGE_ENTRY_TO_CAMERA = {
    "observation.images.ego": "ego",
    "observation.images.left_wrist": "left_wrist",
    "observation.images.right_wrist": "right_wrist",
}

# Per-metric axis names. The full sub-feature label is "<metric>.<axis>", e.g.
# "tcp_pose.x" or "tcp_wrench.fz". tcp_pose carries position + quaternion.
_TCP_POSE_AXES = [
    "tcp_pose.x",
    "tcp_pose.y",
    "tcp_pose.z",
    "tcp_pose.q_w",
    "tcp_pose.q_x",
    "tcp_pose.q_y",
    "tcp_pose.q_z",
]
_TCP_TWIST_AXES = [
    "tcp_twist.vx",
    "tcp_twist.vy",
    "tcp_twist.vz",
    "tcp_twist.wx",
    "tcp_twist.wy",
    "tcp_twist.wz",
]
_TCP_WRENCH_AXES = [
    "tcp_wrench.fx",
    "tcp_wrench.fy",
    "tcp_wrench.fz",
    "tcp_wrench.mx",
    "tcp_wrench.my",
    "tcp_wrench.mz",
]

# Ordered (recording-entry key, snapshot field, axis names) for each metric.
# State and action share the same axis naming so a left_arm observation lines up
# with its left_arm action component-for-component.
_STATE_METRICS: list[tuple[str, str, list[str]]] = [
    ("observation.state.tcp_pose", "tcp_pose", _TCP_POSE_AXES),
    ("observation.state.tcp_twist", "tcp_vel", _TCP_TWIST_AXES),
    ("observation.state.tcp_wrench", "ext_wrench_in_world", _TCP_WRENCH_AXES),
]
_ACTION_METRICS: list[tuple[str, str, list[str]]] = [
    ("action.tcp_pose", "tcp_pose_d", _TCP_POSE_AXES),
    ("action.tcp_twist", "tcp_vel_d", _TCP_TWIST_AXES),
    ("action.tcp_wrench", "ext_wrench_d", _TCP_WRENCH_AXES),
]

STATE_ENTRY_KEYS: set[str] = {entry for entry, _, _ in _STATE_METRICS}
ACTION_ENTRY_KEYS: set[str] = {entry for entry, _, _ in _ACTION_METRICS}


def arm_side_label(index: int) -> str:
    """Human side label for a robot by its capture order (no serials).

    Robots are captured in teleop-pair order, so index 0 is the left arm and
    index 1 the right arm for a bimanual rig; anything beyond falls back to a
    generic name.
    """
    if index == 0:
        return "left_arm"
    if index == 1:
        return "right_arm"
    return f"arm_{index + 1}"


def _normalize_unique(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in items:
        value = str(raw).strip()
        if not value:
            continue
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def resolve_recording_entries(entries: list[str] | None = None) -> list[str]:
    if entries is None:
        return list(DEFAULT_RECORDING_ENTRY_KEYS)

    resolved = _normalize_unique(entries)
    allowed = set(DEFAULT_RECORDING_ENTRY_KEYS)
    for entry in resolved:
        if entry not in allowed:
            raise ValueError(f"Unsupported recording entry: {entry}")
    return resolved


def resolve_recording_image_names(entries: list[str] | None = None) -> list[str]:
    resolved_entries = resolve_recording_entries(entries)
    camera_names: list[str] = []
    seen: set[str] = set()
    for entry in resolved_entries:
        camera_name = _IMAGE_ENTRY_TO_CAMERA.get(entry)
        if not camera_name or camera_name in seen:
            continue
        camera_names.append(camera_name)
        seen.add(camera_name)
    return camera_names


def extract_recording_images(
    images: dict[str, np.ndarray], entries: list[str] | None = None
) -> dict[str, np.ndarray]:
    selected = resolve_recording_image_names(entries)
    return {name: images[name] for name in selected if name in images}


def _collect_arm_group(
    section: Any,
    metrics: list[tuple[str, str, list[str]]],
    enabled_entries: set[str],
) -> tuple[list[float], list[str]]:
    """Concatenate the enabled metrics of one arm into a single flat vector.

    Returns (values, axis_names) where axis_names labels each scalar, e.g.
    ["tcp_pose.x", ..., "tcp_pose.q_z", "tcp_twist.vx", ..., "tcp_wrench.mz"].
    """
    values: list[float] = []
    names: list[str] = []
    for entry, field, axis_names in metrics:
        if entry not in enabled_entries:
            continue
        vector = section.get(field) if isinstance(section, dict) else None
        if not isinstance(vector, (list, tuple)):
            continue
        for index, value in enumerate(vector):
            values.append(float(value))
            names.append(axis_names[index] if index < len(axis_names) else f"{field}.{index}")
    return values, names


def _iter_arm_groups(
    robot_snapshot: dict[str, Any], enabled_entries: set[str]
):
    """Yield (feature_key, values, axis_names) for each arm's state and action.

    Robots are grouped per arm by side (left_arm/right_arm) rather than by
    serial number, so no hardware identifiers leak into the dataset.
    """
    robots = robot_snapshot.get("robots") if isinstance(robot_snapshot, dict) else None
    if not isinstance(robots, dict):
        return
    for index, payload in enumerate(robots.values()):
        if not isinstance(payload, dict):
            continue
        side = arm_side_label(index)
        state_values, state_names = _collect_arm_group(
            payload.get("states"), _STATE_METRICS, enabled_entries
        )
        if state_values:
            yield f"observation.state.{side}", state_values, state_names
        action_values, action_names = _collect_arm_group(
            payload.get("actions"), _ACTION_METRICS, enabled_entries
        )
        if action_values:
            yield f"action.{side}", action_values, action_names


def extract_recording_frame_values(
    robot_snapshot: dict[str, Any], entries: list[str] | None = None
) -> dict[str, list[float]]:
    """Per-frame grouped vectors keyed by arm feature (state and action)."""
    resolved_entries = set(resolve_recording_entries(entries))
    return {
        key: values
        for key, values, _ in _iter_arm_groups(robot_snapshot, resolved_entries)
    }


def build_features_from_sample(
    robot_snapshot: dict[str, Any],
    images: dict[str, np.ndarray],
    entries: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    resolved_entries = set(resolve_recording_entries(entries))
    selected_images = extract_recording_images(images, resolved_entries)

    features: dict[str, dict[str, Any]] = {}

    for camera_name, image in selected_images.items():
        array = np.asarray(image)
        height = int(array.shape[0]) if array.ndim >= 2 else 0
        width = int(array.shape[1]) if array.ndim >= 2 else 0
        channels = int(array.shape[2]) if array.ndim >= 3 else 1
        # Use the "video" dtype so camera feeds are stored as MP4 (the LeRobot
        # v3 format), not loose PNG frames. With "image", use_videos=True is
        # effectively ignored: no video_keys are registered and every frame is
        # written as a PNG instead of encoded into a video.
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": [height, width, channels],
        }

    # Each arm's state/action is one grouped vector feature (e.g.
    # observation.state.left_arm with shape (19,)), with per-axis `names`. The
    # shape is a tuple so it matches the numpy value's `.shape` in LeRobot's
    # add_frame validator (a list shape never compares equal to a tuple).
    state_keys: list[str] = []
    action_keys: list[str] = []
    for key, values, axis_names in _iter_arm_groups(robot_snapshot, resolved_entries):
        features[key] = {
            "dtype": "float32",
            "shape": (len(values),),
            "names": axis_names,
        }
        (action_keys if key.startswith("action.") else state_keys).append(key)

    return features, state_keys, action_keys


@dataclass(slots=True)
class EpisodeManifest:
    root: Path
    repo_id: str
    task: str = "Dual-arm Flexiv teleoperation demonstration"
    fps: int | None = None

    @classmethod
    def from_path(cls, root: Path) -> "EpisodeManifest":
        dataset_root = Path(root).expanduser().resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

        manifest_path = dataset_root / "episode.json"
        if not manifest_path.exists():
            manifest_path = dataset_root / "combined.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No episode manifest found under: {dataset_root}")

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        repo_id = str(payload.get("repo_id") or f"local/{dataset_root.name}")
        task = str(
            payload.get("task")
            or payload.get("sample_task")
            or "Dual-arm Flexiv teleoperation demonstration"
        )

        fps_raw = payload.get("fps")
        fps = int(fps_raw) if isinstance(fps_raw, (int, float)) else None

        return cls(root=dataset_root, repo_id=repo_id, task=task, fps=fps)
