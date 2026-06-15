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

_OBSERVATION_ENTRY_SPECS: dict[str, tuple[str, str]] = {
    "observation.state.tcp_pose": ("states", "tcp_pose"),
    "observation.state.tcp_twist": ("states", "tcp_vel"),
    "observation.state.tcp_wrench": ("states", "ext_wrench_in_world"),
}

_ACTION_ENTRY_SPECS: dict[str, tuple[str, str]] = {
    "action.tcp_pose": ("actions", "tcp_pose_d"),
    "action.tcp_twist": ("actions", "tcp_vel_d"),
    "action.tcp_wrench": ("actions", "ext_wrench_d"),
}


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


def _flatten_numeric_labels(prefix: str, values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [f"{prefix}.{index}" for index, _ in enumerate(values)]


def extract_recording_feature_values(
    robot_snapshot: dict[str, Any], entries: list[str] | None = None
) -> tuple[list[str], list[str]]:
    resolved_entries = resolve_recording_entries(entries)
    robots = robot_snapshot.get("robots") if isinstance(robot_snapshot, dict) else None
    if not isinstance(robots, dict):
        return [], []

    observation_values: list[str] = []
    action_values: list[str] = []

    for robot_name, robot_payload in robots.items():
        if not isinstance(robot_payload, dict):
            continue

        for entry, (parent_key, source_key) in _OBSERVATION_ENTRY_SPECS.items():
            if entry not in resolved_entries:
                continue
            section = robot_payload.get(parent_key)
            values = section.get(source_key) if isinstance(section, dict) else None
            metric = entry.rsplit(".", 1)[-1]
            observation_values.extend(
                _flatten_numeric_labels(f"{robot_name}.state.{metric}", values)
            )

        for entry, (parent_key, source_key) in _ACTION_ENTRY_SPECS.items():
            if entry not in resolved_entries:
                continue
            section = robot_payload.get(parent_key)
            values = section.get(source_key) if isinstance(section, dict) else None
            metric = entry.rsplit(".", 1)[-1]
            action_values.extend(
                _flatten_numeric_labels(f"{robot_name}.command.{metric}", values)
            )

    return observation_values, action_values


def build_features_from_sample(
    robot_snapshot: dict[str, Any],
    images: dict[str, np.ndarray],
    entries: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    resolved_entries = resolve_recording_entries(entries)
    selected_images = extract_recording_images(images, resolved_entries)
    observation_values, action_values = extract_recording_feature_values(
        robot_snapshot, resolved_entries
    )

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

    # LeRobot's add_frame validator compares the feature shape directly against
    # the numpy value's `.shape`, which is always a tuple. A list shape ([1])
    # never equals the tuple shape ((1,)), so every add_frame would raise and the
    # capture loop would silently record zero frames. Use a tuple to match.
    for key in observation_values:
        features[key] = {"dtype": "float32", "shape": (1,)}
    for key in action_values:
        features[key] = {"dtype": "float32", "shape": (1,)}

    return features, observation_values, action_values


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
