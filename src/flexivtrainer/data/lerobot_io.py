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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Camera MP4s are previewed in the browser across Ubuntu/macOS/Windows on
# x64/arm64/aarch64, so the codec must be H.264 — the one format every browser
# decodes natively (AV1 and HEVC are not universally supported, and AV1 has no
# hardware decode on many ARM boards). Hardware H.264 encoders in cross-platform
# preference order; software libx264 ("h264") is the universal fallback and is
# present in every PyAV wheel.
_H264_HW_ENCODERS: tuple[str, ...] = (
    "h264_videotoolbox",  # macOS
    "h264_nvenc",  # NVIDIA (Linux/Windows, incl. Jetson)
    "h264_vaapi",  # Linux Intel/AMD
    "h264_qsv",  # Intel Quick Sync
)
_SOFTWARE_H264 = "h264"  # libx264


def _encoder_available(name: str) -> bool:
    """True if FFmpeg (via PyAV) exposes ``name`` as a video encoder.

    Note: presence in the build does not guarantee a working device at runtime
    (e.g. ``h264_nvenc`` exists but no NVIDIA GPU is installed), which is why
    hardware encoders are opt-in rather than the default.
    """
    try:
        import av

        av.codec.Codec(name, "w")
        return True
    except Exception:
        return False


def resolve_recording_vcodec(preference: str) -> str:
    """Resolve a configured codec to a concrete, browser-playable H.264 encoder.

    - ``"auto"`` -> the first available hardware H.264 encoder for this platform,
      else software ``"h264"``. Never resolves to AV1/HEVC.
    - an explicit codec -> used as-is if available in this FFmpeg build; otherwise
      a warning is logged and it falls back to software ``"h264"`` so a config
      shared across machines never hard-fails recording when an encoder is
      missing.
    """
    if preference == "auto":
        for name in _H264_HW_ENCODERS:
            if _encoder_available(name):
                logger.info("Auto-selected hardware video codec: %s", name)
                return name
        return _SOFTWARE_H264
    if _encoder_available(preference):
        return preference
    logger.warning(
        "Configured video codec %r is not available in this FFmpeg build; "
        "falling back to software %r.",
        preference,
        _SOFTWARE_H264,
    )
    return _SOFTWARE_H264


_WRIST_CAMERA_BY_SIDE = {
    "left_arm": "left_wrist",
    "right_arm": "right_wrist",
    "single_arm": "wrist",
}


def active_camera_names(sides: list[str] | None = None) -> list[str]:
    return ["ego"] + [_WRIST_CAMERA_BY_SIDE[side] for side in _resolve_sides(sides)]


def _image_entry_to_camera(sides: list[str]) -> dict[str, str]:
    cameras = ["ego"] + [_WRIST_CAMERA_BY_SIDE[side] for side in sides]
    return {f"observation.images.{name}": name for name in cameras}


def _depth_entry_to_camera(sides: list[str]) -> dict[str, str]:
    cameras = ["ego"] + [_WRIST_CAMERA_BY_SIDE[side] for side in sides]
    return {f"observation.images.{name}_depth": name for name in cameras}


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

# Gripper width/force is measured state, but the request records the same
# measured values for both observation and action (there is no separate
# commanded-gripper signal in the snapshot). Its axes are shared by the state
# and action vectors, like the other metrics' axis naming.
_GRIPPER_LABEL = "gripper"
_GRIPPER_AXES = ["gripper.width", "gripper.force"]
# Fields read out of the follower payload's "gripper" section, in axis order.
_GRIPPER_FIELDS = ["width", "force"]

# Each metric: (label, state snapshot field, action snapshot field, axis names).
# The label appears in entry keys and axis names (e.g. "tcp_pose"). State and
# action share axis naming so an arm's observation lines up with its action
# component-for-component.
_METRICS: list[tuple[str, str, str, list[str]]] = [
    ("tcp_pose", "tcp_pose", "tcp_pose_d", _TCP_POSE_AXES),
    ("tcp_twist", "tcp_vel", "tcp_vel_d", _TCP_TWIST_AXES),
    ("tcp_wrench", "ext_wrench_in_world", "ext_wrench_d", _TCP_WRENCH_AXES),
]

# Arms in capture order (index 0 = left). Each arm's selected metrics are
# concatenated into one grouped vector feature.
_DUAL_SIDES: list[str] = ["left_arm", "right_arm"]


def _resolve_sides(sides: list[str] | None) -> list[str]:
    return list(sides) if sides else list(_DUAL_SIDES)


# Per-side metric labels available for state and action. The gripper label is
# appended to both: its (width, force) is recorded into observation.state and
# action alike from the same measured gripper states.
def _state_metric_labels() -> list[str]:
    return [label for label, _, _, _ in _METRICS] + [_GRIPPER_LABEL]


def _action_metric_labels() -> list[str]:
    return [label for label, _, _, _ in _METRICS] + [_GRIPPER_LABEL]


def state_entry_keys(sides: list[str] | None = None) -> set[str]:
    return {
        f"observation.state.{side}.{label}"
        for side in _resolve_sides(sides)
        for label in _state_metric_labels()
    }


def action_entry_keys(sides: list[str] | None = None) -> set[str]:
    return {
        f"action.{side}.{label}"
        for side in _resolve_sides(sides)
        for label in _action_metric_labels()
    }


# Recording entries mirror the dataset's plottable vectors one-for-one: each
# arm's wrist camera (plus the shared ego) and its per-metric state/action.
def default_recording_entry_keys(sides: list[str] | None = None) -> list[str]:
    resolved = _resolve_sides(sides)
    return (
        list(_image_entry_to_camera(resolved))
        + list(_depth_entry_to_camera(resolved))
        + [
            f"observation.state.{side}.{label}"
            for side in resolved
            for label in _state_metric_labels()
        ]
        + [
            f"action.{side}.{label}"
            for side in resolved
            for label in _action_metric_labels()
        ]
    )


STATE_ENTRY_KEYS: set[str] = state_entry_keys()
ACTION_ENTRY_KEYS: set[str] = action_entry_keys()
DEFAULT_RECORDING_ENTRY_KEYS: list[str] = default_recording_entry_keys()


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


def resolve_recording_entries(
    entries: list[str] | None = None, sides: list[str] | None = None
) -> list[str]:
    default = default_recording_entry_keys(sides)
    if entries is None:
        return default

    resolved = _normalize_unique(entries)
    allowed = set(default)
    for entry in resolved:
        if entry not in allowed:
            raise ValueError(f"Unsupported recording entry: {entry}")
    return resolved


def resolve_recording_image_names(
    entries: list[str] | None = None, sides: list[str] | None = None
) -> list[str]:
    resolved_entries = resolve_recording_entries(entries, sides)
    entry_to_camera = _image_entry_to_camera(_resolve_sides(sides))
    camera_names: list[str] = []
    seen: set[str] = set()
    for entry in resolved_entries:
        camera_name = entry_to_camera.get(entry)
        if not camera_name or camera_name in seen:
            continue
        camera_names.append(camera_name)
        seen.add(camera_name)
    return camera_names


def extract_recording_images(
    images: dict[str, np.ndarray],
    entries: list[str] | None = None,
    sides: list[str] | None = None,
) -> dict[str, np.ndarray]:
    selected = resolve_recording_image_names(entries, sides)
    return {name: images[name] for name in selected if name in images}


def resolve_recording_depth_names(
    entries: list[str] | None = None, sides: list[str] | None = None
) -> list[str]:
    resolved_entries = resolve_recording_entries(entries, sides)
    entry_to_camera = _depth_entry_to_camera(_resolve_sides(sides))
    camera_names: list[str] = []
    seen: set[str] = set()
    for entry in resolved_entries:
        camera_name = entry_to_camera.get(entry)
        if not camera_name or camera_name in seen:
            continue
        camera_names.append(camera_name)
        seen.add(camera_name)
    return camera_names


def extract_recording_depths(
    depths: dict[str, np.ndarray],
    entries: list[str] | None = None,
    sides: list[str] | None = None,
) -> dict[str, np.ndarray]:
    selected = resolve_recording_depth_names(entries, sides)
    return {name: depths[name] for name in selected if name in depths}


def _collect_arm_group(
    section: Any,
    side: str,
    kind: str,
    enabled_entries: set[str],
) -> tuple[list[float], list[str]]:
    """Concatenate an arm's *enabled* metrics into a single flat vector.

    ``kind`` is "state" or "action"; a metric is included only when its entry
    key ``"<observation.state|action>.<side>.<label>"`` is enabled. Returns
    (values, axis_names) where axis_names labels each scalar, e.g.
    ["tcp_pose.x", ..., "tcp_pose.q_z", "tcp_twist.vx", ..., "tcp_wrench.mz"].
    """
    feature_ns = "observation.state" if kind == "state" else "action"
    field_index = 1 if kind == "state" else 2
    values: list[float] = []
    names: list[str] = []
    for metric in _METRICS:
        label = metric[0]
        if f"{feature_ns}.{side}.{label}" not in enabled_entries:
            continue
        field = metric[field_index]
        axis_names = metric[3]
        vector = section.get(field) if isinstance(section, dict) else None
        if not isinstance(vector, (list, tuple)):
            continue
        for index, value in enumerate(vector):
            values.append(float(value))
            names.append(axis_names[index] if index < len(axis_names) else f"{label}.{index}")
    return values, names


def _collect_arm_gripper(
    payload: dict[str, Any],
    side: str,
    kind: str,
    enabled_entries: set[str],
) -> tuple[list[float], list[str]]:
    """Append a follower gripper's (width, force) to an arm's state or action.

    The same measured ``payload["gripper"]`` width/force feeds both the
    ``observation.state`` and ``action`` vectors, gated per-side by the
    ``"<observation.state|action>.<side>.gripper"`` entry key. Returns empty when
    the side has no gripper telemetry (no gripper configured/enabled) or the
    entry is disabled.
    """
    feature_ns = "observation.state" if kind == "state" else "action"
    if f"{feature_ns}.{side}.{_GRIPPER_LABEL}" not in enabled_entries:
        return [], []
    gripper = payload.get("gripper")
    if not isinstance(gripper, dict):
        return [], []
    values: list[float] = []
    names: list[str] = []
    for axis_index, field in enumerate(_GRIPPER_FIELDS):
        raw = gripper.get(field)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
        names.append(_GRIPPER_AXES[axis_index])
    return values, names


def _iter_combined_features(
    robot_snapshot: dict[str, Any],
    enabled_entries: set[str],
    sides: list[str] | None = None,
):
    """Yield (feature_key, values, axis_names) for the combined state and action.

    Every arm's enabled metrics are concatenated, in capture order (left then
    right), into a single ``observation.state`` vector and a single ``action``
    vector — the layout stock LeRobot policies require (they look up the
    features named exactly ``observation.state`` and ``action``). Axis names are
    prefixed with the arm side (e.g. ``left_arm.tcp_pose.x``) so the two arms
    stay distinguishable within the flat vector, and so state and action line up
    arm-for-arm.
    """
    robots = robot_snapshot.get("robots") if isinstance(robot_snapshot, dict) else None
    if not isinstance(robots, dict):
        return

    resolved_sides = _resolve_sides(sides)
    state_values: list[float] = []
    state_names: list[str] = []
    action_values: list[float] = []
    action_names: list[str] = []
    for index, payload in enumerate(robots.values()):
        if not isinstance(payload, dict):
            continue
        side = (
            resolved_sides[index]
            if index < len(resolved_sides)
            else arm_side_label(index)
        )
        arm_state_values, arm_state_names = _collect_arm_group(
            payload.get("states"), side, "state", enabled_entries
        )
        # The gripper's measured width/force trails the arm's robot-state metrics
        # in the same combined vector, and the identical values trail the action
        # metrics so state and action stay aligned arm-for-arm.
        gripper_state_values, gripper_state_names = _collect_arm_gripper(
            payload, side, "state", enabled_entries
        )
        state_values.extend(arm_state_values + gripper_state_values)
        state_names.extend(
            f"{side}.{name}" for name in arm_state_names + gripper_state_names
        )
        arm_action_values, arm_action_names = _collect_arm_group(
            payload.get("actions"), side, "action", enabled_entries
        )
        gripper_action_values, gripper_action_names = _collect_arm_gripper(
            payload, side, "action", enabled_entries
        )
        action_values.extend(arm_action_values + gripper_action_values)
        action_names.extend(
            f"{side}.{name}" for name in arm_action_names + gripper_action_names
        )

    if state_values:
        yield "observation.state", state_values, state_names
    if action_values:
        yield "action", action_values, action_names


def extract_recording_frame_values(
    robot_snapshot: dict[str, Any],
    entries: list[str] | None = None,
    sides: list[str] | None = None,
) -> dict[str, list[float]]:
    """Per-frame vectors keyed by feature (``observation.state`` and ``action``)."""
    resolved_entries = set(resolve_recording_entries(entries, sides))
    return {
        key: values
        for key, values, _ in _iter_combined_features(
            robot_snapshot, resolved_entries, sides
        )
    }


def build_features_from_sample(
    robot_snapshot: dict[str, Any],
    images: dict[str, np.ndarray],
    entries: list[str] | None = None,
    sides: list[str] | None = None,
    depths: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    resolved_list = resolve_recording_entries(entries, sides)
    resolved_entries = set(resolved_list)
    selected_images = extract_recording_images(images, resolved_list, sides)
    selected_depths = extract_recording_depths(depths or {}, resolved_list, sides)

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
            # LeRobot requires `names` on visual features; it reads ft["names"]
            # unconditionally when building policy features (and uses the last
            # axis to detect (h, w, c) layout). Order must match `shape` above.
            "names": ["height", "width", "channels"],
        }

    for camera_name, depth in selected_depths.items():
        array = np.asarray(depth)
        height = int(array.shape[0]) if array.ndim >= 2 else 0
        width = int(array.shape[1]) if array.ndim >= 2 else 0
        # Depth maps are single-channel; the is_depth_map flag routes LeRobot to
        # the 12-bit depth encoder instead of the RGB video encoder.
        features[f"observation.images.{camera_name}_depth"] = {
            "dtype": "video",
            "shape": [height, width, 1],
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        }

    # Both arms fold into a single observation.state and action vector feature,
    # each with per-axis `names` (e.g. left_arm.tcp_pose.x). The shape is a tuple
    # so it matches the numpy value's `.shape` in LeRobot's add_frame validator
    # (a list shape never compares equal to a tuple).
    state_keys: list[str] = []
    action_keys: list[str] = []
    for key, values, axis_names in _iter_combined_features(
        robot_snapshot, resolved_entries, sides
    ):
        features[key] = {
            "dtype": "float32",
            "shape": (len(values),),
            "names": axis_names,
        }
        (action_keys if key.startswith("action") else state_keys).append(key)

    return features, state_keys, action_keys


def first_dataset_task(dataset_root: Path) -> str:
    """Best-effort read of the first task name from meta/tasks.parquet."""
    default = "Dual-arm Flexiv teleoperation demonstration"
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return default
    try:
        import pandas as pd

        tasks = pd.read_parquet(tasks_path)
        if tasks.index.name == "task" and len(tasks.index):
            return str(tasks.index[0])
        if "task" in tasks.columns and len(tasks):
            return str(tasks["task"].iloc[0])
    except Exception:
        return default
    return default


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

        # Both single recordings and merged datasets are standard LeRobot v3.0
        # datasets, described entirely by their meta/info.json (+ tasks.parquet).
        # The repo id is local/<dir-name>, which is what the recorder/merge write.
        info_path = dataset_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"No dataset metadata found under: {dataset_root}")

        info = json.loads(info_path.read_text(encoding="utf-8"))
        repo_id = f"local/{dataset_root.name}"
        fps_raw = info.get("fps")
        fps = int(fps_raw) if isinstance(fps_raw, (int, float)) else None
        task = cls._first_task(dataset_root)

        return cls(root=dataset_root, repo_id=repo_id, task=task, fps=fps)

    @staticmethod
    def _first_task(dataset_root: Path) -> str:
        return first_dataset_task(dataset_root)
