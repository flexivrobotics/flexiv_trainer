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

"""Resolve checkpoint paths and read metadata baked into LeRobot checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flexivtrainer.data.gripper_command import (
    GRIPPER_COMMAND_FILENAME,
    GripperCommandMetadata,
    read_gripper_command_metadata,
)
from flexivtrainer.data.hub import ACTION_NAMES_FILENAME
from flexivtrainer.data.lerobot_io import first_dataset_task

_LANGUAGE_POLICY_TYPES = {"multi_task_dit", "smolvla", "pi0", "pi05"}
_IMAGE_FEATURE_PREFIX = "observation.images."
_CHANNEL_COUNTS = {1, 3, 4}  # 1: gray, 3: RGB, 4: RGBD


def _checkpoint_model_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    model_dir = path.parent if path.is_file() else path
    if not (model_dir / "config.json").exists():
        nested = model_dir / "pretrained_model"
        if (nested / "config.json").exists():
            model_dir = nested
    return model_dir


def _matching_child(parent: Path, name: str) -> Path | None:
    try:
        for child in parent.iterdir():
            if child.name == name:
                return child
    except OSError:
        return None
    return None


def resolve_checkpoint_path(checkpoint_path: str, storage_root: Path) -> Path:
    """Reject a client checkpoint path that escapes the storage root."""
    root = storage_root.expanduser().resolve()
    root_text = os.fspath(root)
    root_prefix = root_text if root_text.endswith(os.sep) else root_text + os.sep
    if checkpoint_path == root_text:
        return root
    if not checkpoint_path.startswith(root_prefix):
        raise ValueError(f"Access denied: path must be within storage root ({root})")

    parts = checkpoint_path[len(root_prefix) :].split(os.sep)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Access denied: path must be within storage root ({root})")

    resolved = root
    for part in parts:
        child = _matching_child(resolved, part)
        if child is None:
            raise FileNotFoundError("Checkpoint not found")
        resolved = child.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Access denied: path must be within storage root ({root})"
            )
    return resolved


def _default_policy_loader(checkpoint_path: str, device: str) -> Any:
    """Load a LeRobot policy and its processors from a checkpoint directory."""
    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.factory import (  # noqa: PLC0415
        get_policy_class,
        make_pre_post_processors,
    )

    import flexivtrainer.policies.lerobot_plugins  # noqa: F401, PLC0415

    model_dir = _checkpoint_model_dir(checkpoint_path)
    config = PreTrainedConfig.from_pretrained(model_dir)
    policy = get_policy_class(config.type).from_pretrained(model_dir)
    policy.to(device)
    policy.eval()
    # Override the training device for CPU-only rollout hosts.
    device_override = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(model_dir),
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
    )
    return policy, preprocessor, postprocessor


def _positive_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    value = float(value)
    return value if value > 0 else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _dataset_root_candidates(root: str, model_dir: Path) -> list[Path]:
    dataset_root = Path(root).expanduser()
    candidates = [dataset_root]
    if not dataset_root.is_absolute():
        candidates.append(Path.cwd() / dataset_root)
        candidates.extend(parent / dataset_root for parent in model_dir.parents)
    return list(
        dict.fromkeys(candidate.resolve(strict=False) for candidate in candidates)
    )


def _checkpoint_dataset_candidates(
    checkpoint_path: str, storage_root: Path | None = None
) -> list[Path]:
    """Return recoverable roots for the dataset baked into a checkpoint."""

    model_dir = _checkpoint_model_dir(checkpoint_path)
    train_config = _read_json(model_dir / "train_config.json") or {}
    dataset = train_config.get("dataset") if isinstance(train_config, dict) else None
    if not isinstance(dataset, dict):
        return []

    candidates: list[Path] = []
    root = dataset.get("root")
    if isinstance(root, str) and root.strip():
        candidates.extend(_dataset_root_candidates(root, model_dir))

    # LeRobot checkpoints also retain repo_id. It lets a checkpoint survive a
    # moved/deleted absolute dataset path as long as the dataset still exists in
    # this app's storage root.
    repo_id = dataset.get("repo_id")
    if storage_root is not None and isinstance(repo_id, str) and repo_id.strip():
        dataset_name = repo_id.rstrip("/").rsplit("/", 1)[-1]
        if dataset_name:
            storage = Path(storage_root).expanduser().resolve(strict=False)
            candidates.extend(
                [storage / "datasets" / dataset_name, storage / dataset_name]
            )

    return list(
        dict.fromkeys(
            candidate.expanduser().resolve(strict=False) for candidate in candidates
        )
    )


def _config_feature_dim(checkpoint_path: str, group: str, name: str) -> int | None:
    """Read a flat feature width from a checkpoint's config.json."""

    model_dir = _checkpoint_model_dir(checkpoint_path)
    config = _read_json(model_dir / "config.json") or {}
    features = config.get(group)
    feature = features.get(name) if isinstance(features, dict) else None
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if (
        not isinstance(shape, list | tuple)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] <= 0
    ):
        return None
    return int(shape[0])


def checkpoint_action_output_dim(checkpoint_path: str) -> int | None:
    """Read the flat action width declared by a waypoint checkpoint."""
    return _config_feature_dim(checkpoint_path, "output_features", "action")


def checkpoint_state_input_dim(checkpoint_path: str) -> int | None:
    """Read the flat ``observation.state`` width a checkpoint expects."""
    return _config_feature_dim(checkpoint_path, "input_features", "observation.state")


def _validated_action_names(names: Any, origin: str) -> list[str]:
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError(f"Action feature has no valid named axes: {origin}")
    if len(set(names)) != len(names):
        raise ValueError(f"Action axes are not unique: {origin}")
    return list(names)


def checkpoint_sidecar_action_names(checkpoint_path: str) -> list[str] | None:
    """Read the self-describing action-name sidecar this app writes at training.

    A checkpoint that carries its own axis names needs no training dataset at
    rollout, which is what makes Hub checkpoints usable and also survives the
    local dataset being deleted or renamed.
    """
    model_dir = _checkpoint_model_dir(checkpoint_path)
    payload = _read_json(model_dir / ACTION_NAMES_FILENAME)
    if payload is None:
        return None
    return _validated_action_names(
        payload.get("action_names"), str(model_dir / ACTION_NAMES_FILENAME)
    )


def checkpoint_config_action_names(checkpoint_path: str) -> list[str] | None:
    """Read axis names from the policy config when it happens to carry them."""
    model_dir = _checkpoint_model_dir(checkpoint_path)
    config = _read_json(model_dir / "config.json") or {}
    output_features = config.get("output_features")
    action = (
        output_features.get("action") if isinstance(output_features, dict) else None
    )
    names = action.get("names") if isinstance(action, dict) else None
    if names is None:
        return None
    return _validated_action_names(names, str(model_dir / "config.json"))


def _hub_dataset_action_names(
    checkpoint_path: str, settings: Any
) -> list[str] | None:
    """Fetch the training dataset's metadata from the Hub to recover axis names.

    Only ``meta/`` is pulled, so this stays cheap even for video-heavy datasets.
    """
    if settings is None or not getattr(getattr(settings, "hub", None), "enabled", True):
        return None
    from flexivtrainer.data.hub import (  # noqa: PLC0415
        HubError,
        fetch_dataset_metadata,
        is_hub_repo_id,
        parse_hub_ref,
    )

    model_dir = _checkpoint_model_dir(checkpoint_path)
    train_config = _read_json(model_dir / "train_config.json") or {}
    dataset = train_config.get("dataset")
    repo_id = dataset.get("repo_id") if isinstance(dataset, dict) else None
    if not is_hub_repo_id(repo_id):
        return None
    revision = dataset.get("revision") if isinstance(dataset, dict) else None
    try:
        root = fetch_dataset_metadata(settings, parse_hub_ref(repo_id, revision))
    except (HubError, ValueError):
        # Recovery is best-effort; the caller raises an actionable error when
        # every tier misses.
        return None
    return _dataset_action_names(root)


def _dataset_action_names(candidate: Path) -> list[str] | None:
    info_path = candidate / "meta" / "info.json"
    if not info_path.is_file():
        return None
    info = _read_json(info_path) or {}
    features = info.get("features")
    action = features.get("action") if isinstance(features, dict) else None
    if action is None:
        return None
    names = _validated_action_names(
        action.get("names") if isinstance(action, dict) else None,
        str(info_path),
    )
    shape = action.get("shape")
    if not isinstance(shape, list | tuple) or len(shape) != 1 or shape[0] != len(names):
        raise ValueError(
            f"Training dataset action shape does not match its named axes: {info_path}"
        )
    return names


def checkpoint_action_names(
    checkpoint_path: str,
    storage_root: Path | None = None,
    *,
    settings: Any = None,
    override: list[str] | None = None,
) -> list[str] | None:
    """Recover ordered action-axis names for a checkpoint.

    Ordinary LeRobot waypoint configs retain the action width but not the scalar
    axis names, so the names must come from somewhere else. Tried in descending
    order of authority: an explicit caller override, the checkpoint's own
    ``action_names.json`` sidecar, the policy config, a local copy of the
    training dataset, and finally the training dataset's Hub metadata.

    Returns ``None`` only when every source misses; callers must treat that as an
    error for gripper-bearing layouts rather than guessing an axis order.
    """

    if override is not None:
        return _validated_action_names(override, "action_names override")

    sidecar = checkpoint_sidecar_action_names(checkpoint_path)
    if sidecar is not None:
        return sidecar

    from_config = checkpoint_config_action_names(checkpoint_path)
    if from_config is not None:
        return from_config

    for candidate in _checkpoint_dataset_candidates(checkpoint_path, storage_root):
        names = _dataset_action_names(candidate)
        if names is not None:
            return names

    return _hub_dataset_action_names(checkpoint_path, settings)


def resolve_hub_checkpoint(
    repo_id: str, revision: str | None, settings: Any
) -> Path:
    """Materialize a Hub checkpoint, then validate it like any local checkpoint.

    The cache lives inside the storage root, so the existing
    ``resolve_checkpoint_path`` accepts it and still applies its segment-walk and
    symlink-escape checks to the downloaded content.
    """
    from flexivtrainer.data.hub import (  # noqa: PLC0415
        fetch_checkpoint_snapshot,
        parse_hub_ref,
    )

    target = fetch_checkpoint_snapshot(settings, parse_hub_ref(repo_id, revision))
    return resolve_checkpoint_path(str(target), settings.storage.root)


def checkpoint_gripper_command_metadata(
    checkpoint_path: str,
) -> GripperCommandMetadata | None:
    path = _checkpoint_model_dir(checkpoint_path) / GRIPPER_COMMAND_FILENAME
    if not path.is_file():
        return None
    return read_gripper_command_metadata(path)


def _checkpoint_target_hz(checkpoint_path: str) -> float | None:
    """Read of the dataset FPS baked into a LeRobot checkpoint."""
    model_dir = _checkpoint_model_dir(checkpoint_path)

    train_config = _read_json(model_dir / "train_config.json") or {}
    dataset = train_config.get("dataset") if isinstance(train_config, dict) else None
    if isinstance(dataset, dict):
        if fps := _positive_float(dataset.get("fps")):
            return fps
        root = dataset.get("root")
        if isinstance(root, str) and root.strip():
            for candidate in _dataset_root_candidates(root, model_dir):
                info = _read_json(candidate / "meta" / "info.json") or {}
                if fps := _positive_float(info.get("fps")):
                    return fps

    config = _read_json(model_dir / "config.json") or {}
    for key in ("knot_rate_hz", "fps", "dataset_fps", "action_dt_hz"):
        if fps := _positive_float(config.get(key)):
            return fps
    return None


def _checkpoint_task(checkpoint_path: str) -> str | None:
    """Read of the task string of the dataset a checkpoint trained on."""
    model_dir = _checkpoint_model_dir(checkpoint_path)
    train_config = _read_json(model_dir / "train_config.json") or {}
    dataset = train_config.get("dataset") if isinstance(train_config, dict) else None
    if not isinstance(dataset, dict):
        return None
    root = dataset.get("root")
    if not (isinstance(root, str) and root.strip()):
        return None
    for candidate in _dataset_root_candidates(root, model_dir):
        if (candidate / "meta").exists():
            return first_dataset_task(candidate)
    return None


def checkpoint_image_resolutions(checkpoint_path: str) -> dict[str, tuple[int, int]]:
    """Read {camera: (height, width)} of the images a checkpoint was trained on."""
    model_dir = _checkpoint_model_dir(checkpoint_path)
    features = (_read_json(model_dir / "config.json") or {}).get("input_features")
    if not isinstance(features, dict):
        return {}

    resolutions: dict[str, tuple[int, int]] = {}
    for key, feature in features.items():
        if not key.startswith(_IMAGE_FEATURE_PREFIX) or not isinstance(feature, dict):
            continue
        name = key[len(_IMAGE_FEATURE_PREFIX) :]
        shape = feature.get("shape")
        if not name or not isinstance(shape, list) or len(shape) != 3:
            continue
        # LeRobot stores VISUAL shapes channels-first; refuse anything else
        # rather than read a channels-last shape as (height, width).
        if shape[0] in _CHANNEL_COUNTS and shape[2] not in _CHANNEL_COUNTS:
            resolutions[name] = (int(shape[1]), int(shape[2]))
    return resolutions


def _checkpoint_policy_type(checkpoint_path: str) -> str | None:
    model_dir = _checkpoint_model_dir(checkpoint_path)
    config = _read_json(model_dir / "config.json") or {}
    value = config.get("type")
    return value if isinstance(value, str) and value.strip() else None


def _checkpoint_requires_task(checkpoint_path: str) -> bool:
    policy_type = _checkpoint_policy_type(checkpoint_path)
    return policy_type is None or policy_type in _LANGUAGE_POLICY_TYPES
