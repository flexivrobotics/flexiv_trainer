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
