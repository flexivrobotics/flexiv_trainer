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

"""Create a LeRobot dataset whose action vector has no TCP wrench axes.

Only action axes named ``<side>.tcp_wrench.<axis>`` are removed. Observation
features, including wrench values embedded in ``observation.state``, are left
unchanged.

Run this module directly or with ``-m``::

    python -m flexivtrainer.data.remove_action_wrench SOURCE [OUTPUT]

When OUTPUT is omitted, a sibling directory named ``SOURCE_no_wrench`` is
created. The source dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_WRENCH_ACTION_MARKER = ".tcp_wrench."


def _restore_feature_order(root: Path, source_order: list[str]) -> None:
    """Keep converted feature metadata in the same order as the source."""

    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset metadata: {info_path}") from exc
    features = info.get("features") if isinstance(info, dict) else None
    if not isinstance(features, dict):
        raise ValueError(f"Dataset metadata has no features object: {info_path}")

    missing = [name for name in source_order if name not in features]
    if missing:
        raise ValueError(f"Converted metadata is missing features: {missing}")
    ordered_names = [
        *source_order,
        *(name for name in features if name not in source_order),
    ]
    info["features"] = {name: features[name] for name in ordered_names}
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")


def _action_projection(action_names: list[str]) -> tuple[list[int], list[int]]:
    """Return the retained and wrench indices in a named action vector."""

    if not action_names:
        raise ValueError("The action feature must have named axes")
    if len(set(action_names)) != len(action_names):
        raise ValueError("Action feature names must be unique")

    wrench_indices = [
        index
        for index, name in enumerate(action_names)
        if _WRENCH_ACTION_MARKER in name
    ]
    if not wrench_indices:
        raise ValueError(
            "No TCP wrench action axes found. Expected names such as "
            "'left_arm.tcp_wrench.fx'."
        )
    wrench = set(wrench_indices)
    retained_indices = [
        index for index in range(len(action_names)) if index not in wrench
    ]
    if not retained_indices:
        raise ValueError("Removing wrench axes would leave an empty action vector")
    return retained_indices, wrench_indices


def _refresh_action_statistics(
    root: Path,
    actions: np.ndarray,
    episode_indices: np.ndarray,
    action_feature: dict[str, Any],
    existing_stats: dict[str, Any],
) -> None:
    """Recompute global and per-episode statistics for the sliced action."""

    try:
        import pandas as pd
        from lerobot.datasets.dataset_tools import (
            compute_episode_stats,
            write_stats,
        )
    except ImportError as exc:  # pragma: no cover - installed with LeRobot
        raise RuntimeError(
            "LeRobot 0.6 and pandas are required to update dataset statistics"
        ) from exc

    features = {"action": action_feature}

    def compute(values: np.ndarray) -> dict[str, Any]:
        return compute_episode_stats({"action": values}, features)["action"]

    updated_stats = dict(existing_stats)
    updated_stats["action"] = compute(actions)
    write_stats(updated_stats, root)

    per_episode = {
        int(episode_index): compute(actions[episode_indices == episode_index])
        for episode_index in np.unique(episode_indices)
    }
    updated_episodes: set[int] = set()
    metadata_paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not metadata_paths:
        raise ValueError(f"No episode metadata found under {root / 'meta/episodes'}")

    stat_names = set().union(*(stats.keys() for stats in per_episode.values()))
    for metadata_path in metadata_paths:
        metadata = pd.read_parquet(metadata_path)
        for stat_name in stat_names:
            column = f"stats/action/{stat_name}"
            if column not in metadata:
                metadata[column] = None

        for row_index, episode_index in metadata["episode_index"].items():
            index = int(episode_index)
            stats = per_episode.get(index)
            if stats is None:
                continue
            for stat_name, value in stats.items():
                serialized = value.tolist() if hasattr(value, "tolist") else value
                metadata.at[row_index, f"stats/action/{stat_name}"] = serialized
            updated_episodes.add(index)

        temporary = metadata_path.with_suffix(".tmp.parquet")
        metadata.to_parquet(temporary, index=False)
        temporary.replace(metadata_path)

    missing = set(per_episode) - updated_episodes
    if missing:
        raise ValueError(
            f"Episode metadata is missing rows for episodes: {sorted(missing)}"
        )


def _validate_output(
    root: Path,
    expected_frames: int,
    expected_action_names: list[str],
) -> None:
    """Load the result through LeRobot and verify its action schema."""

    try:
        from datasets import config as datasets_config
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - installed with project
        raise RuntimeError("LeRobot 0.6 is required to validate the output") from exc

    original_cache = datasets_config.HF_DATASETS_CACHE
    with tempfile.TemporaryDirectory(prefix="flexivtrainer-hf-") as cache_dir:
        datasets_config.HF_DATASETS_CACHE = cache_dir
        try:
            dataset = LeRobotDataset(
                repo_id=f"local/{root.name}",
                root=root,
                download_videos=False,
            )
        finally:
            datasets_config.HF_DATASETS_CACHE = original_cache

    if len(dataset) != expected_frames:
        raise ValueError(
            f"Converted dataset has {len(dataset)} frames, expected {expected_frames}"
        )
    feature = dataset.meta.features.get("action")
    if not feature:
        raise ValueError("Converted dataset has no action feature")
    actual_names = list(feature.get("names") or [])
    if actual_names != expected_action_names:
        raise ValueError("Converted action names do not match the requested projection")
    if tuple(feature.get("shape") or ()) != (len(expected_action_names),):
        raise ValueError(
            f"Converted action shape is {feature.get('shape')}, expected "
            f"({len(expected_action_names)},)"
        )


def remove_action_wrench(
    source_root: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy a LeRobot dataset while removing named TCP wrench action axes.

    The public LeRobot feature-editing API operates on complete features rather
    than individual vector axes. This helper therefore removes the original
    ``action`` feature and adds a projected ``action`` feature back. Work is
    staged in a temporary directory and the source is never modified.
    """

    try:
        from lerobot.datasets.dataset_tools import add_features, remove_feature
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - installed with project
        raise RuntimeError("LeRobot 0.6 is required for dataset conversion") from exc

    source = Path(source_root).expanduser().resolve()
    if output_root is None:
        output = source.with_name(f"{source.name}_no_wrench")
    else:
        output = Path(output_root).expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"No LeRobot metadata found under: {source}")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    if output == source or output.is_relative_to(source):
        raise ValueError("Output dataset cannot be the source or inside it")

    dataset = LeRobotDataset(
        repo_id=f"local/{source.name}",
        root=source,
        download_videos=False,
    )
    action_feature = dataset.meta.features.get("action")
    if not action_feature:
        raise ValueError("Source dataset has no action feature")
    source_feature_order = list(dataset.meta.features)
    action_names = list(action_feature.get("names") or [])
    retained_indices, wrench_indices = _action_projection(action_names)

    formatted = dataset.hf_dataset.with_format("numpy")
    actions = np.asarray(formatted["action"], dtype=np.float32)
    episode_indices = np.asarray(formatted["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(formatted["frame_index"], dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] != len(action_names):
        raise ValueError(
            f"Action data has shape {actions.shape}, expected "
            f"[frames, {len(action_names)}]"
        )
    if len(actions) == 0:
        raise ValueError("Source dataset contains no action frames")
    if len(episode_indices) != len(actions):
        raise ValueError("Action and episode-index columns have different lengths")
    if len(frame_indices) != len(actions):
        raise ValueError("Action and frame-index columns have different lengths")

    projected_actions = actions[:, retained_indices]
    projected_names = [action_names[index] for index in retained_indices]
    removed_names = [action_names[index] for index in wrench_indices]
    projected_feature = {
        "dtype": "float32",
        "shape": (len(projected_names),),
        "names": projected_names,
    }
    removed_values = actions[:, wrench_indices]
    max_abs_removed = float(np.max(np.abs(removed_values)))
    projected_by_frame = {
        (int(episode_index), int(frame_index)): action
        for episode_index, frame_index, action in zip(
            episode_indices,
            frame_indices,
            projected_actions,
            strict=True,
        )
    }
    if len(projected_by_frame) != len(projected_actions):
        raise ValueError("Dataset contains duplicate episode/frame indices")

    def projected_action(
        _row: dict[str, Any],
        episode_index: int,
        frame_index: int,
    ) -> np.ndarray:
        key = (int(episode_index), int(frame_index))
        try:
            return projected_by_frame[key]
        except KeyError as exc:
            raise ValueError(f"No projected action found for frame {key}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.remove-wrench-",
            dir=output.parent,
        )
    )
    intermediate = temporary_root / "without-action"
    staging = temporary_root / "output"
    try:
        without_action = remove_feature(
            dataset,
            "action",
            output_dir=intermediate,
            repo_id=f"local/{source.name}_without_action",
        )
        add_features(
            without_action,
            {"action": (projected_action, projected_feature)},
            output_dir=staging,
            repo_id=f"local/{output.name}",
        )
        _restore_feature_order(staging, source_feature_order)
        _refresh_action_statistics(
            staging,
            projected_actions,
            episode_indices,
            projected_feature,
            dict(dataset.meta.stats or {}),
        )
        _validate_output(staging, len(actions), projected_names)
        staging.replace(output)
    except Exception:
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    return {
        "source_root": str(source),
        "output_root": str(output),
        "frames": len(actions),
        "source_action_dim": len(action_names),
        "output_action_dim": len(projected_names),
        "removed_action_names": removed_names,
        "max_abs_removed_wrench": max_abs_removed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a LeRobot dataset and remove *.tcp_wrench.* axes from action."
        )
    )
    parser.add_argument("source", type=Path, help="Source LeRobot dataset root")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output root (default: SOURCE_no_wrench)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = remove_action_wrench(args.source, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
