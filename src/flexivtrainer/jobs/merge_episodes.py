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
from pathlib import Path
from typing import Any


def _load_manifest(root: Path) -> Any:
    from flexivtrainer.data.lerobot_io import EpisodeManifest

    return EpisodeManifest.from_path(root)


def _feature_keys(root: Path) -> set[str]:
    info_path = root / "meta" / "info.json"
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset metadata: {info_path}") from exc
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, dict):
        raise ValueError(f"Dataset metadata has no features object: {info_path}")
    return set(features)


def _validate_matching_feature_keys(episode_roots: list[Path]) -> None:
    expected = _feature_keys(episode_roots[0])
    mismatches: list[str] = []
    for root in episode_roots[1:]:
        actual = _feature_keys(root)
        if actual == expected:
            continue
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        mismatches.append(f"{root.name} ({', '.join(details)})")
    if mismatches:
        raise ValueError(
            "Datasets cannot be merged because their feature keys differ. "
            "Depth-enabled and RGB-only recordings must be merged separately: "
            + "; ".join(mismatches)
        )


def merge_episode_datasets(
    episode_roots: list[Path],
    output_root: Path,
    output_name: str,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Merge episode datasets using LeRobot's built-in merge_datasets.

    This uses file-level copy (parquet + video) rather than frame-by-frame
    decode/encode, making it significantly faster.

    Args:
        on_progress: Optional callback(episode_index, total_episodes, 0, 0)
            called as episodes are loaded.
    """
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episode_roots:
        raise ValueError("At least one episode dataset is required")

    _validate_matching_feature_keys(episode_roots)

    target_root = output_root / output_name
    if target_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {target_root}. "
            "Choose a different output name or remove the existing directory."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = [_load_manifest(root) for root in episode_roots]
    total = len(manifests)

    # Load all source datasets
    datasets: list[LeRobotDataset] = []
    for idx, manifest in enumerate(manifests):
        if on_progress:
            on_progress(idx, total, 0, 1)
        datasets.append(LeRobotDataset(manifest.repo_id, root=manifest.root))

    if on_progress:
        on_progress(total - 1, total, 1, 1)

    # Use LeRobot's optimized merge (file-level copy of parquet + videos).
    # This produces a fully standard LeRobot v3.0 dataset; the merged dataset is
    # identified and loaded via its standard meta/info.json (no extra manifest).
    merge_datasets(
        datasets=datasets,
        output_repo_id=f"local/{output_name}",
        output_dir=target_root,
    )

    return {
        "output_name": output_name,
        "root": str(target_root),
        "episodes": len(episode_roots),
    }
