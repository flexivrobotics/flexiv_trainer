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

from pathlib import Path
from typing import Any


def _load_manifest(root: Path) -> Any:
    from flexivtrainer.data.lerobot_io import EpisodeManifest

    return EpisodeManifest.from_path(root)


def combine_episode_datasets(
    episode_roots: list[Path],
    output_root: Path,
    output_name: str,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Combine episode datasets using LeRobot's built-in merge_datasets.

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
