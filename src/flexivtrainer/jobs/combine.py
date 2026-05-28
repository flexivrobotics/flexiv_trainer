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


def combine_episode_datasets(
    episode_roots: list[Path], output_root: Path, output_name: str
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episode_roots:
        raise ValueError("At least one episode dataset is required")

    manifests = [_load_manifest(root) for root in episode_roots]
    first = manifests[0]
    first_dataset = LeRobotDataset(first.repo_id, root=first.root)

    target_root = output_root / output_name
    if target_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {target_root}. "
            "Choose a different output name or remove the existing directory."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    target = LeRobotDataset.create(
        repo_id=f"local/{output_name}",
        fps=first_dataset.fps,
        features=first_dataset.features,
        root=target_root,
        robot_type="flexiv_rizon_dual",
        use_videos=True,
    )

    # Metadata keys injected by LeRobot that must not be passed to add_frame
    _META_KEYS = {"index", "episode_index", "frame_index", "task_index", "timestamp"}

    for manifest in manifests:
        dataset = LeRobotDataset(manifest.repo_id, root=manifest.root)
        if dataset.fps != first_dataset.fps:
            raise ValueError(
                f"FPS mismatch for {manifest.root}: {dataset.fps} != {first_dataset.fps}"
            )
        if dataset.features != first_dataset.features:
            raise ValueError(f"Feature schema mismatch for {manifest.root}")

        image_keys = {
            key
            for key, feat in dataset.features.items()
            if feat.get("dtype") in ("image", "video")
        }

        for index in range(dataset.num_frames):
            item = dataset.get_raw_item(index)
            frame: dict[str, Any] = {}
            for key in dataset.features:
                if key in _META_KEYS or key not in item:
                    continue
                value = item[key]
                # LeRobot returns images as CHW tensors; add_frame expects HWC numpy
                if key in image_keys:
                    import numpy as np
                    import torch

                    if isinstance(value, torch.Tensor):
                        value = value.numpy()
                    if (
                        hasattr(value, "ndim")
                        and value.ndim == 3
                        and value.shape[0] in (1, 3, 4)
                    ):
                        value = np.moveaxis(value, 0, -1)
                frame[key] = value
            frame["task"] = item.get("task", manifest.task)
            target.add_frame(frame)

        target.save_episode()

    target.finalize()
    (target_root / "combined.json").write_text(
        json.dumps(
            {
                "repo_id": f"local/{output_name}",
                "root": str(target_root),
                "episodes": [str(root) for root in episode_roots],
                "fps": first_dataset.fps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "output_name": output_name,
        "root": str(target_root),
        "episodes": len(episode_roots),
    }
