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

"""Tests for path traversal protection in dataset and rollout operations."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from flexivtrainer.config import AppSettings, RobotSerialConfig, StorageConfig
from flexivtrainer.rollout.service import resolve_checkpoint_path
from flexivtrainer.runtime.manager import RuntimeManager


def make_manager_with_storage(tmp_path: Path) -> RuntimeManager:
    manager = RuntimeManager.__new__(RuntimeManager)
    storage = StorageConfig(root=tmp_path)
    storage.ensure()
    manager.settings = AppSettings(storage=storage)
    manager._robot_config = RobotSerialConfig().normalized()
    manager.teleop = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            available=False, initialized=False, started=False, error=None, fault=None
        )
    )
    manager.cameras = SimpleNamespace(
        status=lambda: {"available": False, "cameras": {}, "errors": {}}
    )
    return manager


class TestBrowsePathSecurity:
    def test_browse_within_storage_root_succeeds(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)
        subdir = tmp_path / "episodes" / "demo1"
        subdir.mkdir(parents=True)

        result = manager.browse_path(subdir)

        assert result["path"] == str(subdir)

    def test_browse_outside_storage_root_raises(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)

        with pytest.raises(ValueError, match="Access denied"):
            manager.browse_path(Path("/etc"))

    def test_browse_with_parent_traversal_raises(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)

        with pytest.raises(ValueError, match="Access denied"):
            manager.browse_path(tmp_path / ".." / ".." / "etc")

    def test_browse_default_path_is_storage_root(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)

        result = manager.browse_path(None)

        assert result["path"] == str(tmp_path)

    def test_browse_nonexistent_path_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        manager = make_manager_with_storage(tmp_path)

        with pytest.raises(FileNotFoundError):
            manager.browse_path(tmp_path / "no_such_dir")


class TestPreviewDatasetSecurity:
    def test_preview_outside_storage_root_raises(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)

        with pytest.raises(ValueError, match="Access denied"):
            manager.preview_dataset(Path("/tmp/malicious_dataset"))

    def test_preview_with_traversal_raises(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)

        with pytest.raises(ValueError, match="Access denied"):
            manager.preview_dataset(tmp_path / ".." / ".." / "etc" / "passwd")

    def test_preview_with_storage_prefix_collision_raises(
        self, tmp_path: Path
    ) -> None:
        manager = make_manager_with_storage(tmp_path)
        sibling = tmp_path.with_name(f"{tmp_path.name}-other")
        sibling.mkdir()

        with pytest.raises(ValueError, match="Access denied"):
            manager._resolve_dataset_repo(sibling)


class TestDatasetVideoPathSecurity:
    @staticmethod
    def _video_path(
        manager: RuntimeManager, camera_key: str = "observation.images.ego"
    ) -> Path:
        video = (
            manager.settings.storage.merged_root
            / "rgbd"
            / "videos"
            / camera_key
            / "chunk-000"
            / "file-000.mp4"
        )
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        return video

    def test_existing_camera_video_resolves(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)
        video = self._video_path(manager)
        dataset = manager.settings.storage.merged_root / "rgbd"

        assert manager.dataset_video_path(
            dataset, "observation.images.ego"
        ) == video.resolve()

    @pytest.mark.parametrize(
        "camera_key", ["../outside", "../../etc/passwd", "/etc/passwd", r"..\outside"]
    )
    def test_camera_key_traversal_raises(
        self, tmp_path: Path, camera_key: str
    ) -> None:
        manager = make_manager_with_storage(tmp_path)
        video = self._video_path(manager)

        with pytest.raises(ValueError, match="Access denied"):
            manager.dataset_video_path(video.parents[3], camera_key)

    def test_camera_directory_symlink_escape_raises(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)
        dataset = manager.settings.storage.merged_root / "rgbd"
        videos = dataset / "videos"
        videos.mkdir(parents=True)
        outside = tmp_path.parent / "outside-camera-videos"
        outside.mkdir()
        (videos / "observation.images.ego").symlink_to(
            outside, target_is_directory=True
        )

        with pytest.raises(ValueError, match="Access denied"):
            manager.dataset_video_path(dataset, "observation.images.ego")

    def test_video_file_symlink_escape_raises(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)
        dataset = manager.settings.storage.merged_root / "rgbd"
        chunk = dataset / "videos" / "observation.images.ego" / "chunk-000"
        chunk.mkdir(parents=True)
        outside = tmp_path.parent / "outside-video.mp4"
        outside.write_bytes(b"video")
        (chunk / "file-000.mp4").symlink_to(outside)

        with pytest.raises(ValueError, match="Access denied"):
            manager.dataset_video_path(dataset, "observation.images.ego")


class TestCheckpointPathSecurity:
    def test_checkpoint_within_storage_root_succeeds(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "training" / "run" / "checkpoints" / "034800"
        ckpt.mkdir(parents=True)

        assert resolve_checkpoint_path(str(ckpt), tmp_path) == ckpt.resolve()

    def test_checkpoint_outside_storage_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Access denied"):
            resolve_checkpoint_path("/etc", tmp_path)

    def test_checkpoint_with_traversal_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Access denied"):
            resolve_checkpoint_path(str(tmp_path / ".." / ".." / "etc"), tmp_path)

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            resolve_checkpoint_path(str(tmp_path / "missing"), tmp_path)

    def test_checkpoint_with_prefix_collision_raises(self, tmp_path: Path) -> None:
        sibling = tmp_path.with_name(f"{tmp_path.name}-other")
        sibling.mkdir()

        with pytest.raises(ValueError, match="Access denied"):
            resolve_checkpoint_path(str(sibling), tmp_path)

    def test_checkpoint_symlink_escape_raises(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside-checkpoint"
        outside.mkdir()
        link = tmp_path / "linked-checkpoint"
        link.symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match="Access denied"):
            resolve_checkpoint_path(str(link), tmp_path)


class TestMergeEpisodesSecurity:
    def test_merge_with_path_outside_storage_raises(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)

        with pytest.raises(ValueError, match="Access denied"):
            manager.merge_episodes(["/etc/malicious"], "output")

    def test_merge_with_traversal_path_raises(self, tmp_path: Path) -> None:
        manager = make_manager_with_storage(tmp_path)
        evil_path = str(tmp_path / ".." / ".." / "etc")

        with pytest.raises(ValueError, match="Access denied"):
            manager.merge_episodes([evil_path], "output")
