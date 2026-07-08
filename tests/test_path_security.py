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

"""Tests for path traversal protection in browse_path, preview_dataset, merge_episodes."""

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
