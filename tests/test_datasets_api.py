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

"""Tests for the datasets API routes (browse, preview, merge) including security."""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from flexivtrainer.api.app import create_app
from flexivtrainer.config import AppSettings, RobotSerialConfig, StorageConfig
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager


def _make_manager(tmp_path: Path) -> RuntimeManager:
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
        status=lambda: {"available": False, "cameras": {}, "errors": {}},
        stop_streams=lambda: None,
    )
    manager.recording = SimpleNamespace(shutdown=lambda: None, status=lambda: {})
    manager.training = SimpleNamespace(shutdown=lambda: None)
    return manager


def test_browse_path_within_storage_succeeds(tmp_path: Path) -> None:
    app = create_app()
    manager = _make_manager(tmp_path)
    app.dependency_overrides[get_runtime_manager] = lambda: manager
    client = TestClient(app)
    subdir = tmp_path / "episodes"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "demo1").mkdir()

    response = client.get(f"/datasets/browse?path={subdir}")

    assert response.status_code == 200
    data = response.json()
    assert data["path"] == str(subdir)
    assert any(item["name"] == "demo1" for item in data["items"])


def test_browse_path_traversal_returns_403(tmp_path: Path) -> None:
    app = create_app()
    manager = _make_manager(tmp_path)
    app.dependency_overrides[get_runtime_manager] = lambda: manager
    client = TestClient(app)

    response = client.get("/datasets/browse?path=/etc")

    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


def test_browse_directories_only_filters_files(tmp_path: Path) -> None:
    app = create_app()
    manager = _make_manager(tmp_path)
    app.dependency_overrides[get_runtime_manager] = lambda: manager
    client = TestClient(app)
    (tmp_path / "afile.txt").write_text("hello")
    (tmp_path / "adir").mkdir()

    response = client.get(f"/datasets/browse?path={tmp_path}&directories_only=true")

    assert response.status_code == 200
    items = response.json()["items"]
    assert all(item["is_dir"] for item in items)
    assert any(item["name"] == "adir" for item in items)


def test_list_episodes_returns_episodes(tmp_path: Path) -> None:
    app = create_app()
    manager = _make_manager(tmp_path)
    app.dependency_overrides[get_runtime_manager] = lambda: manager
    client = TestClient(app)
    # An episode is recognized as a standard LeRobot dataset via meta/info.json.
    ep_dir = tmp_path / "episodes" / "ep_001"
    (ep_dir / "meta").mkdir(parents=True, exist_ok=True)
    (ep_dir / "meta" / "info.json").write_text("{}")

    response = client.get("/datasets/episodes")

    assert response.status_code == 200
    episodes = response.json()["episodes"]
    assert any("ep_001" in ep["name"] for ep in episodes)
