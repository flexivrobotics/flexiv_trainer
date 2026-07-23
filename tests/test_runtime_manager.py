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

import json
from types import SimpleNamespace

import numpy as np
import pytest

from flexivtrainer.config import AppSettings, RobotSerialConfig, StorageConfig
from flexivtrainer.runtime.manager import RuntimeManager


def make_manager(tmp_path, started_camera_count: int) -> RuntimeManager:
    manager = RuntimeManager.__new__(RuntimeManager)
    manager.settings = AppSettings(storage=StorageConfig(root=tmp_path))
    manager._robot_config = RobotSerialConfig(
        leader_robot_serials=["LEADER_A", "LEADER_B"],
        follower_robot_serials=["FOLLOWER_A", "FOLLOWER_B"],
    ).normalized()
    manager.teleop = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            available=True,
            initialized=False,
            started=False,
            error=None,
            fault=None,
        )
    )
    camera_names = ["ego", "left_wrist", "right_wrist"]
    camera_status = {
        name: {
            "started": index < started_camera_count,
            "streaming": index < started_camera_count,
        }
        for index, name in enumerate(camera_names)
    }
    manager.cameras = SimpleNamespace(
        status=lambda: {
            "available": True,
            "cameras": camera_status,
            "errors": {},
        }
    )
    return manager


@pytest.mark.parametrize(
    ("started_camera_count", "expected_state", "expected_tone"),
    [
        (0, "0/3 connected", "error"),
        (1, "1/3 connected", "working"),
        (3, "3/3 connected", "ok"),
    ],
)
def test_service_summary_reports_camera_count_and_tone(
    tmp_path, started_camera_count: int, expected_state: str, expected_tone: str
) -> None:
    manager = make_manager(tmp_path, started_camera_count)

    summary = manager.service_summary()

    assert summary["cameras"]["state"] == expected_state
    assert summary["cameras"]["tone"] == expected_tone


def test_service_summary_reports_starting_cameras_as_working(tmp_path) -> None:
    # Started but not yet streaming (warming up / watchdog recovering) must show
    # as connecting, not a 0/N failure, so the UI keeps its spinner.
    manager = make_manager(tmp_path, 0)
    manager.cameras = SimpleNamespace(
        status=lambda: {
            "available": True,
            "cameras": {
                "ego": {"started": True, "streaming": False},
                "left_wrist": {"started": True, "streaming": False},
                "right_wrist": {"started": True, "streaming": False},
            },
            "errors": {},
        }
    )

    summary = manager.service_summary()

    assert summary["cameras"]["state"] == "0/3 connected"
    assert summary["cameras"]["tone"] == "working"


def test_bootstrap_teleop_module_is_not_ready_when_camera_start_fails(tmp_path) -> None:
    manager = RuntimeManager.__new__(RuntimeManager)
    manager.settings = AppSettings(storage=StorageConfig(root=tmp_path))
    manager._robot_config = RobotSerialConfig(
        leader_robot_serials=["LEADER_A", "LEADER_B"],
        follower_robot_serials=["FOLLOWER_A", "FOLLOWER_B"],
    ).normalized()
    manager.teleop = SimpleNamespace(
        initialize=lambda: SimpleNamespace(
            configured=True,
            available=True,
            initialized=True,
            started=False,
            stopped=True,
            fault=None,
            error=None,
        ),
        robot_data_snapshot=lambda: {
            "robots": {
                "FOLLOWER_A": {"connected": True},
                "FOLLOWER_B": {"connected": True},
            },
            "errors": {},
        },
    )
    manager.cameras = SimpleNamespace(
        start_streams=lambda: {
            "available": True,
            "errors": {"ego": "No RealSense camera is available for this stream"},
            "cameras": {
                "ego": {"started": False},
                "left_wrist": {"started": False},
                "right_wrist": {"started": False},
            },
        },
        configured_serials=lambda: {},
    )
    manager.recording = SimpleNamespace(status=lambda: {"active": False})

    result = manager.bootstrap_teleop_module()

    assert result["ready"] is False


def _bare_manager(tmp_path) -> RuntimeManager:
    manager = RuntimeManager.__new__(RuntimeManager)
    manager.settings = AppSettings(storage=StorageConfig(root=tmp_path))
    manager.settings.ensure_storage()
    manager._dataset_cache = {}
    return manager


def _make_episode(directory) -> None:
    # Minimal LeRobot-style dataset marker the listing/browser use to validate.
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / "info.json").write_text("{}", encoding="utf-8")


def test_list_episode_datasets_groups_by_job(tmp_path) -> None:
    manager = _bare_manager(tmp_path)
    episodes_root = manager.settings.storage.episodes_root
    _make_episode(episodes_root / "job_0" / "ep_a")
    _make_episode(episodes_root / "job_0" / "ep_b")
    _make_episode(episodes_root / "pick_place" / "ep_c")
    # A flat, ungrouped episode directly under episodes/ (older layout).
    _make_episode(episodes_root / "legacy_ep")

    episodes = manager.list_episode_datasets()
    by_name = {ep["name"]: ep for ep in episodes}

    assert by_name["ep_a"]["job"] == "job_0"
    assert by_name["ep_b"]["job"] == "job_0"
    assert by_name["ep_c"]["job"] == "pick_place"
    assert by_name["legacy_ep"]["job"] is None


def test_browse_path_expands_job_folders_into_episodes(tmp_path) -> None:
    manager = _bare_manager(tmp_path)
    episodes_root = manager.settings.storage.episodes_root
    _make_episode(episodes_root / "job_0" / "ep_a")
    _make_episode(episodes_root / "legacy_ep")

    result = manager.browse_path(
        path=episodes_root,
        directories_only=True,
        root_path=episodes_root,
        annotate_episode_dirs=True,
    )
    items = {item["name"]: item for item in result["items"]}

    # The job folder is flattened into its episode, tagged with the job name.
    assert items["ep_a"]["job"] == "job_0"
    assert items["ep_a"]["is_valid_episode"] is True
    # The flat legacy episode is listed directly, with no job.
    assert items["legacy_ep"]["job"] is None
    assert items["legacy_ep"]["is_valid_episode"] is True
    # No raw "job_0" folder leaks into the listing.
    assert "job_0" not in items


def test_browse_path_annotates_checkpoint_dirs(tmp_path) -> None:
    manager = _bare_manager(tmp_path)
    training = manager.settings.storage.root
    run = training / "act_run"
    model = run / "checkpoints" / "034800" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"type": "act"}', encoding="utf-8")
    (run / "wandb").mkdir()

    # Top level: the run folder is badged with its policy type, not the step.
    top = manager.browse_path(
        path=training,
        directories_only=True,
        root_path=training,
        annotate_checkpoint_dirs=True,
    )
    top_items = {item["name"]: item for item in top["items"]}
    assert top_items["act_run"]["checkpoint_type"] == "act"
    assert "is_checkpoint" not in top_items["act_run"]

    # Step level: the step folder is the terminal target, without a badge.
    steps = manager.browse_path(
        path=run / "checkpoints",
        directories_only=True,
        root_path=training,
        annotate_checkpoint_dirs=True,
    )
    step_items = {item["name"]: item for item in steps["items"]}
    assert step_items["034800"]["is_checkpoint"] is True
    assert "checkpoint_type" not in step_items["034800"]


def test_browse_path_annotates_created_time_for_sorting(tmp_path) -> None:
    # Every browsed entry carries a numeric "created" time so the episode picker
    # can sort by it; the flattened job episodes carry it too.
    manager = _bare_manager(tmp_path)
    episodes_root = manager.settings.storage.episodes_root
    _make_episode(episodes_root / "job_0" / "ep_a")
    _make_episode(episodes_root / "legacy_ep")

    result = manager.browse_path(
        path=episodes_root,
        directories_only=True,
        root_path=episodes_root,
        annotate_episode_dirs=True,
    )

    assert result["items"], "expected at least one browsed entry"
    for item in result["items"]:
        assert isinstance(item["created"], float)


def test_dataset_resolution_from_info(tmp_path) -> None:
    from flexivtrainer.runtime.manager import _dataset_resolution_from_info

    # Reads [height, width] from the first video feature's shape in info.json.
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [7]},
                    "observation.images.ego": {
                        "dtype": "video",
                        "shape": [360, 480, 3],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    assert _dataset_resolution_from_info(tmp_path) == [360, 480]

    # No camera feature -> None; unreadable dir -> None (never raises).
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "meta").mkdir()
    (empty / "meta" / "info.json").write_text(
        json.dumps({"features": {"observation.state": {"dtype": "float32"}}}),
        encoding="utf-8",
    )
    assert _dataset_resolution_from_info(empty) is None
    assert _dataset_resolution_from_info(tmp_path / "does_not_exist") is None


class _PreviewDataset:
    features = {
        "observation.images.ego": {
            "dtype": "video",
            "shape": [2, 3, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.ego_depth": {
            "dtype": "video",
            "shape": [2, 3, 1],
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True, "depth_unit": "mm"},
        },
    }
    fps = 30
    num_frames = 1
    num_episodes = 1
    meta = SimpleNamespace(episodes=None)
    root = "/tmp/_preview_rgbd"

    def get_raw_item(self, index):
        return {"task": "depth test"}

    def __getitem__(self, index):
        return {
            "observation.images.ego": np.zeros((3, 2, 3), dtype=np.float32),
            "observation.images.ego_depth": np.array(
                [[[0, 500, 1000], [1500, 2000, 2500]]], dtype=np.float32
            ),
        }


def _preview_manager(tmp_path, monkeypatch) -> RuntimeManager:
    manager = _bare_manager(tmp_path)
    dataset_path = manager.settings.storage.merged_root / "rgbd"
    dataset_path.mkdir()
    monkeypatch.setattr(
        manager,
        "_resolve_dataset_repo",
        lambda path: (dataset_path, "local/rgbd"),
    )
    monkeypatch.setattr(manager, "_load_dataset", lambda path: _PreviewDataset())
    return manager


def test_preview_dataset_lists_depth_separately(tmp_path, monkeypatch) -> None:
    manager = _preview_manager(tmp_path, monkeypatch)

    preview = manager.preview_dataset(manager.settings.storage.merged_root / "rgbd")

    assert preview["camera_keys"] == ["observation.images.ego"]
    assert preview["resolution"] == [2, 3]  # [height, width] from camera shape
