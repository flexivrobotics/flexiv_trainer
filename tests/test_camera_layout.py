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

"""Operator-configured camera slots: counts, names, and how they propagate."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flexivtrainer.cameras import service as camera_module
from flexivtrainer.cameras.service import CameraService
from flexivtrainer.config import (
    MAX_CAMERAS_BY_ARM_MODE,
    AppSettings,
    CameraConfig,
    RobotSerialConfig,
    StorageConfig,
)
from flexivtrainer.data.lerobot_io import (
    build_features_from_sample,
    default_recording_entry_keys,
    resolve_recording_depth_names,
    resolve_recording_entries,
    resolve_recording_image_names,
)
from flexivtrainer.runtime.manager import RuntimeManager


@pytest.fixture(autouse=True)
def _isolate_camera_sdks(monkeypatch):
    """Keep this module off real hardware (see tests/test_camera_service.py)."""
    monkeypatch.setattr(camera_module, "rs", None)
    monkeypatch.setattr(camera_module, "ob", None)


def _config(**kwargs) -> RobotSerialConfig:
    return RobotSerialConfig(**kwargs).normalized()


def test_defaults_reproduce_the_pre_configuration_layout() -> None:
    # An unconfigured rig must record exactly the camera names it always has,
    # so existing datasets and checkpoints keep matching.
    assert _config(arm_mode="single").active_camera_names() == ["ego", "wrist"]
    assert _config(arm_mode="dual").active_camera_names() == [
        "ego",
        "left_wrist",
        "right_wrist",
    ]


def test_ceiling_is_configured_per_arm_mode() -> None:
    assert _config(arm_mode="single").max_camera_count() == 3
    assert _config(arm_mode="dual").max_camera_count() == 5
    assert MAX_CAMERAS_BY_ARM_MODE == {"single": 3, "dual": 5}
    assert _config(arm_mode="single").camera_slot_names() == [
        "ego",
        "wrist",
        "aux_1",
    ]
    assert _config(arm_mode="dual").camera_slot_names() == [
        "ego",
        "left_wrist",
        "right_wrist",
        "aux_1",
        "aux_2",
    ]


@pytest.mark.parametrize(
    ("mode", "count", "expected"),
    [
        ("single", 3, ["ego", "wrist", "aux_1"]),
        ("single", 1, ["ego"]),
        ("single", 99, ["ego", "wrist", "aux_1"]),
        ("single", 0, ["ego"]),
        ("dual", 5, ["ego", "left_wrist", "right_wrist", "aux_1", "aux_2"]),
        ("dual", 9, ["ego", "left_wrist", "right_wrist", "aux_1", "aux_2"]),
        ("dual", 4, ["ego", "left_wrist", "right_wrist", "aux_1"]),
        ("dual", -5, ["ego"]),
    ],
)
def test_count_selects_a_prefix_and_clamps_to_the_ceiling(
    mode, count, expected
) -> None:
    config = _config(arm_mode=mode, camera_counts={mode: count})
    assert config.active_camera_names() == expected


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("front cam", "spaces are not allowed in a feature name"),
        ("../etc", "must not escape the dataset video directory"),
        ("", "a slot always has a name"),
        ("_leading", "must start alphanumeric"),
        ("x" * 33, "longer than the 32 character limit"),
        ("side_depth", "would collide with side's depth feature"),
    ],
)
def test_unusable_name_falls_back_to_the_slot_default(name, reason) -> None:
    config = _config(
        arm_mode="single",
        camera_counts={"single": 3},
        camera_names={"single": ["ego", "wrist", name]},
    )
    assert config.active_camera_names() == ["ego", "wrist", "aux_1"], reason


def test_duplicate_name_falls_back_rather_than_shadowing_a_slot() -> None:
    # Names key the camera runtimes, so two slots may never share one.
    config = _config(
        arm_mode="single",
        camera_counts={"single": 3},
        camera_names={"single": ["overhead", "overhead", "overhead"]},
    )
    assert config.active_camera_names() == ["overhead", "wrist", "aux_1"]


def test_default_fallback_is_suffixed_when_it_would_also_collide() -> None:
    # Slot 0 took slot 1's default name, so slot 1 cannot fall back onto it.
    config = _config(
        arm_mode="single",
        camera_counts={"single": 2},
        camera_names={"single": ["wrist", "wrist"]},
    )
    assert config.active_camera_names() == ["wrist", "wrist_2"]


def test_both_modes_survive_a_single_dual_single_round_trip() -> None:
    config = _config(
        arm_mode="single",
        camera_counts={"single": 3, "dual": 5},
        camera_names={
            "single": ["overhead", "wrist", "side"],
            "dual": ["a", "b", "c", "d", "e", "f"],  # trimmed to the ceiling
        },
    )
    assert config.active_camera_names() == ["overhead", "wrist", "side"]

    as_dual = config.model_copy(update={"arm_mode": "dual"}).normalized()
    assert as_dual.active_camera_names() == ["a", "b", "c", "d", "e"]

    back = as_dual.model_copy(update={"arm_mode": "single"}).normalized()
    assert back.active_camera_names() == ["overhead", "wrist", "side"]


def test_recording_entries_follow_the_configured_camera_names() -> None:
    cameras = ["overhead", "wrist", "side"]
    entries = default_recording_entry_keys(["single_arm"], cameras)

    assert [entry for entry in entries if entry.startswith("observation.images.")] == [
        "observation.images.overhead",
        "observation.images.wrist",
        "observation.images.side",
        "observation.images.overhead_depth",
        "observation.images.wrist_depth",
        "observation.images.side_depth",
    ]
    assert resolve_recording_image_names(None, ["single_arm"], cameras) == cameras
    assert resolve_recording_depth_names(None, ["single_arm"], cameras) == cameras


def test_entry_for_an_unconfigured_camera_is_rejected() -> None:
    with pytest.raises(ValueError, match="observation.images.ego"):
        resolve_recording_entries(
            ["observation.images.ego"], ["single_arm"], ["overhead", "wrist"]
        )


def test_dataset_features_are_named_after_the_configured_cameras() -> None:
    features, _, _ = build_features_from_sample(
        {"robots": {}},
        {"overhead": np.zeros((4, 5, 3), dtype=np.uint8)},
        ["observation.images.overhead"],
        ["single_arm"],
        cameras=["overhead", "wrist"],
    )

    assert list(features) == ["observation.images.overhead"]
    assert features["observation.images.overhead"]["shape"] == [4, 5, 3]


def _service(tmp_path, cameras: list[CameraConfig]) -> CameraService:
    return CameraService(
        AppSettings(storage=StorageConfig(root=tmp_path), cameras=cameras)
    )


def test_activating_an_unconfigured_name_builds_a_runtime_for_it(tmp_path) -> None:
    # Slot names are operator-chosen, so an active location need not appear in
    # AppSettings.cameras.
    service = _service(
        tmp_path, [CameraConfig(name="ego", width=1280, height=720, fps=15)]
    )

    service.set_active_locations(["ego", "overhead"])

    assert service.configured_serials()["overhead"] is None
    runtime = service._runtimes["overhead"]
    assert (runtime.config.width, runtime.config.height, runtime.config.fps) == (
        1280,
        720,
        15,
    )


def test_rename_moves_the_assigned_device_to_the_new_slot(tmp_path) -> None:
    service = _service(
        tmp_path,
        [
            CameraConfig(name="ego", device_serial="SERIAL_A"),
            CameraConfig(name="wrist", device_serial="SERIAL_B"),
        ],
    )

    service.rename_locations({"ego": "overhead"})

    serials = service.configured_serials()
    assert serials["overhead"] == "SERIAL_A"
    assert "ego" not in serials
    # The renamed slot must not leave a second claim on its serial behind: the
    # uniqueness pass in set_device_serials() would otherwise strip it.
    assert sorted(serials) == ["overhead", "wrist"]


def test_rename_onto_an_existing_slot_is_ignored(tmp_path) -> None:
    service = _service(
        tmp_path,
        [
            CameraConfig(name="ego", device_serial="SERIAL_A"),
            CameraConfig(name="wrist", device_serial="SERIAL_B"),
        ],
    )

    service.rename_locations({"ego": "wrist"})

    assert service.configured_serials() == {"ego": "SERIAL_A", "wrist": "SERIAL_B"}


def test_rename_follows_the_active_location_list(tmp_path) -> None:
    service = _service(tmp_path, [CameraConfig(name="ego")])
    service.set_active_locations(["ego"])

    service.rename_locations({"ego": "overhead"})

    assert service._active_locations == ["overhead"]


def _manager_for_config_updates(tmp_path, *, recording_active: bool = False):
    manager = RuntimeManager.__new__(RuntimeManager)
    manager.settings = AppSettings(storage=StorageConfig(root=tmp_path))
    manager.settings.ensure_storage()
    manager._robot_config = _config(
        arm_mode="single",
        leader_robot_serials=["LEADER_A"],
        follower_robot_serials=["FOLLOWER_A"],
    )
    manager.cameras = _service(
        tmp_path,
        [
            CameraConfig(name="ego", device_serial="SERIAL_A"),
            CameraConfig(name="wrist", device_serial="SERIAL_B"),
        ],
    )
    manager.cameras.set_active_locations(manager._robot_config.active_camera_names())
    manager.recording = SimpleNamespace(
        status=lambda: {"active": recording_active}, shutdown=lambda: None
    )
    manager.teleop = SimpleNamespace(
        shutdown=lambda: None,
        gripper_command_parameter_snapshot=lambda: None,
    )
    manager.service_summary = lambda: {}
    return manager


def test_update_robot_config_carries_a_renamed_slots_device(tmp_path) -> None:
    manager = _manager_for_config_updates(tmp_path)

    manager.update_robot_config(
        manager._robot_config.model_copy(
            update={"camera_names": {"single": ["overhead", "wrist", "aux_1"]}}
        )
    )

    assert manager._robot_config.active_camera_names() == ["overhead", "wrist"]
    serials = manager.cameras.configured_serials()
    assert serials["overhead"] == "SERIAL_A"
    assert "ego" not in serials
    # Persisted under the new name, so the assignment survives a restart.
    saved = manager.settings.storage.camera_config_path.read_text(encoding="utf-8")
    assert '"overhead": "SERIAL_A"' in saved


def test_update_robot_config_activates_a_raised_camera_count(tmp_path) -> None:
    manager = _manager_for_config_updates(tmp_path)

    manager.update_robot_config(
        manager._robot_config.model_copy(update={"camera_counts": {"single": 3}})
    )

    assert manager._robot_config.active_camera_names() == ["ego", "wrist", "aux_1"]
    assert manager.cameras._active_locations == ["ego", "wrist", "aux_1"]


def test_camera_layout_cannot_change_while_recording(tmp_path) -> None:
    manager = _manager_for_config_updates(tmp_path, recording_active=True)

    with pytest.raises(ValueError, match="while recording"):
        manager.update_robot_config(
            manager._robot_config.model_copy(update={"camera_counts": {"single": 3}})
        )


WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "flexivtrainer" / "web"


def test_camera_setup_panel_is_wired_into_the_home_view() -> None:
    index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "<h3>Camera Setup</h3>" in index
    assert 'id="home-camera-setup"' in index
    assert 'id="teleop-camera-feed-row"' in index
    # Camera count is configured independently of the arm count, so no feed may
    # live inside a wrist column: those hold only their arm's telemetry.
    assert 'id="left-wrist-fps"' not in index
    assert 'id="right-wrist-fps"' not in index
    # The hero title follows a renamed slot 0.
    assert 'id="ego-title"' in index

    assert "function renderCameraSetup()" in source
    assert "function getActiveCameraNames()" in source
    assert "function renderCameraFeedRow(" in source
    assert "const MAX_CAMERAS_BY_ARM_MODE = { single: 3, dual: 5 };" in source


def test_ui_no_longer_derives_the_camera_set_from_the_arm_sides() -> None:
    # Every camera slot is renameable, so nothing may key off the seed names.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'renderCameraFps("ego-fps", "ego"' not in source
    assert 'id: "observation.images.ego"' not in source
    assert '["ego", ...panels.map((panel) => panel.camera)]' not in source


def test_feed_layout_tiles_two_per_row_under_an_optional_hero() -> None:
    # An odd camera count would leave the last row half empty, so the first
    # camera takes a full-width hero and the even remainder tiles beneath it.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    assert "function cameraFeedLayout(" in source
    assert "cameraNames.length % 2 === 1 ? cameraNames[0] : null" in source
    assert 'byId("teleop-hero-feed")?.classList.toggle("hidden", !heroCamera)' in source
    assert 'id="teleop-hero-feed"' in index
    # The row must inherit .feed-row's two columns, not override them.
    css = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")
    assert "#teleop-camera-feed-row {" not in css


def _recording_route_client(cameras: list[str]):
    """A client whose runtime reports `cameras`, with recording stubbed out.

    The dependency override matters: get_runtime_manager() builds a real
    RuntimeManager against the project's own .local storage.
    """
    from fastapi.testclient import TestClient

    from flexivtrainer.api.app import create_app
    from flexivtrainer.runtime.manager import get_runtime_manager

    started: dict = {}

    def _start(**kwargs):
        started.update(kwargs)
        return {"episode_name": "ep_0", "job_name": "job", "fps": 30}

    app = create_app()
    app.dependency_overrides[get_runtime_manager] = lambda: SimpleNamespace(
        get_active_sides=lambda: ["single_arm"],
        get_active_cameras=lambda: list(cameras),
        recording=SimpleNamespace(start=_start),
    )
    return TestClient(app), started


def test_recording_start_accepts_a_renamed_cameras_entry() -> None:
    client, started = _recording_route_client(["ego", "wrist_1", "wrist_2"])

    response = client.post(
        "/teleop/recording/start",
        json={"recording_entries": ["observation.images.wrist_1"]},
    )

    assert response.status_code == 200, response.text
    assert started["recording_entries"] == ["observation.images.wrist_1"]


def test_recording_start_still_rejects_an_unconfigured_camera() -> None:
    client, _ = _recording_route_client(["ego", "wrist_1"])

    response = client.post(
        "/teleop/recording/start",
        json={"recording_entries": ["observation.images.nope"]},
    )

    assert response.status_code == 400
    assert "observation.images.nope" in response.json()["detail"]


def test_recording_start_without_entries_uses_the_live_layout() -> None:
    # A frozen default would carry the dual-arm layout into a single-arm rig.
    client, started = _recording_route_client(["ego", "wrist_1"])

    response = client.post("/teleop/recording/start", json={})

    assert response.status_code == 200, response.text
    images = [
        entry
        for entry in started["recording_entries"]
        if entry.startswith("observation.images.") and not entry.endswith("_depth")
    ]
    assert images == ["observation.images.ego", "observation.images.wrist_1"]
    assert not any(".left_arm." in entry for entry in started["recording_entries"])
