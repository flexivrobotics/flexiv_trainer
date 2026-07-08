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
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flexivtrainer.config import AppSettings, StorageConfig
from flexivtrainer.data.recording_service import (
    DEFAULT_JOB_NAME,
    RecordingService,
    sanitize_job_name,
)


def test_grab_images_converts_bgr_capture_to_rgb() -> None:
    # Cameras capture a red pixel as BGR [0, 0, 255]; recorded frames must be
    # RGB [255, 0, 0] so LeRobot playback shows red (not blue/purple).
    bgr_red = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr_red[0, 0] = [0, 0, 255]

    service = RecordingService.__new__(RecordingService)
    service._cameras = SimpleNamespace(
        capture_frame=lambda name, **kwargs: {"image": bgr_red}
    )

    images = service._grab_images(["ego"], require_all=True, attempts=1)

    assert images["ego"][0, 0].tolist() == [255, 0, 0]
    assert images["ego"].flags["C_CONTIGUOUS"]


def _arm_snapshot_payload(base: float, *, gripper: dict | None = None) -> dict:
    payload = {
        "connected": True,
        "states": {
            "tcp_pose": [base + i for i in range(7)],
            "tcp_vel": [base + 10 + i for i in range(6)],
            "ext_wrench_in_world": [base + 20 + i for i in range(6)],
        },
        "actions": {
            "tcp_pose_d": [base + 30 + i for i in range(7)],
            "tcp_vel_d": [base + 40 + i for i in range(6)],
            "ext_wrench_d": [base + 50 + i for i in range(6)],
        },
    }
    if gripper is not None:
        payload["gripper"] = dict(gripper)
    return payload


class _FakeTeleop:
    """Returns a fixed dual-arm snapshot; left arm carries gripper telemetry."""

    def robot_data_snapshot(self, *, include_states=True, include_actions=True):
        return {
            "robots": {
                "FOLLOWER_A": _arm_snapshot_payload(
                    0.0, gripper={"width": 0.03, "force": -2.0}
                ),
                "FOLLOWER_B": _arm_snapshot_payload(100.0),
            }
        }


def _drive_capture(service: RecordingService, frames: int) -> None:
    """Spin until the capture thread has written at least `frames` frames."""
    deadline = time.monotonic() + 5.0
    while service.status()["frames_captured"] < frames:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"capture stalled at {service.status()['frames_captured']} frames"
            )
        time.sleep(0.02)


def test_records_gripper_width_force_into_saved_episode(tmp_path) -> None:
    # End-to-end: a follower configured as a gripper must land its measured
    # width/force in the saved LeRobot dataset, in both observation.state and
    # action, alongside the arm's TCP metrics. Exercises the recording loop and
    # the real LeRobotDataset round-trip (skipped if lerobot isn't installed).
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    settings.ensure_storage()
    service = RecordingService(
        settings,
        teleop=_FakeTeleop(),
        cameras=SimpleNamespace(),
        get_active_sides=lambda: ["left_arm", "right_arm"],
    )

    # Record only the numeric (state/action) entries -- no cameras -- so the
    # test needs no video encoder. The gripper entries are part of the defaults.
    entries = [
        "observation.state.left_arm.tcp_pose",
        "observation.state.left_arm.gripper",
        "observation.state.right_arm.tcp_pose",
        "action.left_arm.tcp_pose",
        "action.left_arm.gripper",
        "action.right_arm.tcp_pose",
    ]

    service.start(task="gripper recording test", fps=30, recording_entries=entries)
    try:
        _drive_capture(service, frames=3)
    finally:
        service.stop()
    result = service.save()

    dataset = LeRobotDataset(
        f"local/{result['episode_name']}", root=result["path"]
    )
    features = dataset.features

    # Both the observation and action vectors carry the left arm's gripper
    # width/force (right arm has no gripper, so no gripper axes for it).
    state_names = features["observation.state"]["names"]
    action_names = features["action"]["names"]
    assert "left_arm.gripper.width" in state_names
    assert "left_arm.gripper.force" in state_names
    assert "left_arm.gripper.width" in action_names
    assert "left_arm.gripper.force" in action_names
    assert not any(name.startswith("right_arm.gripper") for name in state_names)

    # The stored values match the snapshot's gripper states, identical in state
    # and action (recording reuses gripper.states() for both).
    frame = dataset[0]
    # The provided task string is stamped into every recorded frame.
    assert frame["task"] == "gripper recording test"
    state = np.asarray(frame["observation.state"])
    action = np.asarray(frame["action"])
    assert state[state_names.index("left_arm.gripper.width")] == pytest.approx(0.03)
    assert state[state_names.index("left_arm.gripper.force")] == pytest.approx(-2.0)
    assert action[action_names.index("left_arm.gripper.width")] == pytest.approx(0.03)
    assert action[action_names.index("left_arm.gripper.force")] == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("pick_place", "pick_place"),
        ("  spaced name  ", "spaced_name"),
        ("../escape", "escape"),
        ("a/b\\c", "a_b_c"),
        ("", DEFAULT_JOB_NAME),
        ("   ", DEFAULT_JOB_NAME),
        (None, DEFAULT_JOB_NAME),
        ("...", DEFAULT_JOB_NAME),
    ],
)
def test_sanitize_job_name(raw, expected) -> None:
    assert sanitize_job_name(raw) == expected


class _StubDataset:
    """Stands in for a LeRobotDataset so save() can run without lerobot."""

    def save_episode(self) -> None:
        return None

    def finalize(self) -> None:
        return None


def _awaiting_save_service(tmp_path, *, job_name: str) -> RecordingService:
    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    settings.ensure_storage()
    service = RecordingService(
        settings,
        teleop=_FakeTeleop(),
        cameras=SimpleNamespace(),
        get_active_sides=lambda: ["left_arm", "right_arm"],
    )
    # Stage a fake, already-captured episode awaiting save.
    staging_path = settings.storage.staging_root / "20260101_120000"
    staging_path.mkdir(parents=True)
    (staging_path / "marker").write_text("episode", encoding="utf-8")
    service._awaiting_save = True
    service._episode_name = "20260101_120000"
    service._job_name = job_name
    service._staging_path = staging_path
    service._dataset = _StubDataset()
    service._frames_captured = 3
    return service


def test_save_files_episode_under_job_subfolder(tmp_path) -> None:
    service = _awaiting_save_service(tmp_path, job_name="pick_place")

    result = service.save()

    episodes_root = service._settings.storage.episodes_root
    expected_path = episodes_root / "pick_place" / "20260101_120000"
    assert result["job_name"] == "pick_place"
    assert result["path"] == str(expected_path)
    assert (expected_path / "marker").exists()
    # The episode lives under the job folder, not directly under episodes/.
    assert not (episodes_root / "20260101_120000").exists()


def test_save_writes_task_description_into_info_json(tmp_path) -> None:
    service = _awaiting_save_service(tmp_path, job_name="pick_place")
    service._task = "pick up the red cube"
    meta = service._staging_path / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        '{"codebase_version": "v3.0", "fps": 10, "splits": {}}', encoding="utf-8"
    )

    result = service.save()

    info = json.loads(
        (Path(result["path"]) / "meta" / "info.json").read_text(encoding="utf-8")
    )
    assert info["description"] == "pick up the red cube"
    # Existing keys survive and description lands right after fps.
    assert info["fps"] == 10
    keys = list(info)
    assert keys.index("description") == keys.index("fps") + 1


def test_save_skips_description_without_info_json(tmp_path) -> None:
    service = _awaiting_save_service(tmp_path, job_name="pick_place")
    service._task = "pick up the red cube"

    result = service.save()

    assert not (Path(result["path"]) / "meta" / "info.json").exists()


def test_start_sanitizes_and_reports_job_name(tmp_path) -> None:
    # start() resolves the job name eagerly and status() surfaces the sanitized
    # value so the recording panel can confirm the active job. Record only
    # state/action entries (no cameras) so no video encoder is needed.
    pytest.importorskip("lerobot")

    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    settings.ensure_storage()
    service = RecordingService(
        settings,
        teleop=_FakeTeleop(),
        cameras=SimpleNamespace(),
        get_active_sides=lambda: ["left_arm", "right_arm"],
    )
    entries = [
        "observation.state.left_arm.tcp_pose",
        "action.left_arm.tcp_pose",
    ]

    status = service.start(
        task="job name test",
        fps=30,
        recording_entries=entries,
        job_name="  My Job!! ",
    )
    try:
        # The unsanitized name "  My Job!! " collapses to a single safe segment.
        assert status["job_name"] == "My_Job"
        # status() continues to report it while the recording is active.
        assert service.status()["job_name"] == "My_Job"
    finally:
        service.stop()
