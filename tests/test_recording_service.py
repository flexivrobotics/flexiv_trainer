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

import time
from types import SimpleNamespace

import numpy as np
import pytest

from flexivtrainer.config import AppSettings, StorageConfig
from flexivtrainer.data.recording_service import RecordingService


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
    state = np.asarray(frame["observation.state"])
    action = np.asarray(frame["action"])
    assert state[state_names.index("left_arm.gripper.width")] == pytest.approx(0.03)
    assert state[state_names.index("left_arm.gripper.force")] == pytest.approx(-2.0)
    assert action[action_names.index("left_arm.gripper.width")] == pytest.approx(0.03)
    assert action[action_names.index("left_arm.gripper.force")] == pytest.approx(-2.0)
