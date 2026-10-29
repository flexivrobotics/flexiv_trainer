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

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flexivtrainer.data.lerobot_io import (
    POSE_FORMAT_QUATERNION,
    POSE_FORMAT_ROTATION_6D,
)
from flexivtrainer.rollout.observations import (
    _predict_action_chunk,
    build_observation,
    resolve_state_columns,
    resolve_state_layout,
    resolve_state_pose_format,
    select_state,
)

# Mirror the resolution order in observations.py so the patch target tracks the
# installed LeRobot; 0.6.0 has only the `common` module.
try:
    import lerobot.common.control_utils as control_utils
except ImportError:  # pragma: no cover - depends on the installed LeRobot
    import lerobot.utils.control_utils as control_utils


def test_fresh_inference_returns_and_postprocesses_the_complete_chunk(
    monkeypatch,
) -> None:
    queue = deque()
    policy = SimpleNamespace(_action_queue=queue)
    postprocessed: list[torch.Tensor] = []

    def postprocessor(actions):
        postprocessed.append(actions)
        return actions + 10

    def predict_action(*args, **kwargs):
        queue.extend([
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 4.0]]),
        ])
        return torch.tensor([[10.0, 11.0]])

    monkeypatch.setattr(control_utils, "predict_action", predict_action)

    chunk, fresh = _predict_action_chunk(
        {}, policy, "cpu", lambda value: value, postprocessor
    )

    assert fresh is True
    assert torch.equal(
        chunk,
        torch.tensor([
            [10.0, 11.0],
            [11.0, 12.0],
            [13.0, 14.0],
        ]),
    )
    assert len(postprocessed) == 1


def test_cached_action_skips_reprocessing_the_pending_tail(monkeypatch) -> None:
    queue = deque([
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
    ])
    policy = SimpleNamespace(_action_queue=queue)
    postprocessed: list[torch.Tensor] = []

    def postprocessor(actions):
        postprocessed.append(actions)
        return actions + 10

    def predict_action(*args, **kwargs):
        return postprocessor(queue.popleft())

    monkeypatch.setattr(control_utils, "predict_action", predict_action)

    chunk, fresh = _predict_action_chunk(
        {}, policy, "cpu", lambda value: value, postprocessor
    )

    assert fresh is False
    assert torch.equal(chunk, torch.tensor([[11.0, 12.0]]))
    assert len(queue) == 1
    assert len(postprocessed) == 1


def _snapshot(sides: list[str], gripper: bool = True) -> dict:
    pose = [0.1, 0.2, 0.3, 0.0, 1.0, 0.0, 0.0]
    robots = {}
    for index in range(len(sides)):
        payload = {
            "connected": True,
            "states": {
                "tcp_pose": pose,
                "tcp_vel": [0.0] * 6,
                "ext_wrench_in_world": [0.0] * 6,
            },
            "actions": {
                "tcp_pose_d": pose,
                "tcp_vel_d": [0.0] * 6,
                "ext_wrench_d": [0.0] * 6,
            },
        }
        if gripper:
            payload["gripper"] = {"width": 0.05, "force": 1.0}
        robots[f"robot_{index}"] = payload
    return {"robots": robots, "errors": {}}


def test_state_pose_format_falls_back_to_the_trained_quaternion_width() -> None:
    # Single arm with a gripper: 9 + 6 + 6 + 2 rotation-6D, 7 + 6 + 6 + 2 legacy.
    snapshot = _snapshot(["single_arm"])

    assert resolve_state_pose_format(snapshot, ["single_arm"], None, 23) == (
        POSE_FORMAT_ROTATION_6D
    )
    assert resolve_state_pose_format(snapshot, ["single_arm"], None, 21) == (
        POSE_FORMAT_QUATERNION
    )


def test_state_pose_format_reports_the_legacy_fallback_to_the_operator() -> None:
    snapshot = _snapshot(["single_arm"])
    logs: list[tuple] = []

    def append_log(*entry):
        logs.append(entry)

    resolve_state_pose_format(snapshot, ["single_arm"], None, 21, append_log)
    resolve_state_pose_format(snapshot, ["single_arm"], None, 23, append_log)

    # Only the fallback is announced; the current layout is the silent default.
    assert len(logs) == 1
    assert logs[0][:3] == ("INFO", "ROLLOUT", "Legacy quaternion state layout")


def test_state_pose_format_defaults_to_rotation_6d_without_checkpoint_width() -> None:
    snapshot = _snapshot(["single_arm"])

    assert resolve_state_pose_format(snapshot, ["single_arm"], None, None) == (
        POSE_FORMAT_ROTATION_6D
    )


def test_state_pose_format_names_both_widths_when_neither_matches() -> None:
    snapshot = _snapshot(["single_arm"])

    with pytest.raises(RuntimeError) as excinfo:
        resolve_state_pose_format(snapshot, ["single_arm"], None, 99)

    message = str(excinfo.value)
    assert "99" in message
    assert "23" in message and "21" in message


def test_build_observation_emits_the_requested_pose_format() -> None:
    snapshot = _snapshot(["left_arm", "right_arm"], gripper=False)
    images = {name: np.zeros((4, 5, 3), dtype=np.uint8) for name in ("ego",)}

    sides = ["left_arm", "right_arm"]
    rotation_6d = build_observation(snapshot, images, sides, ["ego"])
    legacy = build_observation(
        snapshot, images, sides, ["ego"], POSE_FORMAT_QUATERNION
    )

    assert rotation_6d["observation.state"].shape == (42,)
    assert legacy["observation.state"].shape == (38,)
    # The driver quaternion reaches the policy unconverted and unnormalized.
    assert legacy["observation.state"][:7].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.0, 1.0, 0.0, 0.0]
    )
    assert "observation.images.ego" in legacy


# What a UMI export trains on: the pose and gripper width, plus the grip force
# when it was exported with --tactile-grip-force.
_UMI_STATE = [
    *(f"right_arm.tcp_pose.{axis}" for axis in "xyz"),
    *(
        f"right_arm.tcp_rotation_6d.{axis}"
        for axis in ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
    ),
    "right_arm.gripper.width",
]


@pytest.mark.parametrize(
    "names", [_UMI_STATE, [*_UMI_STATE, "right_arm.gripper.force"]]
)
def test_state_columns_select_the_trained_subset_in_its_order(names) -> None:
    snapshot = _snapshot(["right_arm"])
    full = build_observation(snapshot, {}, ["right_arm"])["observation.state"]
    assert full.shape == (23,)

    index = resolve_state_columns(snapshot, ["right_arm"], None, names)

    assert len(index) == len(names)
    # Pose first, then the gripper's width (0.05) and force (1.0) after the
    # twist and wrench the subset skips.
    assert full[index][:3].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert full[index][9:].tolist() == pytest.approx([0.05, 1.0][: len(names) - 9])
    reordered = resolve_state_columns(
        snapshot, ["right_arm"], None, list(reversed(names))
    )
    assert reordered.tolist() == list(reversed(index.tolist()))


def test_state_columns_name_a_gripper_without_force() -> None:
    snapshot = _snapshot(["right_arm"])
    snapshot["robots"]["robot_0"]["gripper"] = {"width": 0.05}

    resolve_state_columns(snapshot, ["right_arm"], None, _UMI_STATE)
    with pytest.raises(RuntimeError, match="gripper reports no force"):
        resolve_state_columns(
            snapshot, ["right_arm"], None, [*_UMI_STATE, "right_arm.gripper.force"]
        )


def test_state_layout_without_names_keeps_the_width_match() -> None:
    snapshot = _snapshot(["single_arm"])

    assert resolve_state_layout(snapshot, ["single_arm"], None, 21, None) == (
        POSE_FORMAT_QUATERNION,
        None,
    )
    pose_format, index = resolve_state_layout(
        _snapshot(["right_arm"]), ["right_arm"], None, 10, _UMI_STATE
    )
    assert pose_format == POSE_FORMAT_ROTATION_6D
    assert len(index) == 10


def test_select_state_leaves_the_full_observation_for_the_recording() -> None:
    observation = build_observation(_snapshot(["right_arm"]), {}, ["right_arm"])
    index = resolve_state_columns(
        _snapshot(["right_arm"]), ["right_arm"], None, _UMI_STATE
    )

    selected = select_state(observation, index)

    assert selected["observation.state"].shape == (10,)
    assert observation["observation.state"].shape == (23,)
    assert select_state(observation, None) is observation
