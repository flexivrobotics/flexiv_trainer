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
    resolve_state_pose_format,
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
