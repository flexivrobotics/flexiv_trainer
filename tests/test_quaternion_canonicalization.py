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

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flexivtrainer.data.lerobot_io import extract_recording_frame_values
from flexivtrainer.data.quaternion import QuaternionSignCanonicalizer

# [q_w, q_x, q_y, q_z], tool-down with |q_w| ~ 0.01
_TOOL_DOWN = np.array([0.0137, -0.4582, 0.8885, 0.0201])
_TOOL_DOWN /= np.linalg.norm(_TOOL_DOWN)


def _rotmat(quat_wxyz) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _perturbed(quat: np.ndarray, deg: float, axis=(1.0, 0.0, 0.0)) -> np.ndarray:
    base = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    delta = Rotation.from_rotvec(np.deg2rad(deg) * np.asarray(axis, dtype=float))
    out = (delta * base).as_quat()
    return np.array([out[3], out[0], out[1], out[2]])


def test_negated_quaternion_is_the_same_rotation() -> None:
    assert np.allclose(_rotmat(_TOOL_DOWN), _rotmat(-_TOOL_DOWN))


def test_state_sign_is_made_continuous_and_rotation_preserved() -> None:
    canon = QuaternionSignCanonicalizer()
    canon.begin_episode()
    trajectory = [
        _TOOL_DOWN,
        _perturbed(_TOOL_DOWN, 1.0),
        -_perturbed(_TOOL_DOWN, 2.0),
        -_perturbed(_TOOL_DOWN, 3.0),
        _perturbed(_TOOL_DOWN, 4.0),
    ]
    out = [np.asarray(canon.canonical_state("single_arm", q)) for q in trajectory]

    for raw, fixed in zip(trajectory, out):
        assert np.allclose(_rotmat(raw), _rotmat(fixed))
        assert np.allclose(np.abs(raw), np.abs(fixed))
    for a, b in zip(out, out[1:]):
        assert float(np.dot(a, b)) > 0.0
    assert np.allclose(out[0], _TOOL_DOWN)
    assert float(np.dot(out[2], _TOOL_DOWN)) > 0.0


def test_action_follows_the_state_hemisphere_at_the_same_frame() -> None:
    canon = QuaternionSignCanonicalizer()
    canon.begin_episode()
    state = canon.canonical_state("single_arm", _TOOL_DOWN)
    action_raw = -_perturbed(_TOOL_DOWN, 1.0)
    action = np.asarray(canon.canonical_action("single_arm", action_raw))
    assert float(np.dot(action, state)) > 0.0
    assert np.allclose(_rotmat(action), _rotmat(action_raw))


def test_reference_seeds_first_frame_of_every_episode() -> None:
    canon = QuaternionSignCanonicalizer(reference={"single_arm": _TOOL_DOWN})
    for _ in range(3):
        canon.begin_episode()
        first = np.asarray(canon.canonical_state("single_arm", -_TOOL_DOWN))
        assert float(np.dot(first, _TOOL_DOWN)) > 0.0


def test_without_reference_later_episodes_continue_previous_sign() -> None:
    canon = QuaternionSignCanonicalizer()
    canon.begin_episode()
    last = np.asarray(canon.canonical_state("left_arm", _TOOL_DOWN))
    canon.begin_episode()
    first = np.asarray(canon.canonical_state("left_arm", -_perturbed(_TOOL_DOWN, 5.0)))
    assert float(np.dot(first, last)) > 0.0


def test_reference_beats_previous_episode_and_arms_are_independent() -> None:
    canon = QuaternionSignCanonicalizer(reference={"left_arm": _TOOL_DOWN})
    canon.begin_episode()
    canon.canonical_state("left_arm", -_TOOL_DOWN)
    canon.canonical_state("right_arm", -_TOOL_DOWN)
    canon.begin_episode()
    left = np.asarray(canon.canonical_state("left_arm", -_TOOL_DOWN))
    right = np.asarray(canon.canonical_state("right_arm", -_TOOL_DOWN))
    assert float(np.dot(left, _TOOL_DOWN)) > 0.0
    assert float(np.dot(right, -_TOOL_DOWN)) > 0.0


def test_degenerate_values_pass_through_untouched() -> None:
    canon = QuaternionSignCanonicalizer()
    canon.begin_episode()
    canon.canonical_state("single_arm", _TOOL_DOWN)
    assert canon.canonical_state("single_arm", [0.0, 0.0, 0.0, 0.0]) == [0.0] * 4
    nan_quat = canon.canonical_state("single_arm", [float("nan"), 0.0, 0.0, 1.0])
    assert np.isnan(nan_quat[0])
    assert canon.canonical_state("single_arm", [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_invalid_reference_is_rejected() -> None:
    with pytest.raises(ValueError):
        QuaternionSignCanonicalizer(reference={"single_arm": [0.0, 0.0, 0.0, 0.0]})
    with pytest.raises(ValueError):
        QuaternionSignCanonicalizer(reference={"single_arm": [1.0, 0.0, 0.0]})


def test_apply_touches_only_complete_quaternion_groups() -> None:
    canon = QuaternionSignCanonicalizer()
    canon.begin_episode()
    names = [
        "left_arm.tcp_pose.x",
        "left_arm.tcp_pose.y",
        "left_arm.tcp_pose.z",
        "left_arm.tcp_pose.q_w",
        "left_arm.tcp_pose.q_x",
        "left_arm.tcp_pose.q_y",
        "left_arm.tcp_pose.q_z",
        "left_arm.tcp_twist.vx",
        "right_arm.tcp_pose.q_w",
    ]
    canon.canonical_state("left_arm", _TOOL_DOWN)
    values = [1.0, 2.0, 3.0, *(-_TOOL_DOWN), 0.5, -0.9]
    out = canon.apply(values, names, "state")
    assert out[:3] == [1.0, 2.0, 3.0]
    assert np.allclose(out[3:7], _TOOL_DOWN)
    assert out[7] == 0.5
    assert out[8] == -0.9
    with pytest.raises(ValueError):
        canon.apply(values, names, "pose")


def _snapshot(state_quat, action_quat) -> dict:
    return {
        "robots": {
            "FOLLOWER_A": {
                "states": {
                    "tcp_pose": [0.1, 0.2, 0.3, *state_quat],
                    "tcp_vel": [0.0] * 6,
                    "ext_wrench_in_world": [0.0] * 6,
                },
                "actions": {
                    "tcp_pose_d": [0.1, 0.2, 0.31, *action_quat],
                    "tcp_vel_d": [0.0] * 6,
                    "ext_wrench_d": [0.0] * 6,
                },
            }
        }
    }


def test_extract_recording_frame_values_canonicalizes_state_then_action() -> None:
    canon = QuaternionSignCanonicalizer()
    canon.begin_episode()
    entries = ["observation.state.left_arm.tcp_pose", "action.left_arm.tcp_pose"]
    sides = ["left_arm"]

    first = extract_recording_frame_values(
        _snapshot(_TOOL_DOWN, _TOOL_DOWN), entries, sides, canonicalizer=canon
    )
    flipped = extract_recording_frame_values(
        _snapshot(-_perturbed(_TOOL_DOWN, 1.0), -_perturbed(_TOOL_DOWN, 1.5)),
        entries,
        sides,
        canonicalizer=canon,
    )
    s0, s1 = (
        np.asarray(first["observation.state"]),
        np.asarray(flipped["observation.state"]),
    )
    a0, a1 = np.asarray(first["action"]), np.asarray(flipped["action"])
    assert np.allclose(s1[:3], [0.1, 0.2, 0.3])
    assert np.allclose(a1[:3], [0.1, 0.2, 0.31])
    assert float(np.dot(s0[3:7], s1[3:7])) > 0.0
    assert float(np.dot(a0[3:7], a1[3:7])) > 0.0
    assert float(np.dot(s1[3:7], a1[3:7])) > 0.0
    assert np.allclose(_rotmat(s1[3:7]), _rotmat(-_perturbed(_TOOL_DOWN, 1.0)))


def test_extract_recording_frame_values_unchanged_without_canonicalizer() -> None:
    entries = ["observation.state.left_arm.tcp_pose", "action.left_arm.tcp_pose"]
    raw = extract_recording_frame_values(
        _snapshot(-_TOOL_DOWN, -_TOOL_DOWN), entries, ["left_arm"]
    )
    assert np.allclose(np.asarray(raw["observation.state"])[3:7], -_TOOL_DOWN)
