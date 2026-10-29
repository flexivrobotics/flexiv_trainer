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

from flexivtrainer.rollout.observations import (
    build_observation,
    resolve_state_columns,
    select_state,
)
from flexivtrainer.rollout.pose_history import PoseHistory, history_columns

ROWS = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
MOTION = [f"right_arm.tcp_motion_50ms.{a}" for a in ("x", "y", "z", *ROWS)]
WIDTH_CHANGE = "right_arm.gripper.width_change_50ms"
WRT_START = [f"right_arm.tcp_rotation_wrt_start.{a}" for a in ROWS]
POSE = [
    "right_arm.tcp_pose.x",
    "right_arm.tcp_pose.y",
    "right_arm.tcp_pose.z",
    *(f"right_arm.tcp_rotation_6d.{a}" for a in ROWS),
    "right_arm.gripper.width",
]


def _snapshot(position, rotation: Rotation, width: float = 0.05) -> dict:
    x, y, z, w = rotation.as_quat()
    pose = [*map(float, position), float(w), float(x), float(y), float(z)]
    return {
        "robots": {
            "robot_0": {
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
                "gripper": {"width": width, "force": 1.0},
            }
        },
        "errors": {},
    }


def _rows(matrix: np.ndarray) -> list[float]:
    return [*matrix[0], *matrix[1]]


def _history(names: list[str]) -> PoseHistory:
    spec = history_columns(names)
    assert spec is not None
    return PoseHistory(spec, ["right_arm"])


def test_names_are_found_in_state_order_with_their_delay() -> None:
    spec = history_columns([*POSE, *MOTION, WIDTH_CHANGE, *WRT_START])

    assert spec is not None
    assert spec.names == (*MOTION, WIDTH_CHANGE, *WRT_START)
    assert spec.delay_s == pytest.approx(0.05)
    assert history_columns(POSE) is None
    assert history_columns(WRT_START).delay_s is None
    with pytest.raises(ValueError, match="mix motion delays"):
        history_columns([*MOTION, "right_arm.gripper.width_change_100ms"])


def test_constant_motion_at_planner_rate_matches_the_past_pose() -> None:
    # 0.1 m/s along x and 1 rad/s about z, sampled at 120 Hz (not the 40 Hz
    # the dataset was exported at): the past pose is 50 ms back either way.
    history = _history([*MOTION, WIDTH_CHANGE])
    velocity, rate = np.array([0.1, 0.0, 0.0]), 1.0
    for k in range(60):
        t = k / 120.0
        history.update(
            t,
            _snapshot(
                velocity * t, Rotation.from_rotvec([0, 0, rate * t]), 0.05 - 0.02 * t
            ),
        )
    t = 59 / 120.0
    current = Rotation.from_rotvec([0, 0, rate * t]).as_matrix()
    past = Rotation.from_rotvec([0, 0, rate * (t - 0.05)]).as_matrix()
    expected_position = current.T @ (velocity * (t - 0.05) - velocity * t)
    # The gripper closes at 0.02 m/s, so the width fell 1 mm in 50 ms.
    expected = [*expected_position, *_rows(current.T @ past), -0.02 * 0.05]

    assert history.values().tolist() == pytest.approx(expected, abs=1e-6)


def test_a_fresh_history_reads_no_motion() -> None:
    history = _history([*MOTION, WIDTH_CHANGE])
    history.update(0.0, _snapshot([0.4, 0.0, 0.3], Rotation.identity()))

    assert history.values().tolist() == pytest.approx(
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0], abs=1e-9
    )


def test_rotation_since_start_and_reset_keeps_the_start() -> None:
    history = _history(WRT_START)
    start = Rotation.from_rotvec([0.3, -0.2, 0.1])
    turned = Rotation.from_rotvec([0, 0, np.pi / 2]) * start
    history.update(0.0, _snapshot([0, 0, 0], start))
    history.update(1.0, _snapshot([0, 0, 0], turned))
    expected = _rows(start.as_matrix().T @ turned.as_matrix())

    assert history.values().tolist() == pytest.approx(expected, abs=1e-6)
    history.reset()
    history.update(2.0, _snapshot([0, 0, 0], turned))
    assert history.values().tolist() == pytest.approx(expected, abs=1e-6)


def test_width_change_needs_a_gripper_width() -> None:
    history = _history([*MOTION, WIDTH_CHANGE])
    snapshot = _snapshot([0, 0, 0], Rotation.identity())
    del snapshot["robots"]["robot_0"]["gripper"]
    history.update(0.0, snapshot)

    with pytest.raises(RuntimeError, match="reports no width"):
        history.values()


def test_policy_state_takes_the_computed_columns_after_the_robots() -> None:
    names = [*POSE, *MOTION, WIDTH_CHANGE, *WRT_START]
    history = _history(names)
    snapshot = _snapshot([0.4, 0.0, 0.3], Rotation.identity())
    history.update(0.0, snapshot)
    observation = build_observation(snapshot, {}, ["right_arm"])

    index = resolve_state_columns(
        snapshot, ["right_arm"], None, names, history_columns(names).names
    )
    selected = select_state(observation, index, history.values())

    assert selected["observation.state"].shape == (26,)
    assert observation["observation.state"].shape == (23,)
    assert selected["observation.state"][:3].tolist() == pytest.approx([0.4, 0, 0.3])
    # Identity motion, no width change, identity rotation since start.
    assert selected["observation.state"][10:].tolist() == pytest.approx(
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], abs=1e-6
    )
