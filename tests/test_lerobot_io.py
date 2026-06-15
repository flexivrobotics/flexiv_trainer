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

from flexivtrainer.data.lerobot_io import (
    DEFAULT_RECORDING_ENTRY_KEYS,
    arm_side_label,
    build_features_from_sample,
    extract_recording_frame_values,
    extract_recording_images,
    resolve_recording_image_names,
    resolve_recording_entries,
)


def _arm_payload(base: int) -> dict:
    return {
        "states": {
            "tcp_pose": list(range(base, base + 7)),
            "tcp_vel": list(range(base + 10, base + 16)),
            "ext_wrench_in_world": list(range(base + 20, base + 26)),
        },
        "actions": {
            "tcp_pose_d": list(range(base + 30, base + 37)),
            "tcp_vel_d": list(range(base + 40, base + 46)),
            "ext_wrench_d": list(range(base + 50, base + 56)),
        },
    }


def make_snapshot() -> dict:
    return {"robots": {"REMOTE_A": _arm_payload(0)}}


def make_dual_snapshot() -> dict:
    return {"robots": {"REMOTE_A": _arm_payload(0), "REMOTE_B": _arm_payload(100)}}


def make_images() -> dict[str, np.ndarray]:
    return {
        "ego": np.zeros((4, 5, 3), dtype=np.uint8),
        "left_wrist": np.zeros((4, 5, 3), dtype=np.uint8),
        "right_wrist": np.zeros((4, 5, 3), dtype=np.uint8),
    }


def test_resolve_recording_entries_defaults_to_all_options() -> None:
    assert resolve_recording_entries() == list(DEFAULT_RECORDING_ENTRY_KEYS)


def test_resolve_recording_entries_rejects_unknown_values() -> None:
    try:
        resolve_recording_entries(["unsupported.feature"])
    except ValueError as exc:
        assert "Unsupported recording entry" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for unsupported recording entry")


def test_arm_side_label_uses_left_right_then_generic() -> None:
    assert arm_side_label(0) == "left_arm"
    assert arm_side_label(1) == "right_arm"
    assert arm_side_label(2) == "arm_3"


def test_extract_recording_frame_values_groups_selected_metrics_per_arm() -> None:
    values = extract_recording_frame_values(
        make_snapshot(),
        [
            "observation.state.left_arm.tcp_pose",
            "observation.state.left_arm.tcp_twist",
            "observation.state.left_arm.tcp_wrench",
            "action.left_arm.tcp_wrench",
        ],
    )

    # No serial leaks into the keys; the selected metrics are concatenated into
    # the arm's grouped vector (pose 7 + twist 6 + wrench 6 = 19 for state).
    assert set(values) == {"observation.state.left_arm", "action.left_arm"}
    assert values["observation.state.left_arm"] == list(range(0, 7)) + list(
        range(10, 16)
    ) + list(range(20, 26))
    # Only the wrench action metric was selected.
    assert values["action.left_arm"] == list(range(50, 56))


def test_extract_recording_frame_values_drops_unselected_metrics() -> None:
    # Pose + wrench but NOT twist: the grouped vector omits the twist block.
    values = extract_recording_frame_values(
        make_snapshot(),
        [
            "observation.state.left_arm.tcp_pose",
            "observation.state.left_arm.tcp_wrench",
        ],
    )

    assert set(values) == {"observation.state.left_arm"}
    assert values["observation.state.left_arm"] == list(range(0, 7)) + list(
        range(20, 26)
    )


def test_extract_recording_frame_values_maps_two_arms_to_left_right() -> None:
    values = extract_recording_frame_values(
        make_dual_snapshot(),
        [
            "observation.state.left_arm.tcp_pose",
            "observation.state.right_arm.tcp_pose",
        ],
    )

    assert set(values) == {
        "observation.state.left_arm",
        "observation.state.right_arm",
    }
    assert values["observation.state.left_arm"] == [0, 1, 2, 3, 4, 5, 6]
    assert values["observation.state.right_arm"] == [100, 101, 102, 103, 104, 105, 106]


def test_resolve_recording_image_names_filters_selected_cameras() -> None:
    assert resolve_recording_image_names(
        [
            "observation.images.left_wrist",
            "action.left_arm.tcp_pose",
            "observation.images.ego",
        ]
    ) == ["left_wrist", "ego"]


def test_extract_recording_images_filters_to_requested_entries() -> None:
    images = extract_recording_images(
        make_images(),
        [
            "observation.images.ego",
            "observation.images.right_wrist",
            "action.left_arm.tcp_twist",
        ],
    )

    assert sorted(images) == ["ego", "right_wrist"]


def test_build_features_from_sample_groups_arms_with_named_axes() -> None:
    features, state_keys, action_keys = build_features_from_sample(
        make_snapshot(),
        make_images(),
        [
            "observation.images.ego",
            "observation.state.left_arm.tcp_pose",
            "observation.state.left_arm.tcp_twist",
            "observation.state.left_arm.tcp_wrench",
            "action.left_arm.tcp_wrench",
        ],
    )

    assert "observation.images.ego" in features
    assert "observation.images.left_wrist" not in features

    assert state_keys == ["observation.state.left_arm"]
    assert action_keys == ["action.left_arm"]

    # All three state metrics selected -> one 19-dim vector (7 + 6 + 6).
    state_feature = features["observation.state.left_arm"]
    assert state_feature["dtype"] == "float32"
    assert state_feature["shape"] == (19,)
    assert state_feature["names"][:7] == [
        "tcp_pose.x",
        "tcp_pose.y",
        "tcp_pose.z",
        "tcp_pose.q_w",
        "tcp_pose.q_x",
        "tcp_pose.q_y",
        "tcp_pose.q_z",
    ]
    assert state_feature["names"][7] == "tcp_twist.vx"
    assert state_feature["names"][-1] == "tcp_wrench.mz"

    # Only the wrench action metric selected -> a 6-dim action vector.
    action_feature = features["action.left_arm"]
    assert action_feature["shape"] == (6,)
    assert action_feature["names"] == [
        "tcp_wrench.fx",
        "tcp_wrench.fy",
        "tcp_wrench.fz",
        "tcp_wrench.mx",
        "tcp_wrench.my",
        "tcp_wrench.mz",
    ]
