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


def test_extract_recording_frame_values_groups_per_arm_without_serials() -> None:
    values = extract_recording_frame_values(
        make_snapshot(),
        ["observation.state.tcp_pose", "action.tcp_wrench"],
    )

    # No serial number leaks into the keys; metrics are grouped per arm.
    assert set(values) == {"observation.state.left_arm", "action.left_arm"}
    assert values["observation.state.left_arm"] == [0, 1, 2, 3, 4, 5, 6]
    assert values["action.left_arm"] == [50, 51, 52, 53, 54, 55]


def test_extract_recording_frame_values_maps_two_arms_to_left_right() -> None:
    values = extract_recording_frame_values(
        make_dual_snapshot(), ["observation.state.tcp_pose"]
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
            "action.tcp_pose",
            "observation.images.ego",
        ]
    ) == ["left_wrist", "ego"]


def test_extract_recording_images_filters_to_requested_entries() -> None:
    images = extract_recording_images(
        make_images(),
        [
            "observation.images.ego",
            "observation.images.right_wrist",
            "action.tcp_twist",
        ],
    )

    assert sorted(images) == ["ego", "right_wrist"]


def test_build_features_from_sample_groups_arms_with_named_axes() -> None:
    features, state_keys, action_keys = build_features_from_sample(
        make_snapshot(),
        make_images(),
        [
            "observation.images.ego",
            "observation.state.tcp_pose",
            "action.tcp_wrench",
        ],
    )

    assert "observation.images.ego" in features
    assert "observation.images.left_wrist" not in features

    assert state_keys == ["observation.state.left_arm"]
    assert action_keys == ["action.left_arm"]

    state_feature = features["observation.state.left_arm"]
    assert state_feature["dtype"] == "float32"
    assert state_feature["shape"] == (7,)
    assert state_feature["names"] == [
        "tcp_pose.x",
        "tcp_pose.y",
        "tcp_pose.z",
        "tcp_pose.q_w",
        "tcp_pose.q_x",
        "tcp_pose.q_y",
        "tcp_pose.q_z",
    ]

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


def test_build_features_from_sample_combines_state_metrics_into_one_vector() -> None:
    features, state_keys, _ = build_features_from_sample(
        make_snapshot(),
        {},
        [
            "observation.state.tcp_pose",
            "observation.state.tcp_twist",
            "observation.state.tcp_wrench",
        ],
    )

    assert state_keys == ["observation.state.left_arm"]
    feature = features["observation.state.left_arm"]
    # 7 (pose) + 6 (twist) + 6 (wrench) concatenated into a single feature.
    assert feature["shape"] == (19,)
    assert feature["names"][:1] == ["tcp_pose.x"]
    assert feature["names"][7] == "tcp_twist.vx"
    assert feature["names"][-1] == "tcp_wrench.mz"
