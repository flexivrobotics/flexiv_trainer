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
    build_features_from_sample,
    extract_recording_images,
    extract_recording_feature_values,
    resolve_recording_image_names,
    resolve_recording_entries,
)


def make_snapshot() -> dict:
    return {
        "robots": {
            "REMOTE_A": {
                "cartesian_state": {
                    "tcp_pose": [0, 1, 2, 3, 4, 5, 6],
                    "tcp_vel": [10, 11, 12, 13, 14, 15],
                    "ext_wrench_in_world": [20, 21, 22, 23, 24, 25],
                },
                "cartesian_command": {
                    "tcp_pose_des": [30, 31, 32, 33, 34, 35, 36],
                    "tcp_vel_des": [40, 41, 42, 43, 44, 45],
                    "wrench_des_in_ctrl_frame": [50, 51, 52, 53, 54, 55],
                },
            }
        }
    }


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


def test_extract_recording_feature_values_filters_to_requested_entries() -> None:
    observation_values, action_values = extract_recording_feature_values(
        make_snapshot(),
        [
            "observation.state.tcp_pose",
            "action.tcp_wrench",
        ],
    )

    assert sorted(observation_values) == [
        "REMOTE_A.state.tcp_pose.0",
        "REMOTE_A.state.tcp_pose.1",
        "REMOTE_A.state.tcp_pose.2",
        "REMOTE_A.state.tcp_pose.3",
        "REMOTE_A.state.tcp_pose.4",
        "REMOTE_A.state.tcp_pose.5",
        "REMOTE_A.state.tcp_pose.6",
    ]
    assert sorted(action_values) == [
        "REMOTE_A.command.tcp_wrench.0",
        "REMOTE_A.command.tcp_wrench.1",
        "REMOTE_A.command.tcp_wrench.2",
        "REMOTE_A.command.tcp_wrench.3",
        "REMOTE_A.command.tcp_wrench.4",
        "REMOTE_A.command.tcp_wrench.5",
    ]


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


def test_build_features_from_sample_only_includes_selected_images() -> None:
    features, observation_values, action_values = build_features_from_sample(
        make_snapshot(),
        make_images(),
        [
            "observation.images.ego",
            "action.tcp_wrench",
        ],
    )

    assert "observation.images.ego" in features
    assert "observation.images.left_wrist" not in features
    assert "observation.images.right_wrist" not in features
    assert observation_values == []
    assert action_values == [
        "REMOTE_A.command.tcp_wrench.0",
        "REMOTE_A.command.tcp_wrench.1",
        "REMOTE_A.command.tcp_wrench.2",
        "REMOTE_A.command.tcp_wrench.3",
        "REMOTE_A.command.tcp_wrench.4",
        "REMOTE_A.command.tcp_wrench.5",
    ]
