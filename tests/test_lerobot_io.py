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

from flexivtrainer.data import lerobot_io
from flexivtrainer.data.lerobot_io import (
    DEFAULT_RECORDING_ENTRY_KEYS,
    arm_side_label,
    build_features_from_sample,
    extract_recording_frame_values,
    extract_recording_images,
    resolve_recording_image_names,
    resolve_recording_entries,
    resolve_recording_vcodec,
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
    return {"robots": {"FOLLOWER_A": _arm_payload(0)}}


def make_dual_snapshot() -> dict:
    return {"robots": {"FOLLOWER_A": _arm_payload(0), "FOLLOWER_B": _arm_payload(100)}}


def make_images() -> dict[str, np.ndarray]:
    return {
        "ego": np.zeros((4, 5, 3), dtype=np.uint8),
        "left_wrist": np.zeros((4, 5, 3), dtype=np.uint8),
        "right_wrist": np.zeros((4, 5, 3), dtype=np.uint8),
    }


def test_resolve_recording_entries_defaults_to_all_options() -> None:
    assert resolve_recording_entries() == list(DEFAULT_RECORDING_ENTRY_KEYS)


def test_resolve_vcodec_auto_prefers_hardware_h264(monkeypatch) -> None:
    # Only NVENC "available" -> auto picks it over software h264.
    monkeypatch.setattr(
        lerobot_io, "_encoder_available", lambda name: name == "h264_nvenc"
    )
    assert resolve_recording_vcodec("auto") == "h264_nvenc"


def test_resolve_vcodec_auto_falls_back_to_software_h264(monkeypatch) -> None:
    # No hardware encoder available -> software h264, never AV1/HEVC.
    monkeypatch.setattr(lerobot_io, "_encoder_available", lambda name: name == "h264")
    assert resolve_recording_vcodec("auto") == "h264"


def test_resolve_vcodec_explicit_passthrough_when_available(monkeypatch) -> None:
    monkeypatch.setattr(lerobot_io, "_encoder_available", lambda name: True)
    assert resolve_recording_vcodec("h264_videotoolbox") == "h264_videotoolbox"
    # An explicitly chosen AV1 is honoured (the operator opted in).
    assert resolve_recording_vcodec("libsvtav1") == "libsvtav1"


def test_resolve_vcodec_unavailable_explicit_falls_back_to_software(monkeypatch) -> None:
    # A config shared across machines names an encoder this build lacks.
    monkeypatch.setattr(lerobot_io, "_encoder_available", lambda name: False)
    assert resolve_recording_vcodec("h264_nvenc") == "h264"


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


def test_extract_recording_frame_values_groups_selected_metrics() -> None:
    values = extract_recording_frame_values(
        make_snapshot(),
        [
            "observation.state.left_arm.tcp_pose",
            "observation.state.left_arm.tcp_twist",
            "observation.state.left_arm.tcp_wrench",
            "action.left_arm.tcp_wrench",
        ],
    )

    # Every arm's selected metrics are concatenated into a single
    # observation.state / action vector (the layout stock LeRobot policies
    # require). Here one arm: pose 7 + twist 6 + wrench 6 = 19 for state.
    assert set(values) == {"observation.state", "action"}
    assert values["observation.state"] == list(range(0, 7)) + list(
        range(10, 16)
    ) + list(range(20, 26))
    # Only the wrench action metric was selected.
    assert values["action"] == list(range(50, 56))


def test_extract_recording_frame_values_drops_unselected_metrics() -> None:
    # Pose + wrench but NOT twist: the combined vector omits the twist block.
    values = extract_recording_frame_values(
        make_snapshot(),
        [
            "observation.state.left_arm.tcp_pose",
            "observation.state.left_arm.tcp_wrench",
        ],
    )

    assert set(values) == {"observation.state"}
    assert values["observation.state"] == list(range(0, 7)) + list(range(20, 26))


def test_extract_recording_frame_values_concatenates_both_arms() -> None:
    values = extract_recording_frame_values(
        make_dual_snapshot(),
        [
            "observation.state.left_arm.tcp_pose",
            "observation.state.right_arm.tcp_pose",
        ],
    )

    # Both arms fold into one observation.state vector, left then right.
    assert set(values) == {"observation.state"}
    assert values["observation.state"] == [0, 1, 2, 3, 4, 5, 6] + [
        100,
        101,
        102,
        103,
        104,
        105,
        106,
    ]


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


def test_build_features_from_sample_combines_arms_with_named_axes() -> None:
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
    # Visual features must carry axis names; LeRobot reads ft["names"]
    # unconditionally when building policy features.
    assert features["observation.images.ego"]["names"] == [
        "height",
        "width",
        "channels",
    ]

    # Single combined feature each, named exactly as stock policies expect.
    assert state_keys == ["observation.state"]
    assert action_keys == ["action"]

    # All three state metrics selected -> one 19-dim vector (7 + 6 + 6), with
    # axis names prefixed by the arm side.
    state_feature = features["observation.state"]
    assert state_feature["dtype"] == "float32"
    assert state_feature["shape"] == (19,)
    assert state_feature["names"][:7] == [
        "left_arm.tcp_pose.x",
        "left_arm.tcp_pose.y",
        "left_arm.tcp_pose.z",
        "left_arm.tcp_pose.q_w",
        "left_arm.tcp_pose.q_x",
        "left_arm.tcp_pose.q_y",
        "left_arm.tcp_pose.q_z",
    ]
    assert state_feature["names"][7] == "left_arm.tcp_twist.vx"
    assert state_feature["names"][-1] == "left_arm.tcp_wrench.mz"

    # Only the wrench action metric selected -> a 6-dim action vector.
    action_feature = features["action"]
    assert action_feature["shape"] == (6,)
    assert action_feature["names"] == [
        "left_arm.tcp_wrench.fx",
        "left_arm.tcp_wrench.fy",
        "left_arm.tcp_wrench.fz",
        "left_arm.tcp_wrench.mx",
        "left_arm.tcp_wrench.my",
        "left_arm.tcp_wrench.mz",
    ]
