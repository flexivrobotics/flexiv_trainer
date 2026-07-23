import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flexivtrainer.data.bspline import (
    build_episode_spline_targets,
    detect_tcp_action_layouts,
    evaluate_parameter_matrix,
    quaternion_wxyz_to_rotation_6d,
    rotation_6d_to_quaternion_wxyz,
)


def _pose_names(side: str) -> list[str]:
    return [
        f"{side}.tcp_pose.x",
        f"{side}.tcp_pose.y",
        f"{side}.tcp_pose.z",
        f"{side}.tcp_pose.q_w",
        f"{side}.tcp_pose.q_x",
        f"{side}.tcp_pose.q_y",
        f"{side}.tcp_pose.q_z",
    ]


def test_detect_tcp_action_layouts_uses_named_axes() -> None:
    names = [
        "left_arm.tcp_twist.vx",
        *_pose_names("right_arm"),
        *_pose_names("left_arm"),
    ]

    layouts = detect_tcp_action_layouts(names, sides=["left_arm", "right_arm"])

    assert [layout.side for layout in layouts] == ["left_arm", "right_arm"]
    assert layouts[0].pose_indices == tuple(range(8, 15))
    assert layouts[1].pose_indices == tuple(range(1, 8))


def test_detect_tcp_action_layouts_rejects_incomplete_pose() -> None:
    with pytest.raises(ValueError, match="Incomplete TCP pose"):
        detect_tcp_action_layouts(["left_arm.tcp_pose.x"], sides=["left_arm"])


def test_rotation_6d_round_trip_preserves_orientation() -> None:
    rotations = Rotation.from_euler(
        "xyz",
        [[0.1, -0.2, 0.3], [-1.0, 0.4, 2.2], [0.0, 0.0, 0.0]],
    )
    xyzw = rotations.as_quat()
    wxyz = np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)

    rotation_6d = quaternion_wxyz_to_rotation_6d(wxyz)
    recovered_wxyz = rotation_6d_to_quaternion_wxyz(rotation_6d)
    recovered_xyzw = np.concatenate(
        [recovered_wxyz[:, 1:], recovered_wxyz[:, :1]], axis=1
    )
    difference = Rotation.from_quat(recovered_xyzw) * rotations.inv()

    assert np.max(difference.magnitude()) < 1e-10


def test_episode_targets_preserve_fitted_curve_over_each_local_segment() -> None:
    frame_count = 24
    sample = np.linspace(0.0, 1.0, frame_count)
    position = np.stack(
        [
            0.45 + 0.1 * sample,
            -0.08 + 0.03 * np.sin(np.pi * sample),
            0.30 + 0.05 * sample**2,
        ],
        axis=1,
    )
    xyzw = Rotation.from_euler("z", (0.5 * sample)[:, None]).as_quat()
    wxyz = np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)
    controls = np.concatenate(
        [position, quaternion_wxyz_to_rotation_6d(wxyz)],
        axis=1,
    )

    result = build_episode_spline_targets(
        controls,
        degree=3,
        chunk_size=10,
        max_error=1e-4,
    )

    assert result.parameters.shape == (frame_count, 16, 10)
    for frame_index, parameters in enumerate(result.parameters):
        assert np.all(np.diff(parameters[:, 0]) >= 0)
        start = float(parameters[3, 0])
        end = float(parameters[-4, 0])
        local_times = np.linspace(start, end, 7)
        decoded = evaluate_parameter_matrix(parameters, local_times, degree=3)
        np.testing.assert_allclose(
            decoded,
            result.fit.spline(local_times + frame_index),
            atol=1e-5,
        )
    assert result.fit.max_abs_error <= 1e-4
