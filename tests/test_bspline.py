import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flexivtrainer.data.bspline import (
    build_episode_spline_targets,
    detect_tcp_action_layouts,
    evaluate_parameter_matrix,
    extract_cartesian_controls,
    parameter_feature_names,
    parameter_matrix_shape,
    quaternion_wxyz_to_rotation_6d,
    rotation_6d_to_quaternion_wxyz,
    validate_parameter_matrix_shape,
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


def test_layout_includes_gripper_width_and_excludes_force() -> None:
    names = [
        *_pose_names("left_arm"),
        "left_arm.gripper.force",
        "left_arm.gripper.width",
    ]
    layouts = detect_tcp_action_layouts(names)
    actions = np.zeros((2, len(names)), dtype=np.float64)
    actions[:, 3] = 1.0
    actions[:, 7] = 20.0
    actions[:, 8] = [0.01, 0.02]

    controls = extract_cartesian_controls(actions, layouts)

    assert layouts[0].gripper_width_index == 8
    assert controls.shape == (2, 10)
    np.testing.assert_array_equal(controls[:, -1], [0.01, 0.02])
    assert layouts[0].control_names[-1] == "left_arm.gripper.width"
    assert all("force" not in name for name in layouts[0].control_names)


def test_parameter_layout_is_row_major_for_dual_arm() -> None:
    names = [
        *_pose_names("left_arm"),
        "left_arm.gripper.width",
        *_pose_names("right_arm"),
        "right_arm.gripper.width",
    ]
    layouts = detect_tcp_action_layouts(names)

    shape = parameter_matrix_shape(layouts, parameter_rows=16)
    feature_names = parameter_feature_names(layouts, parameter_rows=16)

    assert shape == (16, 21)
    assert len(feature_names) == 336
    assert feature_names[:3] == [
        "bspline.row_00.knot",
        "bspline.row_00.left_arm.tcp_pose.x",
        "bspline.row_00.left_arm.tcp_pose.y",
    ]
    assert feature_names[20] == "bspline.row_00.right_arm.gripper.width"
    assert feature_names[21] == "bspline.row_01.knot"
    validate_parameter_matrix_shape(
        np.zeros((4, 16, 21)),
        layouts,
        parameter_rows=16,
    )
    with pytest.raises(ValueError, match="expected \\(16, 21\\)"):
        validate_parameter_matrix_shape(
            np.zeros((4, 16, 20)),
            layouts,
            parameter_rows=16,
        )


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
    assert np.all(np.isfinite(result.parameters))
    for frame_index, parameters in enumerate(result.parameters):
        assert np.all(np.diff(parameters[:, 0]) >= 0)
        assert parameters[3, 0] <= 0 <= parameters[-4, 0]
        np.testing.assert_allclose(
            evaluate_parameter_matrix(parameters, [0.0], degree=3)[0],
            result.fit.spline(frame_index),
            atol=1e-5,
        )
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
