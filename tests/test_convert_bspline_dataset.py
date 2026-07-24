import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flexivtrainer.data.bspline import (
    detect_tcp_action_layouts,
    evaluate_parameter_matrix,
    extract_cartesian_controls,
)
from flexivtrainer.jobs.convert_bspline_dataset import (
    convert_lerobot_tcp_actions_to_bspline,
)

lerobot = pytest.importorskip("lerobot")


def _action_names(
    sides: tuple[str, ...] = ("left_arm",),
    *,
    include_gripper: bool = False,
) -> list[str]:
    names: list[str] = []
    for side in sides:
        names.extend(
            [
                f"{side}.tcp_pose.x",
                f"{side}.tcp_pose.y",
                f"{side}.tcp_pose.z",
                f"{side}.tcp_pose.q_w",
                f"{side}.tcp_pose.q_x",
                f"{side}.tcp_pose.q_y",
                f"{side}.tcp_pose.q_z",
                f"{side}.tcp_twist.vx",
                f"{side}.tcp_twist.vy",
                f"{side}.tcp_twist.vz",
                f"{side}.tcp_twist.wx",
                f"{side}.tcp_twist.wy",
                f"{side}.tcp_twist.wz",
                f"{side}.tcp_wrench.fx",
                f"{side}.tcp_wrench.fy",
                f"{side}.tcp_wrench.fz",
                f"{side}.tcp_wrench.mx",
                f"{side}.tcp_wrench.my",
                f"{side}.tcp_wrench.mz",
            ]
        )
        if include_gripper:
            names.extend(
                [
                    f"{side}.gripper.width",
                    f"{side}.gripper.force",
                ]
            )
    return names


def _make_source_dataset(
    root,
    *,
    sides: tuple[str, ...] = ("left_arm",),
    include_gripper: bool = False,
) -> list[np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = _action_names(sides, include_gripper=include_gripper)
    dataset = LeRobotDataset.create(
        repo_id="local/source",
        root=root,
        fps=10,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["state.0", "state.1", "state.2"],
            },
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["environment.0", "environment.1"],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(names),),
                "names": names,
            },
        },
        use_videos=False,
    )

    actions: list[np.ndarray] = []
    for frame_index in range(12):
        fraction = frame_index / 11
        action = np.zeros(len(names), dtype=np.float32)
        for side_index, side in enumerate(sides):
            quaternion_xyzw = Rotation.from_euler(
                "z", (side_index + 1) * 0.4 * fraction
            ).as_quat()
            pose_start = names.index(f"{side}.tcp_pose.x")
            action[pose_start : pose_start + 3] = [
                0.45 + 0.08 * fraction,
                -0.05 + 0.1 * side_index,
                0.3 + 0.02 * fraction,
            ]
            action[pose_start + 3 : pose_start + 7] = [
                quaternion_xyzw[3],
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
            ]
            if include_gripper:
                action[names.index(f"{side}.gripper.width")] = (
                    0.01 * (side_index + 1) + 0.02 * fraction
                )
                action[names.index(f"{side}.gripper.force")] = 10.0 + side_index
        actions.append(action.copy())
        dataset.add_frame(
            {
                "observation.state": np.array(
                    [fraction, fraction**2, 1.0],
                    dtype=np.float32,
                ),
                "observation.environment_state": np.array(
                    [fraction, 1.0 - fraction],
                    dtype=np.float32,
                ),
                "action": action,
                "task": "synthetic TCP motion",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return actions


def test_convert_dataset_replaces_action_with_flat_spline_parameters(tmp_path) -> None:
    from datasets import config as datasets_config
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = tmp_path / "source"
    output = tmp_path / "converted"
    source_actions = _make_source_dataset(source)

    result = convert_lerobot_tcp_actions_to_bspline(source, output)

    assert result["frames"] == 12
    assert result["episodes"] == 1
    assert result["sides"] == ["left_arm"]
    assert result["parameter_matrix_shape"] == [16, 10]
    assert result["flattened_action_dim"] == 160

    source_info = json.loads((source / "meta" / "info.json").read_text())
    converted_info = json.loads((output / "meta" / "info.json").read_text())
    spline_info = json.loads((output / "meta" / "bspline.json").read_text())
    assert source_info["features"]["action"]["shape"] == [19]
    assert converted_info["features"]["action"]["shape"] == [160]
    assert len(converted_info["features"]["action"]["names"]) == 160
    assert spline_info["rotation_representation"] == "rotation_6d_rows"

    datasets_config.HF_DATASETS_CACHE = str(tmp_path / "hf-cache")
    converted = LeRobotDataset(
        repo_id="local/converted", root=output, download_videos=False
    )
    flat_parameters = np.asarray(converted[0]["action"])
    assert flat_parameters.shape == (160,)
    parameters = flat_parameters.reshape(16, 10)

    layouts = detect_tcp_action_layouts(_action_names())
    expected = extract_cartesian_controls(
        np.stack(source_actions),
        layouts,
    )[0]
    decoded = evaluate_parameter_matrix(parameters, [0.0])[0]
    np.testing.assert_allclose(decoded, expected, atol=2e-3)

    stats = json.loads((output / "meta" / "stats.json").read_text())
    assert len(stats["action"]["mean"]) == 160


def test_convert_dual_arm_gripper_data_and_ties_row_statistics(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "converted"
    sides = ("left_arm", "right_arm")
    _make_source_dataset(source, sides=sides, include_gripper=True)

    result = convert_lerobot_tcp_actions_to_bspline(source, output)

    assert result["parameter_matrix_shape"] == [16, 21]
    assert result["flattened_action_dim"] == 336

    converted_info = json.loads((output / "meta" / "info.json").read_text())
    action_names = converted_info["features"]["action"]["names"]
    spline_info = json.loads((output / "meta" / "bspline.json").read_text())
    assert spline_info["format_version"] == 2
    assert spline_info["parameter_matrix_shape"] == [16, 21]
    assert spline_info["active_control_rows"] == 12
    assert spline_info["gripper_width_sides"] == list(sides)
    assert spline_info["normalization_mode"] == "tied_per_semantic_channel"
    assert spline_info["parameter_channel_names"] == [
        "knot",
        *[
            name
            for layout in detect_tcp_action_layouts(
                _action_names(sides, include_gripper=True)
            )
            for name in layout.control_names
        ],
    ]
    assert all("gripper.force" not in name for name in action_names)
    assert action_names[20] == "bspline.row_00.right_arm.gripper.width"
    assert action_names[21] == "bspline.row_01.knot"

    stats = json.loads((output / "meta" / "stats.json").read_text())["action"]
    for stat_name, values in stats.items():
        if stat_name == "count":
            continue
        rows = np.asarray(values).reshape(16, 21)
        np.testing.assert_allclose(rows, np.repeat(rows[:1], 16, axis=0))

    mean = np.asarray(stats["mean"])
    scale = np.maximum(np.asarray(stats["std"]), 1e-8)
    import pandas as pd

    table = pd.read_parquet(next(iter(sorted((output / "data").glob("*/*.parquet")))))
    sample = np.asarray(table.iloc[0]["action"])
    normalized = (sample - mean) / scale
    np.testing.assert_allclose(normalized * scale + mean, sample, atol=1e-12)


@pytest.mark.parametrize(
    ("sides", "include_gripper", "matrix_shape", "flat_dim"),
    [
        (("left_arm",), False, [16, 10], 160),
        (("left_arm",), True, [16, 11], 176),
        (("left_arm", "right_arm"), False, [16, 19], 304),
        (("left_arm", "right_arm"), True, [16, 21], 336),
    ],
)
def test_conversion_action_dimensions(
    tmp_path,
    sides,
    include_gripper,
    matrix_shape,
    flat_dim,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "converted"
    _make_source_dataset(
        source,
        sides=sides,
        include_gripper=include_gripper,
    )

    result = convert_lerobot_tcp_actions_to_bspline(source, output)

    assert result["parameter_matrix_shape"] == matrix_shape
    assert result["flattened_action_dim"] == flat_dim
