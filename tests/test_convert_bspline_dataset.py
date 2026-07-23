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


def _action_names() -> list[str]:
    return [
        "left_arm.tcp_pose.x",
        "left_arm.tcp_pose.y",
        "left_arm.tcp_pose.z",
        "left_arm.tcp_pose.q_w",
        "left_arm.tcp_pose.q_x",
        "left_arm.tcp_pose.q_y",
        "left_arm.tcp_pose.q_z",
        "left_arm.tcp_twist.vx",
        "left_arm.tcp_twist.vy",
        "left_arm.tcp_twist.vz",
        "left_arm.tcp_twist.wx",
        "left_arm.tcp_twist.wy",
        "left_arm.tcp_twist.wz",
        "left_arm.tcp_wrench.fx",
        "left_arm.tcp_wrench.fy",
        "left_arm.tcp_wrench.fz",
        "left_arm.tcp_wrench.mx",
        "left_arm.tcp_wrench.my",
        "left_arm.tcp_wrench.mz",
    ]


def _make_source_dataset(root) -> list[np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = _action_names()
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
        quaternion_xyzw = Rotation.from_euler("z", 0.4 * fraction).as_quat()
        action = np.zeros(len(names), dtype=np.float32)
        action[:3] = [0.45 + 0.08 * fraction, -0.05, 0.3 + 0.02 * fraction]
        action[3:7] = [
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        ]
        actions.append(action.copy())
        dataset.add_frame(
            {
                "observation.state": np.array(
                    [fraction, fraction**2, 1.0],
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
