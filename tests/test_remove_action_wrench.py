import json

import numpy as np
import pandas as pd
import pytest

from flexivtrainer.data.remove_action_wrench import remove_action_wrench

lerobot = pytest.importorskip("lerobot")


def _make_dataset(root) -> tuple[list[str], list[np.ndarray]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = [
        "left_arm.tcp_pose.x",
        "left_arm.tcp_twist.vx",
        "left_arm.tcp_wrench.fx",
        "left_arm.tcp_wrench.fy",
        "left_arm.tcp_wrench.fz",
        "left_arm.tcp_wrench.mx",
        "left_arm.tcp_wrench.my",
        "left_arm.tcp_wrench.mz",
        "right_arm.tcp_pose.x",
        "right_arm.tcp_wrench.fx",
        "right_arm.tcp_wrench.fy",
        "right_arm.tcp_wrench.fz",
        "right_arm.tcp_wrench.mx",
        "right_arm.tcp_wrench.my",
        "right_arm.tcp_wrench.mz",
    ]
    dataset = LeRobotDataset.create(
        repo_id="local/source",
        root=root,
        fps=10,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["measured.fx", "measured.fy"],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(names),),
                "names": names,
            },
        },
        use_videos=False,
    )

    actions = []
    for frame_index in range(4):
        action = np.arange(len(names), dtype=np.float32) + frame_index
        actions.append(action)
        dataset.add_frame(
            {
                "observation.state": np.array(
                    [frame_index, -frame_index], dtype=np.float32
                ),
                "action": action,
                "task": "test",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return names, actions


def test_remove_action_wrench_slices_only_action(tmp_path, monkeypatch) -> None:
    from datasets import config as datasets_config
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    monkeypatch.setattr(
        datasets_config,
        "HF_DATASETS_CACHE",
        str(tmp_path / "hf-cache"),
    )
    source = tmp_path / "source"
    output = tmp_path / "output"
    source_names, source_actions = _make_dataset(source)

    result = remove_action_wrench(source, output)

    expected_indices = [
        index for index, name in enumerate(source_names) if ".tcp_wrench." not in name
    ]
    expected_names = [source_names[index] for index in expected_indices]
    converted = LeRobotDataset(
        repo_id="local/output",
        root=output,
        download_videos=False,
    )

    assert result["source_action_dim"] == 15
    assert result["output_action_dim"] == 3
    assert result["max_abs_removed_wrench"] == 17.0
    assert list(converted.meta.features["action"]["names"]) == expected_names
    assert tuple(converted.meta.features["action"]["shape"]) == (3,)
    np.testing.assert_allclose(
        np.asarray(converted[2]["action"]),
        source_actions[2][expected_indices],
    )
    np.testing.assert_allclose(
        np.asarray(converted[2]["observation.state"]),
        [2.0, -2.0],
    )

    source_info = json.loads((source / "meta" / "info.json").read_text())
    output_info = json.loads((output / "meta" / "info.json").read_text())
    assert source_info["features"]["action"]["shape"] == [15]
    assert list(output_info["features"]) == list(source_info["features"])

    stats = json.loads((output / "meta" / "stats.json").read_text())
    assert len(stats["action"]["mean"]) == 3
    episode_path = next((output / "meta" / "episodes").glob("*/*.parquet"))
    episode_metadata = pd.read_parquet(episode_path)
    assert len(episode_metadata.iloc[0]["stats/action/mean"]) == 3


def test_remove_action_wrench_refuses_dataset_without_wrench(
    tmp_path, monkeypatch
) -> None:
    from datasets import config as datasets_config
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    monkeypatch.setattr(
        datasets_config,
        "HF_DATASETS_CACHE",
        str(tmp_path / "hf-cache"),
    )
    source = tmp_path / "source"
    dataset = LeRobotDataset.create(
        repo_id="local/source",
        root=source,
        fps=10,
        features={
            "action": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["left_arm.tcp_pose.x"],
            }
        },
        use_videos=False,
    )
    dataset.add_frame({"action": np.array([1.0], dtype=np.float32), "task": "test"})
    dataset.save_episode()
    dataset.finalize()

    with pytest.raises(ValueError, match="No TCP wrench action axes found"):
        remove_action_wrench(source, tmp_path / "output")
