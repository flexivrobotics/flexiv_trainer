import json

import numpy as np
import pytest

from flexivtrainer.jobs.convert_gripper_actions import (
    convert_legacy_gripper_actions,
)

lerobot = pytest.importorskip("lerobot")


def _make_source(root, sides=("left_arm", "right_arm")) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    state_names: list[str] = []
    action_names = ["task.progress"]
    for side in sides:
        state_names.extend(
            [f"{side}.gripper.width", f"{side}.gripper.force"]
        )
        action_names.extend(
            [f"{side}.gripper.width", f"{side}.gripper.force"]
        )
    dataset = LeRobotDataset.create(
        repo_id="local/legacy_gripper",
        root=root,
        fps=30,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
        },
        use_videos=False,
    )
    widths = [0.07, 0.07, 0.0699, 0.0697, 0.06, 0.06, 0.0601, 0.0603, 0.07]
    for frame_index, width in enumerate(widths):
        force = 0.5 if frame_index < 4 else 15.0
        state: list[float] = []
        action = [frame_index / (len(widths) - 1)]
        for _side in sides:
            state.extend([width, force])
            action.extend([width, force])
        dataset.add_frame(
            {
                "observation.state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "task": "legacy gripper conversion",
            }
        )
    dataset.save_episode()
    dataset.finalize()


def test_conversion_replaces_legacy_actions_and_preserves_observation(
    tmp_path,
) -> None:
    from datasets import config as datasets_config
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_source(source)
    source_info_before = (source / "meta" / "info.json").read_bytes()

    result = convert_legacy_gripper_actions(source, output)

    assert result["sides"] == ["left_arm", "right_arm"]
    assert result["action_dim"] == 3
    assert (source / "meta" / "info.json").read_bytes() == source_info_before
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["features"]["observation.state"]["names"] == [
        "left_arm.gripper.width",
        "left_arm.gripper.force",
        "right_arm.gripper.width",
        "right_arm.gripper.force",
    ]
    assert info["features"]["action"]["names"] == [
        "task.progress",
        "left_arm.gripper.close",
        "right_arm.gripper.close",
    ]

    datasets_config.HF_DATASETS_CACHE = str(tmp_path / "hf-cache")
    converted = LeRobotDataset(
        repo_id="local/converted_gripper", root=output, download_videos=False
    )
    commands = np.stack([np.asarray(converted[index]["action"]) for index in range(9)])
    np.testing.assert_array_equal(commands[:, 1], [0, 0, 1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(commands[:, 2], commands[:, 1])
    assert np.asarray(converted[0]["observation.state"]).shape == (4,)

    audit = json.loads(
        (output / "meta" / "gripper_action_conversion.json").read_text()
    )
    assert audit["motion_threshold_m"] == pytest.approx(0.0002)
    assert audit["episodes"][0]["transitions"] == [
        {"frame_index": 2, "command": "close"},
        {"frame_index": 6, "command": "open"},
    ]
    stats = json.loads((output / "meta" / "stats.json").read_text())
    assert len(stats["action"]["mean"]) == 3


def test_initial_state_manifest_overrides_force_inference(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    manifest = tmp_path / "initial.json"
    _make_source(source, sides=("left_arm",))
    manifest.write_text(
        json.dumps({"episodes": {"0": {"left_arm": "close"}}}),
        encoding="utf-8",
    )

    convert_legacy_gripper_actions(
        source, output, initial_state_manifest=manifest
    )

    audit = json.loads(
        (output / "meta" / "gripper_action_conversion.json").read_text()
    )
    assert audit["episodes"][0]["initial_state"] == "close"
    assert audit["episodes"][0]["initial_source"] == "override"


def test_conversion_rejects_existing_output_without_touching_source(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_source(source, sides=("left_arm",))
    output.mkdir()

    with pytest.raises(FileExistsError):
        convert_legacy_gripper_actions(source, output)

    assert (source / "meta" / "info.json").is_file()
