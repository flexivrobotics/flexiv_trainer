import json

import pytest

from flexivtrainer.data.gripper_command import (
    GripperCommandMetadata,
    write_gripper_command_metadata,
)
from flexivtrainer.jobs.merge_episodes import _validate_matching_feature_keys


def _dataset(
    root,
    name: str,
    feature_keys: list[str],
    *,
    action_names: list[str] | None = None,
):
    path = root / name
    (path / "meta").mkdir(parents=True)
    features = {key: {} for key in feature_keys}
    if "action" in features:
        names = action_names or ["arm.tcp_pose.x"]
        features["action"] = {"shape": [len(names)], "names": names}
    (path / "meta" / "info.json").write_text(
        json.dumps({"features": features}),
        encoding="utf-8",
    )
    return path


def test_merge_feature_guard_accepts_matching_depth_schema(tmp_path) -> None:
    keys = ["observation.images.ego", "observation.images.ego_depth", "action"]
    first = _dataset(tmp_path, "first", keys)
    second = _dataset(tmp_path, "second", keys)

    _validate_matching_feature_keys([first, second])


def test_merge_feature_guard_rejects_depth_and_rgb_only_mix(tmp_path) -> None:
    rgb = _dataset(tmp_path, "rgb", ["observation.images.ego", "action"])
    rgbd = _dataset(
        tmp_path,
        "rgbd",
        ["observation.images.ego", "observation.images.ego_depth", "action"],
    )

    with pytest.raises(ValueError, match="Depth-enabled and RGB-only"):
        _validate_matching_feature_keys([rgb, rgbd])


def test_merge_guard_rejects_legacy_and_target_width_action_schemas(tmp_path) -> None:
    legacy = _dataset(
        tmp_path,
        "legacy",
        ["observation.state", "action"],
        action_names=["single_arm.gripper.width", "single_arm.gripper.force"],
    )
    command = _dataset(
        tmp_path,
        "command",
        ["observation.state", "action"],
        action_names=["single_arm.gripper.target_width"],
    )

    with pytest.raises(ValueError, match="convert-gripper-actions"):
        _validate_matching_feature_keys([legacy, command])


def test_merge_requires_identical_target_width_command_metadata(tmp_path) -> None:
    names = ["single_arm.gripper.target_width"]
    first = _dataset(
        tmp_path,
        "first",
        ["observation.state", "action"],
        action_names=names,
    )
    second = _dataset(
        tmp_path,
        "second",
        ["observation.state", "action"],
        action_names=names,
    )
    expected = GripperCommandMetadata(velocity_m_s=0.2, force_limit_n=20.0)
    write_gripper_command_metadata(first, expected)
    write_gripper_command_metadata(second, expected)

    assert _validate_matching_feature_keys([first, second]) == expected

    write_gripper_command_metadata(
        second,
        GripperCommandMetadata(velocity_m_s=0.1, force_limit_n=20.0),
    )
    with pytest.raises(ValueError, match="metadata differs"):
        _validate_matching_feature_keys([first, second])


def test_gripper_command_metadata_normalizes_hardware_float_noise() -> None:
    metadata = GripperCommandMetadata(
        velocity_m_s=0.20000000298023224,
        force_limit_n=20.0000001,
    )

    assert metadata.to_dict() == {
        "format_version": 1,
        "velocity_m_s": 0.2,
        "force_limit_n": 20.0,
    }
