import json

import pytest

from flexivtrainer.jobs.merge_episodes import _validate_matching_feature_keys


def _dataset(root, name: str, feature_keys: list[str]):
    path = root / name
    (path / "meta").mkdir(parents=True)
    (path / "meta" / "info.json").write_text(
        json.dumps({"features": {key: {} for key in feature_keys}}),
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
