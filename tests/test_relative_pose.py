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

"""Chunk-relative SE(3) pose actions (policies/lerobot_plugins/relative_pose.py)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_training_service import (
    _FakePulse,
    _NoopThread,
    make_dataset,
    make_service,
)

from flexivtrainer.config import TRAIN_RELATIVE_POSE_ENV
from flexivtrainer.data.lerobot_io import ROTATION_6D_AXES
from flexivtrainer.jobs import train_policy
from flexivtrainer.policies.lerobot_plugins import relative_pose
from flexivtrainer.policies.lerobot_plugins.relative_pose import (
    AbsolutePoseProcessorStep,
    RelativePoseProcessorStep,
    checkpoint_pose_layout,
    hold_relative_pose_anchor,
    pose_layout,
    range_normalized_width_stats,
    relative_action_stats,
    state_anchors,
    to_absolute,
    to_relative,
    unnormalized_rotation_stats,
)
from flexivtrainer.rollout.checkpoint import checkpoint_slam_world_frame

UMI_NAMES = ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper"]
TCP_NAMES = [
    *(f"left.tcp_pose.{a}" for a in "xyz"),
    *(f"left.tcp_rotation_6d.{a}" for a in ROTATION_6D_AXES),
    "left.gripper.width",
]


def _features(state_names, action_names=None):
    action_names = state_names if action_names is None else action_names
    return {
        "observation.state": {"names": list(state_names)},
        "action": {"names": list(action_names)},
    }


def _pose_vector(position, rotation, convention, gripper=0.5):
    matrix = rotation.as_matrix()
    matrix = matrix if convention == "rows" else matrix.T
    return np.r_[position, matrix[:2].reshape(6), gripper]


def _random_poses(rng, shape, convention):
    flat = [
        _pose_vector(rng.normal(size=3), Rotation.random(random_state=rng), convention)
        for _ in range(int(np.prod(shape)))
    ]
    return torch.tensor(np.reshape(flat, (*shape, 10)), dtype=torch.float64)


def _move_world(poses, world, offset, convention):
    """The same poses seen from another world frame."""
    out = poses.clone().reshape(-1, 10).numpy()
    for row in out:
        first, second = row[3:6], row[6:9]
        rows = np.stack([first, second, np.cross(first, second)])
        matrix = rows if convention == "rows" else rows.T
        moved = world @ matrix
        moved = moved if convention == "rows" else moved.T
        row[:3] = world @ row[:3] + offset
        row[3:9] = moved[:2].reshape(6)
    return torch.tensor(out.reshape(poses.shape))


def test_layouts_are_found_by_name_in_both_rotation_conventions() -> None:
    umi = pose_layout(_features(UMI_NAMES))
    assert umi["action"] == [
        {
            "prefix": "",
            "position": [0, 1, 2],
            "rotation": [3, 4, 5, 6, 7, 8],
            "convention": "columns",
            "anchor": 0,
        }
    ]
    tcp = pose_layout(_features(["left.gripper.width", *TCP_NAMES[:-1]]))
    assert tcp["state"][0]["convention"] == "rows"
    assert tcp["state"][0]["position"] == [1, 2, 3]
    assert tcp["action"][0]["prefix"] == "left."


def test_a_quaternion_pose_is_refused_by_name() -> None:
    legacy = ["tcp_pose.x", "tcp_pose.y", "tcp_pose.z", "tcp_pose.q_w"]
    with pytest.raises(ValueError, match="tcp_rotation_6d.r1_x"):
        pose_layout(_features(legacy))


def test_an_action_pose_needs_a_state_pose_of_the_same_arm() -> None:
    right = [name.replace("left.", "right.") for name in TCP_NAMES]
    with pytest.raises(ValueError, match="right."):
        pose_layout(_features(TCP_NAMES, right))
    with pytest.raises(ValueError, match="no Cartesian pose"):
        pose_layout(_features([f"joint_{i}" for i in range(7)]))


@pytest.mark.parametrize(
    ("names", "convention"), [(UMI_NAMES, "columns"), (TCP_NAMES, "rows")]
)
def test_relative_actions_round_trip_and_ignore_the_world_frame(
    names, convention
) -> None:
    rng = np.random.default_rng(0)
    layout = pose_layout(_features(names))
    state = _random_poses(rng, (4,), convention)
    actions = _random_poses(rng, (4, 6), convention)

    relative = to_relative(actions, state_anchors(state, layout), layout)
    restored = to_absolute(relative, state_anchors(state, layout), layout)
    assert torch.allclose(restored, actions, atol=1e-9)
    # The gripper is not a pose and passes through.
    assert torch.equal(relative[..., 9], actions[..., 9])

    world = Rotation.random(random_state=rng).as_matrix()
    offset = rng.normal(size=3)
    moved_state = _move_world(state, world, offset, convention)
    moved_actions = _move_world(actions, world, offset, convention)
    moved = to_relative(moved_actions, state_anchors(moved_state, layout), layout)
    assert torch.allclose(moved, relative, atol=1e-9)


def test_the_action_at_the_current_pose_is_the_identity() -> None:
    layout = pose_layout(_features(UMI_NAMES))
    state = _random_poses(np.random.default_rng(1), (3,), "columns")
    relative = to_relative(state[:, None], state_anchors(state, layout), layout)
    identity = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float64)
    assert torch.allclose(relative[:, 0, :9], identity.expand(3, 9), atol=1e-9)


def _act_processors(monkeypatch, layout):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act import processor_act
    from lerobot.policies.act.configuration_act import ACTConfig

    monkeypatch.setenv(TRAIN_RELATIVE_POSE_ENV, json.dumps(layout))
    config = ACTConfig(
        device="cpu",
        chunk_size=6,
        n_action_steps=6,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
            "observation.images.cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 8, 8)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(10,)),
        },
    )
    unit = {"mean": torch.zeros(10), "std": torch.ones(10)}
    stats = {"observation.state": unit, "action": unit}
    return config, *processor_act.make_act_pre_post_processors(
        config, dataset_stats=stats
    )


def test_new_processors_wrap_their_normalizers_and_round_trip(monkeypatch) -> None:
    from lerobot.processor import (
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )

    layout = pose_layout(_features(UMI_NAMES))
    _, preprocessor, postprocessor = _act_processors(monkeypatch, layout)
    pre_types = [type(step) for step in preprocessor.steps]
    post_types = [type(step) for step in postprocessor.steps]
    at = pre_types.index(RelativePoseProcessorStep)
    assert pre_types[at + 1] is NormalizerProcessorStep
    at = post_types.index(AbsolutePoseProcessorStep)
    assert post_types[at - 1] is UnnormalizerProcessorStep

    rng = np.random.default_rng(2)
    state = _random_poses(rng, (2,), "columns").float()
    actions = _random_poses(rng, (2, 6), "columns").float()
    batch = preprocessor({"observation.state": state, "action": actions})
    # Unit statistics, so what the policy sees is the relative chunk itself.
    expected = to_relative(actions, state_anchors(state, layout), layout)
    assert torch.allclose(batch["action"], expected, atol=1e-5)
    identity = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32)
    assert torch.allclose(batch["observation.state"][:, :9], identity.expand(2, 9))
    assert torch.allclose(postprocessor(batch["action"]), actions, atol=1e-5)


def test_without_the_env_var_processors_stay_absolute(monkeypatch) -> None:
    from lerobot.policies.act import processor_act

    config, _, _ = _act_processors(monkeypatch, pose_layout(_features(UMI_NAMES)))
    monkeypatch.setenv(TRAIN_RELATIVE_POSE_ENV, "")
    plain, _ = processor_act.make_act_pre_post_processors(config)
    assert not any(isinstance(s, RelativePoseProcessorStep) for s in plain.steps)


def test_saved_processors_restore_the_pair_and_its_layout(
    monkeypatch, tmp_path: Path
) -> None:
    from lerobot.policies.factory import make_pre_post_processors

    layout = pose_layout(_features(UMI_NAMES))
    config, preprocessor, postprocessor = _act_processors(monkeypatch, layout)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    monkeypatch.setenv(TRAIN_RELATIVE_POSE_ENV, "")

    assert checkpoint_pose_layout(tmp_path) == layout
    restored_pre, restored_post = make_pre_post_processors(
        config, pretrained_path=str(tmp_path)
    )
    pre = next(
        s for s in restored_pre.steps if isinstance(s, RelativePoseProcessorStep)
    )
    post = next(
        s for s in restored_post.steps if isinstance(s, AbsolutePoseProcessorStep)
    )
    original = next(
        s for s in preprocessor.steps if isinstance(s, RelativePoseProcessorStep)
    )
    assert pre.pair_id == post.pair_id == original.pair_id
    assert post.layout == layout


def test_a_held_anchor_keeps_the_pose_the_chunk_was_predicted_from(
    monkeypatch,
) -> None:
    layout = pose_layout(_features(UMI_NAMES))
    _, preprocessor, postprocessor = _act_processors(monkeypatch, layout)
    rng = np.random.default_rng(3)
    predicted_at, moved_to = _random_poses(rng, (2, 1), "columns").float()
    relative = torch.zeros(1, 10)
    relative[0, 3] = relative[0, 7] = 1.0  # the identity pose

    preprocessor({"observation.state": predicted_at})
    hold_relative_pose_anchor(preprocessor, hold=True)
    preprocessor({"observation.state": moved_to})
    restored = postprocessor(relative)
    assert torch.allclose(restored[:, :9], predicted_at[:, :9], atol=1e-5)

    hold_relative_pose_anchor(preprocessor, hold=False)
    preprocessor({"observation.state": moved_to})
    restored = postprocessor(relative)
    assert torch.allclose(restored[:, :9], moved_to[:, :9], atol=1e-5)


def test_postprocessing_before_any_observation_is_refused() -> None:
    step = AbsolutePoseProcessorStep(layout={"action": []}, pair_id="never-seen")
    from lerobot.types import TransitionKey

    with pytest.raises(RuntimeError, match="anchor"):
        step({TransitionKey.ACTION: torch.zeros(1, 10)})


def test_action_stats_are_relative_and_stay_within_each_episode() -> None:
    rng = np.random.default_rng(4)
    layout = pose_layout(_features(UMI_NAMES))
    states = _random_poses(rng, (8,), "columns")
    # Every action is the same small motion in the gripper's own frame.
    step = torch.zeros(1, 10, dtype=torch.float64)
    step[0, :3] = torch.tensor([0.01, -0.02, 0.03])
    step[0, 3], step[0, 7] = 1.0, 1.0
    actions = to_absolute(step.expand(8, 10), state_anchors(states, layout), layout)
    actions[:, 9] = states[:, 9]
    episode_index = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    stats = relative_action_stats(
        actions.numpy(), states.numpy(), episode_index, [0], layout
    )
    assert np.allclose(stats["mean"][:9], step[0, :9].numpy(), atol=1e-9)
    assert np.allclose(stats["std"][:9], 0.0, atol=1e-7)

    wide = relative_action_stats(
        actions.numpy(), states.numpy(), episode_index, [0, 1, 2, 3, 4], layout
    )
    assert int(wide["count"][0]) == 8 * 5


def test_a_job_asks_its_subprocess_for_the_dataset_layout(
    tmp_path: Path, monkeypatch
) -> None:
    service = make_service(tmp_path)
    dataset = make_dataset(tmp_path, state_size=10)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.state"]["names"] = UMI_NAMES
    info["features"]["action"]["names"] = UMI_NAMES
    info_path.write_text(json.dumps(info), encoding="utf-8")
    spawned = {}

    class _Process:
        stdout = iter(())
        pid = 1

        def poll(self):
            return None

    def popen(*args, **kwargs):
        spawned.update(kwargs)
        return _Process()

    monkeypatch.setattr(train_policy.shutil, "which", lambda _: "/bin/lerobot-train")
    monkeypatch.setattr(train_policy.subprocess, "Popen", popen)
    monkeypatch.setattr(train_policy.threading, "Thread", _NoopThread)
    monkeypatch.setattr(train_policy, "Pulse", _FakePulse)
    monkeypatch.setattr(train_policy, "resolve_training_device", lambda _: "cpu")

    service.start(
        dataset_root=dataset,
        output_dir=tmp_path / "training" / "relative",
        policy_type="act",
        relative_pose=True,
    )
    sent = json.loads(spawned["env"][TRAIN_RELATIVE_POSE_ENV])
    assert sent == pose_layout(info["features"])


def test_a_job_refuses_relative_poses_it_cannot_convert(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    joints = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="no Cartesian pose"):
        service.start(
            dataset_root=joints,
            output_dir=tmp_path / "training" / "joints",
            policy_type="act",
            relative_pose=True,
        )
    with pytest.raises(ValueError, match="not supported for smolvla"):
        service._relative_pose_layout(True, "smolvla", joints, None)


ROWS = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
# A UMI export with --recent-motion-ms 50 --rotation-wrt-start: pose 9, width,
# motion 9, width change, rotation since start 6.
UMI_STATE = [
    *TCP_NAMES[:9],
    "left.gripper.width",
    *(f"left.tcp_motion_50ms.{a}" for a in ("x", "y", "z", *ROWS)),
    "left.gripper.width_change_50ms",
    *(f"left.tcp_rotation_wrt_start.{a}" for a in ROWS),
]
UNNORMALIZED = [*range(13, 19), *range(20, 26)]


def _state_stats(rng) -> dict[str, np.ndarray]:
    width = len(UMI_STATE)
    low = rng.uniform(-1, 0, width)
    return {
        "mean": rng.normal(size=width),
        "std": rng.uniform(1e-5, 0.2, width),
        "min": low,
        "max": low + rng.uniform(1e-4, 1, width),
        "q01": low,
        "q99": low + 0.5,
        "q50": rng.normal(size=width),
        "count": np.array([1000]),
    }


def test_only_the_extra_rotation_columns_get_unit_stats() -> None:
    stats = _state_stats(np.random.default_rng(3))
    before = {key: value.copy() for key, value in stats.items()}

    out = unnormalized_rotation_stats(stats, UMI_STATE)

    kept = [i for i in range(len(UMI_STATE)) if i not in UNNORMALIZED]
    expected = {"mean": 0, "std": 1, "min": -1, "max": 1, "q01": -1, "q99": 1}
    for key, value in expected.items():
        assert out[key][UNNORMALIZED].tolist() == [value] * len(UNNORMALIZED)
        assert out[key][kept].tolist() == stats[key][kept].tolist()
    assert out["q50"].tolist() == stats["q50"].tolist()
    assert out["count"].tolist() == [1000]
    assert "q10" not in out
    for key, value in before.items():
        assert stats[key].tolist() == value.tolist()
    # A state without UMI's extra inputs is left alone.
    plain = unnormalized_rotation_stats(stats, [f"s{i}" for i in range(26)])
    assert plain["std"].tolist() == stats["std"].tolist()


@pytest.mark.parametrize("mode", ["MEAN_STD", "MIN_MAX"])
def test_the_normalizer_passes_those_columns_through(mode) -> None:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import NormalizerProcessorStep

    rng = np.random.default_rng(4)
    stats = unnormalized_rotation_stats(_state_stats(rng), UMI_STATE)
    step = NormalizerProcessorStep(
        features={
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(len(UMI_STATE),)
            )
        },
        norm_map={FeatureType.STATE: NormalizationMode[mode]},
        stats={"observation.state": stats},
    )
    state = torch.as_tensor(
        rng.uniform(-1, 1, (2, len(UMI_STATE))), dtype=torch.float32
    )

    out = step._normalize_observation({"observation.state": state}, inverse=False)[
        "observation.state"
    ]

    assert torch.allclose(out[:, UNNORMALIZED], state[:, UNNORMALIZED], atol=1e-5)
    assert not torch.allclose(out[:, :3], state[:, :3])


WIDTH = UMI_STATE.index("left.gripper.width")


def test_only_the_gripper_width_gets_its_range_as_mean_and_std() -> None:
    stats = _state_stats(np.random.default_rng(5))
    before = {key: value.copy() for key, value in stats.items()}

    out = range_normalized_width_stats(stats, UMI_STATE)

    low, high = stats["min"][WIDTH], stats["max"][WIDTH]
    assert out["mean"][WIDTH] == pytest.approx((low + high) / 2)
    assert out["std"][WIDTH] == pytest.approx((high - low) / 2)
    kept = [i for i in range(len(UMI_STATE)) if i != WIDTH]
    for key in stats.keys() - {"count"}:
        assert out[key][kept].tolist() == stats[key][kept].tolist()
    for key, value in before.items():
        assert stats[key].tolist() == value.tolist()
    # The UMI layout's bare opening counts too; its change and force do not.
    names = ["gripper", "left.gripper.width_change_50ms", "left.gripper.force"]
    three = {key: np.array(value[:3]) for key, value in stats.items() if key != "count"}
    out = range_normalized_width_stats(three, names)
    assert out["std"].tolist()[1:] == three["std"].tolist()[1:]
    assert out["std"][0] != three["std"][0]


def test_a_constant_gripper_width_keeps_its_stats() -> None:
    stats = _state_stats(np.random.default_rng(6))
    stats["max"][WIDTH] = stats["min"][WIDTH]

    out = range_normalized_width_stats(stats, UMI_STATE)

    assert out["mean"][WIDTH] == stats["mean"][WIDTH]
    assert out["std"][WIDTH] == stats["std"][WIDTH]


def test_mean_std_maps_the_gripper_width_range_onto_unit_range() -> None:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import NormalizerProcessorStep

    stats = range_normalized_width_stats(
        _state_stats(np.random.default_rng(7)), UMI_STATE
    )
    step = NormalizerProcessorStep(
        features={
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(len(UMI_STATE),)
            )
        },
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={"observation.state": stats},
    )
    state = torch.zeros((2, len(UMI_STATE)), dtype=torch.float32)
    state[0, WIDTH] = float(stats["min"][WIDTH])
    state[1, WIDTH] = float(stats["max"][WIDTH])

    out = step._normalize_observation({"observation.state": state}, inverse=False)[
        "observation.state"
    ]

    assert out[:, WIDTH].tolist() == pytest.approx([-1.0, 1.0], abs=1e-5)


def _umi_export(tmp_path: Path) -> Path:
    """A dataset with the metadata flexiv_gdk's UMI export writes beside it."""
    dataset = make_dataset(tmp_path, state_size=10)
    (dataset / "meta" / "flexiv_extras.json").write_text(
        json.dumps({"world_frame": "IMU frame of the first keyframe"}),
        encoding="utf-8",
    )
    return dataset


def test_a_job_refuses_umi_poses_as_absolute(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    umi = _umi_export(tmp_path)
    model_dir = tmp_path / "absolute_checkpoint"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="SLAM map frame"):
        service._relative_pose_layout(False, "act", umi, None)
    # Nor by fine-tuning a checkpoint trained on absolute poses.
    with pytest.raises(ValueError, match="SLAM map frame"):
        service._relative_pose_layout(
            False, "act", umi, {"model_path": str(model_dir)}
        )
    # A robot recording still trains on absolute poses.
    robot = make_dataset(tmp_path / "robot")
    assert service._relative_pose_layout(False, "act", robot, None) is None


def test_rollout_finds_the_slam_frame_of_a_checkpoints_dataset(
    tmp_path: Path,
) -> None:
    umi = _umi_export(tmp_path)
    robot = make_dataset(tmp_path / "robot")
    for name, dataset, expected in (
        ("umi", umi, "IMU frame of the first keyframe"),
        ("robot", robot, None),
    ):
        model_dir = tmp_path / name / "pretrained_model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "train_config.json").write_text(
            json.dumps({"dataset": {"root": str(dataset)}}), encoding="utf-8"
        )
        assert checkpoint_slam_world_frame(str(model_dir)) == expected


def test_the_patches_are_installed_once() -> None:
    from lerobot.policies.act import processor_act

    first = processor_act.make_act_pre_post_processors
    relative_pose.apply_patches()
    assert processor_act.make_act_pre_post_processors is first
