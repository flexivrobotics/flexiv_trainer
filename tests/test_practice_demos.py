"""The demo converter: teleop recordings into demonstrations in a practice space."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from typer.testing import CliRunner

from flexivtrainer.evaluation.check_labels import LABELS_FILE, write_label
from flexivtrainer.practice.demos import (
    REPORT_RELATIVE_PATH,
    _arm,
    convert_demos,
)
from flexivtrainer.practice.space import (
    LAST_ACTIONS,
    STATE,
    WRENCH,
    PracticeSpace,
    Sample,
    StepTracker,
    read_space,
)

FPS = 30
SIZE = 32
ARM = "single_arm"
#: Frames per episode, and the tool's speed along its own x, in m per frame.
EPISODES = [(20, 0.0004), (17, 0.0004), (25, 0.001)]
#: How far the operator's target leads the tool, along the tool's x.
LEAD_M = 0.0003
#: How far the operator's target is turned about the tool's z.
LEAD_DEG = 0.6
#: The tool faces along base y: a quarter turn about base z.
TURN = Rotation.from_euler("z", 90, degrees=True)
START = np.array([0.4, 0.0, 0.3])

POSE = [
    *(f"tcp_pose.{a}" for a in "xyz"),
    *(f"tcp_rotation_6d.r{r}_{a}" for r in (1, 2) for a in "xyz"),
]
TWIST = [f"tcp_twist.{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
WRENCH_COLUMNS = [f"tcp_wrench.{a}" for a in ("fx", "fy", "fz", "mx", "my", "mz")]


def _space(**overrides):
    values = {
        "rate_hz": 10,
        "step_m": {"x": 0.002, "y": 0.002},
        "step_deg": {"z": 1.0},
        "force_n": {"z": 50.0},
        "views": ("wrist",),
        "last_actions": 2,
        "wrench_samples": 4,
        "sense_hz": FPS,
    }
    return PracticeSpace(**{**values, **overrides})


def _rows(frame: int, speed: float):
    """The tool at `frame`: its position, the operator's target, the wrench."""
    position = START + TURN.apply([speed * frame, 0.0, 0.0])
    target = START + TURN.apply([speed * frame + LEAD_M, 0.0, 0.0])
    target_rotation = TURN * Rotation.from_euler("z", LEAD_DEG, degrees=True)
    # The tool presses harder each frame: the world pushes back along -z.
    wrench = [0.0, 0.0, -0.5 * frame, 0.0, 0.0, 0.0]
    return position, target, target_rotation, wrench


def _rot6d(rotation: Rotation) -> list[float]:
    return rotation.as_matrix()[:2].reshape(6).tolist()


def _recording(root, *, wrench=True, episodes=EPISODES):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from flexivtrainer.data.lerobot_io import resolve_recording_vcodec

    state_names = [f"{ARM}.{n}" for n in POSE + TWIST]
    if wrench:
        state_names += [f"{ARM}.{n}" for n in WRENCH_COLUMNS]
    state_names += [f"{ARM}.gripper.width", f"{ARM}.gripper.force"]
    action_names = [f"{ARM}.{n}" for n in POSE + TWIST] + [
        f"{ARM}.gripper.target_width"
    ]
    camera = {
        "dtype": "video",
        "shape": (SIZE, SIZE, 3),
        "names": ["height", "width", "channels"],
    }
    dataset = LeRobotDataset.create(
        repo_id=f"local/{root.name}",
        fps=FPS,
        root=root,
        features={
            "observation.images.wrist": camera,
            "observation.images.ego": camera,
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
        use_videos=True,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec=resolve_recording_vcodec("h264")),
    )
    for length, speed in episodes:
        for frame in range(length):
            position, target, target_rotation, force = _rows(frame, speed)
            twist = [0.0, speed * FPS, 0.0, 0.0, 0.0, 0.0]
            state = [*position, *_rot6d(TURN), *twist]
            if wrench:
                state += force
            state += [0.03, 20.0]
            grey = np.full((SIZE, SIZE, 3), 8 * frame, dtype=np.uint8)
            dataset.add_frame(
                {
                    "observation.images.wrist": grey,
                    "observation.images.ego": 255 - grey,
                    "observation.state": np.array(state, dtype=np.float32),
                    "action": np.array(
                        [*target, *_rot6d(target_rotation), *twist, 0.03],
                        dtype=np.float32,
                    ),
                    "task": "insert",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    (root / "meta" / "control.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "recorded_in": "teleop",
                "follower_cartesian_stiffness": [2900.0] * 3 + [100.0] * 3,
                "follower_max_contact_wrench": "controller default",
                "wrench_feedback_scale": "controller default",
            }
        )
    )
    return root


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    pytest.importorskip("lerobot")
    return _recording(tmp_path_factory.mktemp("teleop") / "teleop")


@pytest.fixture(scope="module")
def converted(recording, tmp_path_factory):
    output = tmp_path_factory.mktemp("practice") / "demos"
    result = convert_demos(recording, output, _space())
    return output, result


def _read(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(f"local/{root.name}", root=root, video_backend="pyav")


def _columns(root):
    """Every column of a converted recording, as arrays."""
    rows = _read(root).hf_dataset.with_format("numpy")
    return {key: np.asarray(rows[key]) for key in rows.column_names}


def _steps(length):
    return list(range(0, length - 1, 3))


def _expected_action(start, length, speed):
    """The label by hand: the operator's target one step later, from the tool
    now, in the tool frame, in units of the bounds, clipped."""
    ahead = min(start + 3, length - 1)
    move_x = (speed * (ahead - start) + LEAD_M) / 0.002
    force_z = 0.5 * ahead / 50.0
    return np.clip([move_x, 0.0, LEAD_DEG / 1.0, force_z], -1, 1)


# --- what it writes ---------------------------------------------------------------


def test_one_step_per_stride_with_the_action_toward_the_operator_target(converted):
    output, result = converted
    dataset = _read(output)
    assert dataset.fps == 10
    assert dataset.meta.total_episodes == len(EPISODES)
    columns = _columns(output)
    actions, episode = columns["action"], columns["episode_index"]
    for index, (length, speed) in enumerate(EPISODES):
        expected = [_expected_action(s, length, speed) for s in _steps(length)]
        np.testing.assert_allclose(actions[episode == index], expected, atol=1e-4)
    assert result["steps"] == sum(len(_steps(n)) for n, _ in EPISODES)


def test_the_reward_and_the_end_are_on_each_last_step(converted):
    rows = _columns(converted[0])
    reward = rows["next.reward"].reshape(-1)
    done = rows["next.done"].reshape(-1)
    episode = rows["episode_index"].reshape(-1)
    for index, (length, _) in enumerate(EPISODES):
        expected = np.zeros(len(_steps(length)))
        expected[-1] = 1
        np.testing.assert_array_equal(reward[episode == index], expected)
        np.testing.assert_array_equal(done[episode == index], expected.astype(bool))


def test_only_the_listed_cameras_are_kept_at_the_step_starts(converted):
    dataset = _read(converted[0])
    assert dataset.meta.video_keys == ["observation.images.wrist"]
    for row, start in enumerate(_steps(EPISODES[0][0])):
        frame = dataset[row]["observation.images.wrist"]
        assert float(frame.mean()) * 255 == pytest.approx(8 * start, abs=3)


def test_the_features_name_every_number(converted):
    dataset = _read(converted[0])
    space = _space()
    assert dataset.features["action"]["names"] == space.action_names
    for key, names in space.state_dims().items():
        assert dataset.features[key]["names"] == names


def test_a_robot_moving_as_recorded_sees_the_demonstrations(converted):
    """Replay the episode as the arm reports it -- driver quaternions, every
    sample -- and take the converted actions: the live observation at each
    step is the converted one."""
    output, _ = converted
    rows = _columns(output)
    episode = rows["episode_index"].reshape(-1)
    length, speed = EPISODES[2]
    stored = {k: rows[k][episode == 2] for k in (STATE, WRENCH, LAST_ACTIONS)}
    actions = rows["action"][episode == 2]
    tracker = StepTracker(_space())
    x, y, z, w = TURN.as_quat()

    def reading(frame):
        position, _, _, wrench = _rows(frame, speed)
        twist = [0.0, speed * FPS, 0.0, 0.0, 0.0, 0.0]
        return Sample.from_driver([*position, w, x, y, z], twist, wrench)

    tracker.reset(reading(0))
    image = {"wrist": np.zeros((SIZE, SIZE, 3), np.uint8)}
    for step, start in enumerate(_steps(length)):
        for frame in range(max(1, start - 2), start + 1) if start else ():
            tracker.sense(reading(frame))
        seen = tracker.observation(image)
        for key, values in stored.items():
            # The recording stores float32: a position of 0.4 m to ~3e-8 m.
            np.testing.assert_allclose(seen[key], values[step], atol=1e-4)
        tracker.take(actions[step])


def test_a_demonstration_faster_than_the_bounds_is_clipped_and_reported(converted):
    output, result = converted
    moves = result["clipped"]["move.x"]
    # Episode 2 moves 3 mm a step against a 2 mm bound; the others 1.5 mm.
    fast = len(_steps(EPISODES[2][0]))
    assert moves["share"] == pytest.approx(fast / result["steps"], abs=1e-3)
    assert moves["largest"] == pytest.approx((0.003 + LEAD_M) / 0.002, abs=1e-3)
    assert moves["covers_95_m"] == pytest.approx(0.0033, abs=1e-6)
    assert result["clipped"]["move.y"]["share"] == 0
    assert any(w.startswith("move.x") for w in result["warnings"])
    stored = json.loads((output / REPORT_RELATIVE_PATH).read_text())
    assert stored["clipped"] == result["clipped"]


def test_the_space_and_the_stiffness_travel_with_the_demonstrations(
    converted, recording
):
    output, _ = converted
    assert read_space(output / "meta" / "practice_space.json") == _space()
    control = json.loads((output / "meta" / "control.json").read_text())
    assert control["follower_cartesian_stiffness"] == [2900.0] * 3 + [100.0] * 3


def test_lerobot_practice_loop_reads_the_demonstrations(converted):
    buffer = pytest.importorskip("lerobot.rl.buffer")
    dataset = _read(converted[0])
    transitions = buffer.ReplayBuffer._lerobotdataset_to_transitions(
        dataset, state_keys=[STATE, WRENCH, LAST_ACTIONS, "observation.images.wrist"]
    )
    assert len(transitions) == len(dataset)
    ends = [t for t in transitions if t["done"]]
    assert len(ends) == len(EPISODES)
    assert all(t["reward"] == 1.0 for t in ends)
    assert sum(t["reward"] for t in transitions) == len(EPISODES)


# --- a check ends each demonstration -----------------------------------------------


def test_a_check_ends_each_episode_at_its_labelled_frame(tmp_path):
    pytest.importorskip("lerobot")
    source = _recording(tmp_path / "teleop", episodes=EPISODES)
    write_label(source, "seated", 0, 10)
    write_label(source, "seated", 1, None)
    result = convert_demos(source, tmp_path / "out", _space(), check="seated")
    assert [e["episode"] for e in result["episodes"]] == [0]
    assert result["episodes"][0]["steps"] == len(range(0, 10, 3))
    assert result["left_out"] == [
        {"episode": 1, "why": "never reaches seated"},
        {"episode": 2, "why": "no seated label"},
    ]
    rows = _columns(tmp_path / "out")
    np.testing.assert_array_equal(rows["next.reward"].reshape(-1), [0, 0, 0, 1])


def test_a_label_past_the_end_is_refused(tmp_path):
    pytest.importorskip("lerobot")
    source = _recording(tmp_path / "teleop", episodes=EPISODES[:1])
    labels = source / LABELS_FILE
    labels.write_text(
        json.dumps({"format_version": 1, "checks": {"seated": {"0": 99}}})
    )
    with pytest.raises(ValueError, match="frame 99"):
        convert_demos(source, tmp_path / "out", _space(), check="seated")


# --- refusals ------------------------------------------------------------------------


def test_a_recording_without_the_wrench_is_refused_naming_what_to_record(tmp_path):
    pytest.importorskip("lerobot")
    source = _recording(tmp_path / "teleop", wrench=False, episodes=EPISODES[:1])
    with pytest.raises(ValueError, match="tcp_wrench.fx.*record with those entries"):
        convert_demos(source, tmp_path / "out", _space())
    assert not (tmp_path / "out").exists()


def test_a_wrong_rate_camera_or_output_is_refused(recording, tmp_path):
    with pytest.raises(ValueError, match="recorded at 30 Hz"):
        convert_demos(recording, tmp_path / "a", _space(rate_hz=5, sense_hz=15))
    with pytest.raises(ValueError, match=r"no RGB camera \['side'\]"):
        convert_demos(recording, tmp_path / "b", _space(views=("wrist", "side")))
    with pytest.raises(ValueError, match=r"no episodes \[7\]"):
        convert_demos(recording, tmp_path / "c", _space(), episodes=[0, 7])
    (tmp_path / "d").mkdir()
    with pytest.raises(FileExistsError):
        convert_demos(recording, tmp_path / "d", _space())
    with pytest.raises(ValueError, match="inside the source"):
        convert_demos(recording, recording / "nested", _space())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d"]


def test_the_arm_is_named_when_there_are_several():
    two = ["left_arm.tcp_pose.x", "right_arm.tcp_pose.x"]
    assert _arm(two, "right_arm") == "right_arm"
    with pytest.raises(ValueError, match="2 arms"):
        _arm(two, None)
    with pytest.raises(ValueError, match="no arm 'single_arm'"):
        _arm(two, "single_arm")
    assert _arm(["single_arm.tcp_pose.x"], None) == "single_arm"


def test_the_command_line_converts_chosen_episodes(recording, tmp_path):
    from flexivtrainer.cli.practice_demos import app

    space = tmp_path / "space.json"
    space.write_text(json.dumps(_space().to_dict()))
    result = CliRunner().invoke(
        app,
        [
            str(recording),
            str(tmp_path / "out"),
            "--space",
            str(space),
            "--episode",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["steps"] == len(_steps(EPISODES[1][0]))
    assert _read(tmp_path / "out").meta.total_episodes == 1
