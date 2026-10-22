"""The practice space: what a step skill sees and what its action means."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flexivtrainer.data.lerobot_io import extract_recording_frame_values
from flexivtrainer.practice.space import (
    LAST_ACTIONS,
    STATE,
    WRENCH,
    Command,
    PracticeSpace,
    Sample,
    StepTracker,
    read_space,
    write_space,
)
from flexivtrainer.rollout.observations import read_robot_snapshot


def _space(**overrides):
    values = {
        "rate_hz": 10,
        "step_m": {"x": 0.002, "y": 0.002},
        "step_deg": {"x": 0.5, "y": 0.5, "z": 1.0},
        "force_n": {"z": 50.0},
        "views": ("wrist",),
        "last_actions": 3,
        "wrench_samples": 4,
        "sense_hz": 30,
    }
    return PracticeSpace(**{**values, **overrides})


def _pose(rng):
    """A driver pose [x, y, z, qw, qx, qy, qz] somewhere in front of the arm."""
    x, y, z, w = Rotation.random(random_state=rng).as_quat()
    return [*rng.uniform(-0.5, 0.5, 3), w, x, y, z]


def _sample(rng, pose=None):
    return Sample.from_driver(
        pose or _pose(rng), rng.normal(0, 0.05, 6), rng.normal(0, 5, 6)
    )


# --- the space -----------------------------------------------------------------


def test_actions_are_moves_then_rotations_then_forces_in_axis_order():
    space = _space(step_m={"y": 0.001, "x": 0.002}, step_deg={"z": 1.0})
    assert space.action_names == ["move.x", "move.y", "turn.z", "force.z"]
    assert space.action_bounds == [(0.002, "m"), (0.001, "m"), (1.0, "deg"), (50, "N")]
    assert space.action_dim == 4
    assert space.force_axes == [False, False, True, False, False, False]
    assert space.stride == 3


def test_the_space_round_trips_through_its_file(tmp_path):
    space = _space()
    path = write_space(tmp_path, space)
    assert path == tmp_path / "meta" / "practice_space.json"
    assert read_space(path) == space
    beside_a_checkpoint = write_space(tmp_path, space, checkpoint=True)
    assert beside_a_checkpoint == tmp_path / "practice_space.json"
    assert json.loads(beside_a_checkpoint.read_text())["format_version"] == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sense_hz": 25}, "whole multiple"),
        ({"rate_hz": 0}, "positive"),
        ({"step_m": {"z": 0.001}}, "both moved and force controlled"),
        ({"step_m": {"w": 0.001}}, "unknown axes"),
        ({"step_deg": {"x": -1}}, "positive"),
        ({"force_n": {"z": float("nan")}}, "positive"),
        ({"step_m": {}, "step_deg": {}, "force_n": {}}, "at least one action axis"),
        ({"views": ("wrist", "wrist")}, "distinct"),
        ({"last_actions": -1}, ">= 0"),
        ({"wrench_samples": 2.5}, ">= 0"),
    ],
)
def test_a_wrong_space_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        _space(**change)


def test_a_wrong_file_is_refused_naming_it(tmp_path):
    path = tmp_path / "practice_space.json"
    path.write_text(json.dumps({**_space().to_dict(), "step_mm": {}}))
    with pytest.raises(ValueError, match="practice_space.json.*unknown"):
        read_space(path)
    path.write_text(json.dumps({**_space().to_dict(), "format_version": 2}))
    with pytest.raises(ValueError, match="format 2"):
        read_space(path)


# --- samples -------------------------------------------------------------------


def test_a_live_sample_and_a_recorded_one_see_the_same():
    """The arm reports a quaternion; the recorder stores rotation-6D rows. A
    tracker fed either sees the same numbers, to float32 precision."""
    rng = np.random.default_rng(0)
    pose, twist, wrench = _pose(rng), rng.normal(0, 0.1, 6), rng.normal(0, 5, 6)
    robot = type(
        "Robot",
        (),
        {
            "states": lambda self: type(
                "States",
                (),
                {"tcp_pose": pose, "tcp_vel": twist, "ext_wrench_in_world": wrench},
            )()
        },
    )()
    # What the recorder writes for this reading.
    entries = [
        f"observation.state.single_arm.{metric}"
        for metric in ("tcp_pose", "tcp_twist", "tcp_wrench")
    ]
    recorded = extract_recording_frame_values(
        read_robot_snapshot([robot]), entries, ["single_arm"], ["wrist"]
    )["observation.state"]
    live = Sample.from_driver(pose, twist, wrench)
    from_recording = Sample.from_recorded(recorded[:9], recorded[9:15], recorded[15:21])
    image = {"wrist": np.zeros((4, 4, 3), np.uint8)}
    seen = []
    for sample in (live, from_recording):
        tracker = StepTracker(_space())
        tracker.reset(sample)
        seen.append(tracker.observation(image))
    for key in (STATE, WRENCH, LAST_ACTIONS):
        np.testing.assert_allclose(seen[0][key], seen[1][key], atol=1e-5)
    # The state is the recorder's own pose and velocity columns.
    np.testing.assert_allclose(seen[0][STATE], recorded[:15], atol=1e-6)


def test_the_wrench_is_what_the_tool_applies_in_its_own_frame():
    # Tool z points down (tool turned half a turn about base x). The world
    # pushes the tool up, which is the tool pressing down: +z along the tool.
    down = Rotation.from_euler("x", 180, degrees=True).as_quat()
    pose = [0, 0, 0.3, down[3], *down[:3]]
    sample = Sample.from_driver(pose, [0] * 6, [0, 0, 20.0, 0, 0, 0])
    np.testing.assert_allclose(sample.tool_wrench(), [0, 0, 20.0, 0, 0, 0], atol=1e-9)
    sideways = Sample.from_driver(pose, [0] * 6, [3.0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(sideways.tool_wrench()[:3], [-3.0, 0, 0], atol=1e-9)


# --- steps -------------------------------------------------------------------


def test_an_action_and_the_command_it_makes_round_trip():
    rng = np.random.default_rng(1)
    space = _space()
    for _ in range(20):
        tracker = StepTracker(space)
        tracker.reset(_sample(rng))
        action = rng.uniform(-1, 1, space.action_dim)
        command = tracker.take(action)
        again, unclipped = tracker.action_toward(command)
        np.testing.assert_allclose(again, action, atol=1e-6)
        np.testing.assert_allclose(unclipped, action, atol=1e-9)


def test_a_move_is_along_the_tool_and_unlisted_axes_hold():
    # The tool is turned a quarter about base z: tool x is base y.
    quarter = Rotation.from_euler("z", 90, degrees=True).as_quat()
    pose = [0.4, 0.1, 0.2, quarter[3], *quarter[:3]]
    tracker = StepTracker(_space())
    tracker.reset(Sample.from_driver(pose, [0] * 6, [0] * 6))
    command = tracker.take([1.0, 0, 0, 0, 0, 0.5])
    np.testing.assert_allclose(command.position, [0.4, 0.102, 0.2], atol=1e-12)
    np.testing.assert_allclose(command.force, [0, 0, 25.0])
    np.testing.assert_allclose(command.pose()[3:], pose[3:], atol=1e-12)
    # A rotation about the tool's own z.
    turned = tracker.take([0, 0, 0, 0, 1.0, 0])
    expected = Rotation.from_euler("z", 91, degrees=True).as_matrix()
    np.testing.assert_allclose(turned.rotation, expected, atol=1e-12)


def test_the_driver_pose_has_one_quaternion_sign():
    tracker = StepTracker(_space())
    flipped = Rotation.from_euler("y", 170, degrees=True).as_quat() * -1
    tracker.reset(
        Sample.from_driver([0, 0, 0, flipped[3], *flipped[:3]], [0] * 6, [0] * 6)
    )
    assert tracker.take(np.zeros(6)).pose()[3] >= 0


def test_an_action_outside_its_range_is_refused_not_clamped():
    tracker = StepTracker(_space())
    tracker.reset(_sample(np.random.default_rng(2)))
    with pytest.raises(ValueError, match="leaves"):
        tracker.take([1.01, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="not finite"):
        tracker.take([np.nan, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="6 numbers"):
        tracker.take([0, 0])
    tracker.take([1 + 1e-9, 0, 0, 0, 0, -1 - 1e-9])  # float rounding is not a fault


def test_a_demonstration_beyond_the_bounds_is_clipped_and_says_by_how_much():
    rng = np.random.default_rng(3)
    tracker = StepTracker(_space())
    tracker.reset(_sample(rng))
    far = tracker.take([1.0, -0.5, 0, 0, 0, 0])
    tracker.reset(tracker.sample)
    target = Command(
        tracker.sample.position + 3 * (far.position - tracker.sample.position),
        far.rotation,
        np.array([0, 0, 80.0]),
    )
    action, unclipped = tracker.action_toward(target)
    np.testing.assert_allclose(unclipped[[0, 1, 5]], [3.0, -1.5, 1.6], atol=1e-9)
    np.testing.assert_allclose(action[[0, 1, 5]], [1.0, -1.0, 1.0])


def test_the_last_actions_are_newest_first_and_relative_to_now():
    space = _space(step_deg={}, force_n={}, last_actions=2)
    level = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    tracker = StepTracker(space)
    tracker.reset(Sample.from_driver(level, [0] * 6, [0] * 6))
    image = {"wrist": np.zeros((2, 2, 3), np.uint8)}
    # Nothing commanded yet: "stay here".
    np.testing.assert_allclose(tracker.observation(image)[LAST_ACTIONS], [0, 0, 0, 0])
    tracker.take([1.0, 0.0])
    np.testing.assert_allclose(tracker.observation(image)[LAST_ACTIONS], [1, 0, 0, 0])
    # The arm got halfway: the last command now sits half a step ahead.
    tracker.sense(Sample.from_driver([0.001, 0, 0, 1, 0, 0, 0], [0] * 6, [0] * 6))
    np.testing.assert_allclose(
        tracker.observation(image)[LAST_ACTIONS], [0.5, 0, -0.5, 0], atol=1e-9
    )
    tracker.take([0.0, -1.0])
    np.testing.assert_allclose(
        tracker.observation(image)[LAST_ACTIONS], [0, -1, 0.5, 0], atol=1e-9
    )


def test_a_hold_presses_as_the_tool_does_at_the_start():
    down = Rotation.from_euler("x", 180, degrees=True).as_quat()
    pose = [0, 0, 0.3, down[3], *down[:3]]
    tracker = StepTracker(_space(last_actions=1))
    tracker.reset(Sample.from_driver(pose, [0] * 6, [0, 0, 10.0, 0, 0, 0]))
    seen = tracker.observation({"wrist": np.zeros((2, 2, 3), np.uint8)})
    np.testing.assert_allclose(seen[LAST_ACTIONS], [0, 0, 0, 0, 0, 0.2], atol=1e-9)


def test_the_force_window_is_oldest_first_and_starts_full():
    space = _space(wrench_samples=3)
    tracker = StepTracker(space)
    level = [0, 0, 0, 1, 0, 0, 0]
    image = {"wrist": np.zeros((2, 2, 3), np.uint8)}
    tracker.reset(Sample.from_driver(level, [0] * 6, [0, 0, -1.0, 0, 0, 0]))
    window = tracker.observation(image)[WRENCH].reshape(3, 6)
    np.testing.assert_allclose(window[:, 2], [1.0, 1.0, 1.0])
    for fz in (-2.0, -3.0, -4.0):
        tracker.sense(Sample.from_driver(level, [0] * 6, [0, 0, fz, 0, 0, 0]))
    window = tracker.observation(image)[WRENCH].reshape(3, 6)
    np.testing.assert_allclose(window[:, 2], [2.0, 3.0, 4.0])


def test_the_observation_names_a_missing_camera_and_leaves_out_what_is_off():
    tracker = StepTracker(_space(views=("wrist", "ego")))
    tracker.reset(_sample(np.random.default_rng(4)))
    with pytest.raises(KeyError, match="ego"):
        tracker.observation({"wrist": np.zeros((2, 2, 3), np.uint8)})
    bare = StepTracker(_space(views=(), last_actions=0, wrench_samples=0))
    bare.reset(_sample(np.random.default_rng(5)))
    assert set(bare.observation({})) == {STATE}
    assert set(bare.space.state_dims()) == {STATE}


def test_the_named_dims_match_the_observation():
    space = _space()
    tracker = StepTracker(space)
    tracker.reset(_sample(np.random.default_rng(6)))
    seen = tracker.observation({"wrist": np.zeros((2, 2, 3), np.uint8)})
    for key, names in space.state_dims().items():
        assert seen[key].shape == (len(names),)
        assert seen[key].dtype == np.float32
    assert space.state_dims()[LAST_ACTIONS][: space.action_dim] == [
        f"{name}[-1]" for name in space.action_names
    ]
    assert space.state_dims()[WRENCH][-6:] == [
        f"{a}[t-0]" for a in ("fx", "fy", "fz", "mx", "my", "mz")
    ]


def test_a_tracker_refuses_to_work_before_it_is_reset():
    tracker = StepTracker(_space())
    with pytest.raises(RuntimeError, match="reset"):
        tracker.sense(_sample(np.random.default_rng(7)))
    with pytest.raises(RuntimeError, match="reset"):
        tracker.take(np.zeros(6))
