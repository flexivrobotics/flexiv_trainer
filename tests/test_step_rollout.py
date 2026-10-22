"""Rolling a step skill out: one command per step, sensed at the space's rate."""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_rollout_service import _FakeCameras, _FakePolicy, _settings, _teleop
from waits import WAIT_S

from flexivtrainer.config import TeleopRobotPair
from flexivtrainer.practice.env import StepEnv
from flexivtrainer.practice.runner import StepRunner, step_action_names
from flexivtrainer.practice.space import PracticeSpace, write_space
from flexivtrainer.rollout.safety import SafetyEnvelope, SafetyStop
from flexivtrainer.rollout.service import RolloutService

SIZE = 16
#: The tool points down: half a turn about base x. Tool x is base x, tool y
#: is base -y and tool z is base -z.
DOWN = [0.0, 1.0, 0.0, 0.0]  # qw, qx, qy, qz
START = [0.4, 0.0, 0.3, *DOWN]
LIMITS = (0.25, 0.6, 1.0, 2.5)


def _space(**overrides):
    values = {
        "rate_hz": 100,
        "step_m": {"x": 0.002, "y": 0.002},
        "step_deg": {"z": 1.0},
        "force_n": {"z": 50.0},
        "views": ("wrist",),
        "last_actions": 2,
        "wrench_samples": 3,
        "sense_hz": 300,
    }
    return PracticeSpace(**{**values, **overrides})


class _Arm:
    """A follower that goes where it is sent (unless told to stay)."""

    def __init__(self, *, follows: bool = True) -> None:
        self.pose = list(START)
        self.wrench = [0.0] * 6
        self.follows = follows
        #: What the controller is commanding: set by a test as the leader's.
        self.commanded: list[float] | None = None
        self.commands: list[tuple] = []
        self.calls: list[tuple] = []
        self.primitives: list = []
        self.enabled = False

    def fault(self) -> bool:
        return False

    def ClearFault(self) -> bool:  # noqa: N802 - RDK API names
        return True

    def Enable(self) -> None:  # noqa: N802
        self.enabled = True

    def operational(self) -> bool:
        return self.enabled

    def SwitchMode(self, mode) -> None:  # noqa: N802
        self.calls.append(("mode", mode))

    def ExecutePrimitive(self, name, params) -> None:  # noqa: N802
        self.primitives.append(name)

    def primitive_states(self) -> dict:
        return {"reachedTarget": 1}

    def busy(self) -> bool:
        return False

    def SetCartesianImpedance(self, values) -> None:  # noqa: N802
        self.calls.append(("stiffness", list(values)))

    def SetMaxContactWrench(self, values) -> None:  # noqa: N802
        self.calls.append(("contact", list(values)))

    def SetForceControlFrame(self, coord, transform) -> None:  # noqa: N802
        self.calls.append(("frame", coord, list(transform)))

    def SetForceControlAxis(self, axes, speeds) -> None:  # noqa: N802
        self.calls.append(("axes", list(axes), list(speeds)))

    def Stop(self) -> None:  # noqa: N802
        pass

    def states(self):
        return SimpleNamespace(
            tcp_pose=list(self.pose),
            tcp_vel=[0.0] * 6,
            ext_wrench_in_world=list(self.wrench),
        )

    def actions(self):
        return SimpleNamespace(
            tcp_pose_d=list(self.commanded or self.pose),
            tcp_vel_d=[0.0] * 6,
            ext_wrench_d=[0.0] * 6,
        )

    def SendCartesianMotionForce(self, pose, wrench, twist, *limits):  # noqa: N802
        self.commands.append((list(pose), list(wrench), list(twist), limits))
        if self.follows:
            self.pose = list(pose)


class _Cameras(_FakeCameras):
    """Frames whose channels say which is which: blue 10, green 20, red 30."""

    def capture_frame(self, name, **kwargs):
        image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        image[..., 0], image[..., 1], image[..., 2] = 10, 20, 30
        return {"image": image}


def _rotation(pose):
    w, x, y, z = pose[3:]
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _env(*, arm=None, space=None, ticks=None, **kwargs):
    arm = arm or _Arm()
    seen = [] if ticks is None else ticks
    env = StepEnv(
        robot=arm,
        cameras=_Cameras(["wrist"]),
        camera_names=["wrist"],
        space=space or _space(),
        motion_limits=LIMITS,
        stop_event=threading.Event(),
        on_tick=seen.append,
        **kwargs,
    )
    return env, arm, seen


# --- the environment ------------------------------------------------------------


def test_a_step_sends_one_command_toward_the_action_and_senses_until_the_next():
    env, arm, ticks = _env(arm=_Arm(follows=False))
    env.reset()
    result = env.step([1.0, 0.0, 0.0, 0.5])
    assert len(arm.commands) == 1
    pose, wrench, twist, limits = arm.commands[0]
    # +2 mm along the tool's x, which is base x; the rotation is kept.
    np.testing.assert_allclose(pose[:3], [0.402, 0.0, 0.3], atol=1e-12)
    np.testing.assert_allclose(_rotation(pose), _rotation(START), atol=1e-12)
    assert twist == [0.0] * 6 and limits == LIMITS
    assert result.sent and not result.intervention
    np.testing.assert_allclose(result.action, [1.0, 0.0, 0.0, 0.5])
    # The reading at the reset, then one per sample of the step.
    assert len(ticks) == 1 + 3
    assert [t.fresh for t in ticks] == [False, True, False, False]
    np.testing.assert_allclose(np.diff([t.t for t in ticks]), 1 / 300, atol=1e-9)
    # The arm stayed put, so the last command still sits one full step ahead.
    np.testing.assert_allclose(
        result.observation["observation.last_actions"][:4], [1, 0, 0, 0.5], atol=1e-9
    )


def test_the_force_target_goes_out_as_the_force_the_sensor_should_read():
    """The tool is to apply +25 N along its z; the controller is given the
    force the TCP should sense, -25 N, in a frame on the TCP."""
    env, arm, _ = _env()
    env.reset()
    env.step([0.0, 0.0, 0.0, 0.5])
    np.testing.assert_allclose(arm.commands[0][1], [0, 0, -25.0, 0, 0, 0])


def test_each_step_is_measured_from_where_the_arm_now_is():
    env, arm, _ = _env()
    env.reset()
    for _ in range(2):
        result = env.step([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(arm.commands[1][0][:3], [0.404, 0.0, 0.3], atol=1e-12)
    # The arm reached the last target: it no longer sits ahead.
    np.testing.assert_allclose(
        result.observation["observation.last_actions"][:4], 0, atol=1e-9
    )


def test_an_operator_step_is_measured_like_a_demonstration():
    arm = _Arm(follows=False)
    # The leader takes the tool 1 mm along its y (base -y); the tool presses
    # along its z with 10 N (the world pushes the tool up by 10 N).
    arm.commanded = [0.4, -0.001, 0.3, *DOWN]
    arm.wrench = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    env, arm, ticks = _env(arm=arm, control=lambda: "operator")
    env.reset()
    result = env.step([1.0, 0.0, 0.0, 0.0])
    assert arm.commands == []
    assert not result.sent and result.intervention
    np.testing.assert_allclose(result.action, [0.0, 0.5, 0.0, 0.2], atol=1e-9)
    np.testing.assert_allclose(
        result.observation["observation.last_actions"][:4], result.action, atol=1e-6
    )
    assert {t.control for t in ticks} == {"operator"}


def test_nothing_is_sent_while_paused_or_from_a_plan_older_than_a_resume():
    env, arm, _ = _env()
    env.reset()
    epoch = env.epoch
    env.pause()
    assert not env.step([1.0, 0, 0, 0]).sent
    env.resume()
    stale = env.step([1.0, 0, 0, 0], epoch=epoch)
    assert not stale.sent and stale.intervention
    assert arm.commands == []
    assert env.step([1.0, 0, 0, 0], epoch=env.epoch).sent
    assert len(arm.commands) == 1


def test_the_envelope_stops_a_step_that_leaves_the_workspace_before_it_is_sent():
    box = SafetyEnvelope(
        workspace_min_m=(0.3, -0.1, 0.2), workspace_max_m=(0.401, 0.1, 0.4)
    )
    env, arm, _ = _env(safety=box)
    env.reset()
    with pytest.raises(SafetyStop, match="workspace"):
        env.step([1.0, 0, 0, 0])
    assert arm.commands == []


def test_the_envelope_stops_on_a_measured_force():
    arm = _Arm()
    arm.wrench = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
    env, _, _ = _env(arm=arm, safety=SafetyEnvelope(max_force_n=15.0))
    with pytest.raises(SafetyStop, match="force"):
        env.reset()


def test_a_camera_the_skill_sees_must_be_streamed():
    with pytest.raises(ValueError, match=r"camera \['ego'\]"):
        _env(space=_space(views=("wrist", "ego")))


# --- the runner, through the rollout service ----------------------------------------


@pytest.fixture
def rdk(monkeypatch):
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(
            NRT_PRIMITIVE_EXECUTION="prim", NRT_CARTESIAN_MOTION_FORCE="cmf"
        ),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_coord_type",
        lambda: SimpleNamespace(TCP="tcp", WORLD="world"),
    )
    seen: list[dict] = []

    def predict(observation, policy, device, pre, post, task):
        seen.append(observation)
        return np.asarray(policy.select_action(observation), dtype=np.float64)

    monkeypatch.setattr("flexivtrainer.practice.runner._predict", predict)
    return seen


def _skill(tmp_path, space=None):
    model = tmp_path / "skill" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "type": "nm_step",
                "input_features": {
                    "observation.images.wrist": {
                        "type": "VISUAL",
                        "shape": [3, SIZE, SIZE],
                    }
                },
            }
        )
    )
    write_space(model, space or _space(), checkpoint=True)
    return str(tmp_path / "skill")


def _service(tmp_path, arm, *, action=(0.5, 0.0, 0.0, 0.2), sides=("single_arm",)):
    settings = _settings(tmp_path)
    settings.ensure_storage()
    policy = _FakePolicy(list(action))
    policy.config.type = "nm_step"
    followers = [f"F{i}" for i in range(len(sides))]
    service = RolloutService(
        settings,
        _Cameras(["ego", "wrist"]),
        _teleop(initialized=False),
        lambda: [
            TeleopRobotPair(leader_serial=f"L{f}", follower_serial=f) for f in followers
        ],
        lambda: list(sides),
        policy_loader=lambda path, device: (policy, lambda v: v, lambda v: v),
        robot_factory=lambda serial: arm,
        resolve_device=lambda configured: "cpu",
        get_active_cameras=lambda: ["ego", "wrist"],
    )
    return service, policy


def _wait_until_done(service):
    deadline = time.monotonic() + WAIT_S
    while service.status()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service.status()["status"] != "running"


def test_a_checkpoint_with_a_practice_space_rolls_out_as_a_step_skill(tmp_path, rdk):
    arm = _Arm()
    service, policy = _service(tmp_path, arm)
    service._settings.rollout.max_steps = 4
    service._settings.rollout.force_axis_max_speed_m_s = 0.05
    service.start(_skill(tmp_path))
    _wait_until_done(service)

    status = service.status()
    assert status["stop_reason"] == "timeout"
    assert len(arm.commands) == 4
    # Each step 1 mm on from the last, pressing with 10 N along the tool.
    np.testing.assert_allclose(
        [c[0][0] for c in arm.commands], [0.401, 0.402, 0.403, 0.404], atol=1e-12
    )
    assert all(c[1] == pytest.approx([0, 0, -10.0, 0, 0, 0]) for c in arm.commands)
    # Force control on the tool's z, set before the first step, cleared after.
    force_calls = [c for c in arm.calls if c[0] in ("frame", "axes")]
    assert force_calls[0] == ("frame", "tcp", [0, 0, 0, 1, 0, 0, 0])
    assert force_calls[1] == (
        "axes",
        [False, False, True, False, False, False],
        [0.05] * 3,
    )
    assert force_calls[-1][:2] == ("axes", [False] * 6)
    # What the policy saw: its own space, the camera in RGB order.
    assert set(rdk[0]) == {
        "observation.images.wrist",
        "observation.state",
        "observation.wrench",
        "observation.last_actions",
    }
    assert rdk[0]["observation.images.wrist"][0, 0].tolist() == [30, 20, 10]
    assert rdk[0]["observation.wrench"].shape == (18,)
    messages = " ".join(json.dumps(line) for line in service.status()["logs"])
    assert "Step skill loaded" in messages and "100 steps/s" in messages


def test_a_step_skill_drives_one_arm(tmp_path, rdk):
    service, _ = _service(tmp_path, _Arm(), sides=("left_arm", "right_arm"))
    with pytest.raises(RuntimeError, match="drives one arm"):
        service.start(_skill(tmp_path))


def test_a_step_skill_needs_the_cameras_it_sees(tmp_path, rdk):
    service, _ = _service(tmp_path, _Arm())
    with pytest.raises(RuntimeError, match=r"camera \['side'\]"):
        service.start(_skill(tmp_path, _space(views=("wrist", "side"))))


def test_a_recorded_step_rollout_is_a_teleop_recording(tmp_path, rdk):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    arm = _Arm()
    service, _ = _service(tmp_path, arm)
    service._settings.rollout.max_steps = 3
    service.start(_skill(tmp_path), record=True)
    deadline = time.monotonic() + 30.0
    while service.status()["awaiting_verdict"] is None:
        assert time.monotonic() < deadline, service.status()["recording"]
        time.sleep(0.02)
    root = tmp_path / "rollouts" / "adhoc"
    (episode,) = list(root.iterdir())
    dataset = LeRobotDataset(f"local/{episode.name}", root=episode)
    assert dataset.features["action"]["names"] == step_action_names("single_arm")
    assert dataset.features["observation.state"]["names"][0] == "single_arm.tcp_pose.x"
    # The reading at the start, then three per step.
    assert dataset.meta.total_frames == 1 + 3 * 3
    actions = np.asarray(dataset.hf_dataset.with_format("numpy")["action"])
    # Recorded as the pose commanded: the start, then each step's target.
    np.testing.assert_allclose(
        actions[:, 0], [0.4] + [0.401] * 3 + [0.402] * 3 + [0.403] * 3, atol=1e-6
    )
    ticks = [
        json.loads(line)
        for line in (episode / "meta" / "rollout_ticks.jsonl").read_text().splitlines()
    ]
    assert sum(t["fresh"] for t in ticks) == 3


# --- the takeover, on the runner ------------------------------------------------------


class _Takeover:
    """Who has the arm, as a test says."""

    def __init__(self) -> None:
        self.who = "policy"
        self.closed = False

    def control(self) -> str:
        return self.who

    def gripper_targets(self) -> dict:
        return {}

    def drain_events(self) -> list:
        return []

    def close(self) -> None:
        self.closed = True


def _wait(condition, what):
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.005)


def test_a_takeover_stops_the_skill_and_a_hand_back_resumes_it_afresh(rdk):
    arm = _Arm()
    takeover = _Takeover()
    policy = _FakePolicy([0.5, 0.0, 0.0, 0.0])
    finished = threading.Event()
    stop = threading.Event()
    runner = StepRunner(
        policy=policy,
        preprocessor=lambda v: v,
        postprocessor=lambda v: v,
        robots=[arm],
        sides=["single_arm"],
        followers=["F1"],
        cameras=_Cameras(["wrist"]),
        camera_names=["wrist"],
        space=_space(),
        device="cpu",
        task=None,
        motion_limits=LIMITS,
        max_steps=0,
        stop_event=stop,
        append_log=lambda *a: None,
        append_metric=lambda m: None,
        on_error=lambda d: None,
        on_cleanup_error=lambda d: None,
        on_finished=lambda reason, steps: finished.set(),
        release_robots=lambda: None,
        stop_robots=lambda robots: None,
        takeover=takeover,
    )
    assert arm.primitives == ["ZeroFTSensor"]
    runner.start()
    _wait(lambda: runner.ready, "the skill never sent a step")

    runner.pause_policy()
    takeover.who = "operator"
    sent = len(arm.commands)
    assert arm.calls[-1] == ("axes", [False] * 6, [0.02] * 3)
    time.sleep(0.05)
    assert len(arm.commands) == sent  # nothing reaches the arm under the operator

    takeover.who = "switching_to_policy"
    runner.reclaim()
    assert arm.calls[-1] == (
        "axes",
        [False, False, True, False, False, False],
        [0.02] * 3,
    )
    assert arm.primitives == ["ZeroFTSensor"]  # not zeroed again: it may be in contact
    resets = policy.reset_count
    runner.resume_policy()
    takeover.who = "policy"
    _wait(lambda: len(arm.commands) > sent, "the skill never stepped again")
    assert policy.reset_count == resets + 1

    stop.set()
    runner.stop()
    assert finished.wait(WAIT_S)
    assert takeover.closed
    assert arm.calls[-1] == ("axes", [False] * 6, [0.02] * 3)


def test_force_axes_are_cleared_once_when_a_trial_ends_under_the_operator(rdk):
    """The takeover cleared them; the end of the trial does not try again, as
    the arm may be in a mode that refuses the call."""
    arm = _Arm()
    takeover = _Takeover()
    finished = threading.Event()
    stop = threading.Event()
    errors: list[str] = []
    runner = StepRunner(
        policy=_FakePolicy([0.5, 0.0, 0.0, 0.0]),
        preprocessor=lambda v: v,
        postprocessor=lambda v: v,
        robots=[arm],
        sides=["single_arm"],
        followers=["F1"],
        cameras=_Cameras(["wrist"]),
        camera_names=["wrist"],
        space=_space(),
        device="cpu",
        task=None,
        motion_limits=LIMITS,
        max_steps=0,
        stop_event=stop,
        append_log=lambda *a: None,
        append_metric=lambda m: None,
        on_error=errors.append,
        on_cleanup_error=errors.append,
        on_finished=lambda reason, steps: finished.set(),
        release_robots=lambda: None,
        stop_robots=lambda robots: None,
        takeover=takeover,
    )
    runner.start()
    _wait(lambda: runner.ready, "the skill never sent a step")
    runner.pause_policy()
    takeover.who = "operator"
    cleared = arm.calls.count(("axes", [False] * 6, [0.02] * 3))

    def refuse(axes, speeds):
        raise RuntimeError("Robot is not in an applicable control mode")

    arm.SetForceControlAxis = refuse
    stop.set()
    runner.stop()
    assert finished.wait(WAIT_S)
    assert cleared == 1 and errors == []


class _Check:
    """A live check that passes once it has scored `after` ticks."""

    def __init__(self, after: int) -> None:
        self.after = after
        self.scored = 0

    def submit(self, **tick) -> None:
        assert tick["state"][0] == pytest.approx(
            tick["snapshot"]["robots"]["robot_0"]["states"]["tcp_pose"][0]
        )
        self.scored += 1

    def values(self) -> dict:
        return {"seated": min(1.0, self.scored / self.after)}

    def ended(self) -> str | None:
        return "seated" if self.scored >= self.after else None


def test_a_live_check_that_passes_ends_the_skill(rdk):
    arm = _Arm()
    check = _Check(after=7)
    reasons = []
    runner = StepRunner(
        policy=_FakePolicy([0.5, 0.0, 0.0, 0.0]),
        preprocessor=lambda v: v,
        postprocessor=lambda v: v,
        robots=[arm],
        sides=["single_arm"],
        followers=["F1"],
        cameras=_Cameras(["wrist"]),
        camera_names=["wrist"],
        space=_space(),
        device="cpu",
        task=None,
        motion_limits=LIMITS,
        max_steps=0,
        stop_event=threading.Event(),
        append_log=lambda *a: None,
        append_metric=lambda m: None,
        on_error=lambda d: None,
        on_cleanup_error=lambda d: None,
        on_finished=lambda reason, steps: reasons.append((reason, steps)),
        release_robots=lambda: None,
        stop_robots=lambda robots: None,
        live_checks=check,
    )
    runner.start()
    _wait(lambda: reasons, "the skill never ended")
    # Seven ticks: the reset and two steps of three.
    assert reasons == [("check seated passed", 2)]
    assert len(arm.commands) == 2


def test_the_robot_reads_frames_at_the_size_the_strips_refer_to(tmp_path, rdk):
    """The cameras give 16 x 16; the skill's frames are 8 x 8, cut to one strip
    and resized to 4 x 2."""
    arm = _Arm()
    service, _ = _service(tmp_path, arm)
    service._settings.rollout.max_steps = 1
    space = _space(
        frame_size={"wrist": (8, 8)}, crops={"wrist": [(0, 0, 4, 8)]}, image_size=(4, 2)
    )
    service.start(_skill(tmp_path, space))
    _wait_until_done(service)
    assert rdk[0]["observation.images.wrist"].shape == (4, 2, 3)
    assert rdk[0]["observation.images.wrist"][0, 0].tolist() == [30, 20, 10]
