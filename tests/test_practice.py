"""Practice on the robot: the learner link, resets, episodes, the service."""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_step_rollout import (
    DOWN,
    LIMITS,
    START,
    _Arm,
    _Cameras,
    _service,
    _skill,
    _space,
    _Takeover,
    _wait,
    rdk,  # noqa: F401 - a fixture
)
from waits import WAIT_S

from flexivtrainer.practice.env import StepEnv
from flexivtrainer.practice.reset import ResetInterrupted, StartBox, go_to_start
from flexivtrainer.rollout.safety import SafetyEnvelope, SafetyStop

torch = pytest.importorskip("torch")


def _observation(value=0.5):
    return {
        "observation.images.wrist": np.full((4, 6, 3), 255, dtype=np.uint8),
        "observation.state": np.full(15, value, dtype=np.float32),
    }


# --- the link, against LeRobot's own learner service --------------------------------


@pytest.fixture
def learner():
    """LeRobot's learner service, served here: what the trainer talks to."""
    grpc = pytest.importorskip("grpc")
    from lerobot.rl.learner_service import LearnerService
    from lerobot.transport import services_pb2_grpc

    shutdown = threading.Event()
    service = LearnerService(
        shutdown_event=shutdown,
        parameters_queue=queue.Queue(),
        seconds_between_pushes=0.0,
        transition_queue=queue.Queue(),
        interaction_message_queue=queue.Queue(),
        queue_get_timeout=0.05,
    )
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_LearnerServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield SimpleNamespace(service=service, port=port)
    shutdown.set()
    server.stop(grace=1.0)


def _get(source: queue.Queue):
    try:
        return source.get(timeout=WAIT_S)
    except queue.Empty:
        raise AssertionError("the learner received nothing") from None


class _Policy(torch.nn.Module):
    """Shaped like LeRobot's SAC policies: an actor, maybe a gripper critic."""

    def __init__(self, action=(0.5, 0.0, 0.0, 0.2), critic=False):
        super().__init__()
        self.actor = torch.nn.Linear(3, 2)
        self.discrete_critic = torch.nn.Linear(3, 1) if critic else None
        self.config = SimpleNamespace(type="nm_step", sample_actions=False)
        self.action = np.asarray(action)
        self.resets = 0

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        return self.action


def test_steps_and_statistics_reach_the_learner_as_lerobot_reads_them(learner):
    from lerobot.transport.utils import bytes_to_python_object, bytes_to_transitions

    from flexivtrainer.practice.learner import LearnerLink, transition

    link = LearnerLink("127.0.0.1", learner.port)
    link.connect(threading.Event(), attempts=5, wait_s=0.5)
    link.start()
    steps = [
        transition(
            _observation(0.1),
            [0.5, -1.0],
            0.0,
            _observation(0.2),
            done=False,
            truncated=False,
            intervention=True,
        ),
        transition(
            _observation(0.2),
            [0.1, 0.3],
            1.0,
            _observation(0.3),
            done=True,
            truncated=False,
            intervention=False,
        ),
    ]
    link.send_episode(steps)
    link.send_stats({"Episodic reward": 1.0, "Interaction step": 2})
    received = bytes_to_transitions(_get(learner.service.transition_queue))
    assert len(received) == 2
    image = received[0]["state"]["observation.images.wrist"]
    assert image.shape == (1, 3, 4, 6) and image.dtype == torch.float32
    assert float(image.max()) == 1.0
    assert received[0]["state"]["observation.state"].shape == (1, 15)
    assert received[0]["action"].tolist() == [[0.5, -1.0]]
    assert received[0]["complementary_info"]["is_intervention"] is True
    assert received[1]["reward"] == 1.0 and received[1]["done"] is True
    stats = bytes_to_python_object(_get(learner.service.interaction_message_queue))
    assert stats == {"Episodic reward": 1.0, "Interaction step": 2}
    link.close()
    assert link.error is None


def test_an_episode_sent_just_before_the_link_closes_still_arrives(learner):
    from lerobot.transport.utils import bytes_to_transitions

    from flexivtrainer.practice.learner import LearnerLink, transition

    link = LearnerLink("127.0.0.1", learner.port)
    link.connect(threading.Event(), attempts=5, wait_s=0.5)
    link.start()
    last = transition(
        _observation(),
        [0.0, 0.0],
        1.0,
        _observation(),
        done=True,
        truncated=False,
        intervention=False,
    )
    link.send_episode([last])
    link.close()
    assert len(bytes_to_transitions(_get(learner.service.transition_queue))) == 1


def test_the_newest_weights_the_learner_pushes_are_loaded(learner):
    from lerobot.transport.utils import state_to_bytes

    from flexivtrainer.practice.learner import LearnerLink, apply_weights

    link = LearnerLink("127.0.0.1", learner.port)
    link.connect(threading.Event(), attempts=5, wait_s=0.5)
    link.start()
    trained = _Policy(critic=True)
    for value in (1.0, 2.0):  # two pushes; the newer must win
        with torch.no_grad():
            for p in trained.parameters():
                p.fill_(value)
        learner.service.parameters_queue.put(
            state_to_bytes(
                {
                    "policy": trained.actor.state_dict(),
                    "discrete_critic": trained.discrete_critic.state_dict(),
                }
            )
        )
        time.sleep(0.2)
    deadline = time.monotonic() + WAIT_S
    weights = None
    while weights is None and time.monotonic() < deadline:
        weights = link.latest_weights()
        time.sleep(0.02)
    assert weights is not None
    acting = _Policy(critic=True)
    apply_weights(acting, weights)
    assert all(float(p.detach().min()) == 2.0 for p in acting.parameters())
    assert link.latest_weights() is None  # nothing newer since
    link.close()


def test_a_learner_that_never_answers_is_named():
    pytest.importorskip("grpc")
    from flexivtrainer.practice.learner import LearnerLink

    link = LearnerLink("127.0.0.1", 9)
    with pytest.raises(RuntimeError, match="no learner answered at 127.0.0.1:9"):
        link.connect(threading.Event(), attempts=2, wait_s=0.05)
    link.close()


# --- the reset -----------------------------------------------------------------------


def _box(**overrides):
    values = {
        "min_m": (0.39, -0.01, 0.30),
        "max_m": (0.41, 0.01, 0.31),
        "yaw_deg": 5.0,
        "retreat_m": 0.02,
        "speed_m_s": 0.04,
        "timeout_s": 1.0,
    }
    return StartBox(**{**values, **overrides})


def _reset_env(arm):
    return StepEnv(
        robot=arm,
        cameras=_Cameras(["wrist"]),
        camera_names=["wrist"],
        space=_space(),
        motion_limits=LIMITS,
        stop_event=threading.Event(),
    )


def _rotation(pose):
    w, x, y, z = pose[3:]
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def test_a_reset_backs_off_along_the_tool_then_goes_to_a_start_in_the_box():
    arm = _Arm()
    env = _reset_env(arm)
    reference = _rotation(START)
    position, rotation = go_to_start(
        _box(),
        read=env.read,
        send=env.send_pose,
        stop=threading.Event(),
        rng=np.random.default_rng(0),
        rotation=reference,
    )
    (back, *_), (start, *_) = arm.commands
    # The tool points down, so backing off along it goes up.
    np.testing.assert_allclose(back[:3], [0.4, 0.0, 0.32], atol=1e-12)
    np.testing.assert_allclose(start[:3], position)
    assert all(lo <= v <= hi for v, lo, hi in zip(position, _box().min_m, _box().max_m))
    yaw = Rotation.from_matrix(reference.T @ rotation).as_rotvec(degrees=True)
    assert abs(yaw[2]) <= 5.0 and abs(yaw[0]) < 1e-9 and abs(yaw[1]) < 1e-9
    # Slowly, pressing with nothing.
    assert arm.commands[1][1] == [0.0] * 6 and arm.commands[1][3][0] == 0.04


def test_a_reset_that_never_arrives_fails_naming_how_far_it_is():
    arm = _Arm(follows=False)
    env = _reset_env(arm)
    with pytest.raises(
        RuntimeError, match=r"did not reach the start within 0.1 s.*20.0 mm"
    ):
        go_to_start(
            _box(timeout_s=0.1),
            read=env.read,
            send=env.send_pose,
            stop=threading.Event(),
            rng=np.random.default_rng(0),
            rotation=_rotation(START),
        )


def test_a_reset_stops_when_the_arm_is_taken_or_the_start_is_outside_the_envelope():
    env = _reset_env(_Arm())
    env.pause()
    with pytest.raises(ResetInterrupted):
        go_to_start(
            _box(),
            read=env.read,
            send=env.send_pose,
            stop=threading.Event(),
            rng=np.random.default_rng(0),
            rotation=_rotation(START),
        )
    env = _reset_env(_Arm())
    box = SafetyEnvelope(
        workspace_min_m=(0.0, -1.0, 0.0), workspace_max_m=(1.0, 1.0, 0.315)
    )
    with pytest.raises(SafetyStop, match="workspace"):
        go_to_start(
            _box(),
            read=env.read,
            send=env.send_pose,
            stop=threading.Event(),
            rng=np.random.default_rng(0),
            rotation=_rotation(START),
            safety=box,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"min_m": (0.5, 0.0, 0.3)}, "not a box"),
        ({"max_m": (0.4, 0.0)}, r"\[x, y, z\]"),
        ({"yaw_deg": -1.0}, "0 or more"),
        ({"speed_m_s": 0.0}, "positive"),
    ],
)
def test_a_wrong_start_box_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        _box(**change)


# --- the plan -----------------------------------------------------------------------


REWARD = {"seated": {"kind": "learned", "model": "checks/seated", "hold_s": 0.1}}


def test_a_plan_is_rewarded_by_one_learned_check_that_ends_the_episode():
    from flexivtrainer.practice.session import PracticePlan

    plan = PracticePlan.from_dict(
        {"reward": REWARD, "max_episode_steps": 50, "start": _box().to_dict()}
    )
    assert plan.check_name == "seated"
    assert plan.check.end_on_pass and plan.check.live and plan.check.hold_s == 0.1
    assert plan.start == _box()
    with pytest.raises(ValueError, match="one check, not 2"):
        PracticePlan(reward={**REWARD, "b": REWARD["seated"]}, max_episode_steps=5)
    with pytest.raises(ValueError, match="must be a learned check"):
        PracticePlan(
            reward={
                "box": {
                    "kind": "settled_in_box",
                    "box_min_m": [0, 0, 0],
                    "box_max_m": [1, 1, 1],
                }
            },
            max_episode_steps=5,
        )
    with pytest.raises(ValueError, match="needs max_episode_steps"):
        PracticePlan.from_dict({"reward": REWARD})
    with pytest.raises(ValueError, match="unknown practice settings"):
        PracticePlan.from_dict({"reward": REWARD, "max_episode_steps": 5, "speed": 1})


# --- episodes ------------------------------------------------------------------------


class _Check:
    """The reward check: passes once it has scored `after` ticks of an episode."""

    def __init__(self, after=None):
        self.after = after
        self.scored = 0
        self.resets = 0
        self.stopped = False

    def __bool__(self):
        return True

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def reset(self):
        self.resets += 1
        self.scored = 0

    def submit(self, **tick):
        self.scored += 1

    def values(self):
        return {}

    def ended(self):
        return (
            "seated" if self.after is not None and self.scored >= self.after else None
        )


class _Link:
    """A learner as the practice loop sees it, recording what it is sent."""

    address = "test"

    def __init__(self, weights=None):
        self.episodes: list[list[dict]] = []
        self.stats: list[dict] = []
        self.weights = list(weights or [])
        self.error = None
        self.closed = False

    def connect(self, stop):
        pass

    def start(self):
        pass

    def send_episode(self, steps):
        self.episodes.append(list(steps))

    def send_stats(self, stats):
        self.stats.append(dict(stats))

    def latest_weights(self):
        return self.weights.pop(0) if self.weights else None

    def close(self):
        self.closed = True


def _practice(plan, *, arm=None, check=None, link=None, takeover=None, policy=None):
    from flexivtrainer.practice.session import PracticeRunner

    finished = threading.Event()
    reasons = []
    runner = PracticeRunner(
        plan=plan,
        link=link or _Link(),
        policy=policy or _Policy(),
        preprocessor=lambda v: v,
        postprocessor=lambda v: v,
        robots=[arm or _Arm()],
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
        on_error=reasons.append,
        on_cleanup_error=reasons.append,
        on_finished=lambda reason, steps: (
            reasons.append((reason, steps)),
            finished.set(),
        ),
        release_robots=lambda: None,
        stop_robots=lambda robots: None,
        live_checks=check or _Check(after=7),
        takeover=takeover,
    )
    return runner, finished, reasons


def _plan(**overrides):
    from flexivtrainer.practice.session import PracticePlan

    values = {
        "reward": REWARD,
        "max_episode_steps": 5,
        "episodes": 2,
        "wait_after_success": False,
    }
    return PracticePlan(**{**values, **overrides})


def test_each_episode_ends_where_its_check_passes_and_goes_to_the_learner(rdk):  # noqa: F811
    link = _Link()
    policy = _Policy()
    runner, finished, reasons = _practice(_plan(), link=link, policy=policy)
    assert policy.config.sample_actions is True  # practice explores
    runner.start()
    assert finished.wait(WAIT_S)
    assert reasons == [("episodes done", 4)]
    # Seven ticks: the start, then two steps of three.
    assert [len(e) for e in link.episodes] == [2, 2]
    for episode in link.episodes:
        assert [s["reward"] for s in episode] == [0.0, 1.0]
        assert [s["done"] for s in episode] == [False, True]
        assert not any(s["truncated"] for s in episode)
        np.testing.assert_allclose(episode[0]["action"], [[0.5, 0.0, 0.0, 0.2]])
        assert episode[1]["state"]["observation.last_actions"].shape == (1, 8)
    assert [s["Episodic reward"] for s in link.stats] == [1.0, 1.0]
    assert [s["Interaction step"] for s in link.stats] == [2, 4]
    status = runner.practice_status()
    assert status["episodes"] == 2 and status["successes"] == 2
    assert status["state"] == "finished" and link.closed


def test_an_episode_without_success_is_cut_short_at_its_step_limit(rdk):  # noqa: F811
    link = _Link()
    runner, finished, _ = _practice(
        _plan(episodes=1), link=link, check=_Check(after=None)
    )
    runner.start()
    assert finished.wait(WAIT_S)
    (episode,) = link.episodes
    assert len(episode) == 5
    assert [s["truncated"] for s in episode] == [False] * 4 + [True]
    assert (
        not any(s["done"] for s in episode) and sum(s["reward"] for s in episode) == 0
    )


def test_the_newest_weights_are_loaded_between_episodes(rdk):  # noqa: F811
    trained = _Policy()
    with torch.no_grad():
        for p in trained.parameters():
            p.fill_(3.0)
    policy = _Policy()
    link = _Link(weights=[{"policy": trained.actor.state_dict()}])
    runner, finished, _ = _practice(_plan(episodes=1), link=link, policy=policy)
    runner.start()
    assert finished.wait(WAIT_S)
    assert all(float(p.detach().min()) == 3.0 for p in policy.actor.parameters())
    assert runner.practice_status()["weights_loaded"] == 1


def test_each_episode_starts_with_a_reset_with_every_axis_motion_controlled(rdk):  # noqa: F811
    arm = _Arm()
    runner, finished, _ = _practice(_plan(start=_box()), arm=arm)
    runner.start()
    assert finished.wait(WAIT_S)
    resets = [c for c in arm.commands if c[3][0] == 0.04]
    assert len(resets) == 4  # back off, then the start, per episode
    clear = ("axes", [False] * 6, [0.02] * 3)
    z_force = ("axes", [False, False, True, False, False, False], [0.02] * 3)
    force_calls = [c for c in arm.calls if c[0] == "axes"]
    # Set at the start; around each reset cleared, then set again; cleared at the end.
    assert force_calls == [z_force, clear, z_force, clear, z_force, clear]


def test_the_operators_steps_are_marked_as_interventions(rdk):  # noqa: F811
    arm = _Arm()
    arm.commanded = [0.4, -0.001, 0.3, *DOWN]  # the leader holds 1 mm along tool y
    takeover = _Takeover()
    takeover.who = "operator"
    link = _Link()
    runner, finished, _ = _practice(
        _plan(episodes=1), arm=arm, link=link, takeover=takeover, policy=_Policy()
    )
    runner.start()
    assert finished.wait(WAIT_S)
    (episode,) = link.episodes
    assert all(s["complementary_info"]["is_intervention"] for s in episode)
    assert episode[0]["action"][0, 1] == pytest.approx(0.5)
    assert link.stats[0]["Intervention rate"] == 1.0
    assert arm.commands == []  # the policy sent nothing


def test_after_a_success_it_waits_for_the_operator(rdk):  # noqa: F811
    link = _Link()
    runner, finished, _ = _practice(_plan(wait_after_success=True), link=link)
    runner.start()
    _wait(
        lambda: runner.practice_status()["state"] == "waiting for the operator",
        "it never waited",
    )
    time.sleep(0.05)
    assert len(link.episodes) == 1
    runner.go_on()
    _wait(lambda: len(link.episodes) == 2, "it did not go on")
    runner.go_on()
    assert finished.wait(WAIT_S)


# --- through the service ------------------------------------------------------------


def test_the_service_practises_a_step_skill_with_a_real_link(
    tmp_path,
    rdk,  # noqa: F811 - a fixture
    learner,
    monkeypatch,
):
    from lerobot.transport.utils import bytes_to_transitions

    from flexivtrainer.practice.session import PracticePlan

    check = _Check(after=7)
    monkeypatch.setattr(
        "flexivtrainer.rollout.service.LiveChecks", lambda *a, **k: check
    )
    arm = _Arm()
    service, _ = _service(tmp_path, arm)
    plan = PracticePlan(
        reward=REWARD,
        max_episode_steps=5,
        episodes=1,
        learner_port=learner.port,
        wait_after_success=False,
    )
    service.start(_skill(tmp_path), practice=plan)
    deadline = time.monotonic() + WAIT_S
    while service.status()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    status = service.status()
    assert status["stop_reason"] == "episodes done", status["error"]
    assert status["practice"]["episodes"] == 1 and status["practice"]["successes"] == 1
    received = bytes_to_transitions(_get(learner.service.transition_queue))
    assert [t["reward"] for t in received] == [0.0, 1.0]
    assert check.stopped
    assert arm.calls[-1][:2] == ("axes", [False] * 6)


def test_practice_needs_a_step_skill_and_no_recording(tmp_path, rdk):  # noqa: F811
    from flexivtrainer.practice.session import PracticePlan

    plan = PracticePlan(reward=REWARD, max_episode_steps=5)
    service, _ = _service(tmp_path, _Arm())
    with pytest.raises(ValueError, match="without recording"):
        service.start(_skill(tmp_path), practice=plan, record=True)
    plain = tmp_path / "plain" / "pretrained_model"
    plain.mkdir(parents=True)
    (plain / "config.json").write_text('{"type": "nm_step"}')
    with pytest.raises(RuntimeError, match="needs a step skill"):
        service.start(str(tmp_path / "plain"), practice=plan)
    with pytest.raises(RuntimeError, match="No practice is running"):
        service.practice_continue()
