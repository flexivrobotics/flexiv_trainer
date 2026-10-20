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

"""Recorded rollouts, verdicts, evaluation sessions and the safety envelope.

The loop tests drive the real RolloutService with the fake robot, cameras and
policy of test_rollout_service, so what is asserted is what an operator gets:
a dataset on disk in the teleoperation schema, an outcome waiting for a verdict,
a blind trial that does not show its checkpoint, and a rollout the envelope
stops.
"""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_rollout_service import (
    _cameras,
    _checkpoint,
    _checkpoint_of_type,
    _fake_loader,
    _FakeBSplinePolicy,
    _FakePolicy,
    _FakeRobot,
    _settings,
    _single_arm_pairs,
    _teleop,
)
from waits import WAIT_S

from flexivtrainer.evaluation.outcome import (
    RolloutOutcome,
    Verdict,
    read_outcome,
    write_outcome,
)
from flexivtrainer.evaluation.session import (
    EvaluationPlan,
    SessionStore,
    schedule,
    summarize,
)
from flexivtrainer.evaluation.stats import (
    paired_exact_p,
    pairs_to_detect,
    trials_to_detect,
    wilson_interval,
)
from flexivtrainer.rollout.executors.waypoint import WaypointExecutor
from flexivtrainer.rollout.recorder import TICKS_FILE
from flexivtrainer.rollout.safety import SafetyEnvelope, SafetyStop
from flexivtrainer.rollout.service import RolloutService

# --- statistics ---------------------------------------------------------------


def test_wilson_interval_stays_inside_zero_one_at_the_extremes() -> None:
    low, high = wilson_interval(0, 20)
    assert low == 0.0 and 0.15 < high < 0.17
    low, high = wilson_interval(20, 20)
    assert 0.83 < low < 0.85 and high == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_only_discordant_pairs_decide_a_paired_comparison() -> None:
    assert paired_exact_p(10, 0) == pytest.approx(2 / 1024)
    assert paired_exact_p(5, 5) == 1.0
    assert paired_exact_p(0, 0) == 1.0


def test_trial_counts_match_the_readme_table() -> None:
    """30 points: ~39 independent trials per arm, ~33 pairs; 20 points: ~80."""
    assert trials_to_detect(0.5, 0.8) == 39
    assert pairs_to_detect(0.3, 0.4) == 33
    assert 78 <= trials_to_detect(0.6, 0.8) <= 84
    with pytest.raises(ValueError):
        pairs_to_detect(0.5, 0.4)


# --- outcomes -------------------------------------------------------------------


def _outcome(**overrides) -> RolloutOutcome:
    base = dict(
        episode="e1", checkpoint="ckpt", task=None, started_at="2026-10-15T10:00:00"
    )
    base.update(overrides)
    return RolloutOutcome(**base)


def test_an_outcome_round_trips_through_its_file(tmp_path) -> None:
    outcome = _outcome(verdict=Verdict("failure", "stalled", "on the rim"))
    write_outcome(tmp_path, outcome)
    again = read_outcome(tmp_path)
    assert again == outcome
    assert read_outcome(tmp_path / "missing") is None


def test_a_verdict_must_be_one_of_three_and_success_has_no_failure_mode() -> None:
    with pytest.raises(ValueError, match="not one of"):
        Verdict("probably")
    with pytest.raises(ValueError, match="no failure mode"):
        Verdict("success", "stalled")


def test_an_outcome_from_another_schema_is_refused(tmp_path) -> None:
    write_outcome(tmp_path, _outcome())
    path = tmp_path / "meta" / "rollout_outcome.json"
    data = json.loads(path.read_text())
    data["schema_version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        read_outcome(tmp_path)


# --- sessions -------------------------------------------------------------------


PLAN = {
    "about": "defaults against defaults",
    "arms": {"A": {"checkpoint": "ckpt_a"}, "B": {"checkpoint": "ckpt_b"}},
    "conditions": [
        {"id": "home", "description": "from HOME"},
        {"id": "rim", "description": "on the rim"},
    ],
    "trials_per_condition": 3,
    "seed": 7,
    "failure_modes": ["hovered", "stalled"],
}


def test_a_schedule_is_fixed_by_its_seed_and_covers_every_cell_per_block() -> None:
    plan = EvaluationPlan.from_dict("s", PLAN)
    trials = schedule(plan)
    assert trials == schedule(EvaluationPlan.from_dict("s", PLAN))
    assert len(trials) == 2 * 2 * 3
    for repetition in range(3):
        block = [(t.arm, t.condition) for t in trials if t.repetition == repetition]
        assert sorted(block) == sorted((a, c) for a in "AB" for c in ("home", "rim"))
    other = schedule(EvaluationPlan.from_dict("s", {**PLAN, "seed": 8}))
    assert [(t.arm, t.condition) for t in other] != [
        (t.arm, t.condition) for t in trials
    ]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"arms": {}}, "at least one arm"),
        ({"arms": {"A b": {"checkpoint": "x"}}}, "arm label"),
        ({"conditions": []}, "at least one start condition"),
        ({"conditions": ["x", "x"]}, "repeat"),
        ({"trials_per_condition": 0}, "at least 1"),
        ({"failure_modes": ["a", "a"]}, "distinct"),
        ({"retries": 3}, "unknown session keys"),
        ({"arms": {"A": {"source": "hub"}}}, "repo_id"),
    ],
)
def test_a_plan_that_cannot_mean_what_it_says_is_refused(change, match) -> None:
    with pytest.raises(ValueError, match=match):
        EvaluationPlan.from_dict("s", {**PLAN, **change})


def _judge(plan, outcomes_by_trial: dict[int, str]) -> list[RolloutOutcome]:
    trials = {t.index: t for t in schedule(plan)}
    out = []
    for index, verdict in outcomes_by_trial.items():
        mode = "stalled" if verdict == "failure" else None
        out.append(
            _outcome(
                episode=f"e{index}",
                session="s",
                trial=index,
                arm=trials[index].arm,
                condition=trials[index].condition,
                verdict=Verdict(verdict, mode),
            )
        )
    return out


def test_the_summary_counts_judged_trials_and_keeps_unknown_out_of_the_rates() -> None:
    plan = EvaluationPlan.from_dict("s", PLAN)
    trials = schedule(plan)
    verdicts = {}
    for trial in trials:
        verdicts[trial.index] = "success" if trial.arm == "A" else "failure"
    first_b = next(t.index for t in trials if t.arm == "B")
    verdicts[first_b] = "unknown"
    summary = summarize(plan, _judge(plan, verdicts))
    assert (
        summary["arms"]["A"]["successes"] == 6 and summary["arms"]["A"]["trials"] == 6
    )
    assert summary["arms"]["B"]["trials"] == 5 and summary["arms"]["B"]["unknown"] == 1
    assert summary["arms"]["B"]["failure_modes"] == {"stalled": 5}
    comparison = summary["comparisons"][0]
    assert (
        comparison["pairs"] == 5
        and comparison["only_A"] == 5
        and comparison["only_B"] == 0
    )
    assert comparison["p_value"] == pytest.approx(paired_exact_p(5, 0))


def test_progress_holds_the_next_trial_until_the_last_one_has_a_verdict(
    tmp_path,
) -> None:
    session = tmp_path / "s"
    session.mkdir()
    (session / "session.json").write_text(json.dumps(PLAN))
    store = SessionStore(tmp_path)
    assert store.sessions() == ["s"]
    progress = store.progress("s")
    assert progress["next"]["trial"] == 0 and progress["judged"] == 0
    first = schedule(store.plan("s"))[0]
    pending = _outcome(
        episode="e0", session="s", trial=0, arm=first.arm, condition=first.condition
    )
    write_outcome(session / "e0", pending)
    held = store.progress("s")
    assert held["next"] is None
    assert held["awaiting_verdict"] == [{"trial": 0, "episode": "e0"}]
    write_outcome(
        session / "e0",
        RolloutOutcome(**{**pending.__dict__, "verdict": Verdict("success")}),
    )
    moved = store.progress("s")
    assert moved["judged"] == 1 and moved["next"]["trial"] == 1


def test_session_names_cannot_leave_the_rollouts_root(tmp_path) -> None:
    with pytest.raises(ValueError):
        SessionStore(tmp_path).plan("../escape")
    with pytest.raises(ValueError):
        SessionStore(tmp_path).plan("adhoc")


# --- the safety envelope --------------------------------------------------------


def test_non_finite_commands_always_stop() -> None:
    with pytest.raises(SafetyStop) as stop:
        SafetyEnvelope().check_finite([[0.0, float("nan")]])
    assert stop.value.code == "non_finite"


def test_each_bound_stops_with_its_own_code() -> None:
    envelope = SafetyEnvelope(
        workspace_min_m=(0.2, -0.3, 0.1),
        workspace_max_m=(0.7, 0.3, 0.6),
        max_reference_offset_m=0.03,
        max_force_n=90.0,
        stale_plan_s=1.0,
    )
    envelope.check_workspace([[[0.5, 0.0, 0.3]]])
    with pytest.raises(SafetyStop, match="workspace"):
        envelope.check_workspace([[[0.5, 0.0, 0.05]]])
    envelope.check_reference([[0.5, 0.0, 0.3]], [[0.5, 0.0, 0.32]])
    with pytest.raises(SafetyStop, match="reference_offset"):
        envelope.check_reference([[0.5, 0.0, 0.3]], [[0.5, 0.0, 0.34]])
    envelope.check_force([[0.0, 0.0, -80.0, 0, 0, 0]])
    with pytest.raises(SafetyStop, match="force"):
        envelope.check_force([[30.0, 40.0, -80.0, 0, 0, 0]])
    envelope.check_stale(10.5, 10.0)
    with pytest.raises(SafetyStop, match="stale_plan"):
        envelope.check_stale(11.5, 10.0)


def test_an_unset_envelope_checks_only_finiteness() -> None:
    envelope = SafetyEnvelope()
    envelope.check_workspace([[[99.0, 99.0, 99.0]]])
    envelope.check_reference([[9.0, 9.0, 9.0]], [[0.0, 0.0, 0.0]])
    envelope.check_force([[1e6, 0.0, 0.0]])
    envelope.check_stale(1e6, 0.0)
    assert envelope.describe() == "finite commands"


def test_an_envelope_that_is_not_a_box_is_refused() -> None:
    with pytest.raises(ValueError, match="both workspace corners"):
        SafetyEnvelope(workspace_min_m=(0, 0, 0))
    with pytest.raises(ValueError, match="not a box"):
        SafetyEnvelope(workspace_min_m=(0, 0, 0), workspace_max_m=(1, -1, 1))
    with pytest.raises(ValueError, match="positive"):
        SafetyEnvelope(max_force_n=0.0)


# --- the executor ---------------------------------------------------------------


def _executor(robot, *, send_twist=True) -> WaypointExecutor:
    layout = [
        {
            "side": "single_arm",
            "pose": slice(0, 7),
            "pose_format": "quaternion",
            "twist": slice(7, 13),
            "wrench": slice(13, 19),
        }
    ]
    return WaypointExecutor(
        [robot],
        layout,
        threading.Event(),
        (0.25, 0.6, 1.0, 2.5),
        action_dim=19,
        send_twist=send_twist,
    )


def test_the_executor_refuses_a_non_finite_pose_before_the_arm() -> None:
    robot = _FakeRobot("F1")
    action = [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0] + [0.01] * 12
    action[1] = float("nan")
    with pytest.raises(ValueError, match="non-finite value at position 1"):
        _executor(robot).validate_actions([action])


def test_without_the_feedforward_the_twist_sent_is_zero() -> None:
    robot = _FakeRobot("F1")
    executor = _executor(robot, send_twist=False)
    action = [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0] + [0.05] * 6 + [0.0] * 6
    executor.replace_waypoints(
        [action], [time.monotonic() - 1 + 2], now=time.monotonic()
    )
    executor._send_waypoint(executor._waypoints[0])
    assert robot.commands[0][2] == [0.0] * 6


# --- the loop, end to end -------------------------------------------------------


def _service(tmp_path, policy, robot, *, configure=None) -> RolloutService:
    settings = _settings(tmp_path)
    settings.ensure_storage()
    if configure is not None:
        configure(settings)
    return RolloutService(
        settings,
        _cameras(),
        _teleop(initialized=False),
        _single_arm_pairs,
        lambda: ["single_arm"],
        policy_loader=_fake_loader(policy),
        robot_factory=lambda serial: robot,
        resolve_device=lambda configured: "cpu",
    )


@pytest.fixture
def fake_loop(monkeypatch):
    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._predict_action_chunk",
        lambda obs, pol, dev, pre, post, **kwargs: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )


def _action() -> list[float]:
    # A single arm's legacy layout: pose (7, unit quaternion), twist (6), wrench (6).
    return [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0] + [0.02] * 6 + [0.0] * 6


def _run_ticks(service, robot, start, ticks: int = 5) -> None:
    start()
    deadline = time.monotonic() + WAIT_S
    while len(robot.commands) < ticks and time.monotonic() < deadline:
        if service.status()["status"] != "running":
            break
        time.sleep(0.01)
    service.stop()


def _wait_for_verdict_slot(service) -> dict:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        status = service.status()
        if status["awaiting_verdict"] is not None:
            return status
        time.sleep(0.02)
    raise AssertionError(
        f"the recording never finished: {service.status()['recording']}"
    )


def test_a_recorded_rollout_is_a_dataset_in_the_teleop_schema(
    tmp_path, fake_loop
) -> None:
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    robot = _FakeRobot("F1")
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    _run_ticks(
        service, robot, lambda: service.start(_checkpoint(tmp_path), record=True)
    )
    status = _wait_for_verdict_slot(service)

    awaiting = status["awaiting_verdict"]
    episode_dir = tmp_path / "rollouts" / "adhoc" / awaiting["episode"]
    dataset = LeRobotDataset(f"local/{episode_dir.name}", root=episode_dir)
    assert dataset.num_frames == awaiting["frames"] >= 1
    names = dataset.features["observation.state"]["names"]
    assert names[0] == "single_arm.tcp_pose.x"
    assert len(dataset.features["action"]["names"]) == 19
    ticks = (episode_dir / TICKS_FILE).read_text().splitlines()
    assert len(ticks) == dataset.num_frames
    first = json.loads(ticks[0])
    assert first["frame"] == 0 and len(first["command"]) == 19
    assert first["measured"][0]["wrench"]
    outcome = read_outcome(episode_dir)
    assert outcome.verdict is None and outcome.session is None
    assert outcome.checkpoint.endswith("ckpt") and outcome.frames == dataset.num_frames


def test_a_new_start_waits_for_the_verdict_and_the_verdict_is_saved(
    tmp_path, fake_loop
) -> None:
    pytest.importorskip("lerobot")
    robot = _FakeRobot("F1")
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    checkpoint = _checkpoint(tmp_path)
    _run_ticks(service, robot, lambda: service.start(checkpoint, record=True))
    status = _wait_for_verdict_slot(service)
    with pytest.raises(RuntimeError, match="waiting for its verdict"):
        service.start(checkpoint)
    with pytest.raises(ValueError, match="failure mode"):
        service.verdict("failure")
    with pytest.raises(ValueError, match="not one of"):
        service.verdict("failure", failure_mode="made up")
    service.verdict("failure", failure_mode="stalled", note="rested on the rim")
    assert service.status()["awaiting_verdict"] is None
    outcome = read_outcome(
        tmp_path / "rollouts" / "adhoc" / status["awaiting_verdict"]["episode"]
    )
    assert (
        outcome.verdict.outcome == "failure"
        and outcome.verdict.note == "rested on the rim"
    )
    with pytest.raises(RuntimeError, match="No recorded rollout"):
        service.verdict("success")


def test_a_blind_trial_shows_its_label_not_its_checkpoint(tmp_path, fake_loop) -> None:
    pytest.importorskip("lerobot")
    robot = _FakeRobot("F1")
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    checkpoint = _checkpoint(tmp_path)
    session = tmp_path / "rollouts" / "cmp"
    session.mkdir(parents=True)
    plan = {
        **PLAN,
        "arms": {"A": {"checkpoint": checkpoint}, "B": {"checkpoint": checkpoint}},
    }
    (session / "session.json").write_text(json.dumps(plan))

    seen: dict = {}

    def start():
        result = service.start_next_trial("cmp")
        seen.update(result)

    _run_ticks(service, robot, start)
    assert seen["checkpoint_path"] is None and seen["checkpoint_repo_id"] is None
    first = schedule(EvaluationPlan.from_dict("cmp", plan))[0]
    assert seen["checkpoint_label"] == first.arm
    assert seen["trial"]["condition"] == first.condition and seen["trial"]["blind"]
    status = _wait_for_verdict_slot(service)
    assert status["awaiting_verdict"]["trial"]["trial"] == 0
    assert status["awaiting_verdict"]["failure_modes"] == ["hovered", "stalled"]
    with pytest.raises(RuntimeError, match="waiting for its verdict"):
        service.start_next_trial("cmp")
    service.verdict("success")
    progress = service.evaluation_session("cmp")
    assert progress["judged"] == 1 and progress["next"]["trial"] == 1
    assert service.evaluation_sessions() == [
        {"name": "cmp", "about": PLAN["about"], "total": 12, "judged": 1}
    ]


def test_the_force_bound_stops_the_rollout_and_the_outcome_says_why(
    tmp_path, fake_loop
) -> None:
    pytest.importorskip("lerobot")
    robot = _FakeRobot("F1")

    def configure(settings):
        settings.rollout.max_force_n = 5.0  # the fake arm reads ~37 N

    service = _service(tmp_path, _FakePolicy(_action()), robot, configure=configure)
    service.start(_checkpoint(tmp_path), record=True)
    deadline = time.monotonic() + WAIT_S
    while service.status()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    status = service.status()
    assert status["status"] == "failed"
    assert "Safety stop (force)" in status["error"]
    assert not robot.commands, "the force check runs before any command is sent"
    service.stop()
    finished = _wait_for_verdict_slot(service)
    assert finished["stop_reason"] == "safety"
    awaiting = finished["awaiting_verdict"]
    outcome = read_outcome(tmp_path / "rollouts" / "adhoc" / awaiting["episode"])
    assert outcome.safety["stop"]["code"] == "force"
    assert outcome.safety["max_force_n"] == 5.0
    assert outcome.frames == 0, (
        "a rollout stopped before its first tick still leaves its outcome"
    )


def test_a_non_finite_policy_output_never_reaches_the_arm(tmp_path, fake_loop) -> None:
    robot = _FakeRobot("F1")
    action = _action()
    action[2] = float("nan")
    service = _service(tmp_path, _FakePolicy(action), robot)
    service.start(_checkpoint(tmp_path))
    deadline = time.monotonic() + WAIT_S
    while service.status()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    status = service.status()
    assert "Safety stop (non_finite)" in (status["error"] or "")
    assert not robot.commands
    service.stop()


def test_the_feedforward_can_be_switched_off_per_rollout(tmp_path, fake_loop) -> None:
    robot = _FakeRobot("F1")
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    _run_ticks(
        service, robot, lambda: service.start(_checkpoint(tmp_path), send_twist=False)
    )
    assert robot.commands and all(
        velocity == [0.0] * 6 for _, _, velocity in robot.commands
    )
    assert service.status()["send_twist"] is False


def test_recording_a_bspline_rollout_is_refused_before_the_robot(tmp_path) -> None:
    robot = _FakeRobot("F1")
    service = _service(
        tmp_path, _FakeBSplinePolicy(np.zeros((1, 1)), knot_rate_hz=10), robot
    )
    with pytest.raises(RuntimeError, match="B-spline rollouts is not supported"):
        service.start(_checkpoint_of_type(tmp_path, "bspline_diffusion"), record=True)
    assert not robot.enabled


def test_the_api_reports_no_rollout_waiting_as_a_conflict(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from flexivtrainer.api.app import create_app
    from flexivtrainer.runtime.manager import get_runtime_manager

    service = _service(tmp_path, _FakePolicy(_action()), _FakeRobot("F1"))
    app = create_app()
    app.dependency_overrides[get_runtime_manager] = lambda: SimpleNamespace(
        rollout=service
    )
    client = TestClient(app)
    response = client.post("/rollout/verdict", json={"outcome": "success"})
    assert response.status_code == 409
    assert client.post("/rollout/verdict", json={"outcome": "maybe"}).status_code == 422
    assert client.get("/rollout/evaluations").json() == {"sessions": []}
    assert client.get("/rollout/evaluations/nope").status_code == 404
    assert client.post("/rollout/evaluations/nope/next").status_code == 404


class _SpringRobot(_FakeRobot):
    """A fake arm that also takes a stiffness and a contact-wrench ceiling."""

    def __init__(self, serial: str) -> None:
        super().__init__(serial)
        self.calls: list[tuple] = []

    def SwitchMode(self, mode) -> None:  # noqa: N802
        super().SwitchMode(mode)
        self.calls.append(("mode", mode))

    def SetCartesianImpedance(self, stiffness) -> None:  # noqa: N802
        self.calls.append(("stiffness", list(stiffness), len(self.commands)))

    def SetMaxContactWrench(self, wrench) -> None:  # noqa: N802
        self.calls.append(("wrench", list(wrench), len(self.commands)))


def test_the_spring_is_set_in_its_mode_before_the_first_target(
    tmp_path, fake_loop
) -> None:
    pytest.importorskip("lerobot")
    robot = _SpringRobot("F1")
    stiffness = (2900.0, 2900.0, 2900.0, 250.0, 250.0, 250.0)
    wrench = (15.0, 25.0, 90.0, 10.0, 10.0, 10.0)

    def configure(settings):
        settings.rollout.cartesian_stiffness = stiffness
        settings.rollout.max_contact_wrench = wrench

    service = _service(tmp_path, _FakePolicy(_action()), robot, configure=configure)
    _run_ticks(
        service, robot, lambda: service.start(_checkpoint(tmp_path), record=True)
    )
    kinds = [call[0] for call in robot.calls]
    assert kinds[-3:] == ["mode", "stiffness", "wrench"]
    assert robot.calls[-2] == ("stiffness", list(stiffness), 0)
    assert robot.calls[-1] == ("wrench", list(wrench), 0)
    awaiting = _wait_for_verdict_slot(service)["awaiting_verdict"]
    outcome = read_outcome(tmp_path / "rollouts" / "adhoc" / awaiting["episode"])
    assert outcome.safety["cartesian_stiffness"] == list(stiffness)
    assert outcome.safety["max_contact_wrench"] == list(wrench)


def test_without_settings_the_controller_keeps_its_own_spring(
    tmp_path, fake_loop
) -> None:
    robot = _SpringRobot("F1")
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    _run_ticks(service, robot, lambda: service.start(_checkpoint(tmp_path)))
    assert [call[0] for call in robot.calls] == ["mode"] * len(robot.calls)
