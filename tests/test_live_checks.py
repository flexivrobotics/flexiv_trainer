"""Learned checks scored on each tick of a running rollout."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_learned_check import EPISODES, Z, _dataset, _label_all, _options
from test_rollout_evaluation import (
    PLAN,
    _action,
    _run_ticks,
    _service,
    _wait_for_verdict_slot,
    fake_loop,  # noqa: F401 - a fixture
)
from test_rollout_service import _checkpoint, _FakePolicy, _FakeRobot
from test_rollout_takeover import _take_over, _takeover, _wait, make_rig  # noqa: F401

from flexivtrainer.evaluation.checks import LearnedCheck, parse_checks, read_ticks
from flexivtrainer.evaluation.outcome import read_outcome
from flexivtrainer.rollout.live_checks import LiveChecks


# The stand-in loader never reads it; the checks name their models relative to
# it, as a plan does, so the paths mean the same on every OS.
STORAGE = Path("storage")


class _Model:
    """Stands in for a trained check: answers from one state column."""

    def __init__(self, column="z", threshold=0.5, cameras=(), gate=None):
        self.inputs = SimpleNamespace(cameras=tuple(cameras), state=(column,))
        self.threshold = threshold
        self.calls = 0
        self.gate = gate

    def probabilities(self, images, state):
        if self.gate is not None:
            self.gate.wait(5.0)
        self.calls += 1
        for camera in self.inputs.cameras:
            assert images[camera].shape[0] == 1
        return np.asarray(state[:, 0], dtype=np.float32)


def _live(checks, models, names=("x", "z")):
    live = LiveChecks(
        checks,
        STORAGE,
        state_names=lambda snapshot, images, pose_format, width: list(names),
        loader=lambda path: models[path.name],
    )
    live.start()
    return live


def _feed(live, ps, dt=0.1, column=1):
    for k, p in enumerate(ps):
        state = [0.0, 0.0]
        state[column] = p
        live.submit(t=k * dt, state=state, images={}, snapshot={}, pose_format="r")
        deadline = time.monotonic() + 5.0
        while live.scored + live.skipped < k + 1 and time.monotonic() < deadline:
            time.sleep(0.001)


def _check(name, **options):
    return LearnedCheck(model=f"checks/{name}", **options)


def test_a_check_passes_once_its_probability_holds_long_enough() -> None:
    model = _Model()
    live = _live({"c": _check("m", hold_s=0.2, end_on_pass=True)}, {"m": model})
    try:
        _feed(live, [0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.2])
        summary = live.summary()["checks"]["c"]
        assert summary["passed"] is True and summary["first_at_s"] == pytest.approx(0.4)
        assert summary["last_p"] == pytest.approx(0.2)
        assert live.ended() == "c" and live.values() == {"c": 0.2}
    finally:
        live.stop()


def test_a_check_not_marked_to_end_the_rollout_never_ends_it() -> None:
    live = _live({"c": _check("m", hold_s=0.0, live=True)}, {"m": _Model()})
    try:
        _feed(live, [0.9, 0.9])
        assert live.summary()["checks"]["c"]["passed"] is True
        assert live.ended() is None
    finally:
        live.stop()


def test_the_threshold_comes_from_the_model_unless_the_plan_sets_one() -> None:
    own = _live(
        {"c": _check("m", hold_s=0.0, live=True)}, {"m": _Model(threshold=0.95)}
    )
    set_ = _live(
        {"c": _check("m", hold_s=0.0, live=True, threshold=0.3)},
        {"m": _Model(threshold=0.95)},
    )
    try:
        _feed(own, [0.5])
        _feed(set_, [0.5])
        assert own.summary()["checks"]["c"]["passed"] is False
        assert set_.summary()["checks"]["c"]["passed"] is True
    finally:
        own.stop()
        set_.stop()


def test_the_planner_never_waits_and_only_the_newest_tick_is_scored() -> None:
    gate = threading.Event()
    model = _Model(gate=gate)
    live = _live({"c": _check("m", live=True)}, {"m": model})
    try:
        started = time.monotonic()
        for k in range(50):
            live.submit(
                t=k * 0.01, state=[0.0, 0.9], images={}, snapshot={}, pose_format="r"
            )
        assert time.monotonic() - started < 0.5  # the worker is blocked meanwhile
        gate.set()
        deadline = time.monotonic() + 5.0
        while live.scored + live.skipped < 50 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert live.skipped > 0 and live.scored + live.skipped == 50
        assert model.calls == live.scored < 50
    finally:
        live.stop()


def test_columns_are_found_by_name_and_a_missing_one_is_reported() -> None:
    model = _Model(column="x")
    live = _live({"c": _check("m", hold_s=0.0, live=True)}, {"m": model})
    try:
        _feed(live, [0.9], column=0)
        assert live.summary()["checks"]["c"]["passed"] is True
    finally:
        live.stop()
    absent = _live({"c": _check("m", live=True)}, {"m": _Model(column="fz")})
    try:
        _feed(absent, [0.9])
        assert "no state column ['fz']" in absent.summary()["checks"]["c"]["error"]
        assert absent.ended() is None
    finally:
        absent.stop()


def test_a_model_that_does_not_load_refuses_the_start() -> None:
    live = LiveChecks(
        {"c": _check("gone", live=True)},
        STORAGE,
        state_names=lambda *args: [],
        loader=lambda path: (_ for _ in ()).throw(FileNotFoundError(str(path))),
    )
    with pytest.raises(FileNotFoundError):
        live.start()
    live.stop()  # stopping one that never started is harmless


def test_the_plan_marks_which_checks_run_live() -> None:
    checks = parse_checks(
        {
            "a": {"kind": "learned", "model": "m"},
            "b": {"kind": "learned", "model": "m", "live": True},
            "c": {"kind": "learned", "model": "m", "end_on_pass": True},
        }
    )
    assert [n for n, c in checks.items() if c.scored_live] == ["b", "c"]
    with pytest.raises(ValueError, match="live must be true or false"):
        parse_checks({"x": {"kind": "learned", "model": "m", "live": "yes"}})


# --- through a trial -------------------------------------------------------------


def _session_with_live_check(tmp_path, checkpoint=None, **options):
    from flexivtrainer.evaluation.learned_check import CheckInputs, train_check

    demos = _dataset(tmp_path / "demos", EPISODES)
    _label_all(demos, EPISODES, check="high")
    train_check(
        [demos],
        "high",
        CheckInputs(state=(Z,)),
        tmp_path / "checks" / "high",
        _options(holdout=0.0),
    )
    checkpoint = checkpoint or _checkpoint(tmp_path)
    session = tmp_path / "rollouts" / "cmp"
    session.mkdir(parents=True)
    plan = {
        **PLAN,
        "arms": {"A": {"checkpoint": checkpoint}, "B": {"checkpoint": checkpoint}},
        "checks": {
            "high": {
                "kind": "learned",
                "model": "checks/high",
                "hold_s": 0.0,
                **options,
            }
        },
    }
    (session / "session.json").write_text(json.dumps(plan))
    return session


def test_a_trial_records_its_live_check_on_every_tick(
    tmp_path,
    fake_loop,  # noqa: F811
) -> None:
    pytest.importorskip("lerobot")
    session = _session_with_live_check(tmp_path, live=True)
    robot = _FakeRobot("F1")  # holds its tool at z = 3: the state is reached
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    _run_ticks(service, robot, lambda: service.start_next_trial("cmp"), ticks=8)
    status = _wait_for_verdict_slot(service)
    episode = session / status["awaiting_verdict"]["episode"]
    ticks = read_ticks(episode)
    scored = [t["checks"]["high"] for t in ticks if "checks" in t]
    assert scored and all(p > 0.5 for p in scored)
    outcome = read_outcome(episode)
    assert outcome.live_checks["checks"]["high"]["passed"] is True
    assert outcome.live_checks["scored"] >= 1
    assert outcome.stop_reason != "check high passed"  # live only, not end_on_pass
    assert outcome.checks["high"]["passed"] is True  # and scored again offline


def test_a_check_marked_end_on_pass_ends_the_trial(
    tmp_path,
    fake_loop,  # noqa: F811
) -> None:
    pytest.importorskip("lerobot")
    session = _session_with_live_check(tmp_path, end_on_pass=True)
    robot = _FakeRobot("F1")
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    service.start_next_trial("cmp")
    status = _wait_for_verdict_slot(service)  # ends without service.stop()
    outcome = read_outcome(session / status["awaiting_verdict"]["episode"])
    assert outcome.stop_reason == "check high passed"


def test_a_live_check_that_cannot_load_refuses_the_trial(
    tmp_path,
    fake_loop,  # noqa: F811
) -> None:
    pytest.importorskip("lerobot")
    checkpoint = _checkpoint(tmp_path)
    session = tmp_path / "rollouts" / "cmp"
    session.mkdir(parents=True)
    plan = {
        **PLAN,
        "arms": {"A": {"checkpoint": checkpoint}, "B": {"checkpoint": checkpoint}},
        "checks": {"gone": {"kind": "learned", "model": "checks/gone", "live": True}},
    }
    (session / "session.json").write_text(json.dumps(plan))
    robot = _FakeRobot("F1")
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    with pytest.raises(RuntimeError, match="A live check could not be loaded"):
        service.start_next_trial("cmp")
    assert service.status()["status"] != "running"
    assert not list(session.glob("*/meta"))  # no trial left behind


def test_the_operator_reaching_the_state_ends_the_trial_too(
    tmp_path,
    make_rig,  # noqa: F811
) -> None:
    pytest.importorskip("lerobot")
    rig = make_rig()
    session = _session_with_live_check(
        tmp_path, checkpoint=rig.checkpoint, end_on_pass=True
    )
    rig.follower.position = [0.5, 0.1, 0.0]  # z = 0: not there under the policy
    rig.service.start_next_trial("cmp")
    _wait(lambda: _takeover(rig)["action"] == "take_over", "the policy's first plan")
    time.sleep(0.3)
    assert rig.service.status()["status"] == "running"
    _take_over(rig)
    rig.follower.position = [0.5, 0.1, 3.0]  # the operator gets it there
    status = _wait_for_verdict_slot(rig.service)
    episode = session / status["awaiting_verdict"]["episode"]
    outcome = read_outcome(episode)
    assert outcome.stop_reason == "check high passed"
    live = outcome.live_checks["checks"]["high"]
    assert live["passed"] is True and live["last_p"] > 0.5
    assert outcome.interventions["count"] >= 1
    under_policy = [
        t["checks"]["high"]
        for t in read_ticks(episode)
        if t["control"] == "policy" and "checks" in t
    ]
    assert under_policy and max(under_policy) < 0.5
    _wait(lambda: not rig.teleop.snapshot().lent, "the robots to be returned")
