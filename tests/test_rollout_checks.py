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

"""Automatic checks: scored from the tick log, compared with the verdicts."""

import json

import pytest
from test_rollout_evaluation import (
    PLAN,
    _action,
    _run_ticks,
    _service,
    _wait_for_verdict_slot,
    fake_loop,  # noqa: F401 - a fixture
)
from test_rollout_service import _checkpoint, _FakePolicy, _FakeRobot

from flexivtrainer.evaluation import checks as checks_module
from flexivtrainer.evaluation.checks import (
    SettledInBox,
    agreement,
    parse_checks,
    read_ticks,
    run_checks,
)
from flexivtrainer.evaluation.outcome import (
    RolloutOutcome,
    Verdict,
    read_outcome,
    write_outcome,
)
from flexivtrainer.evaluation.rescore import main as rescore_main
from flexivtrainer.evaluation.rescore import rescore
from flexivtrainer.evaluation.session import EvaluationPlan, summarize
from flexivtrainer.rollout.recorder import TICKS_FILE

BOX = {
    "kind": "settled_in_box",
    "box_min_m": [0.40, -0.05, 0.10],
    "box_max_m": [0.44, -0.01, 0.12],
    "hold_s": 0.5,
    "max_speed_m_s": 0.003,
}
INSIDE = [0.42, -0.03, 0.11]
OUTSIDE = [0.42, -0.03, 0.20]


def _tick(t, position, velocity=(0.0, 0.0, 0.0), arm=0):
    measured = [{"tcp_pose": [9.0, 9.0, 9.0, 1, 0, 0, 0], "tcp_vel": [0.0] * 6}] * arm
    measured.append(
        {"tcp_pose": [*position, 1.0, 0.0, 0.0, 0.0], "tcp_vel": [*velocity, 0, 0, 0]}
    )
    return {"t": t, "measured": measured}


def _track(*segments, dt=0.1):
    """(position, seconds) segments -> ticks every `dt`."""
    ticks, t = [], 0.0
    for position, seconds in segments:
        for _ in range(round(seconds / dt)):
            ticks.append(_tick(round(t, 4), position))
            t += dt
    return ticks


def _check(**overrides) -> SettledInBox:
    return parse_checks({"c": {**BOX, **overrides}})["c"]


# --- one check, on synthetic tick logs ---------------------------------------------


def test_resting_inside_long_enough_passes_and_says_when() -> None:
    result = _check().evaluate(_track((OUTSIDE, 1.0), (INSIDE, 1.0), (OUTSIDE, 0.5)))
    assert result["passed"] is True
    assert result["first_at_s"] == pytest.approx(1.0)
    assert result["held_s"] == pytest.approx(0.9)


def test_passing_through_the_box_does_not_count() -> None:
    result = _check().evaluate(_track((OUTSIDE, 1.0), (INSIDE, 0.3), (OUTSIDE, 1.0)))
    assert result["passed"] is False and result["first_at_s"] is None


def test_moving_inside_the_box_does_not_count() -> None:
    ticks = [_tick(i * 0.1, INSIDE, velocity=(0.01, 0.0, 0.0)) for i in range(20)]
    assert _check().evaluate(ticks)["passed"] is False


def test_without_a_reported_velocity_the_speed_comes_from_the_positions() -> None:
    still = [_tick(i * 0.1, INSIDE) for i in range(10)]
    for tick in still:
        del tick["measured"][0]["tcp_vel"]
    assert _check().evaluate(still)["passed"] is True
    drifting = [_tick(i * 0.1, [0.40 + 0.001 * i, -0.03, 0.11]) for i in range(10)]
    for tick in drifting:
        del tick["measured"][0]["tcp_vel"]
    assert _check().evaluate(drifting)["passed"] is False  # 10 mm/s


def test_at_the_end_needs_the_stretch_to_last_until_the_end() -> None:
    middle = _track((INSIDE, 1.0), (OUTSIDE, 0.5))
    end = _track((OUTSIDE, 0.5), (INSIDE, 1.0))
    assert _check(when="end").evaluate(middle)["passed"] is False
    assert _check(when="any").evaluate(middle)["passed"] is True
    assert _check(when="end").evaluate(end)["passed"] is True


def test_a_tick_without_a_measurement_breaks_the_stretch() -> None:
    ticks = _track((INSIDE, 0.4)) + [{"t": 0.4, "measured": []}]
    ticks += [_tick(0.5 + i * 0.1, INSIDE) for i in range(4)]
    assert _check().evaluate(ticks)["passed"] is False


def test_the_arm_index_picks_the_arm() -> None:
    ticks = [_tick(i * 0.1, INSIDE, arm=1) for i in range(10)]
    assert _check(arm=1).evaluate(ticks)["passed"] is True
    assert _check(arm=0).evaluate(ticks)["passed"] is False


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({**BOX, "kind": "latched"}, "kind 'latched'"),
        ({**BOX, "radius": 1}, "unknown keys"),
        ({**BOX, "box_min_m": [0.5, 0, 0]}, "below box_max_m"),
        ({**BOX, "hold_s": 0}, "positive"),
        ({**BOX, "when": "start"}, "'any' or 'end'"),
        ({k: v for k, v in BOX.items() if k != "box_max_m"}, "needs box_min_m"),
        ({**BOX, "box_max_m": [1, 2]}, "three values"),
    ],
)
def test_a_check_that_cannot_mean_what_it_says_is_refused(spec, match) -> None:
    with pytest.raises(ValueError, match=match):
        parse_checks({"c": spec})


def test_a_check_that_cannot_be_scored_is_unscored_not_failed() -> None:
    results = run_checks({"c": _check()}, [])
    assert "passed" not in results["c"] and results["c"]["error"] == "no tick log"
    broken = run_checks({"c": _check()}, [{"t": "soon"}])
    assert broken["c"]["passed"] is False  # a malformed tick is skipped, not fatal


def test_agreement_counts_each_cell_and_keeps_unscored_apart() -> None:
    yes, no = {"passed": True}, {"passed": False}
    pairs = [(yes, "success"), (yes, "failure"), (no, "success"), (no, "failure")]
    pairs += [
        (yes, "success"),
        ({"error": "no tick log"}, "success"),
        (None, "failure"),
    ]
    result = agreement(pairs)
    assert result["trials"] == 5 and result["agree"] == 3
    assert result["passed_but_failure"] == 1 and result["failed_but_success"] == 1
    assert result["unscored"] == 2
    assert result["ci95"][0] < 0.6 < result["ci95"][1]


def test_the_tick_log_name_matches_the_recorders() -> None:
    assert checks_module.TICKS_FILE == TICKS_FILE


# --- in a session ------------------------------------------------------------------


def test_a_plan_parses_its_checks_and_the_summary_reports_agreement() -> None:
    plan = EvaluationPlan.from_dict("cmp", {**PLAN, "checks": {"placed": BOX}})
    assert isinstance(plan.checks["placed"], SettledInBox)
    outcomes = [
        RolloutOutcome(
            episode=f"e{i}",
            checkpoint="x",
            task=None,
            started_at="t",
            session="cmp",
            trial=i,
            verdict=Verdict(outcome=verdict, failure_mode=mode),
            checks={"placed": {"passed": passed}},
        )
        for i, (verdict, mode, passed) in enumerate(
            [
                ("success", None, True),
                ("failure", "stalled", True),
                ("success", None, True),
            ]
        )
    ]
    summary = summarize(plan, outcomes)["checks"]["placed"]
    assert summary["trials"] == 3 and summary["agree"] == 2
    assert summary["passed_but_failure"] == 1


def test_a_plan_without_checks_reports_none() -> None:
    plan = EvaluationPlan.from_dict("cmp", PLAN)
    assert plan.checks == {} and summarize(plan, [])["checks"] == {}


def test_a_trials_checks_are_scored_before_it_waits_for_its_verdict(
    tmp_path,
    fake_loop,  # noqa: F811
) -> None:
    pytest.importorskip("lerobot")
    robot = _FakeRobot("F1")  # reports its tool at (1, 2, 3)
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    checkpoint = _checkpoint(tmp_path)
    session = tmp_path / "rollouts" / "cmp"
    session.mkdir(parents=True)
    near = {
        "kind": "settled_in_box",
        "box_min_m": [0.9, 1.9, 2.9],
        "box_max_m": [1.1, 2.1, 3.1],
        "hold_s": 0.01,
        "max_speed_m_s": 100.0,
    }
    far = {**near, "box_min_m": [5, 5, 5], "box_max_m": [6, 6, 6]}
    plan = {
        **PLAN,
        "arms": {"A": {"checkpoint": checkpoint}, "B": {"checkpoint": checkpoint}},
        "checks": {"near": near, "far": far},
    }
    (session / "session.json").write_text(json.dumps(plan))

    _run_ticks(service, robot, lambda: service.start_next_trial("cmp"))
    status = _wait_for_verdict_slot(service)
    episode = session / status["awaiting_verdict"]["episode"]
    scored = read_outcome(episode).checks
    assert scored["near"]["passed"] is True and scored["far"]["passed"] is False
    assert "checks" not in json.dumps(status["awaiting_verdict"])
    service.verdict("failure", failure_mode="stalled")
    judged = read_outcome(episode)
    assert judged.checks == scored and judged.verdict.outcome == "failure"
    agreement_now = service.evaluation_session("cmp")["summary"]["checks"]
    assert agreement_now["near"]["passed_but_failure"] == 1
    assert agreement_now["far"]["agree"] == 1


# --- scoring again, offline ----------------------------------------------------------


def _recorded_session(root, verdicts_and_tracks):
    session = root / "cmp"
    session.mkdir(parents=True)
    (session / "session.json").write_text(
        json.dumps({**PLAN, "checks": {"placed": BOX}})
    )
    for trial, (verdict, track) in enumerate(verdicts_and_tracks):
        episode = session / f"e{trial}"
        (episode / "meta").mkdir(parents=True)
        (episode / TICKS_FILE).write_text("".join(json.dumps(t) + "\n" for t in track))
        write_outcome(
            episode,
            RolloutOutcome(
                episode=f"e{trial}",
                checkpoint="x",
                task=None,
                started_at="t",
                session="cmp",
                trial=trial,
                verdict=Verdict(
                    outcome=verdict,
                    failure_mode=None if verdict == "success" else "stalled",
                ),
            ),
        )
    return session


def test_rescoring_tries_new_thresholds_without_touching_the_trials(tmp_path) -> None:
    rested = _track((OUTSIDE, 1.0), (INSIDE, 0.7))
    brief = _track((OUTSIDE, 1.0), (INSIDE, 0.3))
    session = _recorded_session(tmp_path, [("success", rested), ("failure", brief)])
    assert read_ticks(session / "e0") == rested
    before = (session / "e0" / "meta" / "rollout_outcome.json").read_text()

    report = rescore(tmp_path, "cmp")
    assert report["agreement"]["placed"]["agree"] == 2 and not report["written"]
    assert (session / "e0" / "meta" / "rollout_outcome.json").read_text() == before

    loose = parse_checks({"placed": {**BOX, "hold_s": 0.15}})
    report = rescore(tmp_path, "cmp", loose, write=True)
    assert report["agreement"]["placed"]["passed_but_failure"] == 1
    stored = read_outcome(session / "e1")
    assert stored.checks["placed"]["passed"] is True
    assert stored.verdict.outcome == "failure"


def test_the_rescore_command_prints_json_and_refuses_a_missing_session(
    tmp_path, capsys
) -> None:
    _recorded_session(tmp_path, [("success", _track((INSIDE, 1.0)))])
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"checks": {"placed": {**BOX, "hold_s": 5.0}}}))
    assert rescore_main(["cmp", "--root", str(tmp_path), "--checks", str(other)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["trials"][0]["checks"]["placed"]["passed"] is False
    assert rescore_main(["nope", "--root", str(tmp_path)]) == 2
