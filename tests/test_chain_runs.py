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

"""Chain runs judged by the operator, and how many in a row were clean."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_chain import _chain, _Rig
from test_rollout_plans import LIFT, _write
from test_step_rollout import (
    _Arm,
    _service,
    rdk,  # noqa: F401 - a fixture
)
from waits import WAIT_S

from flexivtrainer.api.app import create_app
from flexivtrainer.evaluation.outcome import DEFAULT_FAILURE_MODES
from flexivtrainer.rollout.chain import CHAINS_DIR, ChainPlan
from flexivtrainer.rollout.chain_runs import judge_run, read_runs, summarize_runs
from flexivtrainer.runtime.manager import get_runtime_manager


def _run(directory, name, *, result="done", verdict=None, events=(), plan="pick"):
    record = {
        "plan": plan,
        "about": "",
        "started_at": "2026-10-23T22:00:00+08:00",
        "steps": ["pick"],
        "failure_modes": ["dropped it", "other"],
        "events": list(events),
        "result": result,
    }
    if verdict is not None:
        record["verdict"] = {"outcome": verdict, "failure_mode": None, "note": ""}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(record))


TAKEOVER = {"event": "try", "outcome": {"passed": True, "takeovers": 1}}
CALL = {"event": "operator called", "why": "3 tries failed"}
BACK_OFF = {"event": "back-off", "outcome": {"passed": True}}


def test_a_plan_names_what_may_go_wrong():
    steps = [{"name": "lift", "primitive": {}}]
    plan = ChainPlan.from_dict({"steps": steps, "failure_modes": ["dropped it"]})
    assert plan.failure_modes == ("dropped it",)
    assert ChainPlan.from_dict({"steps": steps}).failure_modes == ()
    for modes, message in (
        ("dropped it", "a list of texts"),
        (["dropped it", "dropped it"], "repeat"),
        (["", "x"], "non-empty texts"),
    ):
        with pytest.raises(ValueError, match=message):
            ChainPlan.from_dict({"steps": steps, "failure_modes": modes})


def test_a_record_carries_its_plan_and_what_may_go_wrong(tmp_path):
    chain, _ = _chain(tmp_path, [{"name": "lift", "primitive": {}}], _Rig())
    chain.join(WAIT_S)
    record = json.loads((tmp_path / "chains" / "run.json").read_text())
    # Started from a plan given whole: no plan file's name; the default modes.
    assert record["plan"] is None
    assert record["failure_modes"] == list(DEFAULT_FAILURE_MODES)


def test_the_streak_counts_clean_runs_up_to_the_newest(tmp_path):
    chains = tmp_path / "chains"
    _run(chains, "20261023_100000", verdict="success")
    _run(chains, "20261023_100100", verdict="failure")
    _run(chains, "20261023_100200", verdict="success", events=[BACK_OFF])
    _run(chains, "20261023_100300", verdict="success")
    _run(chains, "20261023_100400", verdict="success")
    # Helped: not clean, though a success.
    _run(chains, "20261023_100500", verdict="success", events=[TAKEOVER])
    _run(chains, "20261023_100600", verdict="success")
    _run(chains, "20261023_100700", verdict="success", events=[BACK_OFF])
    # Still going: not counted.
    _run(chains, "20261023_100800", result=None)
    _run(chains, "20261023_090000", verdict="success", plan="other")
    (chains / "20261023_090100.json").write_text("{broken")
    runs = read_runs(chains)
    assert [run["name"] for run in runs][:2] == ["20261023_100800", "20261023_100700"]
    assert runs[1]["back_offs"] == 1 and runs[1]["clean"] is True
    assert runs[2]["clean"] is True and runs[3]["takeovers"] == 1
    assert runs[3]["clean"] is False
    broken = next(run for run in runs if run["name"] == "20261023_090100")
    assert "error" in broken
    summary = summarize_runs(runs)
    assert summary["pick"] == {
        "runs": 8,
        "judged": 8,
        "successes": 7,
        "clean": 6,
        "streak": 2,
        "longest_streak": 3,
    }
    assert summary["other"]["streak"] == 1
    # A finished run nobody judged ends the streak.
    _run(chains, "20261023_100900")
    assert summarize_runs(read_runs(chains))["pick"]["streak"] == 0
    assert len(read_runs(chains, limit=3)) == 3


def test_a_verdict_is_written_into_the_finished_run(tmp_path):
    chains = tmp_path / "chains"
    _run(chains, "a")
    _run(chains, "going", result=None)
    with pytest.raises(FileNotFoundError, match="no chain run named 'nope'"):
        judge_run(chains, "nope", "success")
    with pytest.raises(FileNotFoundError):
        judge_run(chains, "../a", "success")
    with pytest.raises(RuntimeError, match="has not finished"):
        judge_run(chains, "going", "success")
    with pytest.raises(ValueError, match="needs a failure mode"):
        judge_run(chains, "a", "failure")
    with pytest.raises(ValueError, match="not one of"):
        judge_run(chains, "a", "failure", failure_mode="hovered")
    failed = judge_run(chains, "a", "failure", failure_mode="dropped it", note=" slid ")
    assert failed["verdict"]["failure_mode"] == "dropped it"
    assert failed["verdict"]["note"] == "slid" and failed["clean"] is False
    # A new verdict replaces the old; a success has no failure mode.
    judged = judge_run(chains, "a", "success", failure_mode="dropped it")
    assert judged["clean"] is True and judged["verdict"]["failure_mode"] is None
    stored = json.loads((chains / "a.json").read_text())
    assert stored["verdict"]["outcome"] == "success" and stored["result"] == "done"
    assert not list(chains.glob("*.tmp"))


def test_a_run_started_from_a_plan_file_is_counted_under_its_name(tmp_path, rdk):  # noqa: F811
    service, _ = _service(tmp_path, _Arm())
    rollouts = service._settings.storage.rollouts_root
    _write(rollouts, "lift", {**LIFT, "failure_modes": ["dropped it"]})
    service.start_plan("lift")
    service._chain.join(WAIT_S)
    listing = service.chain_runs()
    run = listing["runs"][0]
    assert run["plan"] == "lift" and run["result"] == "done"
    assert run["failure_modes"] == ["dropped it"]
    assert listing["plans"]["lift"]["streak"] == 0
    judged = service.chain_verdict(run["name"], "success")
    assert judged["clean"] is True
    assert service.chain_runs()["plans"]["lift"]["streak"] == 1
    assert (rollouts / CHAINS_DIR / f"{run['name']}.json").is_file()


def test_the_routes_list_runs_and_pass_on_a_verdict():
    seen = []

    def chain_verdict(run, outcome, *, failure_mode=None, note=""):
        seen.append((run, outcome, failure_mode, note))
        if run == "missing":
            raise FileNotFoundError("no chain run named 'missing'")
        if run == "bad":
            raise ValueError("a failure needs a failure mode")
        if run == "going":
            raise RuntimeError("chain run 'going' has not finished")
        return {"name": run, "verdict": {"outcome": outcome}}

    app = create_app()
    rollout = SimpleNamespace(
        chain_runs=lambda: {"runs": [], "plans": {}}, chain_verdict=chain_verdict
    )
    app.dependency_overrides[get_runtime_manager] = lambda: SimpleNamespace(
        rollout=rollout
    )
    client = TestClient(app)
    assert client.get("/rollout/chains").json() == {"runs": [], "plans": {}}
    body = {"run": "a", "outcome": "failure", "failure_mode": "other", "note": "x"}
    assert client.post("/rollout/chain/verdict", json=body).status_code == 200
    assert seen[0] == ("a", "failure", "other", "x")
    for run, code in (("missing", 404), ("bad", 400), ("going", 409)):
        reply = client.post("/rollout/chain/verdict", json={**body, "run": run})
        assert reply.status_code == code
    wrong = client.post("/rollout/chain/verdict", json={**body, "outcome": "fine"})
    assert wrong.status_code == 422
