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

"""Chain and practice plans kept on disk, listed and started by name."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_step_rollout import (
    _Arm,
    _service,
    rdk,  # noqa: F401 - a fixture
)
from waits import WAIT_S

from flexivtrainer.api.app import create_app
from flexivtrainer.practice.session import PracticePlan
from flexivtrainer.rollout.chain import ChainPlan
from flexivtrainer.rollout.plans import PLANS_DIR, list_plans, read_plan
from flexivtrainer.runtime.manager import get_runtime_manager

LIFT = {
    "about": "lift the part",
    "steps": [{"name": "lift", "primitive": {"moves": [{"tool_m": [0, 0, -0.02]}]}}],
}
PRACTICE = {
    "checkpoint_path": "training/insert/pretrained_model",
    "plan": {
        "reward": {"seated": {"kind": "learned", "model": "checks/seated"}},
        "max_episode_steps": 100,
        "learner_host": "10.0.0.2",
        "episodes": 50,
    },
}


def _write(rollouts, name, data):
    directory = rollouts / PLANS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else json.dumps(data)
    (directory / f"{name}.json").write_text(text)


def test_plans_are_listed_with_what_they_run_or_why_they_cannot(tmp_path):
    assert list_plans(tmp_path) == []
    _write(tmp_path, "lift", LIFT)
    _write(tmp_path, "insert_practice", PRACTICE)
    _write(tmp_path, "broken", "{not json")
    _write(tmp_path, "empty_chain", {"steps": []})
    _write(tmp_path, "neither", {"moves": []})
    _write(tmp_path, "no_checkpoint", {"plan": PRACTICE["plan"]})
    _write(tmp_path, ".hidden", LIFT)
    plans = {entry["name"]: entry for entry in list_plans(tmp_path)}
    assert set(plans) == {
        "lift",
        "insert_practice",
        "broken",
        "empty_chain",
        "neither",
        "no_checkpoint",
    }
    assert plans["lift"] == {
        "name": "lift",
        "kind": "chain",
        "about": "lift the part",
        "steps": ["lift"],
    }
    assert plans["insert_practice"] == {
        "name": "insert_practice",
        "kind": "practice",
        "checkpoint": "training/insert/pretrained_model",
        "task": "",
        "reward": "seated",
        "episodes": 50,
        "learner": "10.0.0.2:50051",
    }
    assert plans["broken"]["kind"] is None and "Expecting" in plans["broken"]["error"]
    assert "at least one step" in plans["empty_chain"]["error"]
    assert "not a chain plan" in plans["neither"]["error"]
    assert "checkpoint_path" in plans["no_checkpoint"]["error"]


def test_a_plan_is_read_only_from_the_files_on_disk(tmp_path):
    _write(tmp_path, "lift", LIFT)
    kind, plan, data = read_plan(tmp_path, "lift")
    assert kind == "chain" and isinstance(plan, ChainPlan) and data == LIFT
    (tmp_path / "outside.json").write_text(json.dumps(LIFT))
    for name in ("nope", "../outside", f"{PLANS_DIR}/../outside"):
        with pytest.raises(FileNotFoundError, match="no plan named"):
            read_plan(tmp_path, name)


def test_a_chain_plan_runs_from_disk_and_reports_its_events(tmp_path, rdk):  # noqa: F811
    arm = _Arm()
    service, _ = _service(tmp_path, arm)
    _write(service._settings.storage.rollouts_root, "lift", LIFT)
    assert [entry["name"] for entry in service.plans()] == ["lift"]
    service.start_plan("lift")
    service._chain.join(WAIT_S)
    chain = service.status()["chain"]
    assert chain["state"] == "done"
    assert [(e["step"], e["event"]) for e in chain["recent"]] == [("lift", "try")]
    assert arm.pose[2] == pytest.approx(0.32)


def test_a_practice_plan_takes_its_checkpoint_from_the_storage_root(tmp_path, rdk):  # noqa: F811
    service, _ = _service(tmp_path, _Arm())
    _write(service._settings.storage.rollouts_root, "practice", PRACTICE)
    started = []
    service.start = lambda checkpoint, **options: started.append(
        (checkpoint, options)
    ) or {"status": "running"}
    service.start_plan("practice")
    checkpoint, options = started[0]
    assert checkpoint == str(tmp_path / "training/insert/pretrained_model")
    assert options["task"] == ""
    assert isinstance(options["practice"], PracticePlan)
    assert options["practice"].episodes == 50


def test_the_routes_list_plans_and_start_one_by_name():
    started = []

    def start_plan(name):
        started.append(name)
        if name == "missing":
            raise FileNotFoundError("no plan named 'missing'")
        if name == "bad":
            raise ValueError("a chain plan needs at least one step")
        if name == "busy":
            raise RuntimeError("Rollout is already running")
        return {"status": "running"}

    app = create_app()
    rollout = SimpleNamespace(
        plans=lambda: [{"name": "lift", "kind": "chain", "steps": ["lift"]}],
        start_plan=start_plan,
    )
    app.dependency_overrides[get_runtime_manager] = lambda: SimpleNamespace(
        rollout=rollout
    )
    client = TestClient(app)
    assert client.get("/rollout/plans").json() == {
        "plans": [{"name": "lift", "kind": "chain", "steps": ["lift"]}]
    }
    assert client.post("/rollout/plans/lift/start").status_code == 200
    assert client.post("/rollout/plans/missing/start").status_code == 404
    assert client.post("/rollout/plans/bad/start").status_code == 400
    busy = client.post("/rollout/plans/busy/start")
    assert busy.status_code == 409 and "already running" in busy.json()["detail"]
    assert started == ["lift", "missing", "bad", "busy"]
