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

"""Training a learned check from the web UI, as its own process."""

import sys
import time

import pytest
from fastapi.testclient import TestClient
from test_datasets_api import _make_manager, _recording
from test_learned_check import EPISODES, FZ, _dataset, _label_all

from flexivtrainer.api.app import create_app
from flexivtrainer.api.routes import checks as checks_routes
from flexivtrainer.evaluation.check_training import CheckTraining, list_checks
from flexivtrainer.runtime.manager import get_runtime_manager

REQUEST = {"name": "closed", "check": "closed", "cameras": ["wrist"], "state": []}


def _wait(job, seconds=120.0):
    deadline = time.monotonic() + seconds
    while job.status()["state"] in {"running", "stopping"}:
        assert time.monotonic() < deadline, job.status()["log"][-5:]
        time.sleep(0.05)
    return job.status()


def test_the_command_is_the_one_the_readme_gives(tmp_path):
    job = CheckTraining(tmp_path, python="python")
    data = [tmp_path / "datasets" / "a", tmp_path / "datasets" / "b"]
    command = job.command(
        {
            **REQUEST,
            "crops": {"wrist": [0.15, 0.1, 0.85, 0.7]},
            "state": [FZ],
            "epochs": 5,
        },
        data,
    )
    assert command == [
        "python", "-m", "flexivtrainer.evaluation.learned_check", "train",
        "--check", "closed", "--out", str(tmp_path / "checks" / "closed"),
        "--data", str(data[0]), "--data", str(data[1]),
        "--camera", "wrist", "--crop", "wrist=0.15,0.1,0.85,0.7",
        "--state", FZ, "--epochs", "5",
    ]  # fmt: skip
    for request, message in (
        ({**REQUEST, "name": "../x"}, "a check's name"),
        ({**REQUEST, "cameras": [], "state": []}, "looks at a camera"),
        ({**REQUEST, "crops": {"ego": [0, 0, 1, 1]}}, "does not look at"),
        ({**REQUEST, "crops": {"wrist": [0.5, 0, 0.4, 1]}}, "each low below"),
        ({**REQUEST, "cameras": ["../wrist"]}, "not a camera name"),
    ):
        with pytest.raises(ValueError, match=message):
            job.command(request, data)
    with pytest.raises(ValueError, match="at least one labelled dataset"):
        job.command(REQUEST, [])


def test_a_check_trains_in_its_own_process_and_is_listed(tmp_path):
    pytest.importorskip("lerobot")
    root = _dataset(tmp_path / "datasets" / "demos", EPISODES)
    _label_all(root, EPISODES)
    job = CheckTraining(tmp_path)
    started = job.start({**REQUEST, "encoder": "pixels", "epochs": 3}, [root])
    assert started["state"] == "running" and started["model"] == "checks/closed"
    status = _wait(job)
    assert status["state"] == "done", status["log"][-5:]
    assert any("epoch 3/3" in line for line in status["log"])
    held_out = status["result"]["held_out"]
    assert held_out["episodes_reached"] + held_out["episodes_never"] >= 1
    assert "false_passes" in held_out
    listed = list_checks(tmp_path)
    assert [c["name"] for c in listed] == ["closed"]
    assert listed[0]["check"] == "closed" and listed[0]["cameras"] == ["wrist"]
    assert listed[0]["model"] == "checks/closed"
    with pytest.raises(FileExistsError, match="exists"):
        job.start(REQUEST, [root])


def test_a_failed_training_says_why_and_a_running_one_can_be_stopped(
    tmp_path, monkeypatch
):
    job = CheckTraining(tmp_path)
    data = [tmp_path / "datasets" / "a"]
    monkeypatch.setattr(
        job,
        "command",
        lambda request, data: [
            sys.executable, "-c",
            "import sys; print('reading', file=sys.stderr); "
            "print('error: no episode has a label', file=sys.stderr); sys.exit(2)",
        ],
    )  # fmt: skip
    job.start(REQUEST, data)
    status = _wait(job)
    assert status["state"] == "failed"
    assert status["error"] == "no episode has a label"
    monkeypatch.setattr(
        job,
        "command",
        lambda request, data: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    job.start({**REQUEST, "name": "slow"}, data)
    with pytest.raises(RuntimeError, match="already training"):
        job.start({**REQUEST, "name": "other"}, data)
    assert job.stop()["state"] == "stopping"
    assert _wait(job, 10.0)["state"] == "stopped"
    with pytest.raises(RuntimeError, match="No check is training"):
        job.stop()


def test_the_routes_keep_training_to_the_storage_root(tmp_path, monkeypatch):
    app = create_app()
    manager = _make_manager(tmp_path / "storage")
    app.dependency_overrides[get_runtime_manager] = lambda: manager
    client = TestClient(app)
    monkeypatch.setattr(checks_routes, "_jobs", {})
    outside = _recording(tmp_path / "elsewhere", [10])
    body = {**REQUEST, "data": [str(outside)]}
    assert client.post("/checks/train", json=body).status_code == 403
    missing = {**body, "data": [str(tmp_path / "storage" / "datasets" / "nope")]}
    assert client.post("/checks/train", json=missing).status_code == 404
    inside = _recording(tmp_path / "storage" / "datasets" / "demos", [10])
    bad = {**REQUEST, "data": [str(inside)], "cameras": [], "state": []}
    reply = client.post("/checks/train", json=bad)
    assert reply.status_code == 400 and "looks at a camera" in reply.json()["detail"]
    assert client.get("/checks/train").json()["state"] == "idle"
    assert client.post("/checks/train/stop").status_code == 409
    assert client.get("/checks").json() == {"checks": []}
