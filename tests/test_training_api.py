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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from flexivtrainer.api.routes.training import (
    StartTrainingRequest,
    checkpoint_info,
    start_training,
)
from flexivtrainer.config import AppSettings, StorageConfig
from flexivtrainer.jobs.train_policy import TrainingService


def _checkpoint(tmp_path: Path) -> Path:
    step = tmp_path / "training" / "run" / "checkpoints" / "001000"
    model = step / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {},
                "output_features": {},
                "optimizer_lr": 1e-5,
            }
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"weights")
    return step


def test_checkpoint_info_returns_public_fine_tune_schema(tmp_path: Path) -> None:
    storage = StorageConfig(root=tmp_path)
    storage.ensure()
    service = TrainingService(AppSettings(storage=storage))
    runtime = SimpleNamespace(training=service)
    checkpoint = _checkpoint(tmp_path)

    payload = checkpoint_info(str(checkpoint), runtime)

    assert payload["checkpoint_path"] == str(checkpoint.resolve())
    assert payload["policy_type"] == "act"
    assert "model_path" not in payload
    assert "policy_config" not in payload
    names = {field["name"] for field in payload["fields"]}
    assert "optimizer_lr" in names
    assert "chunk_size" not in names


def test_checkpoint_info_rejects_path_outside_training_root(tmp_path: Path) -> None:
    storage = StorageConfig(root=tmp_path)
    storage.ensure()
    service = TrainingService(AppSettings(storage=storage))
    runtime = SimpleNamespace(training=service)

    with pytest.raises(HTTPException) as error:
        checkpoint_info(str(tmp_path / "datasets"), runtime)
    assert error.value.status_code == 403
    assert "Access denied" in error.value.detail


def test_start_forwards_fine_tune_contract(tmp_path: Path) -> None:
    captured = {}

    class _Training:
        def start(self, **kwargs):
            captured.update(kwargs)
            return {"status": "running", "job_id": "job-1"}

    runtime = SimpleNamespace(training=_Training())
    request = StartTrainingRequest.model_validate(
        {
            "dataset_path": str(tmp_path / "datasets" / "data"),
            "output_dir": str(tmp_path / "training" / "fine-tuned"),
            "policy_type": "act",
            "training_mode": "fine_tune",
            "checkpoint_path": str(tmp_path / "training" / "source"),
            "extra_args": ["--batch_size", "32"],
        }
    )
    response = start_training(request, runtime)

    assert response["status"] == "running"
    assert captured["training_mode"] == "fine_tune"
    assert captured["checkpoint_path"] == tmp_path / "training" / "source"
    assert captured["extra_args"] == ["--batch_size", "32"]


def _hub_dataset_runtime(exc: Exception) -> SimpleNamespace:
    def raiser(*args, **kwargs):
        raise exc

    return SimpleNamespace(training=SimpleNamespace(inspect_hub_dataset=raiser))


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        # These three statuses are what make the UI's token prompt fire, and the
        # untagged case must beat the generic HubError 502.
        ("HubDatasetNotTaggedError", 400),
        ("HubNotFoundError", 404),
        ("HubAuthError", 403),
        ("HubUnavailableError", 502),
        ("HubError", 502),
    ],
)
def test_hub_dataset_info_maps_hub_errors(exc: str, status: int) -> None:
    import flexivtrainer.data.hub as hub_module
    from flexivtrainer.api.routes.training import hub_dataset_info

    runtime = _hub_dataset_runtime(getattr(hub_module, exc)("boom"))

    with pytest.raises(HTTPException) as error:
        hub_dataset_info("acme/demo", None, None, runtime)
    assert error.value.status_code == status


def test_hub_dataset_info_returns_the_service_payload(tmp_path: Path) -> None:
    from flexivtrainer.api.routes.training import hub_dataset_info

    payload = {"repo_id": "acme/demo", "dataset_ok": "fine.", "dataset_warning": None}
    runtime = SimpleNamespace(
        training=SimpleNamespace(inspect_hub_dataset=lambda *a, **k: payload)
    )

    assert hub_dataset_info("acme/demo", None, None, runtime) is payload
