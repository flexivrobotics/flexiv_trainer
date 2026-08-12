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

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from flexivtrainer.data.hub import (
    HubAuthError,
    HubDatasetNotTaggedError,
    HubError,
    HubNotFoundError,
    HubUnavailableError,
)
from flexivtrainer.observability import info, ok, warn
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/training", tags=["training"])


class StartTrainingRequest(BaseModel):
    # Hub identifiers travel on their own fields so a repo id can never reach a
    # path resolver, and a filesystem path can never reach a downloader.
    dataset_source: Literal["local", "hub"] = "local"
    dataset_path: str | None = None
    dataset_repo_id: str | None = None
    dataset_revision: str | None = None
    output_dir: str
    policy_type: str = "diffusion"
    extra_args: list[str] = Field(default_factory=list)
    training_mode: Literal["new", "fine_tune"] = "new"
    checkpoint_source: Literal["local", "hub"] = "local"
    checkpoint_path: str | None = None
    checkpoint_repo_id: str | None = None
    checkpoint_revision: str | None = None

    @model_validator(mode="after")
    def _check_sources(self) -> StartTrainingRequest:
        if self.dataset_source == "hub":
            if not self.dataset_repo_id:
                raise ValueError("dataset_repo_id is required for dataset_source='hub'")
            if self.dataset_path:
                raise ValueError(
                    "dataset_path is not allowed with dataset_source='hub'"
                )
        elif not self.dataset_path:
            raise ValueError("dataset_path is required for dataset_source='local'")

        if self.checkpoint_source == "hub":
            if not self.checkpoint_repo_id:
                raise ValueError(
                    "checkpoint_repo_id is required for checkpoint_source='hub'"
                )
            if self.checkpoint_path:
                raise ValueError(
                    "checkpoint_path is not allowed with checkpoint_source='hub'"
                )
        return self


class TrainingDeviceRequest(BaseModel):
    device: str = "auto"


@router.post("/bootstrap")
def bootstrap(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.bootstrap_training_module()
    ok("Training module bootstrapped")
    return result


@router.get("/policies")
def policies(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.training.list_policies()


@router.get("/checkpoint-info")
def checkpoint_info(
    path: str | None = None,
    runtime: RuntimeManager = Depends(get_runtime_manager),
    source: Literal["local", "hub"] = "local",
    repo_id: str | None = None,
    revision: str | None = None,
) -> dict:
    try:
        if source == "hub":
            if not repo_id:
                raise ValueError("repo_id is required when source='hub'")
            result = runtime.training.inspect_hub_checkpoint(repo_id, revision)
        elif not path:
            raise ValueError("path is required when source='local'")
        else:
            result = runtime.training.inspect_checkpoint(Path(path))
    except HubNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HubAuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        status = 403 if str(exc).startswith("Access denied:") else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        key: result[key]
        for key in (
            "checkpoint_path",
            "policy_type",
            "policy_label",
            "fields",
        )
    }


@router.get("/devices")
def training_devices(
    force: bool = False,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    # ``force`` re-runs the probe for the manual "Evaluate devices" button;
    # normal page loads reuse the cached (warmed-up) result.
    return runtime.training.evaluate_devices(force=force)


@router.put("/devices")
def set_training_device(
    request: TrainingDeviceRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    try:
        result = runtime.training.set_default_device(request.device)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ok("Training device updated", f"device={result.get('configured', 'auto')}")
    return result


@router.post("/start")
def start_training(
    request: StartTrainingRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    info(
        "Training job requested",
        " ".join(
            [
                f"mode={request.training_mode}",
                f"policy={request.policy_type}",
                f"dataset={request.dataset_repo_id or request.dataset_path}",
                f"source={request.dataset_source}",
                f"output={request.output_dir}",
            ]
        ),
    )
    try:
        result = runtime.training.start(
            dataset_root=(
                Path(request.dataset_path) if request.dataset_path else None
            ),
            output_dir=Path(request.output_dir),
            policy_type=request.policy_type,
            extra_args=request.extra_args,
            training_mode=request.training_mode,
            checkpoint_path=(
                Path(request.checkpoint_path) if request.checkpoint_path else None
            ),
            dataset_source=request.dataset_source,
            dataset_repo_id=request.dataset_repo_id,
            dataset_revision=request.dataset_revision,
            checkpoint_source=request.checkpoint_source,
            checkpoint_repo_id=request.checkpoint_repo_id,
            checkpoint_revision=request.checkpoint_revision,
        )
    # Hub errors subclass RuntimeError, so they must precede the 409 handler.
    except HubDatasetNotTaggedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HubNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HubAuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HubUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except HubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        status = 403 if str(exc).startswith("Access denied:") else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except (FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if result.get("status") == "running":
        ok("Training job started", f"job_id={result.get('job_id', 'unknown')}")
    else:
        warn(
            "Training job returned unexpected initial state",
            str(result.get("status", "unknown")),
        )
    return result


@router.post("/pause")
def pause_training(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    try:
        result = runtime.training.pause()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Training job paused")
    return result


@router.post("/resume")
def resume_training(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    try:
        result = runtime.training.resume()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Training job resumed")
    return result


@router.post("/stop")
def stop_training(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    try:
        result = runtime.training.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Training job stopped")
    return result


@router.get("/status")
def status(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.training.status()
