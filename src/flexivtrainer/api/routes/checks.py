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

"""Learned checks: the trained ones, and training one from the labelling page."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from flexivtrainer.evaluation.check_training import CheckTraining, list_checks
from flexivtrainer.observability import info
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/checks", tags=["checks"])

_jobs_lock = threading.Lock()
_jobs: dict[Path, CheckTraining] = {}


class TrainCheckRequest(BaseModel):
    # The folder under <storage>/checks/ the check is saved to.
    name: str
    # The label the check learns (meta/check_labels.json).
    check: str
    data: list[str] = Field(min_length=1)
    cameras: list[str] = Field(default_factory=list)
    # Per camera, x0, y0, x1, y1 as fractions of the frame.
    crops: dict[str, list[float]] = Field(default_factory=dict)
    state: list[str] = Field(default_factory=list)
    encoder: Literal["resnet18", "pixels"] = "resnet18"
    epochs: int | None = Field(default=None, ge=1, le=500)


def _root(runtime: RuntimeManager) -> Path:
    return runtime.settings.storage.root.expanduser().resolve()


def _training(runtime: RuntimeManager) -> CheckTraining:
    root = _root(runtime)
    with _jobs_lock:
        if root not in _jobs:
            _jobs[root] = CheckTraining(root)
        return _jobs[root]


def shutdown() -> None:
    """Stop any check still training: the server is stopping."""
    with _jobs_lock:
        jobs = list(_jobs.values())
    for job in jobs:
        job.shutdown()


@router.get("")
def checks(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    """The trained checks, newest first, with their held-out reports."""
    return {"checks": list_checks(_root(runtime))}


@router.get("/train")
def training_status(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return _training(runtime).status()


@router.post("/train")
def train_check(
    request: TrainCheckRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    data = []
    for path in request.data:
        try:
            data.append(runtime.dataset_root(Path(path)))
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    info("Check training requested", f"check={request.check} name={request.name}")
    try:
        return _training(runtime).start(request.model_dump(), data)
    except (ValueError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/train/stop")
def stop_training(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    try:
        return _training(runtime).stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
