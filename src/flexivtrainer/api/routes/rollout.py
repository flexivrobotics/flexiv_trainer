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

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from flexivtrainer.observability import info, ok
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/rollout", tags=["rollout"])


class StartRolloutRequest(BaseModel):
    checkpoint_path: str
    task: str = ""


class RolloutDeviceRequest(BaseModel):
    device: str = "auto"


@router.get("/devices")
def rollout_devices(
    force: bool = False,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    # Rollout inference and training share one computation-device setting; reuse
    # the (warmed-up) training device probe rather than running a second one.
    return runtime.training.evaluate_devices(force=force)


@router.put("/devices")
def set_rollout_device(
    request: RolloutDeviceRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    try:
        result = runtime.training.set_default_device(request.device)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ok("Rollout device updated", f"device={result.get('configured', 'auto')}")
    return result


@router.post("/start")
def start_rollout(
    request: StartRolloutRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    info("Rollout requested", f"checkpoint={request.checkpoint_path}")
    try:
        result = runtime.rollout.start(request.checkpoint_path, task=request.task)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Rollout started")
    return result


@router.get("/checkpoint-info")
def rollout_checkpoint_info(path: str) -> dict:
    from flexivtrainer.rollout.service import (
        _checkpoint_policy_type,
        _checkpoint_requires_task,
        _checkpoint_task,
    )

    return {
        "task": _checkpoint_task(path),
        "policy_type": _checkpoint_policy_type(path),
        "requires_task": _checkpoint_requires_task(path),
    }


@router.post("/stop")
def stop_rollout(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.rollout.stop()
    ok("Rollout stopped")
    return result


@router.get("/status")
def rollout_status(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.rollout.status()
