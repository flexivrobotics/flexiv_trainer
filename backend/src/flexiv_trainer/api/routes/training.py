from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from flexiv_trainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/training", tags=["training"])


class StartTrainingRequest(BaseModel):
    dataset_path: str
    output_dir: str
    policy_type: str = "diffusion"
    extra_args: list[str] = Field(default_factory=list)


@router.post("/bootstrap")
def bootstrap(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.bootstrap_training_module()


@router.get("/policies")
def policies(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.training.list_policies()


@router.post("/start")
def start_training(
    request: StartTrainingRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    return runtime.training.start(
        dataset_root=Path(request.dataset_path).resolve(),
        output_dir=Path(request.output_dir).resolve(),
        policy_type=request.policy_type,
        extra_args=request.extra_args,
    )


@router.get("/status")
def status(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.training.status()
