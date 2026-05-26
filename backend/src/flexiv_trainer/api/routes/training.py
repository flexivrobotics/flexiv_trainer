from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from flexiv_trainer.runtime.manager import RuntimeManager, get_runtime_manager
from flexiv_trainer.terminal import info, ok, warn

router = APIRouter(prefix="/training", tags=["training"])


class StartTrainingRequest(BaseModel):
    dataset_path: str
    output_dir: str
    policy_type: str = "diffusion"
    extra_args: list[str] = Field(default_factory=list)


@router.post("/bootstrap")
def bootstrap(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.bootstrap_training_module()
    ok("Training module bootstrapped")
    return result


@router.get("/policies")
def policies(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.training.list_policies()


@router.post("/start")
def start_training(
    request: StartTrainingRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    info(
        "Training job requested",
        f"policy={request.policy_type} dataset={request.dataset_path} output={request.output_dir}",
    )
    result = runtime.training.start(
        dataset_root=Path(request.dataset_path).resolve(),
        output_dir=Path(request.output_dir).resolve(),
        policy_type=request.policy_type,
        extra_args=request.extra_args,
    )
    if result.get("status") == "running":
        ok("Training job started", f"job_id={result.get('job_id', 'unknown')}")
    else:
        warn(
            "Training job returned unexpected initial state",
            str(result.get("status", "unknown")),
        )
    return result


@router.get("/status")
def status(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.training.status()
