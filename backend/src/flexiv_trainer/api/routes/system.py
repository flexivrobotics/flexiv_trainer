from __future__ import annotations

from fastapi import APIRouter, Depends

from flexiv_trainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/summary")
def get_system_summary(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.system_summary()
