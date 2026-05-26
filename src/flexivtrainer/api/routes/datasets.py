from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from flexivtrainer.observability import info, ok
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/datasets", tags=["datasets"])


class CombineRequest(BaseModel):
    episode_paths: list[str] = Field(default_factory=list)
    output_name: str


@router.get("/episodes")
def list_episodes(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return {"episodes": runtime.list_episode_datasets()}


@router.get("/preview")
def preview(path: str, runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.preview_dataset(Path(path))


@router.get("/browse")
def browse(
    path: str | None = None,
    directories_only: bool = False,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    return runtime.browse_path(
        Path(path) if path else None, directories_only=directories_only
    )


@router.post("/combine")
def combine(
    request: CombineRequest, runtime: RuntimeManager = Depends(get_runtime_manager)
) -> dict:
    info(
        "Combining episode datasets",
        f"count={len(request.episode_paths)} output={request.output_name}",
    )
    result = runtime.combine_episodes(request.episode_paths, request.output_name)
    ok(
        "Combined dataset ready",
        " ".join(
            [
                f"root={result.get('root', request.output_name)}",
                f"episodes={result.get('episodes', len(request.episode_paths))}",
                f"output={result.get('output_name', request.output_name)}",
            ]
        ),
    )
    return result
