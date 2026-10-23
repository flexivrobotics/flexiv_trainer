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

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator

from flexivtrainer.data.hub import HubAuthError, HubError, HubNotFoundError
from flexivtrainer.observability import info, ok
from flexivtrainer.runtime.manager import RuntimeManager, get_runtime_manager

router = APIRouter(prefix="/rollout", tags=["rollout"])


class StartRolloutRequest(BaseModel):
    source: Literal["local", "hub"] = "local"
    checkpoint_path: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    task: str = ""
    # Escape hatch for a checkpoint that carries no action-name metadata; the
    # layout is validated against the policy's action width before use.
    action_names: list[str] | None = None
    # Record this rollout as a dataset with an outcome awaiting a verdict.
    record: bool = False
    # Override rollout.send_twist for this rollout; None keeps the setting.
    send_twist: bool | None = None

    @model_validator(mode="after")
    def _check_source(self) -> StartRolloutRequest:
        if self.source == "hub":
            if not self.repo_id:
                raise ValueError("repo_id is required for source='hub'")
            if self.checkpoint_path:
                raise ValueError("checkpoint_path is not allowed with source='hub'")
        elif not self.checkpoint_path:
            raise ValueError("checkpoint_path is required for source='local'")
        return self


class StartPracticeRequest(BaseModel):
    """Practise a step skill on the robot (practice/session.py)."""

    checkpoint_path: str
    task: str = ""
    # The practice plan: reward, max_episode_steps, learner_host, learner_port,
    # episodes, start, wait_after_success, seed.
    plan: dict[str, Any]


class StartChainRequest(BaseModel):
    """Run a task as a chain of skills and primitives (rollout/chain.py)."""

    # The chain plan: steps, checks, stiffness, check_s, about.
    plan: dict[str, Any]


class ContinueChainRequest(BaseModel):
    # "retry" runs the step again from its start check; "skip" counts it done.
    action: Literal["retry", "skip"] = "retry"


class RolloutDeviceRequest(BaseModel):
    device: str = "auto"


class VerdictRequest(BaseModel):
    outcome: Literal["success", "failure", "unknown"]
    failure_mode: str | None = None
    note: str = ""
    # Name a recorded rollout on disk instead of the one waiting in memory --
    # for a verdict that was not given before the application restarted.
    session: str | None = None
    episode: str | None = None


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
    info(
        "Rollout requested",
        f"checkpoint={request.repo_id or request.checkpoint_path} "
        f"source={request.source}",
    )
    try:
        result = runtime.rollout.start(
            request.checkpoint_path,
            task=request.task,
            source=request.source,
            repo_id=request.repo_id,
            revision=request.revision,
            action_names=request.action_names,
            record=request.record,
            send_twist=request.send_twist,
        )
    # Hub errors subclass RuntimeError, so they must precede the 409 handler.
    except HubNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HubAuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Rollout started")
    return result


@router.post("/practice")
def start_practice(
    request: StartPracticeRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    from flexivtrainer.practice.session import PracticePlan  # noqa: PLC0415

    try:
        plan = PracticePlan.from_dict(request.plan)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    info("Practice requested", f"checkpoint={request.checkpoint_path}")
    try:
        result = runtime.rollout.start(
            request.checkpoint_path, task=request.task, practice=plan
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Practice started")
    return result


@router.post("/practice/continue")
def continue_practice(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    """The operator has done what the reset cannot (a part latched home)."""
    try:
        return runtime.rollout.practice_continue()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/chain")
def start_chain(
    request: StartChainRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    from flexivtrainer.rollout.chain import ChainPlan  # noqa: PLC0415

    try:
        plan = ChainPlan.from_dict(request.plan)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    info("Chain requested", " -> ".join(step.name for step in plan.steps))
    try:
        result = runtime.rollout.start_chain(plan)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Chain started")
    return result


@router.post("/chain/continue")
def continue_chain(
    request: ContinueChainRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    """The operator has done what the chain could not: go on."""
    try:
        return runtime.rollout.chain_continue(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/plans")
def list_plans(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    """The chain and practice plans in <storage>/rollouts/plans/."""
    return {"plans": runtime.rollout.plans()}


@router.post("/plans/{name}/start")
def start_plan(
    name: str, runtime: RuntimeManager = Depends(get_runtime_manager)
) -> dict:
    info("Plan requested", f"plan={name}")
    try:
        result = runtime.rollout.start_plan(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Plan started", f"plan={name}")
    return result


@router.get("/checkpoint-info")
def rollout_checkpoint_info(
    path: str | None = None,
    source: Literal["local", "hub"] = "local",
    repo_id: str | None = None,
    revision: str | None = None,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    from flexivtrainer.rollout.checkpoint import (
        _checkpoint_policy_type,
        _checkpoint_requires_task,
        _checkpoint_task,
        checkpoint_action_names,
        checkpoint_action_output_dim,
        resolve_checkpoint_path,
        resolve_hub_checkpoint,
    )

    try:
        if source == "hub":
            if not repo_id:
                raise ValueError("repo_id is required when source='hub'")
            checkpoint_path = resolve_hub_checkpoint(
                repo_id, revision, runtime.settings
            )
        elif not path:
            raise ValueError("path is required when source='local'")
        else:
            checkpoint_path = resolve_checkpoint_path(
                path, runtime.settings.storage.root
            )
    except HubNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HubAuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Checkpoint not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    resolved_path = str(checkpoint_path)
    try:
        action_names = checkpoint_action_names(
            resolved_path, runtime.settings.storage.root, settings=runtime.settings
        )
    except ValueError:
        action_names = None
    action_dim = checkpoint_action_output_dim(resolved_path)
    return {
        "task": _checkpoint_task(resolved_path),
        "policy_type": _checkpoint_policy_type(resolved_path),
        "requires_task": _checkpoint_requires_task(resolved_path),
        "repo_id": repo_id if source == "hub" else None,
        "action_names": action_names,
        "action_dim": action_dim,
        # Decide here rather than making the operator reason about whether an
        # unknown layout happens to be inferable for their arm count.
        "layout_warning": _layout_warning(
            action_names, action_dim, runtime.get_active_sides()
        ),
        "layout_ok": _layout_ok(action_names, action_dim, runtime.get_active_sides()),
    }


def _layout_warning(
    action_names: list[str] | None, action_dim: int | None, sides: list[str]
) -> str | None:
    """Explain up front why a rollout would refuse to start, or return None."""
    if action_dim is None or not sides:
        return None

    from flexivtrainer.rollout.executors.waypoint import layout_problem

    # This string is returned to an HTTP client, so it is composed from the
    # arguments above rather than from caught exception text.
    problem = layout_problem(action_names, action_dim, sides)
    return f"{problem}." if problem else None


def _layout_ok(
    action_names: list[str] | None, action_dim: int | None, sides: list[str]
) -> str | None:
    """Confirm a usable layout, structured like its warning counterpart."""
    if action_dim is None or not sides:
        return None

    from flexivtrainer.rollout.executors.waypoint import (
        build_action_layout,
        canonical_action_names,
        layout_confirmation,
        recorded_layout_confirmation,
    )

    # Run the rollout's own validation, so this never claims a match it rejects.
    try:
        if action_names is None:
            names = canonical_action_names(action_dim, sides)
            confirmation = layout_confirmation(action_dim, len(sides))
        else:
            names = action_names
            confirmation = recorded_layout_confirmation(action_dim, len(sides))
        build_action_layout(names, sides, action_dim)
    except ValueError:
        return None
    return f"{confirmation}."


@router.post("/stop")
def stop_rollout(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    result = runtime.rollout.stop()
    ok("Rollout stopped")
    return result


@router.get("/status")
def rollout_status(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return runtime.rollout.status()


@router.post("/verdict")
def rollout_verdict(
    request: VerdictRequest,
    runtime: RuntimeManager = Depends(get_runtime_manager),
) -> dict:
    try:
        result = runtime.rollout.verdict(
            request.outcome,
            failure_mode=request.failure_mode,
            note=request.note,
            session=request.session,
            episode=request.episode,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Rollout verdict saved", f"{result['episode']}: {result['verdict']}")
    return result


@router.post("/takeover")
def take_over_rollout(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    """Stop the policy and give the follower to the leader arm.

    Returns at once; `takeover` in the status follows the switch.
    """
    try:
        result = runtime.rollout.take_over()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    info("Rollout takeover requested")
    return result


@router.post("/hand-back")
def hand_back_rollout(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    """Give the follower back to the policy, which plans afresh."""
    try:
        result = runtime.rollout.hand_back()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    info("Rollout hand-back requested")
    return result


@router.get("/evaluations")
def list_evaluations(runtime: RuntimeManager = Depends(get_runtime_manager)) -> dict:
    return {"sessions": runtime.rollout.evaluation_sessions()}


@router.get("/evaluations/{name}")
def get_evaluation(
    name: str, runtime: RuntimeManager = Depends(get_runtime_manager)
) -> dict:
    try:
        return runtime.rollout.evaluation_session(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/evaluations/{name}/next")
def start_next_trial(
    name: str, runtime: RuntimeManager = Depends(get_runtime_manager)
) -> dict:
    info("Evaluation trial requested", f"session={name}")
    try:
        result = runtime.rollout.start_next_trial(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HubNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HubAuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    ok("Evaluation trial started")
    return result
