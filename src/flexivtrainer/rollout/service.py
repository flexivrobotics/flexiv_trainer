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

"""Run trained LeRobot policies on follower robots through the RDK API."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from flexivtrainer.config import AppSettings, TeleopRobotPair
from flexivtrainer.data.control_settings import ControlSettings
from flexivtrainer.data.lerobot_io import active_camera_names
from flexivtrainer.evaluation.checks import (
    Check,
    GripperWidth,
    read_ticks,
    run_checks,
)
from flexivtrainer.evaluation.outcome import (
    DEFAULT_FAILURE_MODES,
    RolloutOutcome,
    Verdict,
    read_outcome,
    write_outcome,
)
from flexivtrainer.evaluation.session import ADHOC_SESSION, SessionStore
from flexivtrainer.jobs.train_policy import _encode_ui_log
from flexivtrainer.observability import describe_exception, ok, warn
from flexivtrainer.policies import act as act_policy
from flexivtrainer.policies import bspline_diffusion as bspline_policy
from flexivtrainer.policies import diffusion as diffusion_policy
from flexivtrainer.policies import dit as dit_policy
from flexivtrainer.policies import import_policy_plugins
from flexivtrainer.practice.runner import StepRunner, step_action_names
from flexivtrainer.practice.session import PracticePlan, PracticeRunner
from flexivtrainer.practice.space import PracticeSpace
from flexivtrainer.rollout import hardware
from flexivtrainer.rollout.chain import (
    CHAINS_DIR,
    ChainPlan,
    ChainRunner,
    ChainStep,
    Outcome,
)
from flexivtrainer.rollout.checkpoint import (
    _checkpoint_target_hz,
    _default_policy_loader,
    _positive_float,
    checkpoint_action_names,
    checkpoint_action_output_dim,
    checkpoint_control,
    checkpoint_gripper_command_metadata,
    checkpoint_image_resolutions,
    checkpoint_space,
    checkpoint_state_input_dim,
    resolve_checkpoint_path,
    resolve_hub_checkpoint,
)
from flexivtrainer.rollout.executors.bspline import (
    BSplineActionLayout,
    parse_bspline_action_layout,
)
from flexivtrainer.rollout.executors.gripper import initialize_gripper_executor
from flexivtrainer.rollout.executors.waypoint import (
    build_action_layout,
    canonical_action_names,
    layout_confirmation,
    recorded_layout_confirmation,
)
from flexivtrainer.rollout.hardware import _default_robot_factory
from flexivtrainer.rollout.live_checks import LiveChecks
from flexivtrainer.rollout.plans import CHAIN, list_plans, read_plan
from flexivtrainer.rollout.primitives import Primitive, PrimitiveRunner
from flexivtrainer.rollout.recorder import (
    RecordingPlan,
    RolloutRecorder,
    recorded_state_names,
)
from flexivtrainer.rollout.runners.bspline import BSplineRunner
from flexivtrainer.rollout.runners.waypoint import WaypointRunner
from flexivtrainer.rollout.safety import SafetyEnvelope
from flexivtrainer.rollout.stiffness import (
    NOT_RECORDED,
    StiffnessChoice,
    choose_rollout_stiffness,
)
from flexivtrainer.rollout.takeover import LeaderTrigger, Takeover, TakeoverRefused
from flexivtrainer.runtime.gripper_session import GripperInitializationRegistry

_ROLLOUT_OVERRIDES = {
    "act": act_policy.apply_rollout_overrides,
    "bspline_diffusion": bspline_policy.apply_rollout_overrides,
    "diffusion": diffusion_policy.apply_rollout_overrides,
    "multi_task_dit": dit_policy.apply_rollout_overrides,
}


@dataclass(frozen=True)
class TrialContext:
    """The planned trial a rollout runs as (evaluation/session.py)."""

    session: str
    trial: int
    arm: str
    condition: str
    description: str
    blind: bool
    failure_modes: tuple[str, ...]


def _as_list(values: Any) -> list[float] | None:
    return None if values is None else [float(v) for v in values]


def _takeover_wrench(settings: AppSettings) -> list[float] | None:
    """The contact-wrench ceiling, also applied to the follower under the
    leader arm (SetFollowerMaxContactWrench takes the same six numbers)."""
    return _as_list(settings.rollout.max_contact_wrench)


def _borrowed_factory(
    followers: list[str], handles: tuple[Any, ...]
) -> Callable[[str], Any]:
    """A robot factory that hands out the teleop connection's own handles."""
    by_serial = dict(zip(followers, handles, strict=True))

    def borrow(serial: str) -> Any:
        robot = by_serial[serial]
        # Stop() leaves the arm holding in IDLE, the mode the rollout starts
        # from (gripper preparation requires it); the teleop loop is stopped.
        stop = getattr(robot, "Stop", None)
        if callable(stop):
            stop()
        return robot

    return borrow


def _episode_name(trial: TrialContext | None) -> str:
    """Timestamp plus trial number. Never the checkpoint: a blind trial's
    directory name is shown to the operator."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_t{trial.trial:03d}" if trial is not None else f"{stamp}_rollout"


def _describe_space(space: PracticeSpace) -> str:
    """One line on how a step skill acts: rate, bounds, force axes, inputs."""
    moves = [f"{a} {b * 1000:g} mm" for a, b in space.step_m.items()]
    turns = [f"r{a} {b:g} deg" for a, b in space.step_deg.items()]
    forces = [f"{a} {b:g} N" for a, b in space.force_n.items()]
    return (
        f"{space.rate_hz:g} steps/s; per step at most "
        f"{', '.join(moves + turns) or 'no move'}; force along tool "
        f"{', '.join(forces) or 'none'}; cameras {list(space.views)}; "
        f"last {space.last_actions} commands"
    )


def _describe_rollout_overrides(rollout_cfg: Any) -> str:
    """Summarize an applied override; families expose different knobs."""
    parts: list[str] = []
    scheduler = getattr(rollout_cfg, "noise_scheduler_type", "")
    if scheduler:
        steps = getattr(rollout_cfg, "num_denoise_steps", 0)
        parts.append(f"scheduler={scheduler} inference_steps={steps}")
    if getattr(rollout_cfg, "disable_temporal_ensemble", False):
        parts.append(
            "temporal ensembling disabled; executing "
            f"n_action_steps={rollout_cfg.n_action_steps} per chunk"
        )
    if getattr(rollout_cfg, "compile_model", False):
        # Compilation is lazy, so the cost lands on the first inference step.
        parts.append("model compilation enabled; first inference includes compilation")
    return "; ".join(parts) or "applied"


class RolloutService:
    """Lifecycle and background control loop for policy rollout."""

    def __init__(
        self,
        settings: AppSettings,
        cameras: Any,
        teleop: Any,
        get_robot_pairs: Callable[[], list[TeleopRobotPair]],
        get_active_sides: Callable[[], list[str]],
        get_active_cameras: Callable[[], list[str]] | None = None,
        *,
        get_end_effector_config: Callable[[], dict[str, Any]] | None = None,
        get_gripper_default_width: Callable[[], float | None] | None = None,
        gripper_initialization_registry: GripperInitializationRegistry | None = None,
        policy_loader: Callable[[str, str], Any] = _default_policy_loader,
        robot_factory: Callable[[str], Any] = _default_robot_factory,
        resolve_device: Callable[[str], str] | None = None,
    ) -> None:
        self._settings = settings
        self._cameras = cameras
        self._teleop = teleop
        self._get_robot_pairs = get_robot_pairs
        self._get_active_sides = get_active_sides
        # Lets a caller construct the service without a camera configuration.
        self._get_active_cameras = get_active_cameras or (
            lambda: active_camera_names(get_active_sides())
        )
        self._get_end_effector_config = get_end_effector_config or (lambda: {})
        self._get_gripper_default_width = get_gripper_default_width or (lambda: None)
        self._gripper_initialization_registry = gripper_initialization_registry
        self._policy_loader = policy_loader
        self._robot_factory = robot_factory
        if resolve_device is None:
            from flexivtrainer.jobs.train_policy import resolve_training_device

            resolve_device = resolve_training_device
        self._resolve_device = resolve_device

        self._lock = threading.Lock()
        self._running = False
        self._stopping = False
        self._error: str | None = None
        self._stop_reason: str | None = None
        self._checkpoint_path: str | None = None
        self._checkpoint_repo_id: str | None = None
        self._task: str | None = None
        self._robots: list[Any] = []
        self._device = "cpu"
        self._target_hz: float | None = None
        self._sides: list[str] = []
        self._runner: (
            WaypointRunner | BSplineRunner | StepRunner | PrimitiveRunner | None
        ) = None
        self._practice_live: LiveChecks | None = None
        # The current or last chain of skills and primitives.
        self._chain: ChainRunner | None = None
        self._stop_event = threading.Event()
        self._logs: deque[str] = deque(maxlen=2000)
        self._metrics: deque[dict[str, Any]] = deque(maxlen=300)
        # ~3 s at the sampled rate; only the newest drives the UI gauges.
        self._wrench: deque[dict[str, Any]] = deque(maxlen=30)
        self._send_twist: bool | None = None
        self._safety: SafetyEnvelope | None = None
        # The Cartesian stiffness of the current or last rollout, and its source.
        self._stiffness: StiffnessChoice | None = None
        self._trial: TrialContext | None = None
        # The recording of the current or last rollout while it is written.
        self._recording: dict[str, Any] | None = None
        # The recorded rollout whose verdict is still to be given.
        self._awaiting: dict[str, Any] | None = None
        # Leader-arm takeover of the current or last rollout, or why the
        # current one has none.
        self._takeover: Takeover | None = None
        self._takeover_reason: str | None = None

    def status(self) -> dict[str, Any]:
        chain = self._chain
        chain_alive = chain is not None and chain.is_alive()
        with self._lock:
            if self._stopping or (chain_alive and chain.stop_event.is_set()):
                status = "stopping"
            elif self._running or chain_alive:
                status = "running"
            elif self._error:
                status = "failed"
            else:
                status = "idle"
            blind = self._trial is not None and self._trial.blind
            return {
                "status": status,
                # A blind trial shows only its arm's label; the checkpoint would
                # tell the operator which policy they are judging.
                "checkpoint_path": None if blind else self._checkpoint_path,
                # Set for Hub rollouts so the UI shows "acme/policy" rather than
                # an opaque cache directory.
                "checkpoint_repo_id": None if blind else self._checkpoint_repo_id,
                "checkpoint_label": (
                    self._trial.arm if self._trial is not None else None
                ),
                "trial": self._public_trial(self._trial),
                "send_twist": self._send_twist,
                "send_twist_default": bool(self._settings.rollout.send_twist),
                "safety": self._safety.describe() if self._safety is not None else None,
                "stiffness": (
                    self._stiffness.to_dict() if self._stiffness is not None else None
                ),
                "recording": self._recording_status(),
                "takeover": self._takeover_status(),
                "awaiting_verdict": self._public_awaiting(),
                "task": self._task,
                "error": self._error,
                "stop_reason": self._stop_reason,
                "logs": list(self._logs),
                "log_lines": len(self._logs),
                "metrics": list(self._metrics),
                "wrench": list(self._wrench),
                "sides": list(self._sides),
                "target_hz": self._target_hz,
                "practice": (
                    self._runner.practice_status()
                    if isinstance(self._runner, PracticeRunner)
                    else None
                ),
                "chain": chain.status() if chain is not None else None,
            }

    def practice_continue(self) -> dict[str, Any]:
        """The operator has done what a practice reset cannot: go on."""
        runner = self._runner
        if not isinstance(runner, PracticeRunner) or not runner.is_alive():
            raise RuntimeError("No practice is running")
        runner.go_on()
        return self.status()

    def _on_practice_finished(self, stop_reason: str | None, steps: int) -> None:
        live, self._practice_live = self._practice_live, None
        if live is not None:
            live.stop()
        self._on_runner_finished(stop_reason, steps)

    def _append_log(
        self, level: str, source: str, message: str, detail: str = ""
    ) -> None:
        self._logs.append(_encode_ui_log(level, source, message, detail))

    def start(
        self,
        checkpoint_path: str | None = None,
        task: str | None = None,
        *,
        source: str = "local",
        repo_id: str | None = None,
        revision: str | None = None,
        action_names: list[str] | None = None,
        record: bool = False,
        send_twist: bool | None = None,
        trial: TrialContext | None = None,
        practice: PracticePlan | None = None,
    ) -> dict[str, Any]:
        """Roll a policy out; with `practice`, practise a step skill instead
        (practice/session.py): episodes judged by its reward check, their steps
        sent to a learner."""
        self._refuse_during_chain()
        return self._start(
            checkpoint_path,
            task,
            source=source,
            repo_id=repo_id,
            revision=revision,
            action_names=action_names,
            record=record,
            send_twist=send_twist,
            trial=trial,
            practice=practice,
        )

    def _start(
        self,
        checkpoint_path: str | None = None,
        task: str | None = None,
        *,
        source: str = "local",
        repo_id: str | None = None,
        revision: str | None = None,
        action_names: list[str] | None = None,
        record: bool = False,
        send_twist: bool | None = None,
        trial: TrialContext | None = None,
        practice: PracticePlan | None = None,
        live_checks: LiveChecks | None = None,
        keep_logs: bool = False,
    ) -> dict[str, Any]:
        """`start`, and for a chain's step (rollout/chain.py): `live_checks`
        scored on every tick, one of which ends the rollout, and the logs of
        the steps before kept."""
        task = task.strip() if isinstance(task, str) else None
        task = task or None
        if practice is not None and (record or trial is not None):
            raise ValueError(
                "Practice keeps its steps with the learner: start it without "
                "recording and outside an evaluation session"
            )
        record = record or trial is not None
        with self._lock:
            if self._awaiting is not None:
                raise RuntimeError(
                    "The last recorded rollout is waiting for its verdict; give it "
                    "one (or mark it unknown) before starting another."
                )
            if self._recording is not None and self._recording.get("saving"):
                raise RuntimeError("The last rollout's recording is still being saved.")
            # A Python thread cannot be killed and torch.compile is not
            # interruptible, so refuse rather than let two planners overlap.
            active_runner = self._runner
            planner_alive = active_runner is not None and active_runner.is_alive()
            if self._stopping or (planner_alive and not self._running):
                raise RuntimeError(
                    "The previous rollout planner has not exited yet; wait for "
                    "it to finish and retry."
                )
            if self._running:
                raise RuntimeError("Rollout is already running")

        lend, takeover_reason = self._preflight_teleop(self._follower_serials())

        checkpoint_origin = checkpoint_path
        try:
            if source == "hub":
                if not repo_id:
                    raise ValueError("repo_id is required when source='hub'")
                checkpoint_origin = repo_id
                self._append_log(
                    "INFO", "HUB", "Fetching checkpoint from HuggingFace", repo_id
                )
                resolved_checkpoint = resolve_hub_checkpoint(
                    repo_id, revision, self._settings
                )
            elif source != "local":
                raise ValueError(f"Unsupported checkpoint source: {source!r}")
            elif not checkpoint_path:
                raise ValueError("checkpoint_path is required when source='local'")
            else:
                resolved_checkpoint = resolve_checkpoint_path(
                    checkpoint_path, self._settings.storage.root
                )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Checkpoint not found: {checkpoint_origin}") from exc
        # HubError propagates as-is so the route can map its subclasses to
        # distinct status codes rather than a blanket 409.
        checkpoint_path = str(resolved_checkpoint)
        # Before anything is loaded or moved: a rollout with no known stiffness
        # is refused here.
        stiffness = choose_rollout_stiffness(
            self._settings.rollout.cartesian_stiffness,
            checkpoint_control(checkpoint_path),
            allow_controller_default=(
                self._settings.rollout.allow_controller_default_stiffness
            ),
        )

        device = self._resolve_device(self._settings.training.default_device)
        sides = self._get_active_sides()
        camera_names = self._get_active_cameras()
        self._preflight_cameras(camera_names)
        followers = self._follower_serials()
        if not followers:
            raise RuntimeError("No follower robot serial is configured")

        try:
            import_policy_plugins(self._settings.policies.plugin_packages)
            policy, preprocessor, postprocessor = self._policy_loader(
                checkpoint_path, device
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load policy: {describe_exception(exc)}"
            ) from exc
        policy_type = getattr(getattr(policy, "config", None), "type", None) or getattr(
            policy, "name", ""
        )
        is_bspline = policy_type == "bspline_diffusion"
        # A checkpoint with a practice space is a step skill, whatever its type.
        space = None if is_bspline else checkpoint_space(checkpoint_path)
        if practice is not None and space is None:
            raise RuntimeError(
                "Practice needs a step skill: a checkpoint with a "
                "practice_space.json beside its weights"
            )
        if is_bspline and record:
            raise RuntimeError(
                "Recording B-spline rollouts is not supported yet; start it without "
                "recording."
            )
        if is_bspline and live_checks is not None:
            raise RuntimeError(
                "A B-spline policy cannot be ended by a check yet, so it cannot "
                "be a step of a chain"
            )
        # A step skill's frames are read at the size its strips refer to; what
        # it was trained on is smaller, cut from them.
        image_resolutions = (
            dict(space.frame_size)
            if space is not None
            else checkpoint_image_resolutions(checkpoint_path)
        )
        rollout_cfg = self._settings.policies.rollout_for(policy_type)
        if space is not None:
            # Its arm is read at the space's own rate, and its step is part of
            # what it learned: no playback speed.
            target_hz = float(space.sense_hz)
        else:
            dataset_hz = self._resolve_target_hz(
                checkpoint_path, policy, require_metadata=is_bspline
            )
            target_hz = self._apply_playback_speed(dataset_hz, rollout_cfg)
        override_fn = _ROLLOUT_OVERRIDES.get(policy_type)
        overrides_applied = (
            override_fn(policy, rollout_cfg) if override_fn is not None else False
        )
        compile_act = bool(
            policy_type == "act" and getattr(rollout_cfg, "compile_model", False)
        )
        compile_mode = str(getattr(rollout_cfg, "compile_mode", "reduce-overhead"))
        overrides_applied |= compile_act
        uses_cuda_graphs = bool(
            compile_act
            and compile_mode == "reduce-overhead"
            and str(device).startswith("cuda")
        )
        bspline_layout: BSplineActionLayout | None = None
        # config.json is what from_pretrained loaded the policy from, so its
        # declared state width is the one the policy's normalizer will enforce.
        state_dim = checkpoint_state_input_dim(checkpoint_path)
        waypoint_layout: list[dict[str, Any]] | None = None
        waypoint_action_dim: int | None = None
        waypoint_layout_inferred = False
        waypoint_action_names: list[str] | None = None
        waypoint_gripper_sides: tuple[str, ...] = ()
        waypoint_gripper_target_mode = "width"
        gripper_command_parameters = None
        end_effector_config: dict[str, Any] = {}
        gripper_default_width_m = self._get_gripper_default_width()
        if is_bspline:
            bspline_layout = self._preflight_bspline(
                policy, sides, followers, target_hz
            )
            end_effector_config = dict(self._get_end_effector_config() or {})
            self._preflight_grippers(
                bspline_layout.gripper_sides,
                end_effector_config,
                policy_label="B-spline",
            )
            if bspline_layout.gripper_target_mode == "target_width":
                gripper_command_parameters = checkpoint_gripper_command_metadata(
                    checkpoint_path
                )
                if gripper_command_parameters is None:
                    raise ValueError(
                        "B-spline checkpoint predicts gripper.target_width but "
                        "has no gripper_command.json"
                    )
        elif space is not None:
            if len(sides) != 1 or len(followers) != 1:
                raise RuntimeError(
                    f"A step skill drives one arm, but {len(followers)} are "
                    "configured; set single-arm mode"
                )
            missing = [v for v in space.views if v not in camera_names]
            if missing:
                raise RuntimeError(
                    f"The step skill sees camera {missing}, which is not "
                    f"configured; the rig has {camera_names}"
                )
            # What its recording stores as the action: the commanded pose.
            waypoint_action_names = step_action_names(sides[0])
        else:
            (
                waypoint_layout,
                waypoint_action_dim,
                waypoint_layout_inferred,
                waypoint_action_names,
            ) = self._preflight_waypoint(
                checkpoint_path,
                policy,
                sides,
                followers,
                action_names=action_names,
            )
            waypoint_gripper_sides = tuple(
                str(arm["side"])
                for arm in waypoint_layout
                if isinstance(arm.get("gripper_width"), int)
            )
            modes = {
                str(arm.get("gripper_target_mode"))
                for arm in waypoint_layout
                if arm.get("gripper_target_mode") is not None
            }
            if modes:
                waypoint_gripper_target_mode = next(iter(modes))
            if waypoint_gripper_target_mode == "target_width":
                gripper_command_parameters = checkpoint_gripper_command_metadata(
                    checkpoint_path
                )
                if gripper_command_parameters is None:
                    raise ValueError(
                        "Waypoint checkpoint predicts gripper.target_width but "
                        "has no gripper_command.json"
                    )
            if waypoint_gripper_sides:
                end_effector_config = dict(self._get_end_effector_config() or {})
                self._preflight_grippers(
                    waypoint_gripper_sides,
                    end_effector_config,
                    policy_label="Waypoint",
                )
            self._apply_n_action_steps(policy, rollout_cfg)

        trigger_arm: int | None = None
        if bspline_layout is not None:
            takeover_reason = "B-spline policies cannot be taken over yet"
        elif lend:
            trigger_arm = self._preflight_takeover_trigger(sides)
        safety = SafetyEnvelope.from_settings(self._settings.rollout)
        effective_twist = (
            self._settings.rollout.send_twist
            if send_twist is None
            else bool(send_twist)
        )

        # Per run: a zombie planner keeps the old event, which stays set, so
        # clearing a shared one can no longer un-stop it.
        self._stop_event = threading.Event()
        robots, lease = self._connect_robots(followers, lend)

        takeover: Takeover | None = None
        if lease is not None and bspline_layout is None:
            takeover = Takeover(
                teleop=self._teleop,
                lease=lease,
                sides=sides,
                max_contact_wrench=_takeover_wrench(self._settings),
                cartesian_stiffness=stiffness.values,
                append_log=self._append_log,
            )
            if trigger_arm is not None:
                self._attach_takeover_trigger(takeover, lease, trigger_arm)

        app_rollout = self._settings.rollout
        motion_limits = (
            app_rollout.max_linear_vel,
            app_rollout.max_angular_vel,
            app_rollout.max_linear_acc,
            app_rollout.max_angular_acc,
        )
        recording: dict[str, Any] | None = None
        if record:
            recording = self._open_recording(
                checkpoint_origin=str(checkpoint_origin),
                task=task,
                trial=trial,
                target_hz=target_hz,
                action_names=waypoint_action_names,
                sides=sides,
                camera_names=camera_names,
                send_twist=effective_twist,
                safety=safety,
                stiffness=stiffness,
            )
        if recording is not None:
            recording["takeover"] = takeover
        on_finished = partial(self._on_runner_finished, recording=recording)
        runner: WaypointRunner | BSplineRunner | StepRunner
        try:
            if bspline_layout is not None:
                runner = BSplineRunner(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    robots=robots,
                    sides=sides,
                    followers=followers,
                    cameras=self._cameras,
                    image_resolutions=image_resolutions,
                    rollout_cfg=rollout_cfg,
                    target_hz=target_hz,
                    device=device,
                    task=task,
                    bspline_layout=bspline_layout,
                    state_dim=state_dim,
                    end_effector_config=end_effector_config,
                    motion_limits=motion_limits,
                    max_steps=app_rollout.max_steps,
                    stop_event=self._stop_event,
                    append_log=self._append_log,
                    append_metric=self._metrics.append,
                    append_wrench=self._wrench.append,
                    on_error=self._on_runner_error,
                    on_cleanup_error=self._on_runner_cleanup_error,
                    on_finished=self._on_runner_finished,
                    release_robots=self._make_release_robots(robots, lease),
                    stop_robots=hardware.stop_robots,
                    prepare_motion=partial(
                        self._prepare_motion, stiffness=stiffness.values
                    ),
                    gripper_default_width_m=gripper_default_width_m,
                    gripper_initialization_registry=(
                        self._gripper_initialization_registry
                    ),
                    gripper_command_parameters=gripper_command_parameters,
                    camera_names=camera_names,
                )
            elif space is not None:
                if recording is not None:
                    recording["live"] = self._start_live_checks(
                        trial, sides, camera_names
                    )
                live = (recording.get("live") if recording else None) or live_checks
                runner_class: type[StepRunner] = StepRunner
                extra: dict[str, Any] = {}
                if practice is not None:
                    from flexivtrainer.practice.learner import (  # noqa: PLC0415
                        LearnerLink,
                    )

                    live = self._practice_live = self._live_checks(
                        {practice.check_name: practice.check}, sides, camera_names
                    )
                    runner_class = PracticeRunner
                    extra = {
                        "plan": practice,
                        "link": LearnerLink(
                            practice.learner_host, practice.learner_port
                        ),
                    }
                    on_finished = self._on_practice_finished
                runner = runner_class(
                    **extra,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    robots=robots,
                    sides=sides,
                    followers=followers,
                    cameras=self._cameras,
                    camera_names=camera_names,
                    space=space,
                    device=device,
                    task=task,
                    motion_limits=motion_limits,
                    max_steps=app_rollout.max_steps,
                    stop_event=self._stop_event,
                    append_log=self._append_log,
                    append_metric=self._metrics.append,
                    on_error=self._on_runner_error,
                    on_cleanup_error=self._on_runner_cleanup_error,
                    on_finished=on_finished,
                    release_robots=self._make_release_robots(robots, lease),
                    stop_robots=hardware.stop_robots,
                    image_resolutions=image_resolutions,
                    append_wrench=self._wrench.append,
                    safety=safety,
                    recorder=recording["recorder"] if recording is not None else None,
                    cartesian_stiffness=stiffness.values,
                    max_contact_wrench=app_rollout.max_contact_wrench,
                    zero_ft_max_force_n=app_rollout.ft_zero_max_force_n,
                    force_axis_max_speed_m_s=app_rollout.force_axis_max_speed_m_s,
                    takeover=takeover,
                    live_checks=live,
                )
                if recording is not None:
                    recording["runner"] = runner
                if takeover is not None:
                    takeover.bind(runner)
            else:
                assert waypoint_layout is not None
                assert waypoint_action_dim is not None
                if recording is not None:
                    recording["live"] = self._start_live_checks(
                        trial, sides, camera_names
                    )
                runner = WaypointRunner(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    robots=robots,
                    sides=sides,
                    followers=followers,
                    cameras=self._cameras,
                    image_resolutions=image_resolutions,
                    rollout_cfg=rollout_cfg,
                    target_hz=target_hz,
                    device=device,
                    task=task,
                    action_layout=waypoint_layout,
                    action_dim=waypoint_action_dim,
                    state_dim=state_dim,
                    gripper_sides=waypoint_gripper_sides,
                    gripper_target_mode=waypoint_gripper_target_mode,
                    end_effector_config=end_effector_config,
                    motion_limits=motion_limits,
                    planner_hz_fallback=app_rollout.planner_hz,
                    expected_hz_fallback=app_rollout.action_dt_hz,
                    max_steps=app_rollout.max_steps,
                    stop_event=self._stop_event,
                    append_log=self._append_log,
                    append_metric=self._metrics.append,
                    append_wrench=self._wrench.append,
                    on_error=self._on_runner_error,
                    on_cleanup_error=self._on_runner_cleanup_error,
                    on_finished=on_finished,
                    release_robots=self._make_release_robots(robots, lease),
                    stop_robots=hardware.stop_robots,
                    prepare_policy=(
                        partial(act_policy.compile_model, mode=compile_mode)
                        if compile_act
                        else None
                    ),
                    uses_cuda_graphs=uses_cuda_graphs,
                    gripper_default_width_m=gripper_default_width_m,
                    gripper_initialization_registry=(
                        self._gripper_initialization_registry
                    ),
                    gripper_command_parameters=gripper_command_parameters,
                    camera_names=camera_names,
                    safety=safety,
                    recorder=recording["recorder"] if recording is not None else None,
                    send_twist=effective_twist,
                    cartesian_stiffness=stiffness.values,
                    max_contact_wrench=app_rollout.max_contact_wrench,
                    zero_ft_max_force_n=app_rollout.ft_zero_max_force_n,
                    takeover=takeover,
                    action_names=waypoint_action_names,
                    live_checks=(
                        (recording.get("live") if recording else None) or live_checks
                    ),
                )
                if recording is not None:
                    recording["runner"] = runner
                if takeover is not None:
                    takeover.bind(runner)

        except BaseException:
            # A runner that never started leaves no trial behind.
            if self._practice_live is not None:
                self._practice_live.stop()
                self._practice_live = None
            if recording is not None:
                if recording.get("live") is not None:
                    recording["live"].stop()
                recording["recorder"].abort()
            if lease is not None:
                self._teleop.release_lease(lease)
            raise

        with self._lock:
            self._send_twist = (
                effective_twist if bspline_layout is None and space is None else None
            )
            self._safety = safety if bspline_layout is None else None
            self._stiffness = stiffness
            self._trial = trial
            self._recording = recording
            self._takeover = takeover
            self._takeover_reason = None if takeover is not None else takeover_reason
            self._checkpoint_path = checkpoint_path
            self._checkpoint_repo_id = repo_id if source == "hub" else None
            self._task = task
            self._error = None
            self._stop_reason = None
            self._stopping = False
            self._robots = robots
            self._device = device
            self._target_hz = target_hz
            self._sides = list(sides)
            self._running = True
            # Drop any depth-preview lease now; alignment would otherwise keep
            # stealing the GIL from the policy loop until it lapsed on its own.
            clear_leases = getattr(self._cameras, "clear_depth_alignment_leases", None)
            if callable(clear_leases):
                clear_leases()
            if not keep_logs:
                self._logs.clear()
            self._metrics.clear()
            self._wrench.clear()
            self._logs.append(
                _encode_ui_log(
                    "INFO",
                    "ROLLOUT",
                    "Rollout started",
                    f"device={device} sides={'+'.join(sides)}",
                )
            )
            if space is not None:
                self._logs.append(
                    _encode_ui_log(
                        "INFO", "ROLLOUT", "Safety envelope", safety.describe()
                    )
                )
            elif bspline_layout is None:
                self._logs.append(
                    _encode_ui_log(
                        "INFO",
                        "ROLLOUT",
                        "Safety envelope",
                        f"{safety.describe()}; velocity feedforward "
                        f"{'on' if effective_twist else 'off'}",
                    )
                )
            self._logs.append(
                _encode_ui_log(
                    "INFO", "ROLLOUT", "Cartesian stiffness", stiffness.describe()
                )
            )
            if stiffness.warning:
                self._logs.append(
                    _encode_ui_log(
                        "WARN", "ROLLOUT", "Cartesian stiffness", stiffness.warning
                    )
                )
            if recording is not None:
                where = (
                    f"session {trial.session} trial {trial.trial} (arm {trial.arm}, "
                    f"condition {trial.condition})"
                    if trial is not None
                    else "ad hoc"
                )
                self._logs.append(
                    _encode_ui_log(
                        "INFO",
                        "ROLLOUT",
                        "Recording this rollout",
                        f"{recording['episode']} -- {where}",
                    )
                )
            if space is not None:
                self._logs.append(
                    _encode_ui_log(
                        "OK", "ROLLOUT", "Step skill loaded", _describe_space(space)
                    )
                )
            elif waypoint_layout_inferred:
                confirmation = layout_confirmation(waypoint_action_dim, len(sides))
                # OK, not WARNING: the width matched, so this confirms the
                # inference rather than cautioning about it.
                self._logs.append(
                    _encode_ui_log(
                        "OK",
                        "ROLLOUT",
                        "Waypoint action layout inferred",
                        f"{confirmation}; sides={'+'.join(sides)}",
                    )
                )
                ok("Waypoint action layout inferred", confirmation)
            else:
                confirmation = recorded_layout_confirmation(
                    waypoint_action_dim, len(sides)
                )
                self._logs.append(
                    _encode_ui_log(
                        "OK",
                        "ROLLOUT",
                        "Waypoint action layout loaded",
                        f"{confirmation}; sides={'+'.join(sides)}",
                    )
                )
                ok("Waypoint action layout loaded", confirmation)
            if waypoint_gripper_sides:
                self._logs.append(
                    _encode_ui_log(
                        "INFO",
                        "ROLLOUT",
                        "Waypoint gripper control enabled",
                        "sides="
                        f"{'+'.join(waypoint_gripper_sides)} "
                        "command=width force=device-limited",
                    )
                )
            if overrides_applied:
                self._logs.append(
                    _encode_ui_log(
                        "INFO",
                        "ROLLOUT",
                        "Rollout overrides applied",
                        _describe_rollout_overrides(rollout_cfg),
                    )
                )
        if stiffness.warning:
            warn("Rollout Cartesian stiffness", stiffness.warning)
        self._runner = runner
        runner.start()
        if takeover is not None:
            takeover.start_trigger()
        return self.status()

    # --- chains of skills and primitives (rollout/chain.py) -------------------

    def _refuse_during_chain(self) -> None:
        chain = self._chain
        if chain is not None and chain.is_alive():
            raise RuntimeError("A chain is running; stop it first")

    def start_chain(self, plan: ChainPlan) -> dict[str, Any]:
        """Run `plan`'s steps in order, each as a rollout of its own."""
        self._refuse_during_chain()
        with self._lock:
            if self._awaiting is not None:
                raise RuntimeError(
                    "The last recorded rollout is waiting for its verdict; give it "
                    "one (or mark it unknown) before starting a chain."
                )
            if self._running or self._stopping:
                raise RuntimeError("Rollout is already running")
        if len(self._follower_serials()) != 1:
            raise RuntimeError("A chain drives one arm; set single-arm mode")
        # Every skill is found now, so a wrong path stops the chain before
        # anything moves.
        root = self._settings.storage.root
        skills: dict[str, str] = {}
        for step in plan.steps:
            if step.skill is None:
                continue
            path = Path(step.skill).expanduser()
            try:
                skills[step.name] = str(
                    resolve_checkpoint_path(
                        str(path if path.is_absolute() else root / path), root
                    )
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Step {step.name!r}: checkpoint not found: {step.skill}"
                ) from exc
            except ValueError as exc:
                raise RuntimeError(f"Step {step.name!r}: {exc}") from exc
        for step in plan.steps:
            # Refused here, not mid-chain, when nothing says how stiff to be.
            self._chain_stiffness(plan, skills.get(step.name))
            if self._chain_reads_jaws(plan, step):
                self._gripper_configs()
        directory = self._settings.storage.rollouts_root / CHAINS_DIR
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record = directory / f"{stamp}.json"
        suffix = 1
        while record.exists():
            suffix += 1
            record = directory / f"{stamp}_{suffix}.json"
        chain = ChainRunner(
            plan,
            run_skill=partial(self._chain_skill, skills),
            run_primitive=partial(self._chain_primitive, plan, skills),
            record_path=record,
            append_log=self._append_log,
        )
        with self._lock:
            self._logs.clear()
            self._error = None
            self._stop_reason = None
        self._append_log(
            "INFO",
            "CHAIN",
            "Chain started",
            " -> ".join(step.name for step in plan.steps),
        )
        self._chain = chain
        chain.start()
        return self.status()

    def chain_continue(self, action: str) -> dict[str, Any]:
        """The operator has done what the chain could not: `retry` the step,
        or `skip` it as done."""
        chain = self._chain
        if chain is None or not chain.is_alive():
            raise RuntimeError("No chain is running")
        chain.go_on(action)
        return self.status()

    def _chain_skill(
        self, skills: dict[str, str], step: ChainStep, checks: dict[str, Check]
    ) -> Outcome:
        """One try of a skill step: roll it out until its end check passes or
        its time is up."""
        started = time.monotonic()
        try:
            live = self._live_checks(
                checks,
                self._get_active_sides(),
                self._get_active_cameras(),
                end_when_all=True,
            )
        except Exception as exc:
            return Outcome(False, "not started", error=describe_exception(exc))
        try:
            try:
                self._start(
                    skills[step.name], step.task, live_checks=live, keep_logs=True
                )
            except Exception as exc:
                return Outcome(False, "not started", error=describe_exception(exc))
            # Timed from when it runs: loading the policy is not the skill's.
            timed_out = self._await_runner(time.monotonic() + step.max_s)
            ended = live.ended()
        finally:
            live.stop()
        return self._outcome(
            started,
            passed=ended is not None,
            reason="time limit" if timed_out and ended is None else None,
            live=live,
        )

    def _chain_primitive(
        self,
        plan: ChainPlan,
        skills: dict[str, str],
        primitive: Primitive | None,
        checks: dict[str, Check],
        watch_s: float,
        step: ChainStep,
    ) -> Outcome:
        """A primitive for `step`, then `checks` watched for `watch_s`."""
        started = time.monotonic()
        live: LiveChecks | None = None
        try:
            if checks:
                live = self._live_checks(
                    checks, self._get_active_sides(), self._get_active_cameras()
                )
            skill = skills.get(step.name)
            self._start_primitive(
                primitive or Primitive(),
                stiffness=self._chain_stiffness(plan, skill),
                live_checks=live,
                check_s=watch_s,
                image_resolutions=self._skill_resolutions(skill),
            )
        except Exception as exc:
            if live is not None:
                live.stop()
            return Outcome(False, "not started", error=describe_exception(exc))
        try:
            self._await_runner(None)
        finally:
            if live is not None:
                live.stop()
        with self._lock:
            reason = self._stop_reason
        return self._outcome(
            started, passed=reason in ("done", "checks passed"), live=live
        )

    def _await_runner(self, deadline: float | None) -> bool:
        """Wait for the current rollout to end; stop it at `deadline`, or when
        the chain is stopped. Whether the deadline stopped it."""
        runner, chain = self._runner, self._chain
        timed_out = False
        while runner is not None and runner.is_alive():
            if chain is not None and chain.stop_event.is_set():
                self._stop_event.set()
            elif deadline is not None and time.monotonic() > deadline:
                timed_out = True
                self._stop_event.set()
            time.sleep(0.02)
        return timed_out

    def _outcome(
        self,
        started: float,
        *,
        passed: bool,
        reason: str | None = None,
        live: LiveChecks | None = None,
    ) -> Outcome:
        """How the rollout that has just ended went. A check that could not be
        scored makes a failed try an error: trying again would not help."""
        runner, takeover = self._runner, self._takeover
        summary = takeover.summary() if takeover is not None else {}
        with self._lock:
            error, stop_reason = self._error, self._stop_reason
        if not passed and error is None and live is not None:
            unscored = [
                f"check {name!r}: {result['error']}"
                for name, result in live.summary()["checks"].items()
                if "error" in result and not result["passed"]
            ]
            error = "; ".join(unscored) or None
        return Outcome(
            passed=passed and error is None,
            reason=reason or stop_reason or "stopped",
            seconds=time.monotonic() - started,
            error=error,
            safety_stop=getattr(runner, "safety_stop", None),
            blocked=getattr(runner, "blocked", None),
            takeovers=int(summary.get("count", 0)),
            operator_s=float(summary.get("seconds", 0.0)),
        )

    @staticmethod
    def _chain_reads_jaws(plan: ChainPlan, step: ChainStep) -> bool:
        """Whether a primitive of `step` moves the jaws, or one of its checks
        reads them."""
        primitives = [p for p in (step.primitive, step.back_off) if p is not None]
        return any(p.moves_jaws for p in primitives) or any(
            isinstance(plan.checks[name], GripperWidth)
            for name in (*step.start, *step.end)
        )

    def _gripper_configs(self) -> dict[str, Any]:
        """The end effectors, when the one arm has a follower gripper."""
        configs = dict(self._get_end_effector_config() or {})
        side = self._get_active_sides()[0]
        config = configs.get(side)
        if (
            config is None
            or self._config_value(config, "follower") != "gripper"
            or not self._config_value(config, "gripper_model")
        ):
            raise RuntimeError(
                f"The chain moves or reads the jaws, but no follower gripper is "
                f"configured for {side}"
            )
        return configs

    def _chain_stiffness(self, plan: ChainPlan, skill: str | None) -> StiffnessChoice:
        """A back-off or check moves at its skill's stiffness, a primitive step
        at the plan's; the rollout setting wins over both, as for a skill."""
        setting = self._settings.rollout.cartesian_stiffness
        allow = self._settings.rollout.allow_controller_default_stiffness
        if skill is not None:
            return choose_rollout_stiffness(
                setting, checkpoint_control(skill), allow_controller_default=allow
            )
        if setting is None and plan.stiffness is not None:
            return StiffnessChoice(list(plan.stiffness), "chain plan", NOT_RECORDED)
        return choose_rollout_stiffness(setting, None, allow_controller_default=allow)

    @staticmethod
    def _skill_resolutions(skill: str | None) -> dict[str, tuple[int, int]] | None:
        """The frame sizes a step's skill reads, so a learned check scored
        around it sees what it sees during the skill."""
        if skill is None:
            return None
        space = checkpoint_space(skill)
        if space is not None:
            return dict(space.frame_size)
        return checkpoint_image_resolutions(skill)

    def _start_primitive(
        self,
        primitive: Primitive,
        *,
        stiffness: StiffnessChoice,
        live_checks: LiveChecks | None,
        check_s: float,
        image_resolutions: dict[str, tuple[int, int]] | None,
    ) -> None:
        """Run `primitive` on the arm as a rollout of its own."""
        with self._lock:
            runner = self._runner
            if self._running or (runner is not None and runner.is_alive()):
                raise RuntimeError("Rollout is already running")
        followers = self._follower_serials()
        lend, _ = self._preflight_teleop(followers)
        sides = self._get_active_sides()
        camera_names = self._get_active_cameras()
        safety = SafetyEnvelope.from_settings(self._settings.rollout)
        jaws = primitive.moves_jaws or (
            live_checks is not None and live_checks.reads_jaws
        )
        configs = self._gripper_configs() if jaws else {}
        self._stop_event = threading.Event()
        robots, lease = self._connect_robots(followers, lend)
        try:
            runner = PrimitiveRunner(
                primitive=primitive,
                robots=robots,
                sides=sides,
                followers=followers,
                cameras=self._cameras,
                camera_names=camera_names,
                stop_event=self._stop_event,
                append_log=self._append_log,
                on_error=self._on_runner_error,
                on_cleanup_error=self._on_runner_cleanup_error,
                on_finished=self._on_runner_finished,
                release_robots=self._make_release_robots(robots, lease),
                stop_robots=hardware.stop_robots,
                safety=safety,
                cartesian_stiffness=stiffness.values,
                max_contact_wrench=self._settings.rollout.max_contact_wrench,
                live_checks=live_checks,
                check_s=check_s,
                image_resolutions=image_resolutions,
                open_gripper=(
                    partial(
                        initialize_gripper_executor,
                        robots,
                        sides,
                        configs,
                        (sides[0],),
                        failure_event=self._stop_event,
                        followers=followers,
                        initialization_registry=(self._gripper_initialization_registry),
                        append_log=self._append_log,
                    )
                    if jaws
                    else None
                ),
            )
        except BaseException:
            if lease is not None:
                self._teleop.release_lease(lease)
            raise
        with self._lock:
            self._safety = safety
            self._stiffness = stiffness
            self._trial = None
            self._recording = None
            self._takeover = None
            self._takeover_reason = "A primitive cannot be taken over"
            self._error = None
            self._stop_reason = None
            self._stopping = False
            self._robots = robots
            self._sides = list(sides)
            self._running = True
            self._metrics.clear()
        self._runner = runner
        runner.start()

    def _connect_robots(
        self, followers: list[str], lend: bool
    ) -> tuple[list[Any], Any]:
        """Connect to the followers -- borrowed from teleoperation when it can
        lend them -- and the lease to give back, if any."""
        robots: list[Any] = []
        lease: Any = None
        try:
            robot_factory = self._robot_factory
            if lend:
                # The teleop connection already holds these robots; a second
                # RDK connection cannot. Borrow its handles instead.
                lease = self._teleop.lend_to_rollout(followers)
                robot_factory = _borrowed_factory(followers, lease.followers)
                self._append_log(
                    "INFO",
                    "ROLLOUT",
                    "Borrowed the robots from teleoperation",
                    "teleoperation stopped; it stays connected",
                )
            for serial in followers:
                robots.append(
                    hardware.connect_robot(
                        robot_factory,
                        serial,
                        self._stop_event,
                    )
                )
        except Exception as exc:
            hardware.stop_robots(robots)
            if lease is not None:
                self._teleop.release_lease(lease)
            raise RuntimeError(
                f"Failed to connect to robot: {describe_exception(exc)}"
            ) from exc
        return robots, lease

    # --- recording, verdicts and evaluation sessions -----------------------

    def _open_recording(
        self,
        *,
        checkpoint_origin: str,
        task: str | None,
        trial: TrialContext | None,
        target_hz: float,
        action_names: list[str] | None,
        sides: list[str],
        camera_names: list[str],
        send_twist: bool,
        safety: SafetyEnvelope,
        stiffness: StiffnessChoice,
    ) -> dict[str, Any]:
        storage = self._settings.storage
        rollout = self._settings.rollout
        session = trial.session if trial is not None else ADHOC_SESSION
        episode = _episode_name(trial)
        destination = storage.rollouts_root / session / episode
        suffix = 1
        while destination.exists():
            suffix += 1
            destination = storage.rollouts_root / session / f"{episode}_{suffix}"
        plan = RecordingPlan(
            destination=destination,
            staging=storage.staging_root / f"rollout_{destination.name}",
            fps=max(1, round(target_hz)),
            task=task,
            robot_type=self._settings.robot_type,
            video_codec=self._settings.video_codec,
            action_names=list(action_names or []),
            sides=list(sides),
            cameras=list(camera_names),
            # A recorded rollout can be merged and trained on like a
            # demonstration, so it stores its follower settings the same way.
            control=ControlSettings(
                recorded_in="rollout",
                follower_cartesian_stiffness=stiffness.values,
                follower_max_contact_wrench=rollout.max_contact_wrench,
            ),
        )
        outcome = RolloutOutcome(
            episode=destination.name,
            checkpoint=checkpoint_origin,
            task=task,
            started_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            session=trial.session if trial is not None else None,
            trial=trial.trial if trial is not None else None,
            arm=trial.arm if trial is not None else None,
            condition=trial.condition if trial is not None else None,
            send_twist=send_twist,
            safety={
                **safety.to_dict(),
                "cartesian_stiffness": stiffness.values,
                "max_contact_wrench": _as_list(rollout.max_contact_wrench),
            },
            stiffness=stiffness.to_dict(),
        )
        return {
            "recorder": RolloutRecorder(plan, outcome),
            "episode": destination.name,
            "trial": trial,
            "saving": False,
            "runner": None,
        }

    def _finalize_recording(
        self, recording: dict[str, Any], stop_reason: str | None, steps: int
    ) -> None:
        runner = recording.get("runner")
        with self._lock:
            error = self._error
        try:
            takeover: Takeover | None = recording.get("takeover")
            live: LiveChecks | None = recording.get("live")
            if live is not None:
                live.stop()
            destination, outcome, saved = recording["recorder"].finish(
                stop_reason=stop_reason,
                error=error,
                steps=steps,
                safety_stop=getattr(runner, "safety_stop", None),
                interventions=takeover.summary() if takeover is not None else None,
                live_checks=live.summary() if live is not None else None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            detail = describe_exception(exc)
            warn("Rollout recording could not be finalized", detail)
            with self._lock:
                recording["saving"] = False
                recording["error"] = detail
            return
        trial: TrialContext | None = recording.get("trial")
        if trial is not None:
            # Before `_awaiting` is set, so no verdict can be written meanwhile.
            self._score_checks(destination, outcome, trial.session)
        with self._lock:
            recording["saving"] = False
            recording["path"] = str(destination)
            recording["saved"] = saved
            self._awaiting = {
                "path": destination,
                "episode": destination.name,
                "trial": trial,
                "failure_modes": tuple(trial.failure_modes)
                if trial is not None
                else DEFAULT_FAILURE_MODES,
                "frames": outcome.frames,
                "stop_reason": outcome.stop_reason,
            }
            self._logs.append(
                _encode_ui_log(
                    "OK" if saved else "WARN",
                    "ROLLOUT",
                    "Rollout recorded"
                    if saved
                    else "Rollout outcome saved without data",
                    f"{destination.name}: {outcome.frames} frames, "
                    f"{outcome.dropped_frames} dropped; waiting for a verdict",
                )
            )

    def _start_live_checks(
        self, trial: TrialContext | None, sides: list[str], camera_names: list[str]
    ) -> LiveChecks | None:
        """The trial's checks marked `live` or `end_on_pass`, loaded.

        Refuses the start when one cannot be loaded: a trial meant to end when
        its check passes would otherwise run until someone stops it.
        """
        if trial is None:
            return None
        plan = SessionStore(self._settings.storage.rollouts_root).plan(trial.session)
        checks = {
            name: check for name, check in plan.checks.items() if check.scored_live
        }
        if not checks:
            return None
        return self._live_checks(checks, sides, camera_names)

    def _live_checks(
        self,
        checks: dict[str, Check],
        sides: list[str],
        camera_names: list[str],
        *,
        end_when_all: bool = False,
    ) -> LiveChecks:
        """Load `checks` and start scoring them, or refuse the start."""
        live = LiveChecks(
            checks,
            self._settings.storage.root,
            state_names=lambda snapshot, images, pose_format, width: (
                recorded_state_names(
                    snapshot, images, sides, camera_names, pose_format, width
                )
            ),
            end_when_all=end_when_all,
        )
        try:
            live.start()
        except Exception as exc:
            live.stop()
            raise RuntimeError(
                f"A live check could not be loaded: {describe_exception(exc)}"
            ) from exc
        self._append_log("INFO", "ROLLOUT", "Live checks", ", ".join(sorted(checks)))
        return live

    def _score_checks(
        self, destination: Path, outcome: RolloutOutcome, session: str
    ) -> None:
        """Score the session's automatic checks on this trial's recording."""
        try:
            store = SessionStore(self._settings.storage.rollouts_root)
            checks = store.plan(session).checks
            if not checks:
                return
            results = run_checks(
                checks,
                read_ticks(destination),
                episode_dir=destination,
                storage_root=self._settings.storage.root,
            )
            scored = replace(outcome, checks={**outcome.checks, **results})
            write_outcome(destination, scored)
        except Exception as exc:  # a check must never lose a trial
            warn("Automatic checks were not scored", describe_exception(exc))

    # --- leader-arm takeover ---------------------------------------------------

    def _takeover_status(self) -> dict[str, Any]:
        takeover = self._takeover
        if takeover is not None and self._running and not self._stopping:
            return takeover.status()
        last = takeover.status()["last_switch"] if takeover is not None else None
        return {
            "available": False,
            "state": None,
            "control": "policy" if self._running else None,
            "error": None,
            "count": 0,
            "last_switch": last,
            "action": None,
            "reason": (
                self._takeover_reason or "The leader arm cannot take over"
                if self._running
                else "No rollout is running"
            ),
            "trigger_channel": None,
        }

    def _active_takeover(self) -> Takeover:
        with self._lock:
            running = self._running and not self._stopping
            takeover, reason = self._takeover, self._takeover_reason
        if not running:
            raise TakeoverRefused("No rollout is running")
        if takeover is None:
            raise TakeoverRefused(reason or "The leader arm cannot take over")
        return takeover

    def take_over(self) -> dict[str, Any]:
        """Stop the policy and give the follower to the leader arm."""
        self._active_takeover().take_over()
        return self.status()

    def hand_back(self) -> dict[str, Any]:
        """Give the follower back to the policy."""
        self._active_takeover().hand_back()
        return self.status()

    def _preflight_takeover_trigger(self, sides: list[str]) -> int | None:
        """The arm whose leader input toggles control, after checking it."""
        rollout = self._settings.rollout
        channel = rollout.takeover_trigger_channel
        if channel is None:
            return None
        arm = rollout.takeover_trigger_arm
        if arm >= len(sides):
            raise RuntimeError(
                f"takeover_trigger_arm {arm} is not one of the rollout's "
                f"{len(sides)} arm(s)"
            )
        config = (self._get_end_effector_config() or {}).get(sides[arm])
        if (
            config is not None
            and self._config_value(config, "leader") == "digital_input"
            and self._config_value(config, "leader_channel") == channel
        ):
            raise RuntimeError(
                f"takeover_trigger_channel {channel} is the gripper trigger of the "
                f"{sides[arm]} leader; choose another channel"
            )
        return arm

    def _attach_takeover_trigger(
        self, takeover: Takeover, lease: Any, arm: int
    ) -> None:
        rollout = self._settings.rollout
        read_inputs = getattr(lease.leaders[arm], "digital_inputs", None)
        if not callable(read_inputs):
            self._append_log(
                "WARN",
                "TAKEOVER",
                "The takeover trigger is off",
                "the leader robot does not report its digital inputs",
            )
            return
        takeover.attach_trigger(
            LeaderTrigger(
                read_inputs,
                int(rollout.takeover_trigger_channel),
                takeover.toggle,
                active_high=rollout.takeover_trigger_active == "high",
                debounce_s=rollout.takeover_trigger_debounce_s,
            )
        )

    def _recording_status(self) -> dict[str, Any] | None:
        recording = self._recording
        if recording is None:
            return None
        live = recording["recorder"].status()
        return {
            "episode": recording["episode"],
            "frames": live["frames"],
            "dropped_frames": live["dropped_frames"],
            "saving": bool(recording.get("saving")),
            "error": live["error"] or recording.get("error"),
        }

    @staticmethod
    def _public_trial(trial: TrialContext | None) -> dict[str, Any] | None:
        if trial is None:
            return None
        return {
            "session": trial.session,
            "trial": trial.trial,
            "arm": trial.arm,
            "condition": trial.condition,
            "description": trial.description,
            "blind": trial.blind,
        }

    def _public_awaiting(self) -> dict[str, Any] | None:
        awaiting = self._awaiting
        if awaiting is None:
            return None
        return {
            "episode": awaiting["episode"],
            "trial": self._public_trial(awaiting["trial"]),
            "failure_modes": list(awaiting["failure_modes"]),
            "frames": awaiting["frames"],
            "stop_reason": awaiting["stop_reason"],
        }

    def verdict(
        self,
        outcome: str,
        *,
        failure_mode: str | None = None,
        note: str = "",
        session: str | None = None,
        episode: str | None = None,
    ) -> dict[str, Any]:
        """Judge the recorded rollout that is waiting, or a named one on disk.

        Naming one (`session` and `episode`) is for a trial whose verdict was not
        given before the application restarted; it must not have one yet.
        """
        if outcome == "failure":
            failure_mode = (failure_mode or "").strip() or None
            if failure_mode is None:
                raise ValueError("a failure needs a failure mode")
        else:
            failure_mode = None
        judged = Verdict(outcome=outcome, failure_mode=failure_mode, note=note.strip())
        with self._lock:
            awaiting = self._awaiting
        if episode is not None:
            directory = self._named_episode(session, episode)
            current = read_outcome(directory)
            if current is None:
                raise RuntimeError(f"{episode} has no rollout outcome")
            if current.verdict is not None:
                raise RuntimeError(f"{episode} already has a verdict")
            modes = self._failure_modes_for(current.session)
        elif awaiting is not None:
            directory = awaiting["path"]
            current = read_outcome(directory)
            if current is None:
                raise RuntimeError(f"{awaiting['episode']} lost its rollout outcome")
            modes = tuple(awaiting["failure_modes"])
        else:
            raise RuntimeError("No recorded rollout is waiting for a verdict")
        if failure_mode is not None and failure_mode not in modes:
            raise ValueError(
                f"failure mode {failure_mode!r} is not one of {list(modes)}"
            )
        write_outcome(directory, replace(current, verdict=judged))
        with self._lock:
            if awaiting is not None and directory == awaiting["path"]:
                self._awaiting = None
            self._logs.append(
                _encode_ui_log(
                    "OK",
                    "ROLLOUT",
                    "Verdict saved",
                    f"{directory.name}: {outcome}"
                    + (f" ({failure_mode})" if failure_mode else ""),
                )
            )
        return {"episode": directory.name, "verdict": judged.outcome}

    def _named_episode(self, session: str | None, episode: str) -> Path:
        name = session or ADHOC_SESSION
        root = self._settings.storage.rollouts_root
        if name != ADHOC_SESSION:
            SessionStore(root).plan(name)  # validates the name and that it exists
        if not episode or "/" in episode or "\\" in episode or episode.startswith("."):
            raise ValueError(f"{episode!r} is not an episode name")
        directory = root / name / episode
        if not directory.is_dir():
            raise FileNotFoundError(f"no recorded rollout {name}/{episode}")
        return directory

    def _failure_modes_for(self, session: str | None) -> tuple[str, ...]:
        if session is None:
            return DEFAULT_FAILURE_MODES
        store = SessionStore(self._settings.storage.rollouts_root)
        return store.plan(session).failure_modes

    def evaluation_sessions(self) -> list[dict[str, Any]]:
        store = SessionStore(self._settings.storage.rollouts_root)
        sessions = []
        for name in store.sessions():
            try:
                progress = store.progress(name)
            except (ValueError, KeyError, TypeError) as exc:
                sessions.append({"name": name, "error": str(exc)})
                continue
            sessions.append(
                {key: progress[key] for key in ("name", "about", "total", "judged")}
            )
        return sessions

    def plans(self) -> list[dict[str, Any]]:
        """The chain and practice plans on disk (plans.py)."""
        return list_plans(self._settings.storage.rollouts_root)

    def start_plan(self, name: str) -> dict[str, Any]:
        """Run the chain or practice plan `name` from disk."""
        kind, plan, data = read_plan(self._settings.storage.rollouts_root, name)
        if kind == CHAIN:
            return self.start_chain(plan)
        checkpoint = Path(data["checkpoint_path"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = self._settings.storage.root / checkpoint
        return self.start(str(checkpoint), task=data.get("task", ""), practice=plan)

    def evaluation_session(self, name: str) -> dict[str, Any]:
        return SessionStore(self._settings.storage.rollouts_root).progress(name)

    def start_next_trial(self, session: str) -> dict[str, Any]:
        """Run the session's next planned trial, recorded, as its arm's checkpoint."""
        store = SessionStore(self._settings.storage.rollouts_root)
        progress = store.progress(session)
        if progress["awaiting_verdict"]:
            first = progress["awaiting_verdict"][0]
            raise RuntimeError(
                f"Trial {first['trial']} ({first['episode']}) is waiting for its "
                "verdict; give it one first."
            )
        upcoming = progress["next"]
        if upcoming is None:
            raise RuntimeError(f"Every trial of session {session!r} has a verdict.")
        plan = store.plan(session)
        arm = plan.arms[upcoming["arm"]]
        trial = TrialContext(
            session=session,
            trial=int(upcoming["trial"]),
            arm=upcoming["arm"],
            condition=upcoming["condition"],
            description=upcoming["description"],
            blind=plan.blind,
            failure_modes=tuple(plan.failure_modes),
        )
        return self.start(
            arm.checkpoint,
            task=arm.task,
            source=arm.source,
            repo_id=arm.repo_id,
            revision=arm.revision,
            send_twist=arm.send_twist,
            trial=trial,
        )

    def _on_runner_error(self, detail: str) -> None:
        with self._lock:
            self._error = detail
            self._running = False
            self._logs.append(
                _encode_ui_log("ERROR", "ROLLOUT", "Rollout stopped", detail)
            )

    def _on_runner_cleanup_error(self, detail: str) -> None:
        with self._lock:
            self._error = self._error or detail

    def _on_runner_finished(
        self,
        stop_reason: str | None,
        step: int,
        *,
        recording: dict[str, Any] | None = None,
    ) -> None:
        if recording is not None:
            # Saving flushes the video encoder, which can take seconds; it runs
            # off the planner thread so stop() is not held up by the disk.
            with self._lock:
                recording["saving"] = True
            threading.Thread(
                target=self._finalize_recording,
                args=(recording, stop_reason or self._stop_reason or "stopped", step),
                name="rollout-recording-finalize",
                daemon=True,
            ).start()
        with self._lock:
            self._running = False
            self._stopping = False
            if stop_reason is not None and self._stop_reason is None:
                self._stop_reason = stop_reason
            reason = self._stop_reason or "stopped"
            if self._error is None:
                self._logs.append(
                    _encode_ui_log(
                        "INFO",
                        "ROLLOUT",
                        "Rollout ended",
                        f"reason={reason} steps={step}",
                    )
                )

    def stop(self) -> dict[str, Any]:
        chain = self._chain
        if chain is not None:
            chain.stop()
        self._stop_event.set()
        cleanup_errors: list[str] = []
        runner = self._runner
        if runner is not None:
            cleanup_errors.extend(runner.stop())
            if runner.is_alive():
                # Reporting idle here would let the next start() sail past the
                # _running guard while this planner still owns the GPU and robots.
                with self._lock:
                    self._stopping = True
                    if cleanup_errors and self._error is None:
                        self._error = "; ".join(cleanup_errors)
                return self.status()
        self._runner = None
        self._release_robots()
        with self._lock:
            self._stopping = False
            if cleanup_errors and self._error is None:
                self._error = "; ".join(cleanup_errors)
            # Only attribute the stop to the operator when the run did not
            # already end on its own (max_steps reached or a fault recorded).
            if self._stop_reason is None and self._error is None:
                self._stop_reason = "stopped"
            self._running = False
        return self.status()

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception as exc:  # pragma: no cover - defensive
            warn("Rollout shutdown failed", describe_exception(exc))

    def _teleop_initialized(self) -> bool:
        snapshot = self._teleop.snapshot()
        return bool(getattr(snapshot, "initialized", False))

    def _follower_serials(self) -> list[str]:
        return [
            pair.follower_serial
            for pair in self._get_robot_pairs()
            if pair.follower_serial
        ]

    def _preflight_teleop(self, followers: list[str]) -> tuple[bool, str]:
        """Decide how the rollout reaches the followers.

        Returns whether to borrow them from teleoperation, and otherwise why
        the leader arm will not be able to take over.

        A fresh RDK connection cannot coexist with the TDK controller holding
        the same follower's LAN connection. When the teleop connection can
        lend its robot handles, the rollout borrows them at connect time (and
        the leader arm can take over). Otherwise teleop is disconnected here
        rather than refusing, which keeps the rollout to a single Start click.
        """
        if not self._teleop_initialized():
            return False, (
                "Teleoperation was not connected when the rollout started; "
                "connect it first to take over with the leader arm"
            )
        lend_problem = getattr(self._teleop, "lend_problem", None)
        problem = (
            lend_problem(followers)
            if callable(lend_problem)
            else "teleoperation cannot lend its robots"
        )
        if problem is None:
            return True, ""
        reason = f"The leader arm cannot take over: {problem}"
        self._append_log("INFO", "ROLLOUT", "Disconnecting teleoperation", problem)
        try:
            self._teleop.shutdown()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to disconnect teleoperation: {describe_exception(exc)}"
            ) from exc
        if self._teleop_initialized():
            raise RuntimeError(
                "Failed to disconnect teleoperation: the service is still "
                "connected and holds the robot connection."
            )
        return False, reason

    def _started_cameras(self, status: dict[str, Any]) -> set[str]:
        entries = status.get("cameras") or {}
        return {name for name, entry in entries.items() if entry.get("started")}

    def _preflight_cameras(self, camera_names: list[str]) -> None:
        if not camera_names:
            return
        if self._started_cameras(self._cameras.status()).issuperset(camera_names):
            return
        self._append_log(
            "INFO",
            "ROLLOUT",
            "Connecting cameras before rollout",
            ", ".join(camera_names),
        )
        try:
            status = self._cameras.start_streams(camera_names)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect cameras: {describe_exception(exc)}"
            ) from exc
        missing = [
            name for name in camera_names if name not in self._started_cameras(status)
        ]
        if not missing:
            return
        # An unavailable SDK reports no per-camera entries, only backend errors.
        entries = status.get("cameras") or {}
        reasons = sorted(
            {
                str(entries[name]["error"])
                for name in missing
                if entries.get(name, {}).get("error")
            }
            or {str(value) for value in (status.get("errors") or {}).values() if value}
        )
        detail = "; ".join(reasons) or f"{', '.join(missing)} did not start"
        raise RuntimeError(f"Failed to connect cameras: {detail}")

    def _apply_playback_speed(self, dataset_hz: float, rollout_cfg: Any) -> float:
        """Scale the action rate away from the rate the policy trained at."""
        speed = _positive_float(getattr(rollout_cfg, "playback_speed", None)) or 1.0
        if speed == 1.0:
            return dataset_hz
        target_hz = dataset_hz * speed
        with self._lock:
            self._logs.append(
                _encode_ui_log(
                    "WARN",
                    "ROLLOUT",
                    "Replaying off the trained rate",
                    f"playback_speed={speed:g} dataset={dataset_hz:.1f}Hz "
                    f"target={target_hz:.1f}Hz; commanded velocity scales with it "
                    "and motion limits are not rescaled",
                )
            )
        return target_hz

    def _resolve_target_hz(
        self,
        checkpoint_path: str,
        policy: Any,
        *,
        require_metadata: bool,
    ) -> float:
        config_rate = _positive_float(
            getattr(getattr(policy, "config", None), "knot_rate_hz", None)
        )
        target_hz = config_rate or _checkpoint_target_hz(checkpoint_path)
        if target_hz is not None:
            return target_hz
        if require_metadata:
            raise RuntimeError(
                "B-spline checkpoint has no knot_rate_hz or recoverable "
                "training dataset FPS"
            )
        target_hz = float(self._settings.rollout.action_dt_hz)
        warn(
            "Checkpoint FPS metadata not found",
            f"falling back to rollout.action_dt_hz={target_hz:.1f}",
        )
        return target_hz

    @staticmethod
    def _action_feature_dim(features: Any) -> int | None:
        if not isinstance(features, dict):
            return None
        action = features.get("action")
        if action is None:
            for key, value in features.items():
                if getattr(key, "value", None) == "action":
                    action = value
                    break
        shape = (
            action.get("shape")
            if isinstance(action, dict)
            else getattr(action, "shape", None)
        )
        if (
            not isinstance(shape, list | tuple)
            or len(shape) != 1
            or isinstance(shape[0], bool)
            or not isinstance(shape[0], int)
            or shape[0] <= 0
        ):
            return None
        return int(shape[0])

    def _preflight_waypoint(
        self,
        checkpoint_path: str,
        policy: Any,
        sides: list[str],
        followers: list[str],
        action_names: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int, bool, list[str]]:
        """Resolve a waypoint action contract before any robot is connected."""

        if len(followers) != len(sides):
            raise RuntimeError(
                "Every active waypoint arm must have a follower robot serial"
            )
        config = getattr(policy, "config", None)
        policy_dim = self._action_feature_dim(getattr(config, "output_features", None))
        saved_dim = checkpoint_action_output_dim(checkpoint_path)
        if policy_dim is not None and saved_dim is not None and policy_dim != saved_dim:
            raise RuntimeError(
                "Loaded policy action width does not match checkpoint metadata: "
                f"policy={policy_dim} checkpoint={saved_dim}"
            )
        action_dim = policy_dim or saved_dim
        if action_dim is None:
            raise RuntimeError(
                "Waypoint checkpoint has no valid one-dimensional action output"
            )

        try:
            names = checkpoint_action_names(
                checkpoint_path,
                self._settings.storage.root,
                settings=self._settings,
                override=action_names,
            )
            inferred = names is None
            if names is None:
                # canonical_action_names only covers the two unambiguous widths
                # and raises for anything with a gripper axis. Guessing a gripper
                # axis would drive real hardware from a made-up layout, so turn
                # that into an error that names the fix.
                try:
                    names = canonical_action_names(action_dim, sides)
                except ValueError as exc:
                    raise ValueError(f"{exc}.") from exc
            layout = build_action_layout(names, sides, action_dim)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        return layout, action_dim, inferred, list(names)

    @staticmethod
    def _preflight_bspline(
        policy: Any,
        sides: list[str],
        followers: list[str],
        target_hz: float,
    ) -> BSplineActionLayout:
        config = getattr(policy, "config", None)
        if config is None:
            raise RuntimeError("B-spline policy has no checkpoint configuration")
        names = getattr(config, "action_feature_names", None)
        if not isinstance(names, list | tuple):
            raise RuntimeError("B-spline checkpoint has no action feature names")
        try:
            layout = parse_bspline_action_layout(names)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if layout.rows != getattr(config, "horizon", None):
            raise RuntimeError(
                "B-spline action rows do not match the checkpoint horizon"
            )
        if tuple(layout.sides) != tuple(sides):
            raise RuntimeError(
                "B-spline checkpoint arm layout does not match active sides: "
                f"checkpoint={list(layout.sides)} active={sides}"
            )
        if len(followers) != len(sides):
            raise RuntimeError(
                "Every active B-spline arm must have a follower robot serial"
            )
        if not _positive_float(target_hz):
            raise RuntimeError("B-spline checkpoint knot rate must be positive")
        degree = getattr(config, "spline_degree", 3)
        if (
            isinstance(degree, bool)
            or not isinstance(degree, int)
            or degree < 1
            or layout.rows <= degree + 1
        ):
            raise RuntimeError("B-spline checkpoint has an invalid spline degree")
        for method in ("enqueue_observation", "predict_action_chunk"):
            if not callable(getattr(policy, method, None)):
                raise RuntimeError(f"B-spline policy does not implement {method}()")
        return layout

    @staticmethod
    def _config_value(config: Any, name: str) -> Any:
        if isinstance(config, dict):
            return config.get(name)
        return getattr(config, name, None)

    @classmethod
    def _preflight_grippers(
        cls,
        gripper_sides: tuple[str, ...],
        configs: dict[str, Any],
        *,
        policy_label: str,
    ) -> None:
        for side in gripper_sides:
            config = configs.get(side)
            if (
                config is None
                or cls._config_value(config, "follower") != "gripper"
                or not cls._config_value(config, "gripper_model")
            ):
                raise RuntimeError(
                    f"{policy_label} checkpoint predicts gripper width but no "
                    f"follower gripper is configured for {side}"
                )

    def _apply_n_action_steps(self, policy: Any, rollout_cfg: Any) -> None:
        # reset() rebuilds the policy action queue from this configured length.
        requested = getattr(rollout_cfg, "n_action_steps", 0)
        if requested <= 0:
            return
        config = getattr(policy, "config", None)
        if config is None or not hasattr(config, "n_action_steps"):
            return
        try:
            previous = config.n_action_steps
            value = requested
            horizon = getattr(config, "horizon", None)
            n_obs_steps = getattr(config, "n_obs_steps", None)
            if horizon is not None and n_obs_steps is not None:
                upper = horizon - n_obs_steps + 1
            else:
                # ACT has no horizon; its bound is the action-chunk length.
                upper = getattr(config, "chunk_size", None)
            if upper is not None and value > upper:
                warn(
                    "Clamped rollout n_action_steps to the checkpoint's bound",
                    f"requested={requested} clamped={upper}",
                )
                value = upper
            config.n_action_steps = value
        except Exception as exc:
            warn("Failed to override n_action_steps", describe_exception(exc))
            return
        with self._lock:
            self._logs.append(
                _encode_ui_log(
                    "INFO",
                    "ROLLOUT",
                    "Action chunk length overridden",
                    f"n_action_steps={value} (checkpoint default {previous})",
                )
            )

    def _prepare_motion(
        self, robot: Any, serial: str, *, stiffness: list[float] | None
    ) -> None:
        hardware.prepare_robot_motion(
            robot,
            serial,
            self._stop_event,
            self._append_log,
            stiffness=stiffness,
            max_contact_wrench=self._settings.rollout.max_contact_wrench,
            zero_ft_max_force_n=self._settings.rollout.ft_zero_max_force_n,
        )

    def _release_robots(self) -> None:
        with self._lock:
            robots, self._robots = self._robots, []
        hardware.stop_robots(robots)

    def _make_release_robots(
        self, robots: list[Any], lease: Any = None
    ) -> Callable[[], None]:
        """Release only this run's robots.

        A planner thread that outlives its stop() would otherwise reach the
        shared list and stop the *next* run's robots from its own finally.
        Borrowed robots go back to teleoperation, which stays connected and
        stopped; the operator restarts it as usual.
        """

        def release() -> None:
            with self._lock:
                if self._robots is robots:
                    self._robots = []
            hardware.stop_robots(robots)
            if lease is not None:
                try:
                    self._teleop.release_lease(lease)
                except Exception as exc:  # pragma: no cover - defensive
                    warn("Teleoperation did not take the robots back", str(exc))

        return release
