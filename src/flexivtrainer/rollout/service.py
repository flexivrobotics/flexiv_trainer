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
from collections import deque
from collections.abc import Callable
from functools import partial
from typing import Any

from flexivtrainer.config import AppSettings, TeleopRobotPair
from flexivtrainer.data.lerobot_io import active_camera_names
from flexivtrainer.jobs.train_policy import _encode_ui_log
from flexivtrainer.observability import describe_exception, ok, warn
from flexivtrainer.policies import act as act_policy
from flexivtrainer.policies import bspline_diffusion as bspline_policy
from flexivtrainer.policies import diffusion as diffusion_policy
from flexivtrainer.policies import dit as dit_policy
from flexivtrainer.rollout import hardware
from flexivtrainer.rollout.checkpoint import (
    _checkpoint_target_hz,
    _default_policy_loader,
    _positive_float,
    checkpoint_action_names,
    checkpoint_action_output_dim,
    checkpoint_gripper_command_metadata,
    checkpoint_image_resolutions,
    checkpoint_state_input_dim,
    resolve_checkpoint_path,
    resolve_hub_checkpoint,
)
from flexivtrainer.rollout.executors.bspline import (
    BSplineActionLayout,
    parse_bspline_action_layout,
)
from flexivtrainer.rollout.executors.waypoint import (
    build_action_layout,
    canonical_action_names,
    layout_confirmation,
    recorded_layout_confirmation,
)
from flexivtrainer.rollout.hardware import _default_robot_factory
from flexivtrainer.rollout.runners.bspline import BSplineRunner
from flexivtrainer.rollout.runners.waypoint import WaypointRunner
from flexivtrainer.runtime.gripper_session import GripperInitializationRegistry

_ROLLOUT_OVERRIDES = {
    "act": act_policy.apply_rollout_overrides,
    "bspline_diffusion": bspline_policy.apply_rollout_overrides,
    "diffusion": diffusion_policy.apply_rollout_overrides,
    "multi_task_dit": dit_policy.apply_rollout_overrides,
}


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
        self._runner: WaypointRunner | BSplineRunner | None = None
        self._stop_event = threading.Event()
        self._logs: deque[str] = deque(maxlen=2000)
        self._metrics: deque[dict[str, Any]] = deque(maxlen=300)
        # ~3 s at the sampled rate; only the newest drives the UI gauges.
        self._wrench: deque[dict[str, Any]] = deque(maxlen=30)

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._stopping:
                status = "stopping"
            elif self._running:
                status = "running"
            elif self._error:
                status = "failed"
            else:
                status = "idle"
            return {
                "status": status,
                "checkpoint_path": self._checkpoint_path,
                # Set for Hub rollouts so the UI shows "acme/policy" rather than
                # an opaque cache directory.
                "checkpoint_repo_id": self._checkpoint_repo_id,
                "task": self._task,
                "error": self._error,
                "stop_reason": self._stop_reason,
                "logs": list(self._logs),
                "log_lines": len(self._logs),
                "metrics": list(self._metrics),
                "wrench": list(self._wrench),
                "sides": list(self._sides),
                "target_hz": self._target_hz,
            }

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
    ) -> dict[str, Any]:
        task = task.strip() if isinstance(task, str) else None
        task = task or None
        with self._lock:
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
            # A fresh RDK connection cannot coexist with the TDK controller
            # holding the same follower's LAN connection; require teleop down.
            if self._teleop_initialized():
                raise RuntimeError(
                    "Stop teleoperation before starting a rollout "
                    "(it holds the robot connection)."
                )

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

        device = self._resolve_device(self._settings.training.default_device)
        sides = self._get_active_sides()
        camera_names = self._get_active_cameras()
        followers = [
            pair.follower_serial
            for pair in self._get_robot_pairs()
            if pair.follower_serial
        ]
        if not followers:
            raise RuntimeError("No follower robot serial is configured")

        try:
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
        image_resolutions = checkpoint_image_resolutions(checkpoint_path)
        rollout_cfg = self._settings.policies.rollout_for(policy_type)
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
        else:
            (
                waypoint_layout,
                waypoint_action_dim,
                waypoint_layout_inferred,
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

        # Per run: a zombie planner keeps the old event, which stays set, so
        # clearing a shared one can no longer un-stop it.
        self._stop_event = threading.Event()
        robots: list[Any] = []
        try:
            for serial in followers:
                robots.append(
                    hardware.connect_robot(
                        self._robot_factory,
                        serial,
                        self._stop_event,
                    )
                )
        except Exception as exc:
            hardware.stop_robots(robots)
            raise RuntimeError(
                f"Failed to connect to robot: {describe_exception(exc)}"
            ) from exc

        app_rollout = self._settings.rollout
        motion_limits = (
            app_rollout.max_linear_vel,
            app_rollout.max_angular_vel,
            app_rollout.max_linear_acc,
            app_rollout.max_angular_acc,
        )
        runner: WaypointRunner | BSplineRunner
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
                release_robots=self._make_release_robots(robots),
                stop_robots=hardware.stop_robots,
                prepare_motion=self._prepare_motion,
                gripper_default_width_m=gripper_default_width_m,
                gripper_initialization_registry=(self._gripper_initialization_registry),
                gripper_command_parameters=gripper_command_parameters,
                camera_names=camera_names,
            )
        else:
            assert waypoint_layout is not None
            assert waypoint_action_dim is not None
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
                on_finished=self._on_runner_finished,
                release_robots=self._make_release_robots(robots),
                stop_robots=hardware.stop_robots,
                prepare_policy=(
                    partial(act_policy.compile_model, mode=compile_mode)
                    if compile_act
                    else None
                ),
                uses_cuda_graphs=uses_cuda_graphs,
                gripper_default_width_m=gripper_default_width_m,
                gripper_initialization_registry=(self._gripper_initialization_registry),
                gripper_command_parameters=gripper_command_parameters,
                camera_names=camera_names,
            )

        with self._lock:
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
            if waypoint_layout_inferred:
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
        self._runner = runner
        runner.start()
        return self.status()

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

    def _on_runner_finished(self, stop_reason: str | None, step: int) -> None:
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
    ) -> tuple[list[dict[str, Any]], int, bool]:
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
        return layout, action_dim, inferred

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

    def _prepare_motion(self, robot: Any, serial: str) -> None:
        hardware.prepare_robot_motion(robot, serial, self._stop_event, self._append_log)

    def _release_robots(self) -> None:
        with self._lock:
            robots, self._robots = self._robots, []
        hardware.stop_robots(robots)

    def _make_release_robots(self, robots: list[Any]) -> Callable[[], None]:
        """Release only this run's robots.

        A planner thread that outlives its stop() would otherwise reach the
        shared list and stop the *next* run's robots from its own finally.
        """

        def release() -> None:
            with self._lock:
                if self._robots is robots:
                    self._robots = []
            hardware.stop_robots(robots)

        return release
