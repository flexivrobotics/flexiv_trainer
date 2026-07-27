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
from typing import Any

from flexivtrainer.config import AppSettings, TeleopRobotPair
from flexivtrainer.jobs.train_policy import _encode_ui_log
from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.policies import bspline_diffusion as bspline_policy
from flexivtrainer.policies import diffusion as diffusion_policy
from flexivtrainer.policies import dit as dit_policy
from flexivtrainer.rollout import hardware
from flexivtrainer.rollout.checkpoint import (
    _checkpoint_target_hz,
    _default_policy_loader,
    _positive_float,
    checkpoint_image_resolutions,
    resolve_checkpoint_path,
)
from flexivtrainer.rollout.executors.bspline import (
    BSplineActionLayout,
    parse_bspline_action_layout,
)
from flexivtrainer.rollout.hardware import _default_robot_factory
from flexivtrainer.rollout.runners.bspline import BSplineRunner
from flexivtrainer.rollout.runners.waypoint import WaypointRunner

_ROLLOUT_OVERRIDES = {
    "bspline_diffusion": bspline_policy.apply_rollout_overrides,
    "diffusion": diffusion_policy.apply_rollout_overrides,
    "multi_task_dit": dit_policy.apply_rollout_overrides,
}


class RolloutService:
    """Lifecycle and background control loop for policy rollout."""

    def __init__(
        self,
        settings: AppSettings,
        cameras: Any,
        teleop: Any,
        get_robot_pairs: Callable[[], list[TeleopRobotPair]],
        get_active_sides: Callable[[], list[str]],
        *,
        get_end_effector_config: Callable[[], dict[str, Any]] | None = None,
        policy_loader: Callable[[str, str], Any] = _default_policy_loader,
        robot_factory: Callable[[str], Any] = _default_robot_factory,
        resolve_device: Callable[[str], str] | None = None,
    ) -> None:
        self._settings = settings
        self._cameras = cameras
        self._teleop = teleop
        self._get_robot_pairs = get_robot_pairs
        self._get_active_sides = get_active_sides
        self._get_end_effector_config = get_end_effector_config or (lambda: {})
        self._policy_loader = policy_loader
        self._robot_factory = robot_factory
        if resolve_device is None:
            from flexivtrainer.jobs.train_policy import resolve_training_device

            resolve_device = resolve_training_device
        self._resolve_device = resolve_device

        self._lock = threading.Lock()
        self._running = False
        self._error: str | None = None
        self._stop_reason: str | None = None
        self._checkpoint_path: str | None = None
        self._task: str | None = None
        self._robots: list[Any] = []
        self._device = "cpu"
        self._target_hz: float | None = None
        self._runner: WaypointRunner | BSplineRunner | None = None
        self._stop_event = threading.Event()
        self._logs: deque[str] = deque(maxlen=2000)
        self._metrics: deque[dict[str, Any]] = deque(maxlen=300)

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._running:
                status = "running"
            elif self._error:
                status = "failed"
            else:
                status = "idle"
            return {
                "status": status,
                "checkpoint_path": self._checkpoint_path,
                "task": self._task,
                "error": self._error,
                "stop_reason": self._stop_reason,
                "logs": list(self._logs),
                "log_lines": len(self._logs),
                "metrics": list(self._metrics),
                "target_hz": self._target_hz,
            }

    def _append_log(
        self, level: str, source: str, message: str, detail: str = ""
    ) -> None:
        self._logs.append(_encode_ui_log(level, source, message, detail))

    def start(
        self, checkpoint_path: str, task: str | None = None
    ) -> dict[str, Any]:
        task = task.strip() if isinstance(task, str) else None
        task = task or None
        with self._lock:
            if self._running:
                raise RuntimeError("Rollout is already running")
            # A fresh RDK connection cannot coexist with the TDK controller
            # holding the same follower's LAN connection; require teleop down.
            if self._teleop_initialized():
                raise RuntimeError(
                    "Stop teleoperation before starting a rollout "
                    "(it holds the robot connection)."
                )

        try:
            resolved_checkpoint = resolve_checkpoint_path(
                checkpoint_path, self._settings.storage.root
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Checkpoint not found: {checkpoint_path}") from exc
        checkpoint_path = str(resolved_checkpoint)

        device = self._resolve_device(self._settings.training.default_device)
        sides = self._get_active_sides()
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
        policy_type = getattr(
            getattr(policy, "config", None), "type", None
        ) or getattr(policy, "name", "")
        is_bspline = policy_type == "bspline_diffusion"
        target_hz = self._resolve_target_hz(
            checkpoint_path, policy, require_metadata=is_bspline
        )
        image_resolutions = checkpoint_image_resolutions(checkpoint_path)
        rollout_cfg = self._settings.policies.rollout_for(policy_type)
        override_fn = _ROLLOUT_OVERRIDES.get(policy_type)
        scheduler_overridden = (
            override_fn(policy, rollout_cfg) if override_fn is not None else False
        )
        bspline_layout: BSplineActionLayout | None = None
        end_effector_config: dict[str, Any] = {}
        if is_bspline:
            bspline_layout = self._preflight_bspline(
                policy, sides, followers, target_hz
            )
            end_effector_config = dict(self._get_end_effector_config() or {})
            self._preflight_bspline_grippers(
                bspline_layout, end_effector_config
            )
        else:
            self._apply_n_action_steps(policy, rollout_cfg)

        self._stop_event.clear()
        robots: list[Any] = []
        try:
            for serial in followers:
                robots.append(
                    hardware.connect_robot(
                        self._robot_factory,
                        serial,
                        self._stop_event,
                        prepare_motion=(
                            None if is_bspline else self._prepare_motion
                        ),
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
                end_effector_config=end_effector_config,
                motion_limits=motion_limits,
                max_steps=app_rollout.max_steps,
                stop_event=self._stop_event,
                append_log=self._append_log,
                append_metric=self._metrics.append,
                on_error=self._on_runner_error,
                on_cleanup_error=self._on_runner_cleanup_error,
                on_finished=self._on_runner_finished,
                release_robots=self._release_robots,
                stop_robots=hardware.stop_robots,
                prepare_motion=self._prepare_motion,
            )
        else:
            runner = WaypointRunner(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                robots=robots,
                sides=sides,
                cameras=self._cameras,
                image_resolutions=image_resolutions,
                rollout_cfg=rollout_cfg,
                target_hz=target_hz,
                device=device,
                task=task,
                motion_limits=motion_limits,
                planner_hz_fallback=app_rollout.planner_hz,
                expected_hz_fallback=app_rollout.action_dt_hz,
                max_steps=app_rollout.max_steps,
                stop_event=self._stop_event,
                append_log=self._append_log,
                append_metric=self._metrics.append,
                on_error=self._on_runner_error,
                on_finished=self._on_runner_finished,
                release_robots=self._release_robots,
            )

        with self._lock:
            self._checkpoint_path = checkpoint_path
            self._task = task
            self._error = None
            self._stop_reason = None
            self._robots = robots
            self._device = device
            self._target_hz = target_hz
            self._running = True
            self._logs.clear()
            self._metrics.clear()
            self._logs.append(
                _encode_ui_log(
                    "INFO",
                    "ROLLOUT",
                    "Rollout started",
                    f"device={device} sides={'+'.join(sides)}",
                )
            )
            if scheduler_overridden:
                self._logs.append(
                    _encode_ui_log(
                        "INFO",
                        "ROLLOUT",
                        "Scheduler overridden",
                        "scheduler="
                        f"{rollout_cfg.noise_scheduler_type} "
                        f"inference_steps={rollout_cfg.num_denoise_steps}",
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
        self._runner = None
        self._release_robots()
        with self._lock:
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
            raise RuntimeError(
                "B-spline checkpoint has no action feature names"
            )
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
            raise RuntimeError(
                "B-spline checkpoint has an invalid spline degree"
            )
        for method in ("enqueue_observation", "predict_action_chunk"):
            if not callable(getattr(policy, method, None)):
                raise RuntimeError(
                    f"B-spline policy does not implement {method}()"
                )
        return layout

    @staticmethod
    def _config_value(config: Any, name: str) -> Any:
        if isinstance(config, dict):
            return config.get(name)
        return getattr(config, name, None)

    @classmethod
    def _preflight_bspline_grippers(
        cls,
        layout: BSplineActionLayout,
        configs: dict[str, Any],
    ) -> None:
        for side in layout.gripper_sides:
            config = configs.get(side)
            if (
                config is None
                or cls._config_value(config, "follower") != "gripper"
                or not cls._config_value(config, "gripper_model")
            ):
                raise RuntimeError(
                    "B-spline checkpoint predicts gripper width but no follower "
                    f"gripper is configured for {side}"
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
                if value > upper:
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
        hardware.prepare_robot_motion(
            robot, serial, self._stop_event, self._append_log
        )

    def _release_robots(self) -> None:
        with self._lock:
            robots, self._robots = self._robots, []
        hardware.stop_robots(robots)
