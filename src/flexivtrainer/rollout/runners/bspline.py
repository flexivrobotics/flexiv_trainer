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

"""Run B-spline diffusion policies through a continuous Cartesian executor."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from flexivtrainer.data.gripper_command import GripperCommandMetadata
from flexivtrainer.data.lerobot_io import resolve_recording_image_names
from flexivtrainer.observability import (
    describe_exception,
    describe_traceback,
    error,
    warn,
)
from flexivtrainer.rollout import _cudagraph_state, observations
from flexivtrainer.rollout.executors.bspline import (
    BSplineActionLayout,
    BSplineExecutor,
    BSplineInstallResult,
)
from flexivtrainer.rollout.executors.gripper import GripperExecutor
from flexivtrainer.runtime.gripper_session import GripperInitializationRegistry


class BSplineRunner:
    """Own the B-spline planner thread and its Cartesian/gripper executors."""

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        robots: list[Any],
        sides: list[str],
        followers: list[str],
        cameras: Any,
        rollout_cfg: Any,
        target_hz: float,
        device: str,
        task: str | None,
        bspline_layout: BSplineActionLayout,
        end_effector_config: dict[str, Any],
        motion_limits: tuple[float, float, float, float],
        max_steps: int,
        stop_event: threading.Event,
        append_log: Callable[..., None],
        append_metric: Callable[[dict[str, Any]], None],
        on_error: Callable[[str], None],
        on_cleanup_error: Callable[[str], None],
        on_finished: Callable[[str | None, int], None],
        release_robots: Callable[[], None],
        stop_robots: Callable[[list[Any]], None],
        prepare_motion: Callable[[Any, str], None],
        image_resolutions: dict[str, tuple[int, int]] | None = None,
        append_wrench: Callable[[dict[str, Any]], None] | None = None,
        gripper_default_width_m: float | None = None,
        gripper_initialization_registry: GripperInitializationRegistry | None = None,
        gripper_command_parameters: GripperCommandMetadata | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robots = robots
        self._sides = sides
        self._followers = followers
        self._cameras = cameras
        self._image_resolutions = image_resolutions
        self._rollout_cfg = rollout_cfg
        self._target_hz = target_hz
        self._device = device
        self._task = task
        self._bspline_layout = bspline_layout
        self._end_effector_config = end_effector_config
        self._motion_limits = motion_limits
        self._max_steps = max_steps
        self._stop_event = stop_event
        self._append_log = append_log
        self._append_metric = append_metric
        self._append_wrench = append_wrench
        self._last_wrench_t = 0.0
        self._on_error = on_error
        self._on_cleanup_error = on_cleanup_error
        self._on_finished = on_finished
        self._release_robots = release_robots
        self._stop_robots = stop_robots
        self._prepare_motion = prepare_motion
        self._gripper_default_width_m = gripper_default_width_m
        self._gripper_initialization_registry = gripper_initialization_registry
        self._gripper_command_parameters = gripper_command_parameters

        self._error: str | None = None
        self._stop_reason: str | None = None
        self._thread: threading.Thread | None = None
        self._bspline_executor: BSplineExecutor | None = None
        self._gripper_executor: GripperExecutor | None = None
        self._build_executors()

    def start(self) -> None:
        thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="rollout-policy-planner",
        )
        self._thread = thread
        thread.start()

    def stop(self, timeout: float = 2.0) -> list[str]:
        cleanup_errors: list[str] = []
        bspline_executor = self._bspline_executor
        if bspline_executor is not None and not bspline_executor.join():
            self._stop_robots(list(self._robots))
            if not bspline_executor.join(timeout=0.5):
                cleanup_errors.append(
                    "B-spline Cartesian executor did not stop cleanly"
                )
        gripper_executor = self._gripper_executor
        if gripper_executor is not None:
            try:
                gripper_executor.stop()
            except Exception as exc:
                self._stop_robots(list(self._robots))
                cleanup_errors.append(describe_exception(exc))
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                cleanup_errors.append("Rollout planner did not stop cleanly")
        alive = thread is not None and thread.is_alive()
        self._thread = thread if alive else None
        self._bspline_executor = None
        self._gripper_executor = None
        return cleanup_errors

    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {"error": self._error, "stop_reason": self._stop_reason}

    def _build_executors(self) -> None:
        robots = self._robots
        gripper_executor: GripperExecutor | None = None
        try:
            config = self._policy.config
            rollout_cfg = self._rollout_cfg
            bspline_executor = BSplineExecutor(
                robots,
                config.action_feature_names,
                self._stop_event,
                self._motion_limits,
                checkpoint_fps=self._target_hz,
                degree=config.spline_degree,
                control_hz=rollout_cfg.control_hz,
                speed_scale=rollout_cfg.speed_scale,
                predict_before_end_s=rollout_cfg.predict_before_end_s,
                time_align_error_threshold=(rollout_cfg.time_align_error_threshold),
                time_align_max_fraction=rollout_cfg.time_align_max_fraction,
                handoff_blend_s=getattr(rollout_cfg, "handoff_blend_s", 0.15),
                handoff_max_accel=getattr(rollout_cfg, "handoff_max_accel", 2.0),
            )
            if self._bspline_layout.gripper_sides:
                gripper_kwargs: dict[str, Any] = {
                    "target_source": lambda: (bspline_executor.last_gripper_targets),
                    "failure_event": self._stop_event,
                }
                if self._bspline_layout.gripper_target_mode == "target_width":
                    gripper_kwargs["target_mode"] = "target_width"
                    gripper_kwargs["command_parameters"] = (
                        self._gripper_command_parameters
                    )
                if self._gripper_default_width_m is not None:
                    gripper_kwargs["default_width_m"] = self._gripper_default_width_m
                if self._gripper_initialization_registry is not None:
                    gripper_kwargs.update(
                        followers=self._followers,
                        initialization_registry=(self._gripper_initialization_registry),
                        append_log=self._append_log,
                    )
                gripper_executor = GripperExecutor(
                    robots,
                    self._sides,
                    self._end_effector_config,
                    self._bspline_layout.gripper_sides,
                    **gripper_kwargs,
                )
                gripper_executor.initialize()
            for serial, robot in zip(self._followers, robots, strict=True):
                self._prepare_motion(robot, serial)
        except Exception as exc:
            if gripper_executor is not None:
                gripper_executor.stop()
            self._stop_robots(robots)
            raise RuntimeError(
                f"Failed to connect to robot: {describe_exception(exc)}"
            ) from exc
        self._bspline_executor = bspline_executor
        self._gripper_executor = gripper_executor

    @staticmethod
    def _bspline_action_vector(actions: Any) -> np.ndarray:
        detached = getattr(actions, "detach", None)
        if callable(detached):
            actions = actions.detach().cpu().numpy()
        array = np.asarray(actions, dtype=np.float64)
        if array.ndim == 3 and array.shape[:2] == (1, 1):
            return array[0, 0]
        if array.ndim == 2 and array.shape[0] == 1:
            return array[0]
        if array.ndim == 1:
            return array
        raise ValueError(
            f"B-spline policy must return one flat action, got shape={array.shape}"
        )

    def _infer_bspline_plan(
        self,
        policy: Any,
        postprocessor: Any,
        executor: BSplineExecutor,
        observed_at: float,
    ) -> tuple[float, BSplineInstallResult | None]:
        infer_started = time.monotonic()
        actions = policy.predict_action_chunk()
        observations._cuda_sync(self._device)
        actions = postprocessor(actions)
        observations._cuda_sync(self._device)
        inference_latency = time.monotonic() - infer_started
        if self._stop_event.is_set():
            return inference_latency, None
        # Alignment needs observation staleness, which includes the planner's
        # queueing delay -- not the compute time infer_ms reports.
        result = executor.install(
            self._bspline_action_vector(actions),
            observation_age_s=time.monotonic() - observed_at,
        )
        return inference_latency, result

    def _log_timing_contract(self) -> None:
        """Log what governs timing, so a timing fault is not read as a model fault."""
        cfg = self._rollout_cfg
        config = getattr(self._policy, "config", None)
        # Only speed_scale is guarded, because only it is used arithmetically.
        speed_scale = float(getattr(cfg, "speed_scale", 1.0) or 1.0)
        # apply_rollout_overrides puts the effective step count on the model, not cfg.
        steps = getattr(
            getattr(self._policy, "diffusion", None), "num_inference_steps", None
        )
        if steps is None:
            steps = getattr(config, "num_inference_steps", None)
        fields = {
            "target_hz": self._target_hz,
            "control_hz": getattr(cfg, "control_hz", None),
            "denoise_steps": steps,
            "knot_rate_hz": getattr(config, "knot_rate_hz", None),
            "playback_speed": getattr(cfg, "playback_speed", None),
            "speed_scale": speed_scale,
            "source_rate": self._target_hz * speed_scale,
            "predict_before_end_s": getattr(cfg, "predict_before_end_s", None),
            "handoff_blend_s": getattr(cfg, "handoff_blend_s", None),
        }
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        self._append_log("INFO", "ROLLOUT", "B-spline timing contract", detail)

    def _run(self) -> None:
        policy = self._policy
        preprocessor = self._preprocessor
        postprocessor = self._postprocessor
        robots = self._robots
        sides = self._sides
        rollout_cfg = self._rollout_cfg
        target_hz = self._target_hz
        task = self._task
        executor = self._bspline_executor
        gripper = self._gripper_executor
        assert executor is not None

        policy.reset()
        self._log_timing_contract()
        period = 1.0 / target_hz
        next_observation = time.monotonic()
        observed_at = next_observation
        camera_names = resolve_recording_image_names(None, sides)
        max_steps = self._max_steps
        inference_latency = 0.0
        alignment_error = 0.0
        blend_s = 0.0
        align_searched = False
        align_capped = False
        step = 0
        inference_future: Future[tuple[float, BSplineInstallResult | None]] | None = (
            None
        )
        inference_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rollout-bspline-inference"
        )
        try:
            executor.start()
            if gripper is not None:
                gripper.start()
            while True:
                if gripper is not None and gripper.error is not None:
                    raise RuntimeError(
                        f"B-spline gripper failed: {describe_exception(gripper.error)}"
                    )
                if self._stop_event.is_set():
                    if executor.error is not None:
                        raise RuntimeError(executor.error)
                    break
                for robot in robots:
                    if robot.fault():
                        raise RuntimeError("Fault occurred on a follower robot")

                installed = False
                if inference_future is not None and inference_future.done():
                    inference_latency, result = inference_future.result()
                    inference_future = None
                    if result is not None:
                        alignment_error = result.alignment_error
                        blend_s = result.blend_s
                        align_searched = result.align_searched
                        align_capped = result.align_capped
                        installed = True
                        if result.warning is not None:
                            warn("B-spline handoff warning", result.warning)
                            self._append_log(
                                "WARNING",
                                "ROLLOUT",
                                "B-spline handoff warning",
                                result.warning,
                            )

                now = time.monotonic()
                observed = now >= next_observation
                snapshot: dict[str, Any] | None = None
                if observed:
                    observed_at = now
                    gripper_states = (
                        gripper.measured_states() if gripper is not None else None
                    )
                    images = observations.grab_images(
                        self._cameras, camera_names, self._image_resolutions
                    )
                    snapshot = observations.read_robot_snapshot(
                        robots, gripper_states, sides
                    )
                    observation = observations.build_observation(
                        snapshot, images, sides
                    )
                    prepared = observations._prepare_policy_observation(
                        observation,
                        self._device,
                        preprocessor,
                        task=task,
                    )
                    policy.enqueue_observation(prepared)
                    step += 1
                    missed = max(
                        0, int((time.monotonic() - next_observation) // period)
                    )
                    next_observation += (missed + 1) * period

                if inference_future is None and executor.replan_needed():
                    inference_future = inference_pool.submit(
                        self._infer_bspline_plan,
                        policy,
                        postprocessor,
                        executor,
                        observed_at,
                    )

                executor_status = executor.status()
                self._append_metric(
                    {
                        "t": round(time.monotonic(), 3),
                        "step": step,
                        "send_hz": round(executor_status.achieved_send_hz, 2),
                        "missed_deadlines": executor_status.missed_deadlines,
                        "spline_remaining_s": (
                            None
                            if executor_status.remaining_s is None
                            else round(executor_status.remaining_s, 4)
                        ),
                        "infer_ms": round(inference_latency * 1000.0, 1),
                        "alignment_error": round(alignment_error, 6),
                        "handoff_blend_s": round(blend_s, 4),
                        "align_searched": align_searched,
                        "align_capped": align_capped,
                        "handoff_warnings": executor_status.handoff_warnings,
                        "fresh": installed,
                    }
                )
                self._last_wrench_t = observations.sample_wrench(
                    self._append_wrench,
                    snapshot,
                    sides,
                    time.monotonic(),
                    self._last_wrench_t,
                )
                if max_steps and step >= max_steps:
                    self._stop_reason = "timeout"
                    break

                now = time.monotonic()
                wake_at = next_observation
                if inference_future is not None:
                    wake_at = min(wake_at, now + 0.01)
                else:
                    remaining = executor.remaining_s(now)
                    until_replan = max(
                        0.0,
                        (remaining or 0.0) - rollout_cfg.predict_before_end_s,
                    )
                    wake_at = min(wake_at, now + until_replan)
                self._stop_event.wait(max(0.0, wake_at - now))
        except Exception as exc:
            detail = describe_exception(exc)
            self._error = detail
            self._on_error(detail)
            warn("Rollout stopped", detail)
            error("Rollout planner thread crashed", describe_traceback(exc))
        finally:
            self._stop_event.set()
            inference_pool.shutdown(wait=True, cancel_futures=True)
            if not executor.join():
                self._stop_robots(robots)
                if not executor.join(timeout=0.5):
                    self._on_cleanup_error(
                        "B-spline Cartesian executor did not stop cleanly"
                    )
            if gripper is not None:
                try:
                    gripper.stop()
                except Exception as exc:
                    self._stop_robots(robots)
                    warn("B-spline gripper shutdown failed", describe_exception(exc))
                    self._on_cleanup_error(describe_exception(exc))
            self._bspline_executor = None
            self._gripper_executor = None
            self._policy = self._preprocessor = self._postprocessor = None
            _cudagraph_state.teardown_rollout_gpu_state(
                self._device, cudagraphs_seeded=False
            )
            self._release_robots()
            self._on_finished(self._stop_reason, step)
