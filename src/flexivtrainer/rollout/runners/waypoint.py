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

"""Run discrete action-chunk policies through a waypoint executor."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from flexivtrainer.data.lerobot_io import (
    resolve_recording_image_names,
)
from flexivtrainer.observability import (
    describe_exception,
    describe_traceback,
    error,
    warn,
)
from flexivtrainer.rollout import _cudagraph_state, hardware, observations
from flexivtrainer.rollout.executors.gripper import (
    GripperExecutor,
    initialize_gripper_executor,
)
from flexivtrainer.rollout.executors.waypoint import (
    WaypointExecutor,
    normalize_pose_quaternion,
)
from flexivtrainer.runtime.gripper_session import GripperInitializationRegistry


class WaypointRunner:
    """Own the ACT/Diffusion planner thread and its waypoint executor."""

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
        action_layout: list[dict[str, Any]],
        action_dim: int,
        gripper_sides: tuple[str, ...],
        end_effector_config: dict[str, Any],
        motion_limits: tuple[float, float, float, float],
        planner_hz_fallback: float,
        expected_hz_fallback: float,
        max_steps: int,
        stop_event: threading.Event,
        append_log: Callable[..., None],
        append_metric: Callable[[dict[str, Any]], None],
        on_error: Callable[[str], None],
        on_cleanup_error: Callable[[str], None],
        on_finished: Callable[[str | None, int], None],
        release_robots: Callable[[], None],
        stop_robots: Callable[[list[Any]], None],
        prepare_policy: Callable[[Any], None] | None = None,
        uses_cuda_graphs: bool = False,
        image_resolutions: dict[str, tuple[int, int]] | None = None,
        append_wrench: Callable[[dict[str, Any]], None] | None = None,
        gripper_default_width_m: float | None = None,
        gripper_initialization_registry: GripperInitializationRegistry | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robots = robots
        self._sides = sides
        self._cameras = cameras
        self._image_resolutions = image_resolutions
        self._rollout_cfg = rollout_cfg
        self._target_hz = target_hz
        self._device = device
        self._task = task
        self._action_layout = action_layout
        self._action_dim = action_dim
        self._motion_limits = motion_limits
        self._planner_hz_fallback = planner_hz_fallback
        self._expected_hz_fallback = expected_hz_fallback
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
        self._prepare_policy = prepare_policy
        self._uses_cuda_graphs = uses_cuda_graphs

        self._error: str | None = None
        self._stop_reason: str | None = None
        self._thread: threading.Thread | None = None
        self._waypoint_executor: WaypointExecutor | None = None
        gripper: GripperExecutor | None = None
        try:
            gripper = initialize_gripper_executor(
                robots,
                sides,
                end_effector_config,
                gripper_sides,
                failure_event=stop_event,
                default_width_m=gripper_default_width_m,
                followers=followers,
                initialization_registry=gripper_initialization_registry,
                append_log=append_log,
                executor_factory=GripperExecutor,
            )
            for serial, robot in zip(followers, robots, strict=True):
                hardware.prepare_robot_motion(
                    robot, serial, stop_event, append_log
                )
        except Exception as exc:
            if gripper is not None:
                gripper.stop()
            stop_robots(robots)
            raise RuntimeError(
                "Failed to prepare waypoint rollout hardware: "
                f"{describe_exception(exc)}"
            ) from exc
        self._gripper_executor = gripper

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
        # Stop robot commands before releasing their connections.
        executor = self._waypoint_executor
        if executor is not None:
            executor.join()
        gripper = self._gripper_executor
        if gripper is not None:
            try:
                gripper.stop()
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
        self._waypoint_executor = None
        self._gripper_executor = None
        return cleanup_errors

    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {"error": self._error, "stop_reason": self._stop_reason}

    def _planner_hz(self) -> float:
        return float(self._target_hz or self._planner_hz_fallback)

    def _run(self) -> None:
        policy = self._policy
        preprocessor = self._preprocessor
        postprocessor = self._postprocessor
        robots = self._robots
        sides = self._sides
        rollout_cfg = self._rollout_cfg
        target_hz = self._target_hz
        task = self._task

        period = 1.0 / self._planner_hz()
        # Waypoint spacing follows dataset FPS, not planner frequency.
        dt = 1.0 / float(target_hz)
        anchor = rollout_cfg.action_anchor_offset_steps
        # Auto replan uses half the first effective action chunk.
        replan_steps: int | None = None
        max_steps = self._max_steps
        camera_names = resolve_recording_image_names(None, sides)
        layout = self._action_layout
        gripper = self._gripper_executor
        # Off by default: _actions_to_lists already syncs the tensors it copies,
        # so this only shifts GPU wait out of infer_ms and into to_list.
        debug_timing = bool(getattr(rollout_cfg, "debug_timing", False))
        log_every = max(1, int(self._planner_hz() // 2))
        stage_times: dict[str, deque[float]] = {
            name: deque(maxlen=10)
            for name in (
                "fault_check", "grab_images", "read_states",
                "build_obs", "inference", "to_list", "dispatch",
            )
        }
        infer_raw: deque[float] = deque(maxlen=log_every)
        waypoint_executor: WaypointExecutor | None = None
        previous_loop_start: float | None = None
        step = 0
        actions: Any = None
        seeded = False
        try:
            cuda_graph_step_begin: Callable[[], None] | None = None
            if self._uses_cuda_graphs:
                import torch  # noqa: PLC0415

                # Must precede compilation: torch.compiler.reset() reaches
                # reset_cudagraph_trees() on this thread and asserts unseeded.
                _cudagraph_state.seed_thread_local_state()
                seeded = True
                cuda_graph_step_begin = torch.compiler.cudagraph_mark_step_begin
            if self._prepare_policy is not None:
                self._prepare_policy(policy)
            policy.reset()
            waypoint_executor = WaypointExecutor(
                robots,
                layout,
                self._stop_event,
                self._motion_limits,
                action_dim=self._action_dim,
                submit_gripper=(gripper.submit if gripper is not None else None),
            )
            self._waypoint_executor = waypoint_executor
            if gripper is not None:
                gripper.start()
            waypoint_executor.start()
            while True:
                if self._stop_event.is_set():
                    if waypoint_executor.error is not None:
                        raise RuntimeError(waypoint_executor.error)
                    if gripper is not None and gripper.error is not None:
                        raise RuntimeError(
                            "Waypoint gripper failed: "
                            f"{describe_exception(gripper.error)}"
                        )
                    break
                if gripper is not None and gripper.error is not None:
                    raise RuntimeError(
                        "Waypoint gripper failed: "
                        f"{describe_exception(gripper.error)}"
                    )
                loop_start = time.monotonic()
                loop_period = (
                    loop_start - previous_loop_start
                    if previous_loop_start is not None
                    else 0.0
                )
                actual_hz = 1.0 / loop_period if loop_period > 0 else 0.0
                previous_loop_start = loop_start
                mark = loop_start

                for robot in robots:
                    if robot.fault():
                        raise RuntimeError("Fault occurred on a follower robot")
                now = time.monotonic()
                stage_times["fault_check"].append(now - mark)
                mark = now

                images = observations.grab_images(
                    self._cameras, camera_names, self._image_resolutions
                )
                now = time.monotonic()
                stage_times["grab_images"].append(now - mark)
                mark = now

                gripper_states = (
                    gripper.measured_states() if gripper is not None else None
                )
                snapshot = observations.read_robot_snapshot(
                    robots, gripper_states, sides
                )
                now = time.monotonic()
                stage_times["read_states"].append(now - mark)
                mark = now

                observation = observations.build_observation(snapshot, images, sides)
                now = time.monotonic()
                stage_times["build_obs"].append(now - mark)
                mark = now

                # Replan early enough to retain a committed path during inference.
                force = replan_steps is None or step % replan_steps == 0
                if cuda_graph_step_begin is not None:
                    cuda_graph_step_begin()
                actions, fresh = observations._predict_action_chunk(
                    observation,
                    policy,
                    self._device,
                    preprocessor,
                    postprocessor,
                    force_refresh=force,
                    task=task,
                )
                if debug_timing:
                    observations._cuda_sync(self._device)
                now = time.monotonic()
                infer_seconds = now - mark
                stage_times["inference"].append(infer_seconds)
                infer_raw.append(infer_seconds)
                mark = now

                action_lists = self._actions_to_lists(actions)
                waypoint_executor.validate_actions(action_lists)
                now = time.monotonic()
                stage_times["to_list"].append(now - mark)
                mark = now

                # Fresh chunks replace pending waypoints on an anchored time grid.
                if fresh:
                    if replan_steps is None:
                        effective = len(action_lists)
                        replan_steps = rollout_cfg.replan_steps or max(
                            1, effective // 2
                        )
                        if replan_steps > effective:
                            warn(
                                "Clamped replan_steps to the effective chunk length",
                                f"replan_steps={replan_steps} chunk={effective}",
                            )
                            replan_steps = effective
                    # Anchored on loop_start, not on inference completion: replan
                    # wipes pending waypoints every tick, so waypoint 0 only fires
                    # while inference is faster than dt. Anchoring later (or raising
                    # anchor) pushes it past the next replan and it never fires.
                    target_times = [
                        loop_start + (k + anchor) * dt
                        for k in range(len(action_lists))
                    ]
                    waypoint_executor.replace_waypoints(
                        action_lists, target_times, now=time.monotonic()
                    )
                if waypoint_executor.error is not None:
                    raise RuntimeError(waypoint_executor.error)
                stage_times["dispatch"].append(time.monotonic() - mark)

                self._append_metric({
                    "t": round(loop_start, 3),
                    "step": step,
                    "hz": round(actual_hz, 2),
                    "infer_ms": round(infer_seconds * 1000.0, 1),
                    "fresh": bool(fresh),
                })
                self._last_wrench_t = observations.sample_wrench(
                    self._append_wrench,
                    snapshot,
                    sides,
                    loop_start,
                    self._last_wrench_t,
                )
                if step % log_every == 0:
                    self._log_timing(
                        step,
                        stage_times,
                        infer_raw,
                        waypoint_executor.scheduled_count,
                    )
                    self._log_step(
                        step, snapshot, action_lists[0], layout, sides,
                        images, camera_names, actual_hz,
                    )

                step += 1
                if max_steps and step >= max_steps:
                    self._stop_reason = "timeout"
                    break

                elapsed = time.monotonic() - loop_start
                if period - elapsed > 0:
                    self._stop_event.wait(period - elapsed)
        except Exception as exc:
            detail = describe_exception(exc)
            self._error = detail
            self._on_error(detail)
            warn("Rollout stopped", detail)
            error("Rollout planner thread crashed", describe_traceback(exc))
        finally:
            # Stop commands before releasing robot connections.
            if waypoint_executor is not None:
                self._stop_event.set()
                waypoint_executor.join()
            if gripper is not None:
                try:
                    gripper.stop()
                except Exception as exc:
                    self._stop_robots(robots)
                    detail = describe_exception(exc)
                    warn("Waypoint gripper shutdown failed", detail)
                    self._on_cleanup_error(detail)
            self._waypoint_executor = None
            self._gripper_executor = None
            # Drop anything that can alias a CUDA-graph output buffer before
            # tearing the pool down, or its memory cannot be reclaimed.
            actions = None
            self._policy = self._preprocessor = self._postprocessor = None
            del policy, preprocessor, postprocessor
            _cudagraph_state.teardown_rollout_gpu_state(
                self._device, cudagraphs_seeded=seeded
            )
            self._release_robots()
            self._on_finished(self._stop_reason, step)

    @staticmethod
    def _actions_to_lists(actions: Any) -> list[list[float]]:
        """Convert an action chunk to per-step float vectors."""
        detached = getattr(actions, "detach", None)
        if callable(detached):
            actions = actions.detach().cpu().numpy()
        array = np.asarray(actions, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim == 3:
            array = array.reshape(array.shape[-2], array.shape[-1])
        return [[float(v) for v in row] for row in array]

    def _log_timing(
        self,
        step: int,
        stage_times: dict[str, deque[float]],
        infer_raw: deque[float],
        scheduled: int,
    ) -> None:
        """Log recent mean stage durations and the scheduled waypoint count."""
        parts: list[str] = []
        total_ms = 0.0
        for name, samples in stage_times.items():
            mean_ms = 1000.0 * sum(samples) / len(samples) if samples else 0.0
            total_ms += mean_ms
            parts.append(f"{name}={mean_ms:.1f}ms")
        parts.append(f"total={total_ms:.1f}ms")
        parts.append(f"sched={scheduled}")
        if infer_raw:
            raw_ms = [1000.0 * value for value in infer_raw]
            parts.append(f"infer_max={max(raw_ms):.1f}ms")
        self._append_log("INFO", "ROLLOUT", f"step={step} timing", " ".join(parts))

    def _log_step(
        self,
        step: int,
        snapshot: dict[str, Any],
        action: list[float],
        layout: list[dict[str, Any]],
        sides: list[str],
        images: dict[str, np.ndarray],
        camera_names: list[str],
        actual_hz: float,
    ) -> None:
        """Log observation health and measured versus commanded poses."""
        cam_parts: list[str] = []
        for name in camera_names:
            image = images.get(name)
            if image is None:
                cam_parts.append(f"{name}=MISSING")
            else:
                cam_parts.append(f"{name}=ok(mean={float(np.asarray(image).mean()):.1f})")
        expected_hz = float(self._target_hz or self._expected_hz_fallback)
        cam_parts.append(f"freq={actual_hz:.1f}/{expected_hz:.1f}Hz")
        self._append_log("INFO", "ROLLOUT", f"step={step} obs", " ".join(cam_parts))

        robots_payload = snapshot.get("robots") if isinstance(snapshot, dict) else None
        payloads = (
            list(robots_payload.values()) if isinstance(robots_payload, dict) else []
        )
        for index, plan in enumerate(layout):
            side = plan.get("side") or (
                sides[index] if index < len(sides) else f"arm_{index}"
            )
            pose_slice = plan["pose"]
            twist_slice = plan["twist"]
            commanded = (
                normalize_pose_quaternion(action[pose_slice])
                if pose_slice is not None
                else []
            )
            commanded_twist = (
                list(action[twist_slice]) if twist_slice is not None else []
            )
            measured: list[float] = []
            measured_gripper: dict[str, Any] = {}
            if index < len(payloads) and isinstance(payloads[index], dict):
                states = payloads[index].get("states")
                if isinstance(states, dict):
                    measured = list(states.get("tcp_pose") or [])
                gripper_payload = payloads[index].get("gripper")
                if isinstance(gripper_payload, dict):
                    measured_gripper = gripper_payload
            gripper_width_index = plan.get("gripper_width")
            gripper_detail = ""
            if isinstance(gripper_width_index, int):
                gripper_detail = (
                    f" cmd_gripper_width={float(action[gripper_width_index]):.4f}"
                    " meas_gripper_width="
                    f"{self._fmt_scalar(measured_gripper.get('width'))}"
                    " meas_gripper_force="
                    f"{self._fmt_scalar(measured_gripper.get('force'))}"
                )
            self._append_log(
                "INFO",
                "ROLLOUT",
                f"step={step} {side}",
                (
                    f"cmd_xyz={self._fmt_xyz(commanded)} "
                    f"meas_xyz={self._fmt_xyz(measured)} "
                    f"cmd_twist={self._fmt_vector(commanded_twist)}"
                    f"{gripper_detail}"
                ),
            )

    @staticmethod
    def _fmt_xyz(pose: list[float]) -> str:
        if len(pose) < 3:
            return "n/a"
        return "[" + ", ".join(f"{pose[i]:.3f}" for i in range(3)) + "]"

    @staticmethod
    def _fmt_vector(vector: list[float]) -> str:
        if not vector:
            return "n/a"
        return "[" + ", ".join(f"{value:.3f}" for value in vector) + "]"

    @staticmethod
    def _fmt_scalar(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "n/a"
