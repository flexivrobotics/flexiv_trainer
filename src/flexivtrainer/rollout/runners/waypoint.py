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

from flexivtrainer.data.gripper_command import GripperCommandMetadata
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
    pose_command,
)
from flexivtrainer.rollout.live_checks import LiveChecks
from flexivtrainer.rollout.pose_history import PoseHistory, history_columns
from flexivtrainer.rollout.recorder import RolloutRecorder
from flexivtrainer.rollout.safety import SafetyEnvelope, SafetyStop
from flexivtrainer.rollout.takeover import Takeover, operator_action
from flexivtrainer.runtime.gripper_session import GripperInitializationRegistry


def _measured_arms(snapshot: dict[str, Any], key: str) -> list[list[float]]:
    """One measured vector per arm, in arm order, from a robot snapshot."""
    robots = snapshot.get("robots") if isinstance(snapshot, dict) else None
    out: list[list[float]] = []
    for payload in (robots or {}).values():
        states = payload.get("states") if isinstance(payload, dict) else None
        vector = states.get(key) if isinstance(states, dict) else None
        out.append([float(v) for v in vector] if vector is not None else [])
    return out


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
        state_dim: int | None,
        gripper_sides: tuple[str, ...],
        gripper_target_mode: str,
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
        gripper_command_parameters: GripperCommandMetadata | None = None,
        camera_names: list[str] | None = None,
        state_names: list[str] | None = None,
        safety: SafetyEnvelope | None = None,
        recorder: RolloutRecorder | None = None,
        send_twist: bool = True,
        cartesian_stiffness: Any = None,
        max_contact_wrench: Any = None,
        zero_ft_max_force_n: float | None = hardware.FT_ZERO_MAX_FORCE_N,
        takeover: Takeover | None = None,
        action_names: list[str] | None = None,
        live_checks: LiveChecks | None = None,
    ) -> None:
        self._safety = safety or SafetyEnvelope()
        # Learned checks scored on each tick (rollout/live_checks.py); one
        # marked end_on_pass ends the rollout when it passes.
        self._live_checks = live_checks or None
        # Leader-arm takeover (rollout/takeover.py), when the robots were
        # borrowed from teleoperation; None leaves the policy in control.
        self._takeover = takeover
        self._action_names = list(action_names or [])
        self._followers = list(followers)
        self._cartesian_stiffness = cartesian_stiffness
        self._max_contact_wrench = max_contact_wrench
        # Set once the policy has sent its first plan; a takeover needs one.
        self._ready = False
        # Set by a hand-back; the planner resets the policy before it plans.
        self._reset_pending = False
        self._recorder = recorder
        self._send_twist = send_twist
        #: Set when the envelope stopped the rollout: {"code", "detail"}.
        self.safety_stop: dict[str, str] | None = None
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robots = robots
        self._sides = sides
        self._cameras = cameras
        self._camera_names = camera_names
        self._image_resolutions = image_resolutions
        self._rollout_cfg = rollout_cfg
        self._target_hz = target_hz
        self._device = device
        self._task = task
        self._action_layout = action_layout
        self._action_dim = action_dim
        self._state_dim = state_dim
        # The checkpoint's trained state columns; None feeds the full state.
        self._state_names = state_names
        # Columns computed from recent poses (UMI motion, rotation since start);
        # the history itself starts with each run.
        self._history_spec = history_columns(state_names)
        self._pose_history: PoseHistory | None = None
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
                target_mode=gripper_target_mode,
                command_parameters=gripper_command_parameters,
                executor_factory=GripperExecutor,
            )
            for serial, robot in zip(followers, robots, strict=True):
                hardware.prepare_robot_motion(
                    robot,
                    serial,
                    stop_event,
                    append_log,
                    stiffness=cartesian_stiffness,
                    max_contact_wrench=max_contact_wrench,
                    zero_ft_max_force_n=zero_ft_max_force_n,
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

    # --- leader-arm takeover (called from the takeover's switch thread) -------

    @property
    def ready(self) -> bool:
        return self._ready

    def pause_policy(self) -> None:
        """No policy command reaches the robots after this returns."""
        executor = self._waypoint_executor
        if executor is not None:
            executor.pause()
        gripper = self._gripper_executor
        if gripper is not None:
            gripper.pause()

    def reclaim(self) -> None:
        """Take the followers back into the rollout's control mode.

        The same settings as at the start of the rollout, except that the F/T
        sensor is NOT zeroed: the tool may be in contact mid-trial.
        """
        for serial, robot in zip(self._followers, self._robots, strict=True):
            hardware.prepare_robot_motion(
                robot,
                serial,
                self._stop_event,
                self._append_log,
                stiffness=self._cartesian_stiffness,
                max_contact_wrench=self._max_contact_wrench,
                zero_ft=False,
            )

    def resume_policy(self) -> None:
        """Let the policy command again, re-planning from the current state."""
        # Before the executors resume, so the planner resets the policy before
        # it can plan into them.
        self._reset_pending = True
        executor = self._waypoint_executor
        if executor is not None:
            executor.resume()
        gripper = self._gripper_executor
        if gripper is not None:
            gripper.resume()

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
        cameras = self._camera_names
        camera_names = resolve_recording_image_names(None, sides, cameras)
        pose_format: str | None = None
        state_index: np.ndarray | None = None
        layout = self._action_layout
        gripper = self._gripper_executor
        # Off by default: _actions_to_lists already syncs the tensors it copies,
        # so this only shifts GPU wait out of infer_ms and into to_list.
        debug_timing = bool(getattr(rollout_cfg, "debug_timing", False))
        log_every = max(1, int(self._planner_hz() // 2))
        stage_times: dict[str, deque[float]] = {
            name: deque(maxlen=10)
            for name in (
                "fault_check",
                "grab_images",
                "read_states",
                "build_obs",
                "inference",
                "to_list",
                "dispatch",
            )
        }
        infer_raw: deque[float] = deque(maxlen=log_every)
        waypoint_executor: WaypointExecutor | None = None
        previous_loop_start: float | None = None
        step = 0
        actions: Any = None
        seeded = False
        takeover = self._takeover
        # The policy's last recorded action, which an operator tick falls back
        # to for any axis teleoperation does not command.
        last_action: list[float] = []
        replan_now = False
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
            if self._history_spec is not None:
                self._pose_history = PoseHistory(self._history_spec, self._sides)
            waypoint_executor = WaypointExecutor(
                robots,
                layout,
                self._stop_event,
                self._motion_limits,
                action_dim=self._action_dim,
                submit_gripper=(gripper.submit if gripper is not None else None),
                send_twist=self._send_twist,
            )
            safety = self._safety
            last_fresh: float | None = None
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
                        f"Waypoint gripper failed: {describe_exception(gripper.error)}"
                    )
                ended = self._live_checks.ended() if self._live_checks else None
                if ended is not None:
                    self._stop_reason = f"check {ended} passed"
                    break
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
                if self._pose_history is not None:
                    # Operator ticks too, so the motion stays continuous.
                    self._pose_history.update(time.monotonic(), snapshot)
                safety.check_force(_measured_arms(snapshot, "ext_wrench_in_world"))
                control = takeover.control() if takeover is not None else "policy"
                if control != "policy":
                    # The operator has the arm, or a switch is under way: the
                    # policy plans nothing, so its stale-plan watchdog rests.
                    last_fresh = None
                    if pose_format is None:
                        pose_format, state_index = observations.resolve_state_layout(
                            snapshot,
                            sides,
                            cameras,
                            self._state_dim,
                            self._state_names,
                            self._append_log,
                            self._history_names(),
                        )
                    self._operator_tick(
                        snapshot=snapshot,
                        images=images,
                        step=step,
                        control=control,
                        pose_format=pose_format,
                        last_action=last_action,
                    )
                    self._last_wrench_t = observations.sample_wrench(
                        self._append_wrench,
                        snapshot,
                        sides,
                        loop_start,
                        self._last_wrench_t,
                    )
                    step += 1
                    if max_steps and step >= max_steps:
                        self._stop_reason = "timeout"
                        break
                    elapsed = time.monotonic() - loop_start
                    if period - elapsed > 0:
                        self._stop_event.wait(period - elapsed)
                    continue
                if self._reset_pending:
                    # Handed back: plan afresh from where the operator left
                    # the arm, not from the queue or ensemble of before.
                    self._reset_pending = False
                    policy.reset()
                    if self._pose_history is not None:
                        self._pose_history.reset()
                        self._pose_history.update(time.monotonic(), snapshot)
                    replan_now = True
                safety.check_stale(time.monotonic(), last_fresh)
                now = time.monotonic()
                stage_times["read_states"].append(now - mark)
                mark = now

                if pose_format is None:
                    pose_format, state_index = observations.resolve_state_layout(
                        snapshot,
                        sides,
                        cameras,
                        self._state_dim,
                        self._state_names,
                        self._append_log,
                        self._history_names(),
                    )
                observation = observations.build_observation(
                    snapshot, images, sides, cameras, pose_format
                )
                now = time.monotonic()
                stage_times["build_obs"].append(now - mark)
                mark = now

                # Replan early enough to retain a committed path during inference.
                force = replan_now or replan_steps is None or step % replan_steps == 0
                replan_now = False
                # Read before planning: a takeover during inference changes it,
                # and the plan then describes an arm that has moved since.
                epoch = waypoint_executor.epoch
                if cuda_graph_step_begin is not None:
                    cuda_graph_step_begin()
                actions, fresh = observations._predict_action_chunk(
                    observations.select_state(
                        observation, state_index, self._history_values()
                    ),
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

                if waypoint_executor.epoch != epoch:
                    # Taken over (or handed back) while planning: drop the plan
                    # before any check reads it as the policy's next command.
                    continue

                action_lists = self._actions_to_lists(actions)
                safety.check_finite(action_lists)
                commanded = [
                    [
                        pose_command(plan, action)[:3]
                        for plan in layout
                        if plan["pose"] is not None
                    ]
                    for action in action_lists
                ]
                safety.check_workspace(commanded)
                if fresh:
                    safety.check_reference(
                        commanded[0], _measured_arms(snapshot, "tcp_pose")
                    )
                    last_fresh = time.monotonic()
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
                        loop_start + (k + anchor) * dt for k in range(len(action_lists))
                    ]
                    if waypoint_executor.replace_waypoints(
                        action_lists, target_times, now=time.monotonic(), epoch=epoch
                    ):
                        self._ready = True
                if waypoint_executor.error is not None:
                    raise RuntimeError(waypoint_executor.error)
                stage_times["dispatch"].append(time.monotonic() - mark)

                self._append_metric(
                    {
                        "t": round(loop_start, 3),
                        "step": step,
                        "hz": round(actual_hz, 2),
                        "infer_ms": round(infer_seconds * 1000.0, 1),
                        "fresh": bool(fresh),
                    }
                )
                last_action = list(action_lists[0])
                checks = self._score_live(
                    loop_start, observation, images, snapshot, pose_format
                )
                if self._recorder is not None:
                    self._recorder.add_tick(
                        state=observation.get("observation.state", []),
                        action=action_lists[0],
                        images=images,
                        snapshot=snapshot,
                        step=step,
                        fresh=fresh,
                        infer_ms=infer_seconds * 1000.0,
                        pose_format=pose_format,
                        switches=(
                            takeover.drain_events() if takeover is not None else None
                        ),
                        checks=checks,
                    )
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
                        step,
                        snapshot,
                        action_lists[0],
                        layout,
                        sides,
                        images,
                        camera_names,
                        actual_hz,
                    )

                step += 1
                if max_steps and step >= max_steps:
                    self._stop_reason = "timeout"
                    break

                elapsed = time.monotonic() - loop_start
                if period - elapsed > 0:
                    self._stop_event.wait(period - elapsed)
        except SafetyStop as exc:
            # The operator may have the arm: the robots hold before the trial
            # reads as failed.
            if takeover is not None:
                takeover.close()
            # Not a crash: the envelope did its job. Recorded as the stop
            # reason, and still reported as an error so the run reads as failed.
            self._stop_reason = "safety"
            self.safety_stop = {"code": exc.code, "detail": exc.detail}
            self._error = str(exc)
            self._on_error(str(exc))
            warn("Rollout stopped by the safety envelope", str(exc))
        except Exception as exc:
            if takeover is not None:
                takeover.close()
            detail = describe_exception(exc)
            self._error = detail
            self._on_error(detail)
            warn("Rollout stopped", detail)
            error("Rollout planner thread crashed", describe_traceback(exc))
        finally:
            if takeover is not None:
                # First: the operator may have the arm. Disengage and stop the
                # teleop loop so the robots hold, whatever ended the trial
                # (a no-op if an error above already did).
                takeover.close()
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

    def _history_names(self) -> tuple[str, ...]:
        """The state columns computed from the pose history, if trained on any."""
        return self._history_spec.names if self._history_spec is not None else ()

    def _history_values(self) -> np.ndarray | None:
        history = self._pose_history
        return history.values() if history is not None else None

    def _operator_tick(
        self,
        *,
        snapshot: dict[str, Any],
        images: dict[str, np.ndarray],
        step: int,
        control: str,
        pose_format: str,
        last_action: list[float],
    ) -> None:
        """A tick under the operator: keep the envelope on, keep recording."""
        # Nothing is commanded to check, so the box is checked against where
        # the arm measurably is.
        self._safety.check_measured_workspace(_measured_arms(snapshot, "tcp_pose"))
        takeover = self._takeover
        switches = takeover.drain_events() if takeover is not None else []
        if self._recorder is None and self._live_checks is None:
            return
        observation = observations.build_observation(
            snapshot, images, self._sides, self._camera_names, pose_format
        )
        # The operator can reach the state too: that ends the skill as well.
        checks = self._score_live(
            time.monotonic(), observation, images, snapshot, pose_format
        )
        if self._recorder is None:
            return
        # Mid-switch too: what the controller commands is what the arm does.
        action = operator_action(
            self._robots,
            self._sides,
            self._action_names,
            takeover.gripper_targets() if takeover is not None else {},
            last_action,
        )
        self._recorder.add_tick(
            state=observation.get("observation.state", []),
            action=action,
            images=images,
            snapshot=snapshot,
            step=step,
            fresh=False,
            infer_ms=0.0,
            pose_format=pose_format,
            control=control,
            switches=switches,
            checks=checks,
        )

    def _score_live(
        self,
        t: float,
        observation: dict[str, Any],
        images: dict[str, np.ndarray],
        snapshot: dict[str, Any],
        pose_format: str,
    ) -> dict[str, float] | None:
        """Hand this tick to the live checks; their newest probabilities."""
        if self._live_checks is None:
            return None
        self._live_checks.submit(
            t=t,
            state=observation.get("observation.state", []),
            images=images,
            snapshot=snapshot,
            pose_format=pose_format,
        )
        return self._live_checks.values()

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
                cam_parts.append(
                    f"{name}=ok(mean={float(np.asarray(image).mean()):.1f})"
                )
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
            commanded = pose_command(plan, action) if pose_slice is not None else []
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
                label = (
                    "target_width"
                    if plan.get("gripper_target_mode") == "target_width"
                    else "width"
                )
                gripper_detail = (
                    f" cmd_gripper_{label}="
                    f"{float(action[gripper_width_index]):.4f}"
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
