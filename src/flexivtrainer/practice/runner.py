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

"""Roll a step skill out: the policy acts once a step, through a `StepEnv`.

The rollout service runs this in place of the waypoint runner when the
checkpoint carries a practice space (`practice_space.json`). It offers the same
things around the policy: the safety envelope, the leader-arm takeover, learned
checks scored on every tick (one may end the rollout), and a recording.

THE RECORDING IS A TELEOP RECORDING. Each tick stores the state and frames in
the teleoperation layout, and as its action the pose the controller was given:
the step's target while the policy drives, the leader's while an operator does
(`rollout/takeover.py`). So a recorded step rollout is reviewed, labelled and
scored like any other. Its step-space actions are not in it: a skill's own
steps belong to practice, which keeps them as transitions.

A skill drives one arm, with the jaws as they are: it does not command the
gripper.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from flexivtrainer.data.lerobot_io import POSE_FORMAT_ROTATION_6D, ROTATION_6D_AXES
from flexivtrainer.observability import (
    describe_exception,
    describe_traceback,
    error,
    warn,
)
from flexivtrainer.practice.env import POLICY, StepEnv, Tick
from flexivtrainer.practice.space import PracticeSpace
from flexivtrainer.rollout import hardware, observations
from flexivtrainer.rollout.live_checks import LiveChecks
from flexivtrainer.rollout.recorder import RolloutRecorder
from flexivtrainer.rollout.safety import SafetyEnvelope, SafetyStop
from flexivtrainer.rollout.takeover import Takeover, operator_action

__all__ = ["StepRunner", "step_action_names"]


def step_action_names(side: str) -> list[str]:
    """The recorded action of a step rollout: the commanded pose and twist."""
    return [
        *(f"{side}.tcp_pose.{a}" for a in ("x", "y", "z")),
        *(f"{side}.tcp_rotation_6d.{a}" for a in ROTATION_6D_AXES),
        *(f"{side}.tcp_twist.{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")),
    ]


def _predict(
    observation: dict[str, Any],
    policy: Any,
    device: str,
    preprocessor: Any,
    postprocessor: Any,
    task: str | None,
) -> np.ndarray:
    """One action from the policy, through its own processors."""
    import torch  # noqa: PLC0415

    try:
        from lerobot.common.control_utils import predict_action  # noqa: PLC0415
    except ImportError:
        from lerobot.utils.control_utils import predict_action  # noqa: PLC0415

    action = predict_action(
        observation,
        policy,
        torch.device(device),
        preprocessor,
        postprocessor,
        use_amp=False,
        task=task,
    )
    return np.asarray(action.detach().cpu().numpy(), dtype=np.float64).reshape(-1)


class StepRunner:
    """Own the thread that rolls a step skill out on one arm."""

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
        camera_names: list[str],
        space: PracticeSpace,
        device: str,
        task: str | None,
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
        image_resolutions: dict[str, tuple[int, int]] | None = None,
        append_wrench: Callable[[dict[str, Any]], None] | None = None,
        safety: SafetyEnvelope | None = None,
        recorder: RolloutRecorder | None = None,
        cartesian_stiffness: Any = None,
        max_contact_wrench: Any = None,
        zero_ft_max_force_n: float | None = hardware.FT_ZERO_MAX_FORCE_N,
        force_axis_max_speed_m_s: float = hardware.FORCE_AXIS_MAX_SPEED_M_S,
        takeover: Takeover | None = None,
        live_checks: LiveChecks | None = None,
    ) -> None:
        if len(robots) != 1 or len(sides) != 1 or len(followers) != 1:
            raise ValueError(
                f"a step skill drives one arm; {len(robots)} are configured "
                "(set single-arm mode)"
            )
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robots[0]
        self._robots = robots
        self._side = sides[0]
        self._sides = list(sides)
        self._follower = followers[0]
        self._cameras = cameras
        self._camera_names = list(camera_names)
        self._space = space
        self._device = device
        self._task = task
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
        self._image_resolutions = image_resolutions
        self._safety = safety or SafetyEnvelope()
        self._recorder = recorder
        self._cartesian_stiffness = cartesian_stiffness
        self._max_contact_wrench = max_contact_wrench
        self._force_speed = force_axis_max_speed_m_s
        self._takeover = takeover
        self._live_checks = live_checks or None
        self._action_names = step_action_names(self._side)
        #: Set when the envelope stopped the rollout: {"code", "detail"}.
        self.safety_stop: dict[str, str] | None = None
        self._error: str | None = None
        self._stop_reason: str | None = None
        self._thread: threading.Thread | None = None
        self._env: StepEnv | None = None
        # Set once the policy has sent its first command; a takeover needs one.
        self._ready = False
        # Set by a hand-back; the policy is reset before it acts again.
        self._reset_pending = False
        self._infer_ms = 0.0
        self._last_action: list[float] = []
        # Whether an axis is under force control now, so it is cleared once:
        # after a takeover the arm may be in a mode that refuses the call.
        self._force_set = False
        try:
            self._prepare(zero_ft=True, zero_ft_max_force_n=zero_ft_max_force_n)
        except Exception as exc:
            stop_robots(robots)
            raise RuntimeError(
                "Failed to prepare the step rollout hardware: "
                f"{describe_exception(exc)}"
            ) from exc

    def _prepare(self, *, zero_ft: bool, zero_ft_max_force_n: float | None = None):
        hardware.prepare_robot_motion(
            self._robot,
            self._follower,
            self._stop_event,
            self._append_log,
            stiffness=self._cartesian_stiffness,
            max_contact_wrench=self._max_contact_wrench,
            zero_ft=zero_ft,
            zero_ft_max_force_n=zero_ft_max_force_n,
            force_axes=self._space.force_axes,
            force_axis_max_speed_m_s=self._force_speed,
        )
        self._force_set = any(self._space.force_axes)

    def _clear_force(self) -> None:
        if self._force_set:
            hardware.clear_force_axes(self._robot)
            self._force_set = False

    # --- the service's side -------------------------------------------------

    def start(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True, name="rollout-steps")
        self._thread = thread
        thread.start()

    def stop(self, timeout: float = 2.0) -> list[str]:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                return ["Step rollout did not stop cleanly"]
        self._thread = None
        return []

    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {"error": self._error, "stop_reason": self._stop_reason}

    # --- leader-arm takeover (called from the takeover's switch thread) ------

    @property
    def ready(self) -> bool:
        return self._ready

    def pause_policy(self) -> None:
        """No policy command reaches the arm after this returns, and every axis
        is motion-controlled again for whoever drives it next."""
        env = self._env
        if env is not None:
            env.pause()
        self._clear_force()

    def reclaim(self) -> None:
        """Take the follower back into the skill's control mode, without
        zeroing the force sensor: the tool may be in contact."""
        self._prepare(zero_ft=False)

    def resume_policy(self) -> None:
        self._reset_pending = True
        env = self._env
        if env is not None:
            env.resume()

    def _control(self) -> str:
        return self._takeover.control() if self._takeover is not None else POLICY

    # --- the loop ----------------------------------------------------------------

    def _run(self) -> None:
        policy = self._policy
        env: StepEnv | None = None
        try:
            policy.reset()
            env = StepEnv(
                robot=self._robot,
                cameras=self._cameras,
                camera_names=self._camera_names,
                space=self._space,
                motion_limits=self._motion_limits,
                stop_event=self._stop_event,
                safety=self._safety,
                image_resolutions=self._image_resolutions,
                control=self._control,
                on_tick=self._on_tick,
            )
            self._env = env
            observation = env.reset()
            log_every = max(1, round(self._space.rate_hz))
            previous: float | None = None
            while not self._stop_event.is_set():
                ended = self._live_checks.ended() if self._live_checks else None
                if ended is not None:
                    self._stop_reason = f"check {ended} passed"
                    break
                started = time.monotonic()
                if self._control() == POLICY:
                    if self._reset_pending:
                        # Handed back: act afresh from where the operator left
                        # the arm.
                        self._reset_pending = False
                        policy.reset()
                    epoch = env.epoch
                    action = _predict(
                        observation,
                        policy,
                        self._device,
                        self._preprocessor,
                        self._postprocessor,
                        self._task,
                    )
                    self._infer_ms = (time.monotonic() - started) * 1000.0
                    result = env.step(action, epoch=epoch)
                    self._ready |= result.sent
                else:
                    self._infer_ms = 0.0
                    result = env.step(None)
                observation = result.observation
                hz = 1.0 / (started - previous) if previous is not None else 0.0
                previous = started
                self._append_metric(
                    {
                        "t": round(started, 3),
                        "step": env.steps,
                        "hz": round(hz, 2),
                        "infer_ms": round(self._infer_ms, 1),
                        "fresh": bool(result.sent),
                        "intervention": bool(result.intervention),
                    }
                )
                if env.steps % log_every == 0:
                    self._log_step(env.steps, result.action, result.intervention)
                if self._max_steps and env.steps >= self._max_steps:
                    self._stop_reason = "timeout"
                    break
        except SafetyStop as exc:
            if self._takeover is not None:
                self._takeover.close()
            self._stop_reason = "safety"
            self.safety_stop = {"code": exc.code, "detail": exc.detail}
            self._error = str(exc)
            self._on_error(str(exc))
            warn("Step rollout stopped by the safety envelope", str(exc))
        except Exception as exc:
            if self._takeover is not None:
                self._takeover.close()
            detail = describe_exception(exc)
            self._error = detail
            self._on_error(detail)
            warn("Step rollout stopped", detail)
            error("Step rollout thread crashed", describe_traceback(exc))
        finally:
            if self._takeover is not None:
                self._takeover.close()
            self._stop_event.set()
            try:
                self._clear_force()
            except Exception as exc:
                self._stop_robots(self._robots)
                detail = describe_exception(exc)
                warn("Could not clear the force axes", detail)
                self._on_cleanup_error(detail)
            self._policy = self._preprocessor = self._postprocessor = None
            self._release_robots()
            self._on_finished(self._stop_reason, env.steps if env is not None else 0)

    # --- each tick --------------------------------------------------------------

    def _on_tick(self, tick: Tick) -> None:
        self._last_wrench_t = observations.sample_wrench(
            self._append_wrench, tick.snapshot, self._sides, tick.t, self._last_wrench_t
        )
        if self._recorder is None and self._live_checks is None:
            return
        state = observations.build_observation(
            tick.snapshot,
            tick.images,
            self._sides,
            self._camera_names,
            POSE_FORMAT_ROTATION_6D,
        ).get("observation.state", [])
        checks = None
        if self._live_checks is not None:
            self._live_checks.submit(
                t=tick.t,
                state=state,
                images=tick.images,
                snapshot=tick.snapshot,
                pose_format=POSE_FORMAT_ROTATION_6D,
            )
            checks = self._live_checks.values()
        if self._recorder is None:
            return
        takeover = self._takeover
        if tick.control == POLICY:
            command = tick.command
            if command is not None:
                pose = [*command.position, *command.rotation[:2].reshape(6)]
            else:  # nothing sent yet: the arm holds where it is
                pose = [*tick.sample.position, *tick.sample.rotation[:2].reshape(6)]
            action = [float(v) for v in pose] + [0.0] * 6
        else:
            action = operator_action(
                self._robots,
                self._sides,
                self._action_names,
                takeover.gripper_targets() if takeover is not None else {},
                self._last_action,
            )
        self._last_action = action
        self._recorder.add_tick(
            state=state,
            action=action,
            images=tick.images,
            snapshot=tick.snapshot,
            step=tick.step,
            fresh=tick.fresh,
            infer_ms=self._infer_ms,
            pose_format=POSE_FORMAT_ROTATION_6D,
            control=tick.control,
            switches=takeover.drain_events() if takeover is not None else None,
            checks=checks,
        )

    def _log_step(self, step: int, action: Any, intervention: bool) -> None:
        names = self._space.action_names
        values = [] if action is None else [float(v) for v in action]
        parts = " ".join(f"{n}={v:+.2f}" for n, v in zip(names, values, strict=False))
        who = "operator" if intervention else "policy"
        self._append_log("INFO", "ROLLOUT", f"step={step} {who}", parts or "n/a")
