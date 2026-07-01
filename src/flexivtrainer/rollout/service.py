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

"""Run a trained LeRobot policy on the follower robot(s).

Rollout drives the follower arm(s) directly through the RDK ``Robot`` API, not
the TDK teleop controller -- that controller only mirrors a physical leader and
has no command-injection API, and its LAN connection cannot coexist with a fresh
RDK one, so rollout requires teleoperation to be shut down first.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from flexivtrainer.config import AppSettings, TeleopRobotPair
from flexivtrainer.data.lerobot_io import (
    build_features_from_sample,
    extract_recording_frame_values,
    extract_recording_images,
    resolve_recording_image_names,
)
from flexivtrainer.jobs.train_policy import UI_LOG_PREFIX, _encode_ui_log
from flexivtrainer.observability import describe_exception, warn

# Scalars per metric: tcp_pose is [x,y,z,qw,qx,qy,qz]; tcp_twist (velocity) and
# tcp_wrench are each 6-axis.
_POSE_DIM = 7
_TWIST_DIM = 6
_WRENCH_DIM = 6


def _default_policy_loader(checkpoint_path: str, device: str) -> Any:
    """Load a LeRobot policy and its processors from a checkpoint directory.

    Returns ``(policy, preprocessor, postprocessor)``. Since LeRobot 0.5,
    normalization lives in the processors saved with the checkpoint, not in the
    policy: the preprocessor normalizes the observation and the postprocessor
    un-normalizes the action back into physical units. ``from_pretrained`` wants
    the directory holding ``config.json``, so accept a weights file or a
    ``checkpoints/<step>`` parent and resolve down to it.
    """
    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.factory import (  # noqa: PLC0415
        get_policy_class,
        make_pre_post_processors,
    )

    path = Path(checkpoint_path)
    model_dir = path.parent if path.is_file() else path
    if not (model_dir / "config.json").exists():
        nested = model_dir / "pretrained_model"
        if (nested / "config.json").exists():
            model_dir = nested
    config = PreTrainedConfig.from_pretrained(model_dir)
    policy = get_policy_class(config.type).from_pretrained(model_dir)
    policy.to(device)
    policy.eval()
    # The processors bake in the training device (e.g. cuda); point them at the
    # rollout device so loading on a cpu-only host does not fail.
    device_override = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(model_dir),
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
    )
    return policy, preprocessor, postprocessor


def _default_robot_factory(serial: str) -> Any:
    import flexivrdk  # noqa: PLC0415

    return flexivrdk.Robot(serial)


def _rdk_mode() -> Any:
    import flexivrdk  # noqa: PLC0415

    return flexivrdk.Mode


def _predict_action_chunk(
    observation: dict[str, Any],
    policy: Any,
    device: str,
    preprocessor: Any,
    postprocessor: Any,
) -> tuple[Any, bool]:
    """Run one inference cycle; return ``(chunk, fresh)``.

    Execute-then-refill: ``select_action`` re-runs the U-Net only when its action
    queue has drained (every ``n_action_steps`` calls), otherwise it just pops the
    next cached step. ``fresh`` (queue empty before the call) tells the caller a
    genuinely new chunk exists, so it schedules a new trajectory only then instead
    of re-anchoring the interpolator on the pop-only steps (which stomps the
    trajectory in flight). On a fresh cycle we reuse this call for step 0 and
    un-normalize the cached tail so the whole chunk comes from one diffusion
    sample; the tail sits in the queue before the un-normalizer runs, hence the
    extra ``postprocessor`` call.
    """
    import torch  # noqa: PLC0415
    from lerobot.utils.constants import ACTION  # noqa: PLC0415
    from lerobot.utils.control_utils import predict_action  # noqa: PLC0415

    torch_device = torch.device(device)
    fresh = len(policy._queues.get(ACTION, [])) == 0
    first = predict_action(
        observation, policy, torch_device, preprocessor, postprocessor, use_amp=False
    )
    tail = list(policy._queues.get(ACTION, []))
    if not tail:
        return first.reshape(1, -1), fresh
    with torch.inference_mode():
        tail = postprocessor(torch.cat([t.to(torch_device) for t in tail], dim=0))
    chunk = torch.cat([first.reshape(1, -1), tail.reshape(len(tail), -1)], dim=0)
    return chunk, fresh


def _cuda_sync(device: str) -> None:
    """Block until queued CUDA work finishes so stage timings are attributed
    to the stage that issued the work, not to a later forced sync.

    No-op off cuda or when torch is unavailable (e.g. the cpu/fake-policy tests).
    """
    if not str(device).startswith("cuda"):
        return
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover - torch optional
        pass


class _SenderController:
    """Sender side of the rollout producer/consumer split (see ``RolloutService``).

    Owns the ``rollout-sender`` thread. The planner
    (``RolloutService._planner_loop``) schedules sparse timestamped waypoints via
    ``schedule_chunk``; this thread
    ticks at ``sender_hz`` and sends the interpolated pose, so the robot tracks a
    smooth spline instead of raw waypoints (which is what made motion jerky). The
    per-arm interpolators are the shared state -- both sides touch them under
    ``self._lock``; shutdown is the shared ``stop_event``; a sender fault surfaces
    on ``self.error`` for the producer to poll.
    """

    def __init__(
        self,
        robots: list[Any],
        layout: list[dict[str, Any]],
        sender_hz: int,
        stop_event: threading.Event,
        motion_limits: tuple[float, float, float, float],
    ) -> None:
        from flexivtrainer.rollout.pose_interpolator import (  # noqa: PLC0415
            PoseTrajectoryInterpolator,
        )

        self._robots = robots
        self._layout = layout
        self._period = 1.0 / float(sender_hz)
        self._stop_event = stop_event
        # (max_linear_vel, max_angular_vel, max_linear_acc, max_angular_acc) —
        # hardware ceilings on the motion generator, our speed safety backstop.
        self._motion_limits = motion_limits
        self._lock = threading.Lock()
        # Per-arm interpolator, created lazily on the first scheduled pose.
        self._interp: list[Any | None] = [None] * len(layout)
        # Per-arm time of the last scheduled waypoint; lets the next one splice
        # onto the trajectory end instead of overwriting it.
        self._last_wp_time: list[float] = [0.0] * len(layout)
        self._wrench: list[list[float]] = [[0.0] * _WRENCH_DIM for _ in layout]
        self._make_interp = PoseTrajectoryInterpolator
        self._error: str | None = None
        self._ticks = 0
        # Chunk steps that survived the future-filter last call; a shrinking
        # count means inference is eating into the streamed horizon.
        self._last_scheduled = 0
        self._thread: threading.Thread | None = None

    def schedule_chunk(
        self,
        actions: list[list[float]],
        target_times: list[float],
        now: float,
    ) -> None:
        """Splice a whole action chunk into each arm's interpolator as waypoints.

        ``actions[k]`` is the flat action for step k, reached at ``target_times[k]``;
        steps already in the past (``<= now``) are dropped. Velocity comes from the
        spline, so the twist slice is not read here.
        """
        with self._lock:
            for index, arm_plan in enumerate(self._layout):
                if index >= len(self._robots):
                    break
                pose_slice = arm_plan["pose"]
                if pose_slice is None:
                    continue
                future = [
                    (t, RolloutService._normalize_pose_quaternion(list(a[pose_slice])))
                    for a, t in zip(actions, target_times)
                    if t > now
                ]
                self._last_scheduled = len(future)
                if not future:
                    continue
                wrench_slice = arm_plan["wrench"]
                self._wrench[index] = (
                    list(actions[0][wrench_slice])
                    if wrench_slice is not None
                    else [0.0] * _WRENCH_DIM
                )
                interp = self._interp[index]
                start = 0
                if interp is None:
                    # Seed from the measured pose so the first tick eases from
                    # where the arm actually is, rather than jumping.
                    measured = RolloutService._normalize_pose_quaternion(
                        [float(v) for v in self._robots[index].states().tcp_pose]
                    )
                    t0, pose0 = future[0]
                    interp = self._make_interp(times=[now, t0], poses=[measured, pose0])
                    self._last_wp_time[index] = t0
                    start = 1
                for target_time, pose in future[start:]:
                    interp = interp.schedule_waypoint(
                        pose=pose,
                        time=target_time,
                        curr_time=now,
                        last_waypoint_time=self._last_wp_time[index],
                    )
                    self._last_wp_time[index] = target_time
                self._interp[index] = interp

    def _send_interpolated_action(self) -> None:
        now = time.monotonic()
        with self._lock:
            interps = list(self._interp)
            wrenches = [list(w) for w in self._wrench]
        max_lin_vel, max_ang_vel, max_lin_acc, max_ang_acc = self._motion_limits
        for index, interp in enumerate(interps):
            if interp is None or index >= len(self._robots):
                continue
            pose = interp(now)
            velocity = interp.velocity(now)
            self._robots[index].SendCartesianMotionForce(
                [float(v) for v in pose],
                wrenches[index],
                [float(v) for v in velocity],
                max_lin_vel,
                max_ang_vel,
                max_lin_acc,
                max_ang_acc,
            )

    def _sender_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                start = time.monotonic()
                # Skip until at least one arm has a scheduled trajectory.
                if any(i is not None for i in self._interp):
                    self._send_interpolated_action()
                    self._ticks += 1
                rest = self._period - (time.monotonic() - start)
                if rest > 0:
                    self._stop_event.wait(rest)
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
            self._stop_event.set()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._sender_loop, daemon=True, name="rollout-sender"
        )
        self._thread.start()

    def join(self, timeout: float = 2.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def ticks(self) -> int:
        return self._ticks


class RolloutService:
    """Lifecycle + background control loop for policy rollout.

    Two threads run during a rollout, in a planner/sender (producer/consumer) split:

    - **Planner** -- ``_planner_loop`` (thread ``rollout-planner``): reads cameras
      and robot state, runs policy inference, and schedules the resulting action
      chunk as timestamped waypoints. Ticks at ``planner_hz``.
    - **Sender** -- ``_SenderController._sender_loop`` (thread ``rollout-sender``):
      samples the interpolated pose spline and sends it to the robot via
      ``SendCartesianMotionForce``. Ticks at the higher ``sender_hz``.

    Handshake between them:

    - **Trajectory** -- the planner calls ``controller.schedule_chunk`` and the
      sender reads ``controller(now)``; both touch the controller's per-arm
      interpolators under ``_SenderController._lock``. This is the only shared
      mutable state.
    - **Shutdown** -- ``self._stop_event`` (owned here, passed into the controller)
      signals both threads to exit. ``stop()`` sets it, then joins the sender
      *before* ``_release_robots()`` so the sender can never command a released
      robot.
    - **Errors** -- a fault in the sender is recorded on ``controller.error`` and
      polled by the planner, which raises and tears the run down.
    """

    def __init__(
        self,
        settings: AppSettings,
        cameras: Any,
        teleop: Any,
        get_robot_pairs: Callable[[], list[TeleopRobotPair]],
        get_active_sides: Callable[[], list[str]],
        *,
        policy_loader: Callable[[str, str], Any] = _default_policy_loader,
        robot_factory: Callable[[str], Any] = _default_robot_factory,
        resolve_device: Callable[[str], str] | None = None,
    ) -> None:
        self._settings = settings
        self._cameras = cameras
        self._teleop = teleop
        self._get_robot_pairs = get_robot_pairs
        self._get_active_sides = get_active_sides
        self._policy_loader = policy_loader
        self._robot_factory = robot_factory
        if resolve_device is None:
            from flexivtrainer.jobs.train_policy import resolve_training_device

            resolve_device = resolve_training_device
        self._resolve_device = resolve_device

        self._lock = threading.Lock()
        self._running = False
        self._error: str | None = None
        # Why the last run ended: None while running/never-run, "stopped" for an
        # operator stop, or "timeout" when the max_steps budget was reached.
        self._stop_reason: str | None = None
        self._checkpoint_path: str | None = None
        self._robots: list[Any] = []
        self._device = "cpu"
        self._thread: threading.Thread | None = None
        # Held so stop() can join the sender before releasing the robots.
        self._controller: _SenderController | None = None
        self._stop_event = threading.Event()
        # Ring buffer of UI-encoded log lines (same wire format as the training
        # terminal), surfaced through status() so the rollout tab can stream the
        # per-step measured state and commanded action like the training tab.
        self._logs: deque[str] = deque(maxlen=2000)

    # -- status -------------------------------------------------------------

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
                "error": self._error,
                "stop_reason": self._stop_reason,
                "logs": list(self._logs),
                "log_lines": len(self._logs),
            }

    def _append_log(self, level: str, source: str, message: str, detail: str = "") -> None:
        self._logs.append(_encode_ui_log(level, source, message, detail))

    # -- lifecycle ----------------------------------------------------------

    def start(self, checkpoint_path: str) -> dict[str, Any]:
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

        if not Path(checkpoint_path).exists():
            raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

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
        self._apply_diffusion_scheduler_override(policy)
        robots: list[Any] = []
        try:
            for serial in followers:
                robots.append(self._connect_robot(serial))
        except Exception as exc:
            self._stop_robots(robots)
            raise RuntimeError(
                f"Failed to connect to robot: {describe_exception(exc)}"
            ) from exc

        self._stop_event.clear()
        with self._lock:
            self._checkpoint_path = checkpoint_path
            self._error = None
            self._stop_reason = None
            self._robots = robots
            self._device = device
            self._running = True
            self._logs.clear()
            self._logs.append(
                _encode_ui_log(
                    "INFO",
                    "ROLLOUT",
                    "Rollout started",
                    f"device={device} sides={'+'.join(sides)}",
                )
            )
        thread = threading.Thread(
            target=self._planner_loop,
            args=(policy, preprocessor, postprocessor, robots, sides),
            daemon=True,
            name="rollout-planner",
        )
        self._thread = thread
        thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        # Join the sender before releasing robots so it can't command them after.
        controller = self._controller
        if controller is not None:
            controller.join()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        self._release_robots()
        with self._lock:
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

    # -- internals ----------------------------------------------------------

    def _teleop_initialized(self) -> bool:
        snapshot = self._teleop.snapshot()
        return bool(getattr(snapshot, "initialized", False))

    def _apply_diffusion_scheduler_override(self, policy: Any) -> None:
        """Swap a diffusion policy's denoising sampler per the rollout config.

        Rebuilds ``policy.diffusion.noise_scheduler`` (e.g. DDPM -> DDIM) reusing
        the checkpoint's own schedule kwargs -- only the sampler changes, the
        trained weights do not -- and sets ``num_inference_steps``. DDIM reaches
        the target in far fewer steps, shrinking the per-chunk refill that stalls
        the control loop. Best-effort and no-op for non-diffusion policies or
        when ``scheduler`` is "".
        """
        scheduler = self._settings.rollout.diffusion.scheduler
        if not scheduler:
            return
        diffusion = getattr(policy, "diffusion", None)
        existing = getattr(diffusion, "noise_scheduler", None)
        if diffusion is None or existing is None:
            return  # not a diffusion policy; nothing to override
        steps = self._settings.rollout.diffusion.inference_steps
        try:
            from lerobot.policies.diffusion.modeling_diffusion import (  # noqa: PLC0415
                _make_noise_scheduler,
            )

            # Reuse the trained schedule (beta range, clip, prediction type) so
            # only the sampler family changes. Pass just these kwargs -- the same
            # set LeRobot builds the scheduler with -- since a scheduler's full
            # config carries family-specific keys the other family rejects.
            cfg = existing.config
            kwargs = dict(
                num_train_timesteps=cfg.num_train_timesteps,
                beta_start=cfg.beta_start,
                beta_end=cfg.beta_end,
                beta_schedule=cfg.beta_schedule,
                clip_sample=cfg.clip_sample,
                clip_sample_range=cfg.clip_sample_range,
                prediction_type=cfg.prediction_type,
            )
            diffusion.noise_scheduler = _make_noise_scheduler(scheduler, **kwargs)
            diffusion.num_inference_steps = steps
        except Exception as exc:
            warn("Failed to override diffusion scheduler", describe_exception(exc))
            return
        with self._lock:
            self._logs.append(
                _encode_ui_log(
                    "INFO",
                    "ROLLOUT",
                    "Diffusion scheduler overridden",
                    f"scheduler={scheduler} inference_steps={steps}",
                )
            )

    def _connect_robot(self, serial: str) -> Any:
        robot = self._robot_factory(serial)
        if robot.fault():
            robot.ClearFault()
        robot.Enable()
        while not robot.operational():
            if self._stop_event.wait(0.1):
                break
        mode = _rdk_mode()
        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        return robot

    def _release_robots(self) -> None:
        with self._lock:
            robots, self._robots = self._robots, []
        self._stop_robots(robots)

    @staticmethod
    def _stop_robots(robots: list[Any]) -> None:
        for robot in robots:
            try:
                stop = getattr(robot, "Stop", None)
                if callable(stop):
                    stop()
            except Exception:  # pragma: no cover - hardware specific
                pass

    def _read_robot_snapshot(
        self, robots: list[Any], sides: list[str]
    ) -> dict[str, Any]:
        """Build a robot_data_snapshot-shaped dict from direct RDK reads.

        Mirrors ``TeleopService.robot_data_snapshot`` so the lerobot_io helpers
        consume it unchanged: the follower's ``states()`` provides tcp_pose /
        tcp_vel / ext_wrench_in_world for the observation.
        """
        robots_payload: dict[str, Any] = {}
        for index, robot in enumerate(robots):
            states = robot.states()
            tcp_pose = [float(v) for v in states.tcp_pose]
            tcp_vel = [float(v) for v in states.tcp_vel]
            wrench = [float(v) for v in states.ext_wrench_in_world]
            robots_payload[f"robot_{index}"] = {
                "connected": True,
                "states": {
                    "tcp_pose": tcp_pose,
                    "tcp_vel": tcp_vel,
                    "ext_wrench_in_world": wrench,
                },
                # Placeholder: ``build_features_from_sample`` reads only the axis
                # names/dimensions here to slice the policy output, not the values.
                "actions": {
                    "tcp_pose_d": tcp_pose,
                    "tcp_vel_d": tcp_vel,
                    "ext_wrench_d": wrench,
                },
            }
        return {"robots": robots_payload, "errors": {}}

    def _plan_action_layout(
        self, action_names: list[str], sides: list[str]
    ) -> list[dict[str, Any]]:
        """Map the flat action vector to per-side pose/wrench command slices.

        ``action_names`` are the action feature's axis names, e.g.
        ``left_arm.tcp_pose.x`` ... ``right_arm.tcp_wrench.mz``. We locate each
        side's ``tcp_pose``, ``tcp_twist`` and ``tcp_wrench`` runs by name so the
        slicing tracks exactly what the recorder produced for this checkpoint.
        """
        layout: list[dict[str, Any]] = []
        for side in sides:
            pose_start = self._find_run(action_names, f"{side}.tcp_pose.")
            twist_start = self._find_run(action_names, f"{side}.tcp_twist.")
            wrench_start = self._find_run(action_names, f"{side}.tcp_wrench.")
            pose = (
                None
                if pose_start is None
                else slice(pose_start, pose_start + _POSE_DIM)
            )
            twist = (
                None
                if twist_start is None
                else slice(twist_start, twist_start + _TWIST_DIM)
            )
            wrench = (
                None
                if wrench_start is None
                else slice(wrench_start, wrench_start + _WRENCH_DIM)
            )
            layout.append(
                {"side": side, "pose": pose, "twist": twist, "wrench": wrench}
            )
        return layout

    @staticmethod
    def _find_run(names: list[str], prefix: str) -> int | None:
        for index, name in enumerate(names):
            if name.startswith(prefix):
                return index
        return None

    def _loop_period(self) -> float:
        return 1.0 / float(self._settings.rollout.planner_hz)

    def _planner_loop(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        robots: list[Any],
        sides: list[str],
    ) -> None:
        policy.reset()
        period = self._loop_period()
        # Chunk pose spacing: the dataset's recording period, not the loop period.
        dt = 1.0 / float(self._settings.rollout.action_dt_hz)
        # 0 means "no cap": run until the operator stops it or a fault occurs.
        max_steps = self._settings.rollout.max_steps
        camera_names = resolve_recording_image_names(None, sides)
        layout: list[dict[str, Any]] | None = None
        log_every = max(1, int(self._settings.rollout.planner_hz // 2))
        # Recent per-step work times (sleep excluded) for a smoothed actual Hz,
        # plus the same window per pipeline stage for the timing breakdown.
        work_times: deque[float] = deque(maxlen=10)
        stage_times: dict[str, deque[float]] = {
            name: deque(maxlen=10)
            for name in (
                "fault_check", "grab_images", "read_states",
                "build_obs", "inference", "to_list", "dispatch",
            )
        }
        # Raw (un-smoothed) per-step inference times for one logging interval, to
        # expose the 1-in-n_action_steps refill spike that the smoothed window hides.
        infer_raw: deque[float] = deque(maxlen=log_every)
        sender_hz = self._settings.rollout.sender_hz
        controller: _SenderController | None = None
        step = 0
        try:
            while not self._stop_event.is_set():
                loop_start = time.monotonic()
                # ``mark`` advances after each stage; ``now - mark`` is that
                # stage's duration, recorded into its rolling window.
                mark = loop_start

                for robot in robots:
                    if robot.fault():
                        raise RuntimeError("Fault occurred on a follower robot")
                now = time.monotonic()
                stage_times["fault_check"].append(now - mark)
                mark = now

                images = self._grab_images(camera_names)
                now = time.monotonic()
                stage_times["grab_images"].append(now - mark)
                mark = now

                snapshot = self._read_robot_snapshot(robots, sides)
                now = time.monotonic()
                stage_times["read_states"].append(now - mark)
                mark = now

                observation = self._build_observation(snapshot, images, sides)
                now = time.monotonic()
                stage_times["build_obs"].append(now - mark)
                mark = now

                if layout is None:
                    features, _, _ = build_features_from_sample(
                        snapshot, images, None, sides
                    )
                    action_feature = features.get("action")
                    action_names = action_feature["names"] if action_feature else []
                    layout = self._plan_action_layout(action_names, sides)
                    rollout_cfg = self._settings.rollout
                    controller = _SenderController(
                        robots,
                        layout,
                        sender_hz,
                        self._stop_event,
                        (
                            rollout_cfg.max_linear_vel,
                            rollout_cfg.max_angular_vel,
                            rollout_cfg.max_linear_acc,
                            rollout_cfg.max_angular_acc,
                        ),
                    )
                    self._controller = controller
                    controller.start()

                actions, fresh = _predict_action_chunk(
                    observation, policy, self._device, preprocessor, postprocessor
                )
                # Sync so async cuda inference is timed here, not at to_list's
                # device->host copy.
                _cuda_sync(self._device)
                now = time.monotonic()
                stage_times["inference"].append(now - mark)
                infer_raw.append(now - mark)
                mark = now

                action_lists = self._actions_to_lists(actions)
                now = time.monotonic()
                stage_times["to_list"].append(now - mark)
                mark = now

                # Execute-then-refill: only a fresh inference yields a new chunk; on
                # pop-only steps the sender keeps riding the trajectory already
                # scheduled. Space the poses at the dataset control period and
                # anchor at the observation time so the chunk plays out at real-time
                # speed (with planner_hz == action_dt_hz the chunk abuts the refill).
                assert controller is not None  # created above with the layout
                if fresh:
                    target_times = [
                        loop_start + k * dt for k in range(len(action_lists))
                    ]
                    controller.schedule_chunk(
                        action_lists, target_times, now=time.monotonic()
                    )
                if controller.error is not None:
                    raise RuntimeError(controller.error)
                stage_times["dispatch"].append(time.monotonic() - mark)

                work_times.append(time.monotonic() - loop_start)
                if step % log_every == 0:
                    mean_work = sum(work_times) / len(work_times)
                    actual_hz = 1.0 / mean_work if mean_work > 0 else 0.0
                    self._log_timing(
                        step, stage_times, infer_raw, controller._last_scheduled
                    )
                    self._log_step(
                        step, snapshot, action_lists[0], layout, sides,
                        images, camera_names, actual_hz,
                    )

                step += 1
                if max_steps and step >= max_steps:
                    with self._lock:
                        self._stop_reason = "timeout"
                    break

                elapsed = time.monotonic() - loop_start
                if period - elapsed > 0:
                    self._stop_event.wait(period - elapsed)
        except Exception as exc:
            detail = describe_exception(exc)
            with self._lock:
                self._error = detail
                self._running = False
                self._logs.append(_encode_ui_log("ERROR", "ROLLOUT", "Rollout stopped", detail))
            warn("Rollout stopped", detail)
        finally:
            # Stop the high-rate sender before releasing robots so it cannot
            # command an already-released robot. The break paths above may exit
            # the loop without setting the stop event, so set it here.
            if controller is not None:
                self._stop_event.set()
                controller.join()
            self._controller = None
            self._release_robots()
            with self._lock:
                self._running = False
                reason = self._stop_reason or "stopped"
                if self._error is None:
                    self._logs.append(
                        _encode_ui_log("INFO", "ROLLOUT", "Rollout ended", f"reason={reason} steps={step}")
                    )

    def _grab_images(self, camera_names: list[str]) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for name in camera_names:
            frame = self._cameras.capture_frame(name, block=False, allow_cached=True)
            image = frame.get("image") if isinstance(frame, dict) else None
            if image is None:
                continue
            # Cameras capture BGR; LeRobot policies were trained on RGB frames.
            images[name] = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        return images

    def _build_observation(
        self, snapshot: dict[str, Any], images: dict[str, np.ndarray], sides: list[str]
    ) -> dict[str, Any]:
        observation: dict[str, Any] = {}
        selected = extract_recording_images(images, None, sides)
        for name, image in selected.items():
            observation[f"observation.images.{name}"] = image
        frame_values = extract_recording_frame_values(snapshot, None, sides)
        for key, vector in frame_values.items():
            if key.startswith("observation"):
                observation[key] = np.asarray(vector, dtype=np.float32)
        return observation

    @staticmethod
    def _actions_to_lists(actions: Any) -> list[list[float]]:
        """Convert an action chunk to a list of per-step float vectors.

        Accepts a [n_steps, dim], [1, n_steps, dim] or [dim] tensor/ndarray/
        sequence; a bare single action becomes a one-element list.
        """
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
        """Log the mean per-stage duration (ms) over the recent window.

        One line per ``log_every``, ordered as the stages run in the loop, with a
        ``total`` so the breakdown can be checked against the ``freq=`` line. Use
        it to find which stage (e.g. grab_images, inference) caps the loop rate.
        ``sched`` is how many chunk steps the sender last accepted -- if it falls
        toward 0, inference is outrunning the streamed horizon.
        """
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
        """Log measured vs commanded TCP pose per side, plus an observation row.

        The observation row reports, per expected camera, whether a frame was
        present and its mean pixel value -- a missing camera or a frozen/black
        feed (mean ~0 or unchanging) would starve the policy and is the prime
        suspect for an in-distribution start still diverging.
        """
        cam_parts: list[str] = []
        for name in camera_names:
            image = images.get(name)
            if image is None:
                cam_parts.append(f"{name}=MISSING")
            else:
                cam_parts.append(f"{name}=ok(mean={float(np.asarray(image).mean()):.1f})")
        expected_hz = float(self._settings.rollout.planner_hz)
        cam_parts.append(f"freq={actual_hz:.1f}/{expected_hz:.1f}Hz")
        self._append_log("INFO", "ROLLOUT", f"step={step} obs", " ".join(cam_parts))

        robots_payload = snapshot.get("robots") if isinstance(snapshot, dict) else None
        payloads = list(robots_payload.values()) if isinstance(robots_payload, dict) else []
        for index, plan in enumerate(layout):
            side = plan.get("side") or (sides[index] if index < len(sides) else f"arm_{index}")
            pose_slice = plan["pose"]
            commanded = (
                self._normalize_pose_quaternion(action[pose_slice])
                if pose_slice is not None
                else []
            )
            measured: list[float] = []
            if index < len(payloads) and isinstance(payloads[index], dict):
                states = payloads[index].get("states")
                if isinstance(states, dict):
                    measured = list(states.get("tcp_pose") or [])
            self._append_log(
                "INFO",
                "ROLLOUT",
                f"step={step} {side}",
                f"cmd_xyz={self._fmt_xyz(commanded)} meas_xyz={self._fmt_xyz(measured)}",
            )

    @staticmethod
    def _fmt_xyz(pose: list[float]) -> str:
        if len(pose) < 3:
            return "n/a"
        return "[" + ", ".join(f"{pose[i]:.3f}" for i in range(3)) + "]"

    @staticmethod
    def _normalize_pose_quaternion(pose: list[float]) -> list[float]:
        """Renormalize the orientation quaternion of a ``[x,y,z,qw,qx,qy,qz]`` pose.

        The policy outputs the quaternion as four independently-regressed scalars
        (ACTION uses per-element MIN_MAX normalization), so the result is not
        guaranteed to be unit length. ``SendCartesianMotionForce`` expects a unit
        quaternion, so rescale components 3:7 to unit norm before commanding. A
        near-zero norm is left untouched to avoid dividing by ~0.
        """
        pose = list(pose)
        if len(pose) < _POSE_DIM:
            return pose
        quat = pose[3:7]
        norm = sum(component * component for component in quat) ** 0.5
        if norm > 1e-6:
            pose[3:7] = [component / norm for component in quat]
        return pose
