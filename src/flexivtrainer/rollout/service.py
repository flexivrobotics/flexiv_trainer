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

import json
import os
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
    first_dataset_task,
    resolve_recording_image_names,
)
from flexivtrainer.jobs.train_policy import _encode_ui_log
from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.policies import diffusion as diffusion_policy
from flexivtrainer.policies import dit as dit_policy
from flexivtrainer.rollout.waypoint_executor import (
    WaypointExecutor,
    build_action_layout,
    normalize_pose_quaternion,
)

_ROLLOUT_OVERRIDES = {
    "diffusion": diffusion_policy.apply_rollout_overrides,
    "multi_task_dit": dit_policy.apply_rollout_overrides,
}

_LANGUAGE_POLICY_TYPES = {"multi_task_dit", "smolvla", "pi0", "pi05"}

_FORCE_REFRESH_WARNED = False


def _checkpoint_model_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    model_dir = path.parent if path.is_file() else path
    if not (model_dir / "config.json").exists():
        nested = model_dir / "pretrained_model"
        if (nested / "config.json").exists():
            model_dir = nested
    return model_dir


def _matching_child(parent: Path, name: str) -> Path | None:
    try:
        for child in parent.iterdir():
            if child.name == name:
                return child
    except OSError:
        return None
    return None


def resolve_checkpoint_path(checkpoint_path: str, storage_root: Path) -> Path:
    """Reject a client checkpoint path that escapes the storage root."""
    root = storage_root.expanduser().resolve()
    root_text = os.fspath(root)
    root_prefix = root_text if root_text.endswith(os.sep) else root_text + os.sep
    if checkpoint_path == root_text:
        return root
    if not checkpoint_path.startswith(root_prefix):
        raise ValueError(f"Access denied: path must be within storage root ({root})")

    parts = checkpoint_path[len(root_prefix) :].split(os.sep)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Access denied: path must be within storage root ({root})")

    resolved = root
    for part in parts:
        child = _matching_child(resolved, part)
        if child is None:
            raise FileNotFoundError("Checkpoint not found")
        resolved = child.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Access denied: path must be within storage root ({root})"
            )
    return resolved


def _default_policy_loader(checkpoint_path: str, device: str) -> Any:
    """Load a LeRobot policy and its processors from a checkpoint directory."""
    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.factory import (  # noqa: PLC0415
        get_policy_class,
        make_pre_post_processors,
    )

    model_dir = _checkpoint_model_dir(checkpoint_path)
    config = PreTrainedConfig.from_pretrained(model_dir)
    policy = get_policy_class(config.type).from_pretrained(model_dir)
    policy.to(device)
    policy.eval()
    # Override the training device for CPU-only rollout hosts.
    device_override = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(model_dir),
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
    )
    return policy, preprocessor, postprocessor


def _positive_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    value = float(value)
    return value if value > 0 else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _dataset_root_candidates(root: str, model_dir: Path) -> list[Path]:
    dataset_root = Path(root).expanduser()
    candidates = [dataset_root]
    if not dataset_root.is_absolute():
        candidates.append(Path.cwd() / dataset_root)
        candidates.extend(parent / dataset_root for parent in model_dir.parents)
    return list(
        dict.fromkeys(candidate.resolve(strict=False) for candidate in candidates)
    )


def _checkpoint_target_hz(checkpoint_path: str) -> float | None:
    """Read of the dataset FPS baked into a LeRobot checkpoint."""
    model_dir = _checkpoint_model_dir(checkpoint_path)

    train_config = _read_json(model_dir / "train_config.json") or {}
    dataset = train_config.get("dataset") if isinstance(train_config, dict) else None
    if isinstance(dataset, dict):
        if fps := _positive_float(dataset.get("fps")):
            return fps
        root = dataset.get("root")
        if isinstance(root, str) and root.strip():
            for candidate in _dataset_root_candidates(root, model_dir):
                info = _read_json(candidate / "meta" / "info.json") or {}
                if fps := _positive_float(info.get("fps")):
                    return fps

    config = _read_json(model_dir / "config.json") or {}
    for key in ("fps", "dataset_fps", "action_dt_hz"):
        if fps := _positive_float(config.get(key)):
            return fps
    return None


def _checkpoint_task(checkpoint_path: str) -> str | None:
    """Read of the task string of the dataset a checkpoint trained on."""
    model_dir = _checkpoint_model_dir(checkpoint_path)
    train_config = _read_json(model_dir / "train_config.json") or {}
    dataset = train_config.get("dataset") if isinstance(train_config, dict) else None
    if not isinstance(dataset, dict):
        return None
    root = dataset.get("root")
    if not (isinstance(root, str) and root.strip()):
        return None
    for candidate in _dataset_root_candidates(root, model_dir):
        if (candidate / "meta").exists():
            return first_dataset_task(candidate)
    return None


def _checkpoint_policy_type(checkpoint_path: str) -> str | None:
    model_dir = _checkpoint_model_dir(checkpoint_path)
    config = _read_json(model_dir / "config.json") or {}
    value = config.get("type")
    return value if isinstance(value, str) and value.strip() else None


def _checkpoint_requires_task(checkpoint_path: str) -> bool:
    policy_type = _checkpoint_policy_type(checkpoint_path)
    return policy_type is None or policy_type in _LANGUAGE_POLICY_TYPES


def _default_robot_factory(serial: str) -> Any:
    import flexivrdk  # noqa: PLC0415

    return flexivrdk.Robot(serial)


def _rdk_mode() -> Any:
    import flexivrdk  # noqa: PLC0415

    return flexivrdk.Mode


def _zero_ft_sensor(
    robot: Any, stop_event: threading.Event, timeout: float = 3.0
) -> bool:
    # ZeroFTSensor requires NRT_PRIMITIVE_EXECUTION; unsupported firmware skips it.
    execute = getattr(robot, "ExecutePrimitive", None)
    if not callable(execute):
        return False
    try:
        mode = _rdk_mode()
        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        execute("ZeroFTSensor", {})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stop_event.is_set():
                break
            states = getattr(robot, "primitive_states", None)
            done = False
            if callable(states):
                values = states()
                if isinstance(values, dict):
                    done = any(
                        int(values.get(key, 0)) == 1
                        for key in ("reachedTarget", "terminated")
                    )
            busy = getattr(robot, "busy", None)
            if done or (callable(busy) and not busy()):
                return True
            stop_event.wait(0.05)
    except Exception as exc:
        warn("Failed to zero F/T sensor", describe_exception(exc))
        return False
    return True


def _normalize_actions(preprocessor: Any, actions: Any) -> Any:
    """Map real-unit actions to the model's normalized space via the
    preprocessor's NormalizerProcessorStep (same stats the postprocessor inverts).
    Returns None if the step is not found so the caller can disable RTC.
    """
    from lerobot.processor.normalize_processor import (  # noqa: PLC0415
        NormalizerProcessorStep,
    )

    for step in getattr(preprocessor, "steps", []):
        if isinstance(step, NormalizerProcessorStep):
            return step._normalize_action(actions, inverse=False)
    warn("RTC disabled: normalizer step not found", "")
    return None


def _predict_action_chunk(
    observation: dict[str, Any],
    policy: Any,
    device: str,
    preprocessor: Any,
    postprocessor: Any,
    *,
    force_refresh: bool = False,
    task: str | None = None,
    rtc_prev_actions: Any = None,
    rtc_inference_delay: int | None = None,
) -> tuple[Any, bool]:
    """Return an action chunk and whether it came from fresh inference.

    When RTC is attached to the policy and ``rtc_prev_actions`` is given, it is
    stashed on ``policy.diffusion`` so the RTC sampler conditions the fresh
    chunk's head on the previous chunk's still-executing tail. ``rtc_prev_actions``
    must be laid out on the full-horizon grid (index 0 == first horizon step).
    """
    import torch  # noqa: PLC0415
    from lerobot.utils.constants import ACTION  # noqa: PLC0415
    from lerobot.utils.control_utils import predict_action  # noqa: PLC0415

    torch_device = torch.device(device)
    queues = getattr(policy, "_queues", None)
    action_queue = queues.get(ACTION) if isinstance(queues, dict) else None

    # RTC: hand the previous chunk's unexecuted tail to the attached sampler.
    # The sampler works in normalized action space, but rtc_prev_actions arrives
    # in real units (postprocessed), so normalize it via the preprocessor's own
    # NormalizerProcessorStep (same stats the postprocessor later inverts).
    # Cleared after inference so a stale prefix never leaks into a later chunk.
    diffusion = getattr(policy, "diffusion", None)
    rtc_active = diffusion is not None and getattr(diffusion, "_rtc_attached", False)
    if rtc_active:
        if rtc_prev_actions is not None:
            prev = torch.as_tensor(
                rtc_prev_actions, dtype=torch.float32, device=torch_device
            )
            if prev.ndim == 2:
                prev = prev.unsqueeze(0)  # (1, T, D)
            prev = _normalize_actions(preprocessor, prev)
            diffusion._rtc_prev_actions = prev
            if prev is not None and rtc_inference_delay is not None:
                diffusion._rtc_inference_delay = int(rtc_inference_delay)
        else:
            diffusion._rtc_prev_actions = None

    if force_refresh:
        if action_queue is not None:
            action_queue.clear()  # LeRobot re-infers from the current obs when empty
        else:
            global _FORCE_REFRESH_WARNED
            if not _FORCE_REFRESH_WARNED:
                _FORCE_REFRESH_WARNED = True
                warn(
                    "Cannot force a fresh rollout inference",
                    "policy has no _queues[ACTION]; falling back to drain-refill",
                )

    fresh = action_queue is None or len(action_queue) == 0

    try:
        first = predict_action(
            observation, policy, torch_device, preprocessor, postprocessor,
            use_amp=False, task=task,
        )
    finally:
        # Drop the prefix so it can never condition a later, unrelated chunk.
        if rtc_active:
            diffusion._rtc_prev_actions = None
    tail = list(action_queue) if action_queue is not None else []
    if not tail:
        return first.reshape(1, -1), fresh
    with torch.inference_mode():
        tail = postprocessor(torch.cat([t.to(torch_device) for t in tail], dim=0))
    chunk = torch.cat([first.reshape(1, -1), tail.reshape(len(tail), -1)], dim=0)
    return chunk, fresh


def _cuda_sync(device: str) -> None:
    """Synchronize CUDA so inference timing includes queued work."""
    if not str(device).startswith("cuda"):
        return
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover - torch optional
        pass


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
        self._stop_reason: str | None = None
        self._checkpoint_path: str | None = None
        self._task: str | None = None
        self._robots: list[Any] = []
        self._device = "cpu"
        self._thread: threading.Thread | None = None
        self._target_hz: float | None = None
        self._waypoint_executor: WaypointExecutor | None = None
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
        target_hz = _checkpoint_target_hz(checkpoint_path)
        if target_hz is None:
            target_hz = float(self._settings.rollout.action_dt_hz)
            warn(
                "Checkpoint FPS metadata not found",
                f"falling back to rollout.action_dt_hz={target_hz:.1f}",
            )

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
        rollout_cfg = self._settings.policies.rollout_for(policy_type)
        override_fn = _ROLLOUT_OVERRIDES.get(policy_type)
        scheduler_overridden = (
            override_fn(policy, rollout_cfg) if override_fn is not None else False
        )
        self._apply_n_action_steps(policy, rollout_cfg)
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
        thread = threading.Thread(
            target=self._policy_planner_loop,
            args=(
                policy, preprocessor, postprocessor, robots, sides,
                rollout_cfg, target_hz, task,
            ),
            daemon=True,
            name="rollout-policy-planner",
        )
        self._thread = thread
        thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        # Stop robot commands before releasing their connections.
        executor = self._waypoint_executor
        if executor is not None:
            executor.join()
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

    def _teleop_initialized(self) -> bool:
        snapshot = self._teleop.snapshot()
        return bool(getattr(snapshot, "initialized", False))

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

    def _connect_robot(self, serial: str) -> Any:
        robot = self._robot_factory(serial)
        if robot.fault():
            robot.ClearFault()
        robot.Enable()
        while not robot.operational():
            if self._stop_event.wait(0.1):
                break
        if _zero_ft_sensor(robot, self._stop_event):
            self._append_log("INFO", "ROLLOUT", "F/T sensor zeroed", serial)
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

    def _read_robot_snapshot(self, robots: list[Any]) -> dict[str, Any]:
        """Build the robot snapshot shape consumed by the LeRobot I/O helpers."""
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
                # Values are placeholders; the feature builder reads only axes.
                "actions": {
                    "tcp_pose_d": tcp_pose,
                    "tcp_vel_d": tcp_vel,
                    "ext_wrench_d": wrench,
                },
            }
        return {"robots": robots_payload, "errors": {}}

    def _planner_hz(self) -> float:
        return float(self._target_hz or self._settings.rollout.planner_hz)

    def _policy_planner_loop(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        robots: list[Any],
        sides: list[str],
        rollout_cfg: Any,
        target_hz: float,
        task: str | None = None,
    ) -> None:
        policy.reset()
        period = 1.0 / self._planner_hz()
        # Waypoint spacing follows dataset FPS, not planner frequency.
        dt = 1.0 / float(target_hz)
        anchor = rollout_cfg.action_anchor_offset_steps
        # Auto replan uses half the first effective action chunk.
        replan_steps: int | None = None
        max_steps = self._settings.rollout.max_steps
        camera_names = resolve_recording_image_names(None, sides)
        layout: list[dict[str, Any]] | None = None
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
        # RTC: remember the last fresh chunk and when it was dispatched so the
        # next fresh inference can freeze its head onto the still-executing tail.
        rtc_enabled = bool(getattr(rollout_cfg, "rtc_enabled", False))
        rtc_delay_cfg = int(getattr(rollout_cfg, "rtc_inference_delay", 0) or 0)
        rtc_start_offset = max(
            getattr(getattr(policy, "config", None), "n_obs_steps", 1) - 1, 0
        )
        prev_chunk: np.ndarray | None = None
        prev_chunk_start: float | None = None
        step = 0
        try:
            while not self._stop_event.is_set():
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

                images = self._grab_images(camera_names)
                now = time.monotonic()
                stage_times["grab_images"].append(now - mark)
                mark = now

                snapshot = self._read_robot_snapshot(robots)
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
                    layout = build_action_layout(action_names, sides)
                    app_rollout = self._settings.rollout
                    waypoint_executor = WaypointExecutor(
                        robots,
                        layout,
                        self._stop_event,
                        (
                            app_rollout.max_linear_vel,
                            app_rollout.max_angular_vel,
                            app_rollout.max_linear_acc,
                            app_rollout.max_angular_acc,
                        ),
                    )
                    self._waypoint_executor = waypoint_executor
                    waypoint_executor.start()

                # Replan early enough to retain a committed path during inference.
                force = replan_steps is None or step % replan_steps == 0
                # RTC: freeze the new chunk's head onto the previous chunk's tail,
                # re-based to the horizon grid (index 0 == loop_start + anchor*dt).
                rtc_prev_actions = None
                rtc_delay = None
                seam_gap = None
                if (
                    rtc_enabled and force
                    and prev_chunk is not None and prev_chunk_start is not None
                ):
                    elapsed = max(1, round((loop_start - prev_chunk_start) / dt))
                    if elapsed < len(prev_chunk):
                        tail_start = max(elapsed - rtc_start_offset, 0)
                        rtc_prev_actions = prev_chunk[tail_start:]
                        rtc_delay = rtc_start_offset + (rtc_delay_cfg or 1)
                actions, fresh = _predict_action_chunk(
                    observation,
                    policy,
                    self._device,
                    preprocessor,
                    postprocessor,
                    force_refresh=force,
                    task=task,
                    rtc_prev_actions=rtc_prev_actions,
                    rtc_inference_delay=rtc_delay,
                )
                _cuda_sync(self._device)
                now = time.monotonic()
                infer_seconds = now - mark
                stage_times["inference"].append(infer_seconds)
                infer_raw.append(infer_seconds)
                mark = now

                action_lists = self._actions_to_lists(actions)
                now = time.monotonic()
                stage_times["to_list"].append(now - mark)
                mark = now

                # Fresh chunks replace pending waypoints on an anchored time grid.
                assert waypoint_executor is not None
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
                    target_times = [
                        loop_start + (k + anchor) * dt
                        for k in range(len(action_lists))
                    ]
                    waypoint_executor.replace_waypoints(
                        action_lists, target_times, now=time.monotonic()
                    )
                    if rtc_enabled:
                        if prev_chunk is not None and prev_chunk_start is not None:
                            old_i = min(
                                round((loop_start - prev_chunk_start) / dt),
                                len(prev_chunk) - 1,
                            )
                            seam_gap = float(
                                np.linalg.norm(prev_chunk[old_i] - action_lists[0])
                            )
                        prev_chunk = np.asarray(action_lists, dtype=np.float32)
                        prev_chunk_start = loop_start
                if waypoint_executor.error is not None:
                    raise RuntimeError(waypoint_executor.error)
                stage_times["dispatch"].append(time.monotonic() - mark)

                self._metrics.append({
                    "t": round(loop_start, 3),
                    "step": step,
                    "hz": round(actual_hz, 2),
                    "infer_ms": round(infer_seconds * 1000.0, 1),
                    "fresh": bool(fresh),
                    "rtc_d": rtc_delay,
                    "rtc_len": len(rtc_prev_actions)
                    if rtc_prev_actions is not None else None,
                    "seam_gap": round(seam_gap, 4) if seam_gap is not None else None,
                })
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
                self._logs.append(
                    _encode_ui_log("ERROR", "ROLLOUT", "Rollout stopped", detail)
                )
            warn("Rollout stopped", detail)
        finally:
            # Stop commands before releasing robot connections.
            if waypoint_executor is not None:
                self._stop_event.set()
                waypoint_executor.join()
            self._waypoint_executor = None
            self._release_robots()
            with self._lock:
                self._running = False
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
        expected_hz = float(self._target_hz or self._settings.rollout.action_dt_hz)
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
            if index < len(payloads) and isinstance(payloads[index], dict):
                states = payloads[index].get("states")
                if isinstance(states, dict):
                    measured = list(states.get("tcp_pose") or [])
            self._append_log(
                "INFO",
                "ROLLOUT",
                f"step={step} {side}",
                (
                    f"cmd_xyz={self._fmt_xyz(commanded)} "
                    f"meas_xyz={self._fmt_xyz(measured)} "
                    f"cmd_twist={self._fmt_vector(commanded_twist)}"
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
