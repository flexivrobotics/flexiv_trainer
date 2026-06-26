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

Rollout drives the follower arm(s) directly through the RDK ``Robot`` API
(NRT Cartesian motion-force control), *not* through the TDK teleop controller --
that controller only mirrors a physical leader and has no command-injection API.
Because a fresh RDK connection cannot coexist with the TDK controller holding the
same follower, rollout requires teleoperation to be shut down first.

The control loop, at a fixed rate, reads each follower's RDK state and the
cameras, builds an observation in the *same layout the recorder used for
training* (so the policy sees its training distribution), runs inference, and
unpacks the policy's flat ``action`` vector back into per-side TCP pose / wrench
/ gripper commands by matching the action feature's axis names.
"""

from __future__ import annotations

import threading
import time
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
from flexivtrainer.observability import describe_exception, warn

# Number of scalars per metric, in the order the recorder concatenates them
# within one side: tcp_pose (7) -> tcp_twist (6) -> tcp_wrench (6) -> gripper (2).
# Sliced out of the policy's flat action vector via the action feature's axis
# names (see ``_plan_action_layout``), so this stays robust to which metrics a
# given checkpoint was trained with.
_POSE_DIM = 7
_WRENCH_DIM = 6


def _default_policy_loader(checkpoint_path: str, device: str) -> Any:
    """Load a LeRobot policy from a checkpoint directory onto ``device``.

    ``from_pretrained`` expects the checkpoint *directory* (it holds the policy
    config + weights). A user may pick the weights file inside it, so accept
    either and resolve to the containing directory. The concrete policy class is
    selected from the type recorded in the checkpoint's config — the abstract
    base cannot be instantiated directly.
    """
    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.factory import get_policy_class  # noqa: PLC0415

    path = Path(checkpoint_path)
    model_dir = path.parent if path.is_file() else path
    config = PreTrainedConfig.from_pretrained(model_dir)
    policy = get_policy_class(config.type).from_pretrained(model_dir)
    policy.to(device)
    policy.eval()
    return policy


def _default_robot_factory(serial: str) -> Any:
    import flexivrdk  # noqa: PLC0415

    return flexivrdk.Robot(serial)


def _rdk_mode() -> Any:
    import flexivrdk  # noqa: PLC0415

    return flexivrdk.Mode


class RolloutService:
    """Lifecycle + background control loop for policy rollout."""

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
        self._stop_event = threading.Event()

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
            }

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
            policy = self._policy_loader(checkpoint_path, device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load policy: {describe_exception(exc)}"
            ) from exc
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
        thread = threading.Thread(
            target=self._run,
            args=(policy, robots, sides),
            daemon=True,
            name="rollout-control",
        )
        self._thread = thread
        thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
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
                # The policy produces the action; we never read a commanded
                # action off the robot. But ``build_features_from_sample`` derives
                # the action feature's axis names (used to slice the policy
                # output) from this section, so mirror the state dimensions here.
                # Only the names/dimensions matter, not these placeholder values.
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
        side's ``tcp_pose`` and ``tcp_wrench`` runs by name so the slicing tracks
        exactly what the recorder produced for this checkpoint.
        """
        layout: list[dict[str, Any]] = []
        for side in sides:
            pose_start = self._find_run(action_names, f"{side}.tcp_pose.")
            wrench_start = self._find_run(action_names, f"{side}.tcp_wrench.")
            pose = (
                None
                if pose_start is None
                else slice(pose_start, pose_start + _POSE_DIM)
            )
            wrench = (
                None
                if wrench_start is None
                else slice(wrench_start, wrench_start + _WRENCH_DIM)
            )
            layout.append({"side": side, "pose": pose, "wrench": wrench})
        return layout

    @staticmethod
    def _find_run(names: list[str], prefix: str) -> int | None:
        for index, name in enumerate(names):
            if name.startswith(prefix):
                return index
        return None

    def _loop_period(self) -> float:
        return 1.0 / float(self._settings.rollout.loop_hz)

    def _run(self, policy: Any, robots: list[Any], sides: list[str]) -> None:
        period = self._loop_period()
        # 0 means "no cap": run until the operator stops it or a fault occurs.
        max_steps = self._settings.rollout.max_steps
        camera_names = resolve_recording_image_names(None, sides)
        layout: list[dict[str, Any]] | None = None
        step = 0
        try:
            while not self._stop_event.is_set():
                loop_start = time.monotonic()

                for robot in robots:
                    if robot.fault():
                        raise RuntimeError("Fault occurred on a follower robot")

                images = self._grab_images(camera_names)
                snapshot = self._read_robot_snapshot(robots, sides)
                observation = self._build_observation(snapshot, images, sides)

                if layout is None:
                    features, _, _ = build_features_from_sample(
                        snapshot, images, None, sides
                    )
                    action_feature = features.get("action")
                    action_names = action_feature["names"] if action_feature else []
                    layout = self._plan_action_layout(action_names, sides)

                batch = self._to_policy_batch(observation)
                action = policy.select_action(batch)
                action_vector = self._action_to_list(action)
                self._dispatch_action(action_vector, robots, layout)

                step += 1
                if max_steps and step >= max_steps:
                    with self._lock:
                        self._stop_reason = "timeout"
                    break

                elapsed = time.monotonic() - loop_start
                if period - elapsed > 0:
                    self._stop_event.wait(period - elapsed)
        except Exception as exc:
            with self._lock:
                self._error = describe_exception(exc)
                self._running = False
            warn("Rollout stopped", describe_exception(exc))
        finally:
            self._release_robots()
            with self._lock:
                self._running = False

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

    def _to_policy_batch(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        """Convert the numpy observation into a batched torch tensor dict.

        LeRobot policies expect a batch of torch tensors on the policy's device:
        state vectors as float32 with a leading batch axis, and images as float
        CHW normalized to [0, 1] (the recorder stored HWC uint8 RGB frames).
        ``select_action`` applies the policy's own input normalization on top.
        """
        import torch  # noqa: PLC0415

        batch: dict[str, Any] = {}
        for key, value in observation.items():
            array = np.asarray(value)
            if key.startswith("observation.images."):
                tensor = torch.from_numpy(np.ascontiguousarray(array)).float() / 255.0
                tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            else:
                tensor = torch.from_numpy(array.astype(np.float32))
            batch[key] = tensor.unsqueeze(0).to(self._device)
        return batch

    @staticmethod
    def _action_to_list(action: Any) -> list[float]:
        """Normalize a policy action into a flat ``list[float]``.

        Accepts a torch tensor (any device), numpy array, or sequence, detaches
        and moves tensors to CPU, then flattens to a 1-D list of native floats so
        downstream slicing/commands don't depend on the policy's output type,
        device, or shape.
        """
        detached = getattr(action, "detach", None)
        if callable(detached):
            action = action.detach().cpu().numpy()
        return [float(v) for v in np.asarray(action).reshape(-1)]

    def _dispatch_action(
        self, action: list[float], robots: list[Any], layout: list[dict[str, Any]]
    ) -> None:
        """Slice the flat action vector per arm and command each robot.

        ``layout`` (from ``_plan_action_layout``) maps each side to its pose and
        wrench slices within ``action``; ``layout[i]`` is paired positionally with
        ``robots[i]``. Arms without a pose slice are skipped, a missing wrench
        defaults to zero force/torque, and each arm's pose+wrench is issued via
        ``SendCartesianMotionForce``.
        """
        for index, plan in enumerate(layout):
            if index >= len(robots):
                break
            robot = robots[index]
            pose_slice = plan["pose"]
            wrench_slice = plan["wrench"]
            if pose_slice is None:
                continue
            target_pose = action[pose_slice]
            if wrench_slice is not None:
                target_wrench = action[wrench_slice]
            else:
                target_wrench = [0.0] * _WRENCH_DIM
            robot.SendCartesianMotionForce(target_pose, target_wrench)
