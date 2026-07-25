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

"""Build rollout observations and run policy inference on them."""

from __future__ import annotations

from typing import Any

import numpy as np

from flexivtrainer.data.lerobot_io import (
    extract_recording_frame_values,
    extract_recording_images,
)
from flexivtrainer.observability import warn

_FORCE_REFRESH_WARNED = False


def _predict_action_chunk(
    observation: dict[str, Any],
    policy: Any,
    device: str,
    preprocessor: Any,
    postprocessor: Any,
    *,
    force_refresh: bool = False,
    task: str | None = None,
) -> tuple[Any, bool]:
    """Return an action chunk and whether it came from fresh inference."""
    import torch  # noqa: PLC0415
    from lerobot.utils.constants import ACTION  # noqa: PLC0415

    try:
        from lerobot.common.control_utils import predict_action  # noqa: PLC0415
    except ImportError:
        from lerobot.utils.control_utils import predict_action  # noqa: PLC0415

    torch_device = torch.device(device)
    queues = getattr(policy, "_queues", None)
    action_queue = queues.get(ACTION) if isinstance(queues, dict) else None

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

    first = predict_action(
        observation, policy, torch_device, preprocessor, postprocessor,
        use_amp=False, task=task,
    )
    tail = list(action_queue) if action_queue is not None else []
    if not tail:
        return first.reshape(1, -1), fresh
    with torch.inference_mode():
        tail = postprocessor(torch.cat([t.to(torch_device) for t in tail], dim=0))
    chunk = torch.cat([first.reshape(1, -1), tail.reshape(len(tail), -1)], dim=0)
    return chunk, fresh


def _prepare_policy_observation(
    observation: dict[str, Any],
    device: str,
    preprocessor: Any,
    *,
    task: str | None = None,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    from lerobot.policies.utils import (  # noqa: PLC0415
        prepare_observation_for_inference,
    )

    with torch.inference_mode():
        batch = prepare_observation_for_inference(
            dict(observation),
            torch.device(device),
            task=task,
        )
        return preprocessor(batch)


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


def read_robot_snapshot(
    robots: list[Any],
    gripper_states: dict[str, dict[str, float]] | None = None,
    sides: list[str] | None = None,
) -> dict[str, Any]:
    """Build the robot snapshot shape consumed by the LeRobot I/O helpers."""
    robots_payload: dict[str, Any] = {}
    for index, robot in enumerate(robots):
        states = robot.states()
        tcp_pose = [float(v) for v in states.tcp_pose]
        tcp_vel = [float(v) for v in states.tcp_vel]
        wrench = [float(v) for v in states.ext_wrench_in_world]
        payload = {
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
        if (
            gripper_states is not None
            and sides is not None
            and index < len(sides)
            and sides[index] in gripper_states
        ):
            payload["gripper"] = dict(gripper_states[sides[index]])
        robots_payload[f"robot_{index}"] = payload
    return {"robots": robots_payload, "errors": {}}


def grab_images(cameras: Any, camera_names: list[str]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for name in camera_names:
        frame = cameras.capture_frame(name, block=False, allow_cached=True)
        image = frame.get("image") if isinstance(frame, dict) else None
        if image is None:
            continue
        # Cameras capture BGR; LeRobot policies were trained on RGB frames.
        images[name] = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    return images


def build_observation(
    snapshot: dict[str, Any], images: dict[str, np.ndarray], sides: list[str]
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
