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

from collections.abc import Callable
from typing import Any

import numpy as np

from flexivtrainer.data.lerobot_io import (
    extract_recording_frame_values,
    extract_recording_images,
)
from flexivtrainer.observability import info, warn

_FORCE_REFRESH_WARNED = False
_RESOLUTION_UNKNOWN_WARNED = False
_RESIZE_LOGGED: set[str] = set()
# Aspect ratios closer than this are treated as the same framing.
_ASPECT_TOLERANCE = 0.02


def _policy_action_queue(policy: Any, action_key: str) -> Any:
    """The policy's cached action chunk, or None when it holds no chunk.

    Most LeRobot families key a ``_queues`` dict by feature; ACT instead keeps a
    single ``_action_queue`` (and only when temporal ensembling is disabled).
    """
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict):
        return queues.get(action_key)
    return getattr(policy, "_action_queue", None)


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
    action_queue = _policy_action_queue(policy, ACTION)

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

    # predict_action calls policy.select_action internally. If a cached action
    # queue has entries, it pops the next one; otherwise it runs inference to
    # produce/refill a chunk and returns its first action.
    first = predict_action(
        observation, policy, torch_device, preprocessor, postprocessor,
        use_amp=False, task=task,
    )
    # Only a fresh inference needs the rest of its chunk for waypoint scheduling.
    # Cached-action ticks reuse the schedule already installed by that inference.
    if not fresh or action_queue is None:
        return first.reshape(1, -1), fresh

    tail = list(action_queue)
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


# Matches the teleop telemetry rate; the planner ticks ~3x faster than this.
WRENCH_SAMPLE_PERIOD_S = 0.1


def sample_wrench(
    append: Callable[[dict[str, Any]], None] | None,
    snapshot: dict[str, Any] | None,
    sides: list[str],
    now: float,
    last_sample_t: float,
) -> float:
    """Append a ``{t, <side>: [fx,fy,fz,mx,my,mz]}`` sample; return its timestamp.

    ``read_robot_snapshot`` inserts ``robot_0..robot_N`` in ``sides`` order, so
    payloads zip onto side names positionally. ``snapshot`` is None on ticks that
    skipped observation.
    """
    if append is None or not snapshot:
        return last_sample_t
    # Without the tolerance a 30 Hz tick lands at 0.0999... and halves the rate.
    if now - last_sample_t < WRENCH_SAMPLE_PERIOD_S - 1e-3:
        return last_sample_t

    sample: dict[str, Any] = {"t": round(now, 3)}
    payloads = snapshot.get("robots") or {}
    for side, payload in zip(sides, payloads.values()):
        states = payload.get("states") if isinstance(payload, dict) else None
        wrench = (
            states.get("ext_wrench_in_world") if isinstance(states, dict) else None
        )
        if isinstance(wrench, list) and len(wrench) >= 6:
            sample[side] = [round(float(v), 4) for v in wrench[:6]]
    append(sample)
    return now


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


def _resize_to_trained_resolution(
    name: str, image: np.ndarray, resolution: tuple[int, int]
) -> np.ndarray:
    """Match the checkpoint's image size, reusing the recording-time resize."""
    from flexivtrainer.data.recording_service import _resize_frame  # noqa: PLC0415

    height, width = int(image.shape[0]), int(image.shape[1])
    target_height, target_width = resolution
    target_aspect = target_width / target_height
    if abs(width / height - target_aspect) > _ASPECT_TOLERANCE * target_aspect:
        raise RuntimeError(
            f"Camera '{name}' captures {width}x{height} but the checkpoint was "
            f"trained on {target_width}x{target_height} (different aspect ratio); "
            "set the camera width/height to match the checkpoint"
        )
    if (height, width) != (target_height, target_width) and name not in _RESIZE_LOGGED:
        _RESIZE_LOGGED.add(name)
        info(
            "Resizing rollout frames to the trained resolution",
            f"{name} {height}x{width} -> {target_height}x{target_width}",
        )
    return _resize_frame(image, resolution)


def grab_images(
    cameras: Any,
    camera_names: list[str],
    resolutions: dict[str, tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    import cv2  # noqa: PLC0415

    if not resolutions:
        global _RESOLUTION_UNKNOWN_WARNED
        if not _RESOLUTION_UNKNOWN_WARNED:
            _RESOLUTION_UNKNOWN_WARNED = True
            warn(
                "Checkpoint image resolution unknown",
                "feeding raw camera frames to the policy without resizing",
            )
    images: dict[str, np.ndarray] = {}
    for name in camera_names:
        frame = cameras.capture_frame(name, block=False, allow_cached=True)
        image = frame.get("image") if isinstance(frame, dict) else None
        if image is None:
            continue
        image = np.asarray(image)
        resolution = resolutions.get(name) if resolutions else None
        if resolution is not None:
            # Resize before color conversion so it processes fewer pixels.
            image = _resize_to_trained_resolution(name, image, resolution)
        # Cameras capture BGR; LeRobot policies were trained on RGB frames.
        images[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
