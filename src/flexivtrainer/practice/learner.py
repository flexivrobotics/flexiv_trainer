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

"""The trainer's side of LeRobot's actor-learner link.

Practice splits a skill's learning in two. The trainer acts: it steps the arm,
lets an operator take over, and judges each episode. A learner -- LeRobot's
(`python -m lerobot.rl.learner`), on a GPU machine -- learns: it keeps the
demonstrations and the steps it is sent, updates the policy, and pushes the new
weights back. This module is the trainer's end of that link, in LeRobot's own
protocol (gRPC, `lerobot.transport`), so any policy LeRobot's learner trains
can practise here, with nothing of the policy's own code in the trainer.

What goes to the learner: each finished episode's steps, in order, as LeRobot
transitions -- the observation the step started from, the action it took (the
operator's while they had the arm, marked as an intervention, which the learner
also keeps among its demonstrations), the reward, the next observation, and
whether the episode ended there -- and a line of statistics per episode. What
comes back: the policy's weights, whenever the learner pushes them; the newest
are loaded between episodes, never during one.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["LearnerLink", "apply_weights", "observation_state", "transition"]

#: The intervention flag's key, as LeRobot's learner reads it.
INTERVENTION = "is_intervention"


def observation_state(observation: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """An observation as a learner stores it: tensors with a batch of one,
    images channels-first in [0, 1], as a LeRobot dataset gives them."""
    import torch  # noqa: PLC0415

    state: dict[str, Any] = {}
    for key, value in observation.items():
        array = np.asarray(value)
        if ".images." in key:
            tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
            state[key] = (tensor.float() / 255.0).unsqueeze(0)
        else:
            state[key] = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
    return state


def transition(
    observation: Mapping[str, np.ndarray],
    action: Sequence[float],
    reward: float,
    next_observation: Mapping[str, np.ndarray],
    *,
    done: bool,
    truncated: bool,
    intervention: bool,
) -> dict[str, Any]:
    """One step, as LeRobot's learner takes it."""
    import torch  # noqa: PLC0415

    return {
        "state": observation_state(observation),
        "action": torch.as_tensor(np.asarray(action), dtype=torch.float32)[None],
        "reward": float(reward),
        "next_state": observation_state(next_observation),
        "done": bool(done),
        "truncated": bool(truncated),
        "complementary_info": {
            "discrete_penalty": torch.zeros(1),
            INTERVENTION: bool(intervention),
        },
    }


def apply_weights(policy: Any, weights: Mapping[str, Any]) -> None:
    """Load a learner's push into the acting policy: the actor, and the
    gripper's critic when it has one -- what LeRobot's actor loads."""
    device = next(policy.parameters()).device
    policy.actor.load_state_dict(
        {k: v.to(device) for k, v in weights["policy"].items()}
    )
    critic = getattr(policy, "discrete_critic", None)
    if "discrete_critic" in weights and critic is not None:
        critic.load_state_dict(
            {k: v.to(device) for k, v in weights["discrete_critic"].items()}
        )


class LearnerLink:
    """A connection to a LeRobot learner: steps out, weights in."""

    def __init__(self, host: str, port: int, *, queue_timeout_s: float = 0.2) -> None:
        import grpc  # noqa: PLC0415
        from lerobot.transport import services_pb2_grpc  # noqa: PLC0415
        from lerobot.transport.utils import grpc_channel_options  # noqa: PLC0415

        self.address = f"{host}:{port}"
        self._channel = grpc.insecure_channel(self.address, grpc_channel_options())
        self._stub = services_pb2_grpc.LearnerServiceStub(self._channel)
        self._timeout = queue_timeout_s
        self._shutdown = threading.Event()
        self._transitions: queue.Queue[bytes] = queue.Queue()
        self._interactions: queue.Queue[bytes] = queue.Queue()
        self._weights: queue.Queue[bytes] = queue.Queue()
        self._threads: list[threading.Thread] = []
        #: The first failure of the link, once it has failed.
        self.error: str | None = None

    def connect(
        self, stop: threading.Event, *, attempts: int = 30, wait_s: float = 2.0
    ) -> None:
        """Wait for the learner to answer; raises when it never does."""
        import grpc  # noqa: PLC0415
        from lerobot.transport import services_pb2  # noqa: PLC0415

        last = ""
        for _ in range(attempts):
            if stop.is_set():
                raise RuntimeError("stopped while waiting for the learner")
            try:
                self._stub.Ready(services_pb2.Empty(), timeout=wait_s)
                return
            except grpc.RpcError as exc:
                last = exc.details() if hasattr(exc, "details") else str(exc)
                stop.wait(wait_s)
        raise RuntimeError(f"no learner answered at {self.address}: {last}")

    def start(self) -> None:
        """Start streaming: weights in, steps and statistics out."""
        from lerobot.transport import services_pb2  # noqa: PLC0415
        from lerobot.transport.utils import receive_bytes_in_chunks  # noqa: PLC0415

        def weights() -> None:
            receive_bytes_in_chunks(
                self._stub.StreamParameters(services_pb2.Empty()),
                self._weights,
                self._shutdown,
            )

        def transitions() -> None:
            self._stub.SendTransitions(
                self._stream(self._transitions, services_pb2.Transition)
            )

        def interactions() -> None:
            self._stub.SendInteractions(
                self._stream(self._interactions, services_pb2.InteractionMessage)
            )

        for target in (weights, transitions, interactions):
            thread = threading.Thread(
                target=self._guarded,
                args=(target,),
                name=f"practice-learner-{target.__name__}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def send_episode(self, transitions: Sequence[dict[str, Any]]) -> None:
        """Queue one finished episode's steps, in order."""
        from lerobot.transport.utils import transitions_to_bytes  # noqa: PLC0415

        if transitions:
            self._transitions.put(transitions_to_bytes(list(transitions)))

    def send_stats(self, stats: Mapping[str, Any]) -> None:
        """Queue one episode's statistics, for the learner's log."""
        from lerobot.transport.utils import python_object_to_bytes  # noqa: PLC0415

        self._interactions.put(python_object_to_bytes(dict(stats)))

    def latest_weights(self) -> dict[str, Any] | None:
        """The newest weights pushed since the last call, or None."""
        from lerobot.transport.utils import bytes_to_state_dict  # noqa: PLC0415

        newest = None
        while True:
            try:
                newest = self._weights.get_nowait()
            except queue.Empty:
                break
        return None if newest is None else bytes_to_state_dict(newest)

    def close(self, timeout_s: float = 10.0) -> None:
        """Deliver what is queued, then hang up.

        The last episode is usually queued just before this is called, so the
        outgoing streams are drained first and ended, which lets the learner
        finish receiving; only then is the channel closed, which also ends the
        stream of weights.
        """
        deadline = time.monotonic() + timeout_s
        for source in (self._transitions, self._interactions):
            while source.unfinished_tasks and time.monotonic() < deadline:
                if self.error is not None:
                    break
                time.sleep(0.01)
        self._shutdown.set()
        senders, receiver = self._threads[1:], self._threads[:1]
        for thread in senders:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._channel.close()
        for thread in receiver:
            thread.join(timeout=5.0)

    def _stream(self, source: queue.Queue[bytes], message: Any) -> Iterator[Any]:
        from lerobot.transport.utils import send_bytes_in_chunks  # noqa: PLC0415

        while not self._shutdown.is_set():
            try:
                payload = source.get(timeout=self._timeout)
            except queue.Empty:
                continue
            # Resumed once the last chunk is taken: then it counts as sent.
            yield from send_bytes_in_chunks(payload, message)
            source.task_done()

    def _guarded(self, target: Any) -> None:
        try:
            target()
        except Exception as exc:  # a closed channel at the end is no failure
            if not self._shutdown.is_set() and self.error is None:
                self.error = f"{type(exc).__name__}: {exc}"
