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

"""One arm stepping in a practice space: send a step, sense until the next.

`reset` reads where the arm is and returns what the skill sees. `step` sends
the command for an action once, then reads the arm `stride` times, at
`sense_hz`, until the next step is due, and returns what the skill sees then.
The arm tracks the command in between; a step that sends nothing (an operator
has the arm) leaves it with whatever drives it.

THE FORCE TARGET GOES OUT IN THE CONTROLLER'S SIGN. The space's force is what
the tool applies, along the tool's axes. The controller takes "the wrench
sensed at the TCP" in the force-control frame: commanding +5 N along z moves
the tool towards -z until the sensor reads +5 N. So the command is negated, in
a force-control frame set on the TCP (`rollout/hardware.py`, `set_force_axes`).

AN OPERATOR'S STEP IS MEASURED LIKE A DEMONSTRATION. While someone else has
the arm (a leader-arm takeover), `step(None)` sends nothing; at the end of the
step it reads the pose the controller is commanding and the force the tool
applies, and turns them into the action that heads there from where the step
started -- the label the demo converter gives a teleop recording. That action
becomes the newest of the skill's last commands, as it would in the demos.

NOTHING PLANNED BEFORE A PAUSE IS SENT AFTER IT. `pause` returns only once no
command can reach the arm; `epoch` changes on every pause and resume, and a
step planned under an older epoch sends nothing (`step(action, epoch=...)`).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from flexivtrainer.practice.space import Command, PracticeSpace, Sample, StepTracker
from flexivtrainer.rollout import observations
from flexivtrainer.rollout.safety import SafetyEnvelope
from flexivtrainer.rollout.takeover import controller_command

__all__ = ["POLICY", "StepEnv", "StepResult", "Tick"]

POLICY = "policy"


@dataclass(frozen=True)
class Tick:
    """One reading of the arm, for whoever records or scores the episode."""

    t: float
    step: int
    snapshot: dict[str, Any]
    images: dict[str, np.ndarray]
    sample: Sample
    #: Who had the arm: `policy`, `operator`, or a switch under way.
    control: str
    #: The policy's command in force, or None while it has sent none.
    command: Command | None
    #: The first tick after the policy sent a new command.
    fresh: bool = False


@dataclass(frozen=True)
class StepResult:
    """What one step did, and what the skill sees after it."""

    observation: dict[str, np.ndarray]
    #: The action the step took: the policy's when its command was sent,
    #: otherwise the operator's, measured as a demonstration would be.
    action: np.ndarray
    #: Whether the policy's command was sent at the start of the step.
    sent: bool
    #: Whether anyone but the policy had the arm during the step.
    intervention: bool


class StepEnv:
    """One arm stepping in a practice space."""

    def __init__(
        self,
        *,
        robot: Any,
        cameras: Any,
        camera_names: Sequence[str],
        space: PracticeSpace,
        motion_limits: tuple[float, float, float, float],
        stop_event: threading.Event,
        safety: SafetyEnvelope | None = None,
        image_resolutions: Mapping[str, tuple[int, int]] | None = None,
        control: Callable[[], str] | None = None,
        on_tick: Callable[[Tick], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        missing = [v for v in space.views if v not in camera_names]
        if missing:
            raise ValueError(
                f"the skill sees camera {missing}, which this rig does not stream; "
                f"it streams {list(camera_names)}"
            )
        self.space = space
        self._robot = robot
        self._cameras = cameras
        self._camera_names = list(camera_names)
        self._limits = tuple(float(v) for v in motion_limits)
        self._stop = stop_event
        self._safety = safety or SafetyEnvelope()
        self._resolutions = dict(image_resolutions or {})
        self._control = control or (lambda: POLICY)
        self._on_tick = on_tick
        self._clock = clock
        self._dt = 1.0 / space.sense_hz
        self._tracker = StepTracker(space)
        self._lock = threading.Lock()
        # Held for the whole of each send, so pause() can wait one out.
        self._send_lock = threading.Lock()
        self._paused = False
        self._epoch = 0
        self._step = 0
        self._t0 = 0.0
        self._images: dict[str, np.ndarray] = {}
        self._command: Command | None = None

    # --- control of the arm ------------------------------------------------

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def pause(self) -> None:
        """Send nothing until resume(). Returns once a send under way is done."""
        with self._lock:
            self._paused = True
            self._epoch += 1
        with self._send_lock:
            pass

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._epoch += 1

    @property
    def steps(self) -> int:
        return self._step

    # --- the episode ---------------------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        """Start an episode where the arm is; what the skill sees there."""
        self._step = 0
        self._command = None
        self._t0 = self._clock()
        snapshot, images, sample = self._read()
        self._tracker.reset(sample)
        self._images = images
        self._emit(self._t0, snapshot, images, sample, self._control())
        return self._tracker.observation(images)

    def step(
        self, action: Sequence[float] | None, *, epoch: int | None = None
    ) -> StepResult:
        """Send `action`'s command (None: send nothing), then sense one step.

        `epoch`, read before the action was planned, drops a plan that a
        pause or resume has made stale: the step then runs as one the policy
        did not drive.
        """
        start = self._tracker.sample
        sent = False
        taken: np.ndarray | None = None
        if action is not None and self._control() == POLICY:
            self._safety.check_finite([list(action)])
            command = self._tracker.command(action)
            self._safety.check_workspace([[command.position.tolist()]])
            self._safety.check_reference(
                [command.position.tolist()], [start.position.tolist()]
            )
            if self._send(command, self.epoch if epoch is None else epoch):
                self._tracker.remember(command)
                self._command = command
                sent = True
                taken = np.asarray(action, dtype=np.float32)
        intervention = not sent
        self._step += 1
        for k in range(1, self.space.stride + 1):
            due = self._t0 + k * self._dt
            delay = due - self._clock()
            if delay > 0 and self._stop.wait(delay):
                break
            control = self._control()
            intervention |= control != POLICY
            snapshot, images, sample = self._read()
            if control != POLICY:
                self._safety.check_measured_workspace([sample.position.tolist()])
            self._tracker.sense(sample)
            self._images = images
            self._emit(due, snapshot, images, sample, control, fresh=sent and k == 1)
        if not sent:
            # The step is known only now: where the controller is taking the
            # tool, from where the step started.
            taken, _ = self._tracker.action_toward(self._operator_target(), start)
            self._tracker.remember(self._tracker.command(taken, start))
        self._t0 += self.space.stride * self._dt
        if self._clock() - self._t0 > self._dt:
            # Running late by more than a tick: start the grid again from now
            # rather than sense the next ticks back to back.
            self._t0 = self._clock()
        return StepResult(
            observation=self._tracker.observation(self._images),
            action=taken,
            sent=sent,
            intervention=intervention,
        )

    # --- the arm -------------------------------------------------------------

    def _send(self, command: Command, epoch: int) -> bool:
        wrench = [-float(f) for f in command.force] + [0.0, 0.0, 0.0]
        with self._send_lock:
            with self._lock:
                if self._paused or epoch != self._epoch:
                    return False
            self._robot.SendCartesianMotionForce(
                command.pose(), wrench, [0.0] * 6, *self._limits
            )
        return True

    def _read(self) -> tuple[dict[str, Any], dict[str, np.ndarray], Sample]:
        if self._robot.fault():
            raise RuntimeError("Fault occurred on the follower robot")
        images = observations.grab_images(
            self._cameras, self._camera_names, self._resolutions
        )
        snapshot = observations.read_robot_snapshot([self._robot])
        states = snapshot["robots"]["robot_0"]["states"]
        self._safety.check_force([states["ext_wrench_in_world"]])
        sample = Sample.from_driver(
            states["tcp_pose"], states["tcp_vel"], states["ext_wrench_in_world"]
        )
        return snapshot, images, sample

    def _operator_target(self) -> Command:
        """Where the controller is taking the tool, and the force it applies."""
        commanded = Sample.from_driver(
            controller_command(self._robot)["tcp_pose_d"], [0.0] * 6, [0.0] * 6
        )
        return Command(
            commanded.position,
            commanded.rotation,
            self._tracker.sample.tool_wrench()[:3],
        )

    def _emit(
        self,
        t: float,
        snapshot: dict[str, Any],
        images: dict[str, np.ndarray],
        sample: Sample,
        control: str,
        *,
        fresh: bool = False,
    ) -> None:
        if self._on_tick is not None:
            self._on_tick(
                Tick(
                    t=t,
                    step=self._step,
                    snapshot=snapshot,
                    images=images,
                    sample=sample,
                    control=control,
                    command=self._command,
                    fresh=fresh,
                )
            )
