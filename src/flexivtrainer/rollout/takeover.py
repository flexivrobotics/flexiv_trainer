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

"""Take a running rollout over with the leader arm, and hand it back.

The rollout borrows the teleop connection's robot handles (TeleopService.
lend_to_rollout), so the same transparent teleoperation that records the
demonstrations can drive the follower in the middle of a trial, without
disconnecting anything.

TAKING OVER: the policy stops at once -- its pending waypoints are dropped and
nothing it plans is sent -- then the teleop loop starts WITHOUT zeroing the
force sensors (the tool may be in contact) and every pair is engaged. The
follower copies the leader's relative motion, so the leader need not be moved
onto the follower first. The jaws keep their state until the operator actuates
the leader trigger.

HANDING BACK: the pairs disengage and the teleop loop stops (the robots hold);
the rollout takes the followers back into its own control mode, and the policy
is reset so it plans from where the operator left the arm. No waypoint planned
before the takeover is ever sent.

A switch that fails leaves the robots holding and the policy stopped; the
operator can retry the hand-back or stop the trial. Each switch is timed: on
the rig this is the number that matters most, and it is recorded.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from flexivtrainer.data.lerobot_io import (
    POSE_FORMAT_QUATERNION,
    POSE_FORMAT_ROTATION_6D,
    recorded_action_by_name,
)
from flexivtrainer.observability import describe_exception, warn

__all__ = [
    "LeaderTrigger",
    "Takeover",
    "TakeoverRefused",
    "operator_action",
]

POLICY = "policy"
OPERATOR = "operator"
TO_OPERATOR = "switching_to_operator"
TO_POLICY = "switching_to_policy"
ERROR = "error"

#: How long the end of a trial waits for a switch under way to finish.
CLOSE_TIMEOUT_S = 30.0


class TakeoverRefused(RuntimeError):
    """A takeover or hand-back that does not apply now; the message says why."""


class _Runner(Protocol):
    """What a takeover needs from the rollout runner."""

    @property
    def ready(self) -> bool: ...

    def pause_policy(self) -> None: ...

    def reclaim(self) -> None: ...

    def resume_policy(self) -> None: ...


def _vector(source: Any, field: str) -> list[float] | None:
    raw = (
        source.get(field) if isinstance(source, dict) else getattr(source, field, None)
    )
    if raw is None:
        return None
    return [float(v) for v in raw]


def _commanded(robot: Any) -> dict[str, list[float]]:
    """What the controller is commanding the follower: the teleop action."""
    reader = getattr(robot, "actions", None)
    if callable(reader):
        actions = reader()
        values = {
            field: _vector(actions, field)
            for field in ("tcp_pose_d", "tcp_vel_d", "ext_wrench_d")
        }
        if all(value is not None for value in values.values()):
            return values  # type: ignore[return-value]
    # A handle without actions(): the measured state is the closest record of
    # where the operator put the arm.
    states = robot.states()
    return {
        "tcp_pose_d": _vector(states, "tcp_pose") or [],
        "tcp_vel_d": _vector(states, "tcp_vel") or [],
        "ext_wrench_d": _vector(states, "ext_wrench_in_world") or [],
    }


def operator_action(
    robots: Sequence[Any],
    sides: Sequence[str],
    action_names: Sequence[str],
    gripper_targets: dict[str, float],
    fallback: Sequence[float],
) -> list[float]:
    """The action recorded for a tick the operator drives.

    Defined the way a teleoperated recording defines `action`, so a corrected
    segment reads like the demonstrations: each follower's commanded pose,
    twist and wrench, and the gripper width the leader trigger last commanded.
    An axis that has no such value (a gripper the operator has not actuated
    yet keeps the policy's last target) takes `fallback`'s.
    """
    pose_format = (
        POSE_FORMAT_ROTATION_6D
        if any(".tcp_rotation_6d." in name for name in action_names)
        else POSE_FORMAT_QUATERNION
    )
    payload: dict[str, Any] = {}
    for index, robot in enumerate(robots):
        entry: dict[str, Any] = {"actions": _commanded(robot)}
        side = sides[index] if index < len(sides) else ""
        if side in gripper_targets:
            entry["gripper_command"] = {"target_width": gripper_targets[side]}
        payload[f"robot_{index}"] = entry
    by_name = recorded_action_by_name({"robots": payload}, list(sides), pose_format)
    return [
        float(by_name.get(name, fallback[index] if index < len(fallback) else 0.0))
        for index, name in enumerate(action_names)
    ]


class LeaderTrigger:
    """A leader digital input that toggles control, read edge by edge.

    A press counts once the input has held its active level for `debounce_s`;
    releasing it does nothing. The level read when polling starts is taken as
    the resting state, so a button already held does not fire.
    """

    POLL_S = 0.01

    def __init__(
        self,
        read_inputs: Callable[[], Sequence[bool]],
        channel: int,
        on_press: Callable[[], None],
        *,
        active_high: bool = True,
        debounce_s: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._read_inputs = read_inputs
        self._channel = channel
        self._on_press = on_press
        self._active_high = active_high
        self._debounce_s = debounce_s
        self._clock = clock
        self._stable: bool | None = None
        self._candidate: bool | None = None
        self._since = 0.0
        self._read_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def channel(self) -> int:
        return self._channel

    def _active(self) -> bool:
        level = bool(self._read_inputs()[self._channel])
        return level if self._active_high else not level

    def poll(self) -> bool:
        """Read the input once; True when this reading completed a press."""
        try:
            active = self._active()
        except Exception as exc:
            detail = describe_exception(exc)
            if detail != self._read_error:
                warn("Takeover trigger could not be read", detail)
            self._read_error = detail
            return False
        self._read_error = None
        now = self._clock()
        if self._stable is None:
            self._stable = self._candidate = active
            return False
        if active != self._candidate:
            self._candidate = active
            self._since = now
            return False
        if active == self._stable or now - self._since < self._debounce_s:
            return False
        self._stable = active
        if not active:
            return False
        try:
            self._on_press()
        except Exception as exc:
            warn("Takeover trigger was ignored", describe_exception(exc))
        return True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="rollout-takeover-trigger", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(self.POLL_S)


class Takeover:
    """Who controls the follower during one rollout, and the switches between.

    `take_over()` and `hand_back()` return at once; the switch runs on its own
    thread and `status()` follows it. The planner reads `control()` every tick.
    """

    def __init__(
        self,
        *,
        teleop: Any,
        lease: Any,
        sides: Sequence[str],
        max_contact_wrench: Sequence[float] | None = None,
        append_log: Callable[..., None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._teleop = teleop
        self._lease = lease
        self._sides = list(sides)
        self._max_contact_wrench = (
            [float(v) for v in max_contact_wrench]
            if max_contact_wrench is not None
            else None
        )
        self._append_log = append_log or (lambda *args: None)
        self._clock = clock
        self._runner: _Runner | None = None
        self._trigger: LeaderTrigger | None = None
        self._lock = threading.Lock()
        # Held for the whole of a switch, so a trial that ends waits for it.
        self._switch_lock = threading.Lock()
        self._state = POLICY
        self._error: str | None = None
        self._closed = False
        self._count = 0
        self._switches: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        # Start of the current intervention (policy stopped), and the total of
        # the finished ones.
        self._intervention_start: float | None = None
        self._intervention_s = 0.0
        self._worker: threading.Thread | None = None

    def bind(self, runner: _Runner) -> None:
        self._runner = runner

    def attach_trigger(self, trigger: LeaderTrigger) -> None:
        self._trigger = trigger

    def start_trigger(self) -> None:
        if self._trigger is not None:
            self._trigger.start()

    # --- the planner's side ---------------------------------------------------

    def control(self) -> str:
        """`policy`, `operator`, or `switching` (also after a failed switch)."""
        state = self._state
        if state in (POLICY, OPERATOR):
            return state
        return "switching"

    def drain_events(self) -> list[dict[str, Any]]:
        """Switches finished since the last call, for the tick log."""
        with self._lock:
            events, self._events = self._events, []
        return events

    def gripper_targets(self) -> dict[str, float]:
        """The width the leader trigger last commanded, by arm side."""
        reader = getattr(self._teleop, "gripper_targets_by_index", None)
        if not callable(reader):
            return {}
        try:
            by_pair = reader()
        except Exception:  # pragma: no cover - hardware specific
            return {}
        return {
            side: by_pair[pair]
            for side, pair in zip(self._sides, self._lease.pairs, strict=False)
            if pair in by_pair
        }

    # --- requests -------------------------------------------------------------

    def _refusal(self, wanted: str) -> str | None:
        state = self._state
        if self._closed:
            return "The rollout is ending"
        if state in (TO_OPERATOR, TO_POLICY):
            return "A switch is under way"
        if wanted == OPERATOR:
            if state == OPERATOR:
                return "The operator already has control"
            if state == ERROR:
                return "The last switch failed; hand back or stop the trial"
            runner = self._runner
            if runner is None or not runner.ready:
                return "The policy has not sent its first command yet"
            return None
        if state == POLICY:
            return "The policy already has control"
        return None

    def take_over(self) -> dict[str, Any]:
        """Stop the policy now and give the follower to the leader arm."""
        with self._lock:
            refusal = self._refusal(OPERATOR)
            if refusal is not None:
                raise TakeoverRefused(refusal)
            self._state = TO_OPERATOR
            self._error = None
            self._count += 1
            started = self._clock()
            self._intervention_start = started
        self._append_log("INFO", "TAKEOVER", "Switching to the operator")
        try:
            # Here rather than on the switch thread: the policy stops before
            # this returns, whatever happens to the rest of the switch.
            assert self._runner is not None
            self._runner.pause_policy()
        except Exception as exc:
            self._failed(OPERATOR, started, exc)
            return self.status()
        self._spawn(self._to_operator, started)
        return self.status()

    def hand_back(self) -> dict[str, Any]:
        """Give the follower back to the policy, which plans afresh."""
        with self._lock:
            refusal = self._refusal(POLICY)
            if refusal is not None:
                raise TakeoverRefused(refusal)
            self._state = TO_POLICY
            self._error = None
            started = self._clock()
        self._append_log("INFO", "TAKEOVER", "Switching to the policy")
        self._spawn(self._to_policy, started)
        return self.status()

    def toggle(self) -> None:
        """The leader trigger: take over under the policy, hand back otherwise."""
        state = self._state
        if state == POLICY:
            self.take_over()
        elif state in (OPERATOR, ERROR):
            self.hand_back()

    def _spawn(self, target: Callable[[float], None], started: float) -> None:
        worker = threading.Thread(
            target=target, args=(started,), name="rollout-takeover", daemon=True
        )
        self._worker = worker
        worker.start()

    # --- the switches -----------------------------------------------------------

    def _to_operator(self, started: float) -> None:
        with self._switch_lock:
            if self._closed:
                return
            try:
                self._teleop.resume_for_takeover(
                    self._lease, max_contact_wrench=self._max_contact_wrench
                )
            except Exception as exc:
                self._failed(OPERATOR, started, exc)
                return
            self._switched(OPERATOR, started)

    def _to_policy(self, started: float) -> None:
        with self._switch_lock:
            if self._closed:
                return
            try:
                self._teleop.return_from_takeover(self._lease)
                assert self._runner is not None
                self._runner.reclaim()
                if self._closed:
                    raise RuntimeError("the trial ended during the switch")
                self._runner.resume_policy()
            except Exception as exc:
                self._failed(POLICY, started, exc)
                return
            self._switched(POLICY, started)

    def _switched(self, to: str, started: float) -> None:
        now = self._clock()
        seconds = round(now - started, 3)
        event = {"to": to, "s": seconds}
        with self._lock:
            self._state = to
            self._switches.append(event)
            self._events.append(dict(event))
            if to == POLICY and self._intervention_start is not None:
                self._intervention_s += now - self._intervention_start
                self._intervention_start = None
        label = "operator" if to == OPERATOR else "policy"
        self._append_log(
            "OK", "TAKEOVER", f"The {label} has control", f"switch took {seconds:.3f} s"
        )

    def _failed(self, to: str, started: float, exc: Exception) -> None:
        detail = describe_exception(exc)
        # The robots must be left holding: stop the teleop loop if it runs.
        try:
            self._teleop.return_from_takeover(self._lease)
        except Exception as stop_exc:
            detail = f"{detail}; stopping teleoperation also failed: " + (
                describe_exception(stop_exc)
            )
        seconds = round(self._clock() - started, 3)
        event = {"to": to, "s": seconds, "error": detail}
        with self._lock:
            self._state = ERROR
            self._error = detail
            self._switches.append(event)
            self._events.append(dict(event))
        warn("Takeover switch failed", detail)
        self._append_log(
            "ERROR",
            "TAKEOVER",
            f"Switch to the {'operator' if to == OPERATOR else 'policy'} failed",
            f"{detail}; the robots hold and the policy stays stopped",
        )

    # --- the end of the trial ---------------------------------------------------

    def close(self) -> None:
        """Disengage and stop teleoperation (the robots hold); refuse requests.

        Called by the planner as the trial ends for any reason, including a
        safety stop while the operator has control.
        """
        with self._lock:
            self._closed = True
        if self._trigger is not None:
            self._trigger.stop()
        acquired = self._switch_lock.acquire(timeout=CLOSE_TIMEOUT_S)
        try:
            if self._state != POLICY:
                try:
                    self._teleop.return_from_takeover(self._lease)
                except Exception as exc:
                    warn("Teleoperation did not stop cleanly", describe_exception(exc))
            with self._lock:
                if self._intervention_start is not None:
                    self._intervention_s += self._clock() - self._intervention_start
                    self._intervention_start = None
        finally:
            if acquired:
                self._switch_lock.release()

    # --- reporting --------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            last = dict(self._switches[-1]) if self._switches else None
            closed = self._closed
            error = self._error
            count = self._count
        action: str | None = None
        reason: str | None = None
        if closed:
            reason = "The rollout is not running"
        elif state in (POLICY,):
            action = "take_over"
            reason = self._refusal(OPERATOR)
        elif state in (OPERATOR, ERROR):
            action = "hand_back"
        else:
            reason = "A switch is under way"
        return {
            "available": True,
            "state": state,
            "control": self.control(),
            "error": error,
            "count": count,
            "last_switch": last,
            "action": action if reason is None else None,
            "reason": reason,
            "trigger_channel": (
                self._trigger.channel if self._trigger is not None else None
            ),
        }

    def summary(self) -> dict[str, Any]:
        """For the outcome: how often, how long, and every switch's duration."""
        with self._lock:
            seconds = self._intervention_s
            if self._intervention_start is not None:
                seconds += self._clock() - self._intervention_start
            return {
                "count": self._count,
                "seconds": round(seconds, 3),
                "switch_s": [dict(event) for event in self._switches],
            }
