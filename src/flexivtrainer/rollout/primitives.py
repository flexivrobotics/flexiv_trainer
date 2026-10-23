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

"""Generic moves of the tool, the primitives a chain is built from.

A primitive knows nothing of the task. It moves the tool in straight lines,
and the jaws, each move one of:

    {"tool_m": [0, 0, -0.02]}       by an offset in the tool's own frame
    {"base_m": [0, 0, 0.03]}        by an offset in the arm's base frame
    {"box_min_m": [...], "box_max_m": [...]}
                                    into a box in the base frame, to the point
                                    of it nearest the tool (a point, when the
                                    corners are equal)
    {"gripper_m": 0.08}             the jaws to a width (0: closed as far as
                                    they go)

A tool move may also turn the tool to an orientation (`rotation`, [qw, qx, qy,
qz]) and set its own speed. Each is sent once, slowly, with every axis
motion-controlled, and the arm is read until it is there.

A jaw move may set its speed (`gripper_speed_m_s`) and the force it grips with
(`gripper_force_n`); the defaults are the gripper's, as in a rollout. It is
sent once, and the jaws are read until they stop: at the width, or closed on
something, which a `gripper_width` check then tells apart. The arm holds still
meanwhile.

A FORCE LIMIT STOPS A PRIMITIVE. One that meets more force than its
`max_force_n` -- a part half latched, a jam -- holds where it is and gives up
(`MoveBlocked`): it is for the operator, not for more force.

After its moves, a primitive can watch checks for a while (`check_s`): the
arm is read and each check scored, as in a rollout, until all have passed or
the time is up. A chain asks this way whether a step may start.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from flexivtrainer.data.lerobot_io import POSE_FORMAT_ROTATION_6D
from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.practice.space import Command, Sample
from flexivtrainer.rollout import hardware, observations
from flexivtrainer.rollout.live_checks import LiveChecks
from flexivtrainer.rollout.safety import SafetyEnvelope, SafetyStop

__all__ = [
    "Move",
    "MoveBlocked",
    "MoveInterrupted",
    "Primitive",
    "PrimitiveRunner",
    "move_jaws",
    "move_tool",
]

#: How often a primitive reads the arm while it waits (Hz).
_READ_HZ = 50.0
#: The jaws are at their width within this (m) ...
_JAW_TOLERANCE_M = 0.0005
#: ... or have stopped, once they move less than this (m) for this long (s).
_JAW_STILL_M = 0.0002
_JAW_SETTLE_S = 0.3


class MoveInterrupted(RuntimeError):
    """Someone else took the arm while it was being moved."""


class MoveBlocked(RuntimeError):
    """A move met more force than its limit, and stopped."""

    def __init__(self, force_n: float, limit_n: float, what: str) -> None:
        super().__init__(
            f"{force_n:.1f} N on the tool while it {what}, over the {limit_n:g} N limit"
        )
        self.force_n = force_n
        self.limit_n = limit_n


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} is [x, y, z] in metres, got {value!r}")
    numbers = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numbers  # type: ignore[return-value]


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


@dataclass(frozen=True)
class Move:
    """One move: exactly one of `tool_m`, `base_m`, the box
    `box_min_m`..`box_max_m`, or `gripper_m` (see the module's docstring)."""

    tool_m: tuple[float, float, float] | None = None
    base_m: tuple[float, float, float] | None = None
    box_min_m: tuple[float, float, float] | None = None
    box_max_m: tuple[float, float, float] | None = None
    #: The tool's orientation at the end, [qw, qx, qy, qz]; None keeps it.
    rotation: tuple[float, float, float, float] | None = None
    speed_m_s: float = 0.02
    #: The jaws' width (m), and how they get there; None: the gripper's own.
    gripper_m: float | None = None
    gripper_speed_m_s: float | None = None
    gripper_force_n: float | None = None

    def __post_init__(self) -> None:
        for name in ("tool_m", "base_m", "box_min_m", "box_max_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _triple(value, name))
        if (self.box_min_m is None) != (self.box_max_m is None):
            raise ValueError("a box needs both box_min_m and box_max_m")
        given = [
            self.tool_m is not None,
            self.base_m is not None,
            self.box_min_m is not None,
            self.gripper_m is not None,
        ]
        if sum(given) != 1:
            raise ValueError("a move is one of tool_m, base_m, a box or gripper_m")
        if self.gripper_m is not None:
            width = float(self.gripper_m)
            if not (math.isfinite(width) and width >= 0):
                raise ValueError(f"gripper_m must be 0 or more, got {self.gripper_m!r}")
            object.__setattr__(self, "gripper_m", width)
            if self.rotation is not None:
                raise ValueError("a jaw move does not turn the tool")
        for name in ("gripper_speed_m_s", "gripper_force_n"):
            value = getattr(self, name)
            if value is not None:
                if self.gripper_m is None:
                    raise ValueError(f"{name} belongs to a jaw move (gripper_m)")
                object.__setattr__(self, name, _positive(value, name))
        if self.box_min_m is not None and any(
            a > b for a, b in zip(self.box_min_m, self.box_max_m, strict=True)
        ):
            raise ValueError(
                f"the box {list(self.box_min_m)}..{list(self.box_max_m)} is not a box"
            )
        if self.rotation is not None:
            if len(self.rotation) != 4:
                raise ValueError("rotation is [qw, qx, qy, qz]")
            object.__setattr__(self, "rotation", tuple(float(v) for v in self.rotation))
        object.__setattr__(self, "speed_m_s", _positive(self.speed_m_s, "speed_m_s"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Move:
        if not isinstance(data, Mapping):
            raise ValueError("a move is an object")
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown move settings {unknown}")
        return cls(**dict(data))

    def target(self, here: Sample) -> tuple[np.ndarray, np.ndarray]:
        """Where the move takes the tool from `here`: position and rotation."""
        if self.tool_m is not None:
            position = here.position + here.rotation @ np.asarray(self.tool_m)
        elif self.base_m is not None:
            position = here.position + np.asarray(self.base_m)
        else:
            position = np.clip(here.position, self.box_min_m, self.box_max_m)
        if self.rotation is None:
            return position, here.rotation
        w, x, y, z = self.rotation
        return position, Rotation.from_quat([x, y, z, w]).as_matrix()

    @property
    def moves_jaws(self) -> bool:
        return self.gripper_m is not None

    def describe(self) -> str:
        if self.gripper_m is not None:
            return f"moved the jaws to {self.gripper_m * 1000:g} mm"
        if self.tool_m is not None:
            return f"moved {_mm(self.tool_m)} in the tool frame"
        if self.base_m is not None:
            return f"moved {_mm(self.base_m)} in the base frame"
        return "went into its box"


def _mm(offset: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v * 1000:g}" for v in offset) + "] mm"


@dataclass(frozen=True)
class Primitive:
    """Moves in order, with a force limit; then, optionally, checks."""

    moves: tuple[Move, ...] = ()
    #: More force than this on the tool (N, the force's size) stops the
    #: primitive; None leaves only the safety envelope's limit.
    max_force_n: float | None = None
    #: When the arm counts as there.
    tolerance_m: float = 0.001
    tolerance_deg: float = 0.5
    #: How long one move may take (s).
    timeout_s: float = 20.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "moves", tuple(self.moves))
        if self.max_force_n is not None:
            _positive(self.max_force_n, "max_force_n")
        for name in ("tolerance_m", "tolerance_deg", "timeout_s"):
            _positive(getattr(self, name), name)

    @property
    def moves_jaws(self) -> bool:
        return any(move.moves_jaws for move in self.moves)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Primitive:
        if not isinstance(data, Mapping):
            raise ValueError("a primitive is an object")
        values = dict(data)
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown primitive settings {unknown}")
        moves = values.pop("moves", [])
        if isinstance(moves, str | Mapping) or not isinstance(moves, Sequence):
            raise ValueError("moves is a list of moves")
        return cls(moves=tuple(Move.from_dict(m) for m in moves), **values)


def move_tool(
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    read: Callable[[], Sample],
    send: Callable[[Sequence[float], tuple[float, float, float, float]], bool],
    stop: threading.Event,
    speed_m_s: float,
    tolerance_m: float,
    tolerance_deg: float,
    timeout_s: float,
    what: str,
    goal: str = "the target",
    safety: SafetyEnvelope | None = None,
    max_force_n: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Send the tool to `position`, `rotation` once, and read until it is there.

    Raises MoveInterrupted when the send is refused (someone else has the
    arm), MoveBlocked when the tool feels more than `max_force_n` (the arm is
    told to hold where it was measured), and RuntimeError when it does not get
    there within `timeout_s`. `what` says what the arm was doing, for messages.
    """
    safety = safety or SafetyEnvelope()
    safety.check_workspace([[np.asarray(position).tolist()]])
    limits = (speed_m_s, 0.5, 0.5, 1.0)
    pose = Command(np.asarray(position), rotation, np.zeros(3)).pose()
    if not send(pose, limits):
        raise MoveInterrupted(f"the arm was taken while it {what}")
    deadline = clock() + timeout_s
    while not stop.is_set():
        here = read()
        if max_force_n is not None:
            force = float(np.linalg.norm(here.wrench[:3]))
            if force > max_force_n:
                send(Command(here.position, here.rotation, np.zeros(3)).pose(), limits)
                raise MoveBlocked(force, max_force_n, what)
        off = float(np.linalg.norm(here.position - position))
        turned = Rotation.from_matrix(here.rotation.T @ rotation).magnitude()
        if off <= tolerance_m and math.degrees(turned) <= tolerance_deg:
            return
        if clock() > deadline:
            raise RuntimeError(
                f"the arm did not reach {goal} within {timeout_s:g} s while it "
                f"{what}: {off * 1000:.1f} mm and {math.degrees(turned):.1f} deg away"
            )
        stop.wait(1.0 / _READ_HZ)


def move_jaws(
    gripper: Any,
    side: str,
    move: Move,
    *,
    stop: threading.Event,
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Send the jaws toward `move`'s width once, through the gripper executor
    (`executors/gripper.py`), and read them until they stop: at the width, or
    closed on something. Returns the width they stopped at."""
    assert move.gripper_m is not None
    target, _, _ = gripper.move_now(
        side,
        move.gripper_m,
        velocity=move.gripper_speed_m_s,
        force=move.gripper_force_n,
    )
    started = clock()
    last: float | None = None
    still_since: float | None = None
    while not stop.is_set():
        now = clock()
        width = float(gripper.read_states()[side]["width"])
        if abs(width - target) <= _JAW_TOLERANCE_M:
            return width
        if last is not None and abs(width - last) < _JAW_STILL_M:
            still_since = now if still_since is None else still_since
        else:
            still_since = None
        last = width
        if (
            still_since is not None
            and now - still_since >= _JAW_SETTLE_S
            and gripper.moving(side) is not True
        ):
            return width
        if now - started > timeout_s:
            raise RuntimeError(
                f"the jaws did not stop within {timeout_s:g} s: at "
                f"{width * 1000:.1f} mm, heading for {target * 1000:.1f} mm"
            )
        stop.wait(1.0 / _READ_HZ)
    return last if last is not None else target


class PrimitiveRunner:
    """Own the thread that runs one primitive on one arm.

    It has the lifecycle of a rollout runner -- the service starts it, stops
    it, and is told when it has finished and why -- but no policy. Its stop
    reasons: "done", "checks passed", "check failed", "blocked", "safety",
    "stopped".
    """

    def __init__(
        self,
        *,
        primitive: Primitive,
        robots: list[Any],
        sides: list[str],
        followers: list[str],
        cameras: Any,
        camera_names: list[str],
        stop_event: threading.Event,
        append_log: Callable[..., None],
        on_error: Callable[[str], None],
        on_cleanup_error: Callable[[str], None],
        on_finished: Callable[[str | None, int], None],
        release_robots: Callable[[], None],
        stop_robots: Callable[[list[Any]], None],
        safety: SafetyEnvelope | None = None,
        cartesian_stiffness: Any = None,
        max_contact_wrench: Any = None,
        live_checks: LiveChecks | None = None,
        check_s: float = 1.0,
        image_resolutions: dict[str, tuple[int, int]] | None = None,
        open_gripper: Callable[[], Any] | None = None,
    ) -> None:
        """`open_gripper` prepares the arm's gripper executor, while the arm
        is idle; given when the primitive moves the jaws or a check reads them."""
        if len(robots) != 1 or len(sides) != 1 or len(followers) != 1:
            raise ValueError(
                f"a primitive moves one arm; {len(robots)} are configured "
                "(set single-arm mode)"
            )
        self._primitive = primitive
        self._robot = robots[0]
        self._robots = robots
        self._sides = list(sides)
        self._follower = followers[0]
        self._cameras = cameras
        self._camera_names = list(camera_names)
        self._stop_event = stop_event
        self._append_log = append_log
        self._on_error = on_error
        self._on_cleanup_error = on_cleanup_error
        self._on_finished = on_finished
        self._release_robots = release_robots
        self._stop_robots = stop_robots
        self._safety = safety or SafetyEnvelope()
        self._live_checks = live_checks or None
        self._check_s = check_s
        self._image_resolutions = image_resolutions
        if primitive.moves_jaws and open_gripper is None:
            raise ValueError("the primitive moves the jaws, but has no gripper")
        self._side = sides[0]
        self._gripper: Any = None
        #: Set when a move met more than the force limit: {"force_n", "limit_n"}.
        self.blocked: dict[str, float] | None = None
        #: Set when the envelope stopped it: {"code", "detail"}.
        self.safety_stop: dict[str, str] | None = None
        self._stop_reason: str | None = None
        self._thread: threading.Thread | None = None
        try:
            if open_gripper is not None:
                # Before the mode switch: a gripper is prepared while idle.
                self._gripper = open_gripper()
            # Every axis motion-controlled, and the force sensor left as it
            # is: a back-off starts in contact, where zeroing would offset it.
            hardware.prepare_robot_motion(
                self._robot,
                self._follower,
                stop_event,
                append_log,
                stiffness=cartesian_stiffness,
                max_contact_wrench=max_contact_wrench,
                zero_ft=False,
                force_axes=(False,) * 6,
            )
        except Exception as exc:
            if self._gripper is not None:
                self._gripper.stop()
            stop_robots(robots)
            raise RuntimeError(
                f"Failed to prepare the arm for a primitive: {describe_exception(exc)}"
            ) from exc

    # --- the service's side -------------------------------------------------

    def start(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True, name="primitive")
        self._thread = thread
        thread.start()

    def stop(self, timeout: float = 2.0) -> list[str]:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                return ["The primitive did not stop cleanly"]
        self._thread = None
        return []

    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # --- the primitive -------------------------------------------------------

    def _run(self) -> None:
        done = 0
        try:
            primitive = self._primitive
            for move in primitive.moves:
                if self._stop_event.is_set():
                    break
                if move.moves_jaws:
                    move_jaws(
                        self._gripper,
                        self._side,
                        move,
                        stop=self._stop_event,
                        timeout_s=primitive.timeout_s,
                    )
                    done += 1
                    continue
                position, rotation = move.target(self._read())
                move_tool(
                    position,
                    rotation,
                    read=self._read,
                    send=self._send,
                    stop=self._stop_event,
                    speed_m_s=move.speed_m_s,
                    tolerance_m=primitive.tolerance_m,
                    tolerance_deg=primitive.tolerance_deg,
                    timeout_s=primitive.timeout_s,
                    what=move.describe(),
                    safety=self._safety,
                    max_force_n=primitive.max_force_n,
                )
                done += 1
            if self._stop_event.is_set():
                self._stop_reason = "stopped"
            elif self._live_checks is not None:
                self._stop_reason = (
                    "checks passed" if self._watch_checks() else "check failed"
                )
            else:
                self._stop_reason = "done"
        except MoveBlocked as exc:
            self.blocked = {"force_n": exc.force_n, "limit_n": exc.limit_n}
            self._stop_reason = "blocked"
            self._append_log("WARN", "PRIMITIVE", "Stopped by force", str(exc))
        except SafetyStop as exc:
            self._stop_reason = "safety"
            self.safety_stop = {"code": exc.code, "detail": exc.detail}
            self._on_error(str(exc))
        except Exception as exc:
            detail = describe_exception(exc)
            self._on_error(detail)
            warn("Primitive stopped", detail)
        finally:
            self._stop_event.set()
            try:
                if self._gripper is not None:
                    self._gripper.stop()
                self._release_robots()
            except Exception as exc:  # pragma: no cover - defensive
                self._on_cleanup_error(describe_exception(exc))
            self._on_finished(self._stop_reason, done)

    def _watch_checks(self) -> bool:
        """Read the arm and score every check until all have passed or
        `check_s` is up."""
        live = self._live_checks
        assert live is not None
        live.reset()
        deadline = time.monotonic() + self._check_s
        while not self._stop_event.is_set():
            t = time.monotonic()
            snapshot = self._snapshot()
            images = (
                observations.grab_images(
                    self._cameras, self._camera_names, self._image_resolutions
                )
                if live.needs_frames
                else {}
            )
            state = (
                observations.build_observation(
                    snapshot,
                    images,
                    self._sides,
                    self._camera_names,
                    POSE_FORMAT_ROTATION_6D,
                ).get("observation.state", [])
                if live.needs_frames
                else []
            )
            live.submit(
                t=t,
                state=state,
                images=images,
                snapshot=snapshot,
                pose_format=POSE_FORMAT_ROTATION_6D,
            )
            if live.all_passed():
                return True
            if t > deadline:
                return False
            self._stop_event.wait(1.0 / _READ_HZ)
        return False

    # --- the arm -------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        if self._robot.fault():
            raise RuntimeError("Fault occurred on the follower robot")
        snapshot = observations.read_robot_snapshot(
            [self._robot],
            self._gripper.read_states() if self._gripper is not None else None,
            self._sides,
        )
        states = snapshot["robots"]["robot_0"]["states"]
        self._safety.check_force([states["ext_wrench_in_world"]])
        return snapshot

    def _read(self) -> Sample:
        states = self._snapshot()["robots"]["robot_0"]["states"]
        return Sample.from_driver(
            states["tcp_pose"], states["tcp_vel"], states["ext_wrench_in_world"]
        )

    def _send(
        self, pose: Sequence[float], limits: tuple[float, float, float, float]
    ) -> bool:
        self._robot.SendCartesianMotionForce(list(pose), [0.0] * 6, [0.0] * 6, *limits)
        return True
