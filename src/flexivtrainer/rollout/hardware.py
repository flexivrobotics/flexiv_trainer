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

"""RDK robot construction and hardware preparation helpers for rollout."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from flexivtrainer.observability import describe_exception, warn

#: The default for `rollout.ft_zero_max_force_n`: above this measured force the
#: tool is taken to be touching something, and the F/T sensor is not zeroed.
FT_ZERO_MAX_FORCE_N = 1.5
#: The default for `rollout.force_axis_max_speed_m_s`.
FORCE_AXIS_MAX_SPEED_M_S = 0.02


def _default_robot_factory(serial: str) -> Any:
    import flexivrdk  # noqa: PLC0415

    return flexivrdk.Robot(serial)


def _rdk_mode() -> Any:
    import flexivrdk  # noqa: PLC0415

    return flexivrdk.Mode


def _rdk_coord_type() -> Any:
    import flexivrdk  # noqa: PLC0415

    return flexivrdk.CoordType


#: The force-control frame placed exactly on the TCP: [x, y, z, qw, qx, qy, qz].
_TCP_ITSELF = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def set_force_axes(
    robot: Any, axes: Sequence[bool], max_speed_m_s: float = FORCE_AXIS_MAX_SPEED_M_S
) -> None:
    """Control force along `axes` ([x, y, z, rx, ry, rz]) of the tool's own frame.

    The frame moves with the tool, so "z" stays the tool's z as it turns. The
    other axes stay motion-controlled. `max_speed_m_s` caps the speed along a
    linear force axis when contact is lost; the controller enforces it only
    under active force control, which is its default. Both calls block, and
    both need the Cartesian motion-force mode.
    """
    robot.SetForceControlFrame(_rdk_coord_type().TCP, list(_TCP_ITSELF))
    robot.SetForceControlAxis([bool(a) for a in axes], [float(max_speed_m_s)] * 3)


def clear_force_axes(robot: Any) -> None:
    """Every axis motion-controlled again, as a mode switch alone may not leave
    it: a later rollout that sets no force axes must not inherit these."""
    robot.SetForceControlAxis([False] * 6, [FORCE_AXIS_MAX_SPEED_M_S] * 3)


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


def measured_force_n(robot: Any) -> float | None:
    """The magnitude of the measured external force, or None if unreadable."""
    try:
        force = [float(v) for v in list(robot.states().ext_wrench_in_world)[:3]]
    except Exception:
        return None
    if len(force) != 3 or not all(math.isfinite(v) for v in force):
        return None
    return math.dist(force, (0.0, 0.0, 0.0))


def zero_ft_unless_in_contact(
    robot: Any,
    serial: str,
    stop_event: threading.Event,
    append_log: Callable[[str, str, str, str], None],
    max_force_n: float | None = FT_ZERO_MAX_FORCE_N,
) -> bool:
    """Zero the F/T sensor, unless the tool already measures over `max_force_n`.

    Zeroing takes whatever force is on the sensor as its new zero. Done while
    the tool rests or presses on something, it offsets every later reading by
    that force -- the force ceiling, a force-aware policy's input and a learned
    check all read it -- so it is skipped, with a warning that says how much
    was measured. A sensor that has drifted past the threshold in free air is
    skipped too, and the same warning shows it. `max_force_n` None always
    zeroes; a force that cannot be read does not block zeroing.
    """
    force = measured_force_n(robot) if max_force_n is not None else None
    if force is not None and force > max_force_n:
        detail = (
            f"{serial}: |F| = {force:.1f} N, above {max_force_n:g} N; "
            "the tool is touching something, so the sensor keeps its zero"
        )
        warn("F/T sensor not zeroed", detail)
        append_log("WARN", "ROLLOUT", "F/T sensor not zeroed", detail)
        return False
    if _zero_ft_sensor(robot, stop_event):
        append_log("INFO", "ROLLOUT", "F/T sensor zeroed", serial)
        return True
    return False


def connect_robot(
    robot_factory: Callable[[str], Any],
    serial: str,
    stop_event: threading.Event,
    *,
    prepare_motion: Callable[[Any, str], None] | None = None,
) -> Any:
    robot = robot_factory(serial)
    if robot.fault():
        robot.ClearFault()
    robot.Enable()
    while not robot.operational():
        if stop_event.wait(0.1):
            break
    if prepare_motion is not None:
        prepare_motion(robot, serial)
    return robot


def prepare_robot_motion(
    robot: Any,
    serial: str,
    stop_event: threading.Event,
    append_log: Callable[[str, str, str, str], None],
    *,
    stiffness: Sequence[float] | None = None,
    max_contact_wrench: Sequence[float] | None = None,
    zero_ft: bool = True,
    zero_ft_max_force_n: float | None = FT_ZERO_MAX_FORCE_N,
    force_axes: Sequence[bool] | None = None,
    force_axis_max_speed_m_s: float = FORCE_AXIS_MAX_SPEED_M_S,
) -> None:
    """Zero the F/T sensor, enter Cartesian motion-force mode, set the spring.

    The stiffness is set in the mode it applies to and before any pose is
    sent: the first target is already interpreted through it. `zero_ft=False`
    is for taking the arm back in the middle of a trial, when the tool may be
    in contact and zeroing would offset every later force reading; at a start,
    zeroing is skipped when the tool measures over `zero_ft_max_force_n`.
    `force_axes` puts those axes of the tool frame under force control
    (`set_force_axes`); None leaves the controller's setting alone.
    """
    if zero_ft:
        zero_ft_unless_in_contact(
            robot, serial, stop_event, append_log, zero_ft_max_force_n
        )
    mode = _rdk_mode()
    robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
    if stiffness is not None:
        robot.SetCartesianImpedance([float(v) for v in stiffness])
        append_log(
            "INFO", "ROLLOUT", "Cartesian stiffness set", f"{serial} {list(stiffness)}"
        )
    if max_contact_wrench is not None:
        robot.SetMaxContactWrench([float(v) for v in max_contact_wrench])
        append_log(
            "INFO",
            "ROLLOUT",
            "Contact wrench limit set",
            f"{serial} {list(max_contact_wrench)}",
        )
    if force_axes is not None:
        set_force_axes(robot, force_axes, force_axis_max_speed_m_s)
        axis_names = ("x", "y", "z", "rx", "ry", "rz")
        named = [n for n, on in zip(axis_names, force_axes, strict=True) if on]
        append_log(
            "INFO",
            "ROLLOUT",
            "Force control set",
            f"{serial} tool axes {named or 'none'}, "
            f"at most {force_axis_max_speed_m_s} m/s without contact",
        )


def stop_robots(robots: list[Any]) -> None:
    for robot in robots:
        try:
            stop = getattr(robot, "Stop", None)
            if callable(stop):
                stop()
        except Exception:  # pragma: no cover - hardware specific
            pass
