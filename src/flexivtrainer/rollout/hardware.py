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

import threading
import time
from collections.abc import Callable
from typing import Any

from flexivtrainer.observability import describe_exception, warn


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
) -> None:
    if _zero_ft_sensor(robot, stop_event):
        append_log("INFO", "ROLLOUT", "F/T sensor zeroed", serial)
    mode = _rdk_mode()
    robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)


def stop_robots(robots: list[Any]) -> None:
    for robot in robots:
        try:
            stop = getattr(robot, "Stop", None)
            if callable(stop):
                stop()
        except Exception:  # pragma: no cover - hardware specific
            pass
