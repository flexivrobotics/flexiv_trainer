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

"""Non-blocking gripper command execution."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

try:
    import flexivrdk
except ImportError:  # pragma: no cover - environment-specific
    flexivrdk = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class GripperExecutor:
    """Own configured follower grippers and execute latest-only width targets."""

    DEFAULT_COMMAND_HZ = 30.0
    MAX_COMMAND_HZ = 30.0
    DUPLICATE_TOLERANCE_M = 0.0005
    FORCE_FRACTION = 0.25

    def __init__(
        self,
        robots: Sequence[Any],
        sides: Sequence[str],
        configs: Mapping[str, Any],
        controlled_sides: Collection[str],
        *,
        command_hz: float = DEFAULT_COMMAND_HZ,
        gripper_factory: Callable[[Any], Any] | None = None,
        tool_factory: Callable[[Any], Any] | None = None,
        idle_mode: Any = None,
        target_source: Callable[[], Mapping[str, float]] | None = None,
        failure_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        if len(robots) != len(sides):
            raise ValueError("Robot and side counts must match")
        if not math.isfinite(command_hz) or not 0 < command_hz <= self.MAX_COMMAND_HZ:
            raise ValueError(
                f"command_hz must be in (0, {self.MAX_COMMAND_HZ:g}]"
            )

        self._robots = dict(zip(sides, robots, strict=True))
        self._controlled_sides = tuple(dict.fromkeys(controlled_sides))
        self._configs = {
            side: self._config_for_side(configs, side)
            for side in self._controlled_sides
        }
        for side, config in self._configs.items():
            if side not in self._robots:
                raise ValueError(f"Controlled gripper side has no robot: {side}")
            if config is None or self._config_value(config, "follower") != "gripper":
                raise ValueError(
                    "Controlled gripper side has no configured follower "
                    f"gripper: {side}"
                )
            if not self._config_value(config, "gripper_model"):
                raise ValueError(f"Configured gripper has no model: {side}")

        if flexivrdk is None:
            default_gripper_factory = None
            default_tool_factory = None
            default_idle_mode = None
        else:
            default_gripper_factory = flexivrdk.Gripper
            default_tool_factory = flexivrdk.Tool
            default_idle_mode = flexivrdk.Mode.IDLE
        self._gripper_factory = gripper_factory or default_gripper_factory
        self._tool_factory = tool_factory or default_tool_factory
        self._idle_mode = default_idle_mode if idle_mode is None else idle_mode
        self._target_source = target_source
        self._failure_event = failure_event
        self._clock = clock
        self._wait = wait or (lambda event, timeout: event.wait(timeout))
        self._period = 1.0 / command_hz

        self._lock = threading.Lock()
        self._io_locks = {
            side: threading.Lock() for side in self._controlled_sides
        }
        self._grippers: dict[str, Any] = {}
        self._params: dict[str, Any] = {}
        self._pending: dict[str, float] = {}
        self._last_sent: dict[str, float] = {}
        self._measured: dict[str, dict[str, float]] = {}
        self._error: Exception | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _config_for_side(configs: Mapping[str, Any], side: str) -> Any:
        return configs.get(side)

    @staticmethod
    def _config_value(config: Any, name: str) -> Any:
        if isinstance(config, Mapping):
            return config.get(name)
        return getattr(config, name, None)

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def initialize(self) -> None:
        """Initialize every predicted gripper while its robot is idle."""
        if self._gripper_factory is None or self._tool_factory is None:
            raise RuntimeError("flexivrdk is unavailable; cannot control grippers")
        if self._idle_mode is None:
            raise RuntimeError("Flexiv IDLE mode is unavailable")

        for side in self._controlled_sides:
            robot = self._robots[side]
            mode = getattr(robot, "mode", None)
            if not callable(mode) or mode() != self._idle_mode:
                raise RuntimeError(
                    f"Follower robot must be IDLE to initialize gripper: {side}"
                )
            config = self._configs[side]
            gripper = self._gripper_factory(robot)
            model = self._config_value(config, "gripper_model")
            gripper.Enable(model)
            self._tool_factory(robot).Switch(model)
            gripper.Init()
            params = gripper.params()
            self._grippers[side] = gripper
            self._params[side] = params
        self._refresh_states()

    def measured_states(self) -> dict[str, dict[str, float]]:
        """Return measured width and force keyed by arm side."""
        with self._lock:
            missing = set(self._controlled_sides) - self._measured.keys()
            if missing:
                raise RuntimeError(
                    f"Gripper telemetry is unavailable: {', '.join(sorted(missing))}"
                )
            return {
                side: dict(self._measured[side])
                for side in self._controlled_sides
            }

    def submit(self, widths: Mapping[str, float]) -> None:
        """Replace pending targets without waiting for hardware I/O."""
        updates: dict[str, float] = {}
        for side, value in widths.items():
            if side not in self._configs:
                raise ValueError(f"Unknown controlled gripper side: {side}")
            width = float(value)
            if not math.isfinite(width):
                raise ValueError(f"Gripper width must be finite: {side}")
            updates[side] = width
        with self._lock:
            self._pending.update(updates)

    def start(self) -> None:
        if self._thread is not None:
            return
        missing = set(self._controlled_sides) - self._grippers.keys()
        if missing:
            raise RuntimeError(
                f"Grippers are not initialized: {', '.join(sorted(missing))}"
            )
        self._stop_event.clear()
        with self._lock:
            self._error = None
        self._thread = threading.Thread(
            target=self._run, name="rollout-gripper-executor", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError("Timed out stopping gripper executor")
        self._thread = None

    def _run(self) -> None:
        deadline = self._clock()
        while not self._stop_event.is_set():
            try:
                self._send_pending()
            except Exception as exc:
                with self._lock:
                    self._error = exc
                if self._failure_event is not None:
                    self._failure_event.set()
                self._stop_event.set()
                return

            deadline += self._period
            now = self._clock()
            if now >= deadline:
                deadline += (math.floor((now - deadline) / self._period) + 1) * (
                    self._period
                )
            if self._wait(self._stop_event, max(0.0, deadline - now)):
                return

    def _send_pending(self) -> None:
        with self._lock:
            pending = self._pending
            self._pending = {}
        if self._target_source is not None:
            pending.update(self._target_source())
        for side, requested_width in pending.items():
            params = self._params[side]
            width = _clamp(
                requested_width, float(params.min_width), float(params.max_width)
            )
            last_width = self._last_sent.get(side)
            if (
                last_width is not None
                and abs(width - last_width) < self.DUPLICATE_TOLERANCE_M
            ):
                continue
            velocity = float(params.max_vel)
            force = _clamp(
                float(params.max_force) * self.FORCE_FRACTION,
                float(params.min_force),
                float(params.max_force),
            )
            with self._io_locks[side]:
                self._grippers[side].Move(width, velocity, force)
            self._last_sent[side] = width
        self._refresh_states()

    def _refresh_states(self) -> None:
        measured: dict[str, dict[str, float]] = {}
        for side in self._controlled_sides:
            gripper = self._grippers.get(side)
            if gripper is None:
                raise RuntimeError(f"Gripper is not initialized: {side}")
            with self._io_locks[side]:
                state = gripper.states()
            measured[side] = {
                "width": float(state.width),
                "force": float(state.force),
            }
        with self._lock:
            self._measured = measured
