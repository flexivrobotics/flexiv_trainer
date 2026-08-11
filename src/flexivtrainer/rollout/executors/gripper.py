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
from typing import Any, Literal

from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.runtime.gripper_session import (
    GripperIdentity,
    GripperInitializationRegistry,
)

try:
    import flexivrdk
except ImportError:  # pragma: no cover - environment-specific
    flexivrdk = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class GripperExecutor:
    """Own configured follower grippers and execute latest-only policy targets."""

    DEFAULT_COMMAND_HZ = 30.0
    MAX_COMMAND_HZ = 30.0
    DUPLICATE_TOLERANCE_M = 0.0005
    FORCE_FRACTION = 0.25
    INIT_SETTLE_S = 10.0
    CLOSE_THRESHOLD = 0.6
    OPEN_THRESHOLD = 0.4

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
        target_mode: Literal["width", "close"] = "width",
        failure_event: threading.Event | None = None,
        default_width_m: float | None = None,
        followers: Sequence[str] | None = None,
        initialization_registry: GripperInitializationRegistry | None = None,
        append_log: Callable[..., None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        if len(robots) != len(sides):
            raise ValueError("Robot and side counts must match")
        if followers is not None and len(followers) != len(sides):
            raise ValueError("Follower serial and side counts must match")
        if initialization_registry is not None and followers is None:
            raise ValueError(
                "Follower serials are required with an initialization registry"
            )
        if not math.isfinite(command_hz) or not 0 < command_hz <= self.MAX_COMMAND_HZ:
            raise ValueError(
                f"command_hz must be in (0, {self.MAX_COMMAND_HZ:g}]"
            )
        if default_width_m is not None and (
            not math.isfinite(default_width_m) or default_width_m < 0
        ):
            raise ValueError("default_width_m must be finite and nonnegative")
        if target_mode not in {"width", "close"}:
            raise ValueError(f"Unsupported gripper target mode: {target_mode}")

        self._robots = dict(zip(sides, robots, strict=True))
        self._followers = (
            dict(zip(sides, followers, strict=True)) if followers is not None else {}
        )
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
        self._target_mode = target_mode
        self._failure_event = failure_event
        self._default_width_m = default_width_m
        self._initialization_registry = initialization_registry
        self._append_log = append_log
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
        self._last_close: dict[str, bool] = {}
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
        """Prepare every predicted gripper while its robot is idle."""
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

        registry = self._initialization_registry
        identities = {
            side: GripperIdentity(
                str(self._followers[side]).strip(),
                str(
                    self._config_value(self._configs[side], "gripper_model") or ""
                ).strip(),
            )
            for side in self._controlled_sides
            if registry is not None
        }
        claim = registry.claim(identities.values()) if registry is not None else None
        claimed = set(claim.initialize) if claim else set()
        reused = set(claim.reused) if claim else set()

        prepared_sides: list[str] = []
        initialized_sides: list[str] = []
        initialized_identities: list[GripperIdentity] = []
        errors: dict[str, str] = {}
        for side in self._controlled_sides:
            config = self._configs[side]
            robot = self._robots[side]
            model = self._config_value(config, "gripper_model")
            identity = identities.get(side)
            should_initialize = registry is None or identity in claimed
            try:
                gripper = self._gripper_factory(robot)
                gripper.Enable(model)
                self._tool_factory(robot).Switch(model)
                if should_initialize:
                    gripper.Init()
                params = gripper.params()
                self._grippers[side] = gripper
                self._params[side] = params
                prepared_sides.append(side)
                if should_initialize:
                    initialized_sides.append(side)
                    if identity is not None:
                        initialized_identities.append(identity)
                elif identity in reused:
                    self._log_session(
                        "Preserved gripper width for session handoff",
                        side,
                        identity,
                    )
            except Exception as exc:
                message = describe_exception(exc)
                if identity in reused:
                    message = self._with_reinitialize_instruction(message)
                errors[side] = message
                if registry is not None and identity in claimed:
                    registry.fail([identity])

        wait_for_initialization = bool(initialized_sides) and (
            registry is not None or self._default_width_m is not None
        )
        if wait_for_initialization:
            wait_event = self._failure_event or self._stop_event
            if self._wait(wait_event, self.INIT_SETTLE_S):
                if registry is not None:
                    registry.fail(initialized_identities)
                for side in initialized_sides:
                    errors.setdefault(side, "Gripper initialization was cancelled")
                    if side in prepared_sides:
                        prepared_sides.remove(side)
                initialized_sides = []
                initialized_identities = []
            elif registry is not None:
                registry.complete(initialized_identities)
                for side, identity in zip(
                    initialized_sides, initialized_identities, strict=True
                ):
                    self._log_session(
                        "Initialized gripper for backend session", side, identity
                    )

        # complete() has moved successful claims to READY; fail() deliberately
        # leaves those entries intact while releasing partial failures.
        if registry is not None:
            registry.fail(claimed - set(initialized_identities))

        if self._default_width_m is not None:
            for side in initialized_sides:
                params = self._params[side]
                requested = float(self._default_width_m)
                width = _clamp(
                    requested, float(params.min_width), float(params.max_width)
                )
                if width != requested:
                    warn(
                        f"Clamped rollout gripper startup width for {side}",
                        f"requested={requested:.4f} clamped={width:.4f} "
                        f"range=[{float(params.min_width):.4f}, "
                        f"{float(params.max_width):.4f}]",
                    )
                velocity = float(params.max_vel)
                force = _clamp(
                    float(params.max_force) * self.FORCE_FRACTION,
                    float(params.min_force),
                    float(params.max_force),
                )
                try:
                    with self._io_locks[side]:
                        self._grippers[side].Move(width, velocity, force)
                    self._last_sent[side] = width
                    self._log_session(
                        "Moved gripper to session default width",
                        side,
                        identities.get(side),
                        detail=f"width={width:.4f} m",
                    )
                except Exception as exc:
                    message = describe_exception(exc)
                    if identities.get(side) in reused:
                        message = self._with_reinitialize_instruction(message)
                    errors[side] = message

        if errors:
            detail = "; ".join(
                f"{side}: {message}" for side, message in errors.items()
            )
            raise RuntimeError(f"Gripper preparation failed: {detail}")
        try:
            self._refresh_states()
        except Exception as exc:
            message = describe_exception(exc)
            if reused:
                message = self._with_reinitialize_instruction(message)
            raise RuntimeError(message) from exc

    @staticmethod
    def _with_reinitialize_instruction(message: str) -> str:
        return f"{message}. Reinitialize the gripper from teleop and retry rollout"

    def _log_session(
        self,
        message: str,
        side: str,
        identity: GripperIdentity | None,
        *,
        detail: str = "",
    ) -> None:
        if self._append_log is None:
            return
        identity_detail = identity.describe() if identity is not None else ""
        combined = " ".join(part for part in (identity_detail, detail) if part)
        self._append_log("INFO", "GRIPPER", f"{message}: {side}", combined)

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

    def submit(self, targets: Mapping[str, float]) -> None:
        """Replace pending policy targets without waiting for hardware I/O."""
        updates: dict[str, float] = {}
        for side, value in targets.items():
            if side not in self._configs:
                raise ValueError(f"Unknown controlled gripper side: {side}")
            target = float(value)
            if not math.isfinite(target):
                raise ValueError(f"Gripper target must be finite: {side}")
            updates[side] = target
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
        for side, target in pending.items():
            params = self._params[side]
            requested_width = self._decode_target(side, target, params)
            if requested_width is None:
                continue
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

    def _decode_target(self, side: str, target: float, params: Any) -> float | None:
        if self._target_mode == "width":
            return target
        with self._lock:
            close = self._last_close.get(side)
            if target >= self.CLOSE_THRESHOLD:
                close = True
            elif target <= self.OPEN_THRESHOLD:
                close = False
            elif close is None:
                return None
            self._last_close[side] = close
        if close:
            return float(params.min_width)
        requested = self._default_width_m
        return float(params.max_width) if requested is None else float(requested)

    def describe_target(self, side: str, target: float) -> str:
        if self._target_mode == "width":
            return f"width={target:.4f}"
        with self._lock:
            close = self._last_close.get(side)
        if target >= self.CLOSE_THRESHOLD:
            return "close"
        if target <= self.OPEN_THRESHOLD:
            return "open"
        if close is None:
            return "preserve"
        return "close" if close else "open"

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


def initialize_gripper_executor(
    robots: Sequence[Any],
    sides: Sequence[str],
    configs: Mapping[str, Any],
    controlled_sides: Collection[str],
    *,
    failure_event: threading.Event,
    default_width_m: float | None = None,
    followers: Sequence[str] | None = None,
    initialization_registry: GripperInitializationRegistry | None = None,
    append_log: Callable[..., None] | None = None,
    target_mode: Literal["width", "close"] = "width",
    executor_factory: Callable[..., GripperExecutor] = GripperExecutor,
) -> GripperExecutor | None:
    """Create and initialize predicted grippers while their robots are IDLE."""

    if not controlled_sides:
        return None
    kwargs: dict[str, Any] = {"failure_event": failure_event}
    if target_mode != "width":
        kwargs["target_mode"] = target_mode
    if default_width_m is not None:
        kwargs["default_width_m"] = default_width_m
    if initialization_registry is not None:
        kwargs.update(
            followers=followers,
            initialization_registry=initialization_registry,
            append_log=append_log,
        )
    executor = executor_factory(robots, sides, configs, controlled_sides, **kwargs)
    try:
        executor.initialize()
    except Exception:
        executor.stop()
        raise
    return executor
