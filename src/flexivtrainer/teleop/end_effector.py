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

"""End effector control during teleoperation.

Each arm pair can configure a leader *digital input* device (a trigger such as a
button wired to one of the leader robot's DI ports) and a follower end effector
(a digital output device or a gripper). While the teleop control loop is running
(this controller's mirror thread is started/stopped by teleop Start/Stop, not by
engage -- it keeps running across engage/disengage cycles), it polls every pair's
leader DI and mirrors its state onto the follower: the follower output/gripper is
driven to its configured "activated" state while the leader trigger is active,
and to the opposite state otherwise.

The configuration is the per-side :class:`EndEffectorSideConfig` cached in
``robot_serials.json`` and keyed by arm side ("left_arm"/"right_arm"/
"single_arm"); the side order matches the teleop pair index.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from flexivtrainer.config import EndEffectorSideConfig
from flexivtrainer.observability import describe_exception, info, warn
from flexivtrainer.runtime.gripper_session import (
    GripperIdentity,
    GripperInitializationRegistry,
)

try:
    import flexivrdk
except ImportError:  # pragma: no cover - environment-specific
    flexivrdk = None

Gripper = getattr(flexivrdk, "Gripper", None) if flexivrdk is not None else None
Tool = getattr(flexivrdk, "Tool", None) if flexivrdk is not None else None
Mode = getattr(flexivrdk, "Mode", None) if flexivrdk is not None else None


def _side_config(value: Any) -> EndEffectorSideConfig | None:
    """Coerce a cached entry (model or plain dict) to EndEffectorSideConfig."""
    if value is None:
        return None
    if isinstance(value, EndEffectorSideConfig):
        return value
    if isinstance(value, dict):
        try:
            return EndEffectorSideConfig(**value)
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return max(low, min(high, value))


class EndEffectorController:
    """Background thread that mirrors leader DIs onto follower end effectors."""

    # The poll rate only needs to track button presses, so a modest rate keeps
    # the (blocking) digital-output / gripper writes from saturating the bus.
    POLL_HZ = 30.0
    # Fraction of a gripper's max force used as the default grasping force until
    # the Gripper Control panel sets a value (matches the panel's own default).
    DEFAULT_FORCE_FRACTION = 0.25
    INIT_SETTLE_S = 10.0

    def __init__(
        self,
        controller: Any,
        sides: list[str],
        configs: dict[str, Any],
        default_gripper_width_m: float | None = None,
        *,
        gripper_identities: dict[str, GripperIdentity] | None = None,
        initialization_registry: GripperInitializationRegistry | None = None,
        init_settle_s: float = INIT_SETTLE_S,
    ) -> None:
        if not math.isfinite(init_settle_s) or init_settle_s < 0:
            raise ValueError("init_settle_s must be finite and nonnegative")
        self._controller = controller
        # Arm sides and their config in pair-index order. The side name at index
        # i drives teleop pair i (instances(i)/digital_inputs(i)); a None config
        # means that side configured nothing.
        self._sides: list[str] = list(sides)
        self._configs: list[EndEffectorSideConfig | None] = [
            _side_config(configs.get(side)) for side in self._sides
        ]
        self._default_gripper_width_m = default_gripper_width_m
        identities = gripper_identities or {}
        self._gripper_identities = [identities.get(side) for side in self._sides]
        self._initialization_registry = initialization_registry
        self._init_settle_s = init_settle_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._setup_lock = threading.Lock()
        # Grippers enabled by initialize_grippers() (the panel's Init button) and
        # the last commands issued, keyed by pair index, so we only issue a
        # (blocking) command on an actual state change.
        self._grippers: dict[int, Any] = {}
        self._gripper_params: dict[int, Any] = {}
        # User-set (velocity, force) per pair index from the Gripper Control
        # panel's sliders, applied to the mirror loop's Move() calls. Falls back
        # to the gripper's own params-derived defaults until a slider sets one.
        self._command_params: dict[int, tuple[float, float]] = {}
        self._last_do: dict[int, bool] = {}
        self._last_gripper_target: dict[int, str] = {}
        # A cached rollout -> teleop handoff must never open a held object. For
        # each reused, digitally controlled gripper, ignore Open until the
        # leader has requested Close once; then resume normal mirroring.
        self._takeover_pending: set[int] = set()
        self._errors: dict[int, str] = {}

    @staticmethod
    def _should_mirror(cfg: EndEffectorSideConfig | None) -> bool:
        # Mirroring needs a leader trigger to read and a follower effector to
        # drive. A gripper without a leader trigger is still enabled (for manual
        # control) but is not mirrored.
        return cfg is not None and cfg.leader == "digital_input" and cfg.follower in {
            "digital_output",
            "gripper",
        }

    def has_work(self) -> bool:
        # "Work" for the background thread is the set of mirrored pairs.
        return any(self._should_mirror(cfg) for cfg in self._configs)

    def has_grippers(self) -> bool:
        return any(
            cfg is not None and cfg.follower == "gripper" for cfg in self._configs
        )

    def _index_for_side(self, side: str) -> int:
        try:
            return self._sides.index(side)
        except ValueError as exc:
            raise ValueError(f"Unknown arm side: {side}") from exc

    def is_running(self) -> bool:
        return self._thread is not None

    def initialize_grippers(self, *, force: bool = False) -> dict[str, Any]:
        """Prepare every configured gripper and initialize each device once.

        RDK handles remain local to this teleop connection. The optional shared
        registry decides whether this connection needs to issue the mechanical
        ``Init()`` cycle or can reuse initialization performed by a rollout.
        """

        with self._setup_lock:
            return self._initialize_grippers_locked(force=force)

    def _initialize_grippers_locked(self, *, force: bool) -> dict[str, Any]:
        entries = [
            (index, cfg, self._gripper_identities[index])
            for index, cfg in enumerate(self._configs)
            if cfg is not None and cfg.follower == "gripper"
        ]
        if force:
            for index, _, _ in entries:
                _, follower = self._controller.instances(index)
                if not self._follower_is_idle(follower):
                    side = self._sides[index]
                    raise RuntimeError(
                        f"Follower robot must be IDLE to reinitialize gripper: {side}"
                    )

        registry = self._initialization_registry
        identities = [identity for _, _, identity in entries if identity is not None]
        if registry is not None and len(identities) != len(entries):
            raise RuntimeError(
                "A configured gripper is missing its follower serial or model"
            )
        claim = registry.claim(identities, force=force) if registry else None
        claimed = set(claim.initialize) if claim else set()
        reused = set(claim.reused) if claim else set()

        errors: dict[str, str] = {}
        initialized_identities: list[GripperIdentity] = []
        initialized_sides: list[str] = []
        reused_sides: list[str] = []
        for index, cfg, identity in entries:
            should_initialize = registry is None or identity in claimed
            try:
                self._setup_gripper(index, cfg, initialize=should_initialize)
                self._errors.pop(index, None)
                side = self._sides[index]
                if should_initialize:
                    self._takeover_pending.discard(index)
                    initialized_sides.append(side)
                    if identity is not None:
                        initialized_identities.append(identity)
                elif identity in reused:
                    reused_sides.append(side)
                    if self._should_mirror(cfg):
                        self._takeover_pending.add(index)
                    else:
                        self._takeover_pending.discard(index)
                    info(
                        f"Preserved gripper width for session handoff: {side}",
                        identity.describe() if identity is not None else "",
                    )
            except Exception as exc:
                message = describe_exception(exc)
                if identity in reused:
                    message = self._with_reinitialize_instruction(message)
                warn(f"Gripper setup failed for pair {index}", message)
                self._errors[index] = message
                errors[self._sides[index]] = message
                if registry is not None and identity in claimed:
                    registry.fail([identity])

        if registry is not None and initialized_identities:
            self._stop_event.clear()
            if self._stop_event.wait(self._init_settle_s):
                registry.fail(initialized_identities)
                cancelled_sides = set(initialized_sides)
                for side in cancelled_sides:
                    errors.setdefault(side, "Gripper initialization was cancelled")
                initialized_identities = []
                initialized_sides = []
            else:
                registry.complete(initialized_identities)
                for side, identity in zip(
                    initialized_sides, initialized_identities, strict=True
                ):
                    info(
                        f"Initialized gripper for backend session: {side}",
                        identity.describe(),
                    )

        # Any claimed identity that did not reach Init() successfully must not
        # remain stuck in INITIALIZING after a partial multi-gripper failure.
        if registry is not None:
            registry.fail(claimed - set(initialized_identities))

        widths: dict[str, float] = {}
        if self._default_gripper_width_m is not None and initialized_sides:
            movement = self.move_grippers_to_width(
                self._default_gripper_width_m, sides=initialized_sides
            )
            widths = movement["widths"]
            for side, message in movement["errors"].items():
                errors[side] = (
                    self._with_reinitialize_instruction(message)
                    if side in reused_sides
                    else message
                )
            for side, width in widths.items():
                info(
                    f"Moved gripper to session default width: {side}",
                    f"width={width:.4f} m",
                )

        return {
            "errors": errors,
            "initialized_now": initialized_sides,
            "reused": reused_sides,
            "widths": widths,
        }

    @staticmethod
    def _with_reinitialize_instruction(message: str) -> str:
        return f"{message}. Reinitialize the gripper and retry"

    def start(self) -> None:
        """Start the mirror thread (on teleop Start).

        The thread mirrors every leader-triggered follower (gripper or digital
        output). Gripper sides that were not initialized are skipped per-tick
        (the panel surfaces a warning), so a forgotten Init never blocks teleop.
        """
        if not self.has_work() or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="end-effector-control", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the mirror thread (on teleop Stop); keep grippers enabled/cached.

        The enabled grippers and their params survive a teleop stop/start cycle
        so the panel's sliders stay populated; they are only released by
        shutdown() when the controller is disconnected.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def shutdown(self) -> None:
        """Stop the thread and release all gripper/command state (on disconnect)."""
        self.stop()
        with self._setup_lock:
            self._grippers.clear()
            self._gripper_params.clear()
            self._command_params.clear()
            self._last_do.clear()
            self._last_gripper_target.clear()
            self._takeover_pending.clear()

    def _run(self) -> None:  # pragma: no cover - hardware specific
        period = 1.0 / self.POLL_HZ
        while not self._stop_event.is_set():
            for index, cfg in enumerate(self._configs):
                if cfg is None or not self._should_mirror(cfg):
                    continue
                try:
                    self._tick(index, cfg)
                    self._errors.pop(index, None)
                except Exception as exc:
                    message = describe_exception(exc)
                    # Log only on the first failure (or when it changes) to avoid
                    # flooding the console at the poll rate.
                    if self._errors.get(index) != message:
                        warn(f"End effector control failed for pair {index}", message)
                    self._errors[index] = message
            self._stop_event.wait(period)

    def _tick(self, index: int, cfg: EndEffectorSideConfig) -> None:
        triggered = self._leader_triggered(index, cfg)
        if cfg.follower == "digital_output":
            self._drive_digital_output(index, cfg, triggered)
        elif cfg.follower == "gripper":
            self._drive_gripper(index, cfg, triggered)

    def _leader_triggered(self, index: int, cfg: EndEffectorSideConfig) -> bool:
        # digital_inputs(idx) returns (leader_ports, follower_ports); the leader
        # is the trigger source.
        leader_ports, _ = self._controller.digital_inputs(index)
        port_high = bool(leader_ports[cfg.leader_channel])
        # The leader trigger is "active" when its port matches the activating state.
        return port_high if cfg.leader_activating_state == "high" else not port_high

    def _drive_digital_output(
        self, index: int, cfg: EndEffectorSideConfig, triggered: bool
    ) -> None:
        activated_high = cfg.follower_activated_state == "high"
        # Drive the port to its activated state while triggered, else the opposite.
        value = activated_high if triggered else not activated_high
        if self._last_do.get(index) == value:
            return
        _, follower = self._controller.instances(index)
        follower.SetDigitalOutputs({cfg.follower_channel: value})
        self._last_do[index] = value

    def _drive_gripper(
        self, index: int, cfg: EndEffectorSideConfig, triggered: bool
    ) -> None:
        # Activated state ("close"/"open") is applied while triggered; the
        # opposite is applied otherwise.
        if triggered:
            target = cfg.gripper_activated_state
        else:
            target = "open" if cfg.gripper_activated_state == "close" else "close"
        takeover_pending = index in self._takeover_pending
        if takeover_pending and target != "close":
            return
        if not takeover_pending and self._last_gripper_target.get(index) == target:
            return

        gripper = self._grippers.get(index)
        params = self._gripper_params.get(index)
        if gripper is None or params is None:
            # The gripper was never (fully) initialized -- Init not run, or setup
            # raised after Enable() so the gripper object exists without params --
            # so there is nothing to command. Skip it silently (the panel shows a
            # "not initialized" warning) so the rest of the mirror loop and teleop
            # keep running rather than raising on every tick.
            return
        if target == "close":
            width = params.min_width
        else:
            requested = self._default_gripper_width_m
            width = params.max_width if requested is None else _clamp(
                requested, params.min_width, params.max_width
            )
        # Use the panel slider's velocity/force, falling back to the gripper's
        # own params-derived defaults until a slider sets them.
        velocity, force = self._move_params_for(index)
        gripper.Move(width, velocity, force)
        if takeover_pending:
            self._takeover_pending.discard(index)
        self._last_gripper_target[index] = target

    def _move_params_for(self, index: int) -> tuple[float, float]:
        params = self._gripper_params[index]
        # Until the panel sets a value, default to the gripper's own max velocity
        # (Move() rejects 0) and a fraction of its max force.
        velocity, force = self._command_params.get(
            index, (params.max_vel, params.max_force * self.DEFAULT_FORCE_FRACTION)
        )
        velocity = _clamp(velocity, params.min_vel, params.max_vel)
        force = _clamp(force, params.min_force, params.max_force)
        return velocity, force

    def set_command_params(
        self, side: str, velocity: float, force: float
    ) -> tuple[float, float]:
        """Store the (velocity, force) the panel set for this side's gripper.

        Used by the mirror loop's Move() calls. Clamped into the gripper's range
        when known. Returns the stored values.
        """
        index = self._index_for_side(side)
        params = self._gripper_params.get(index)
        if params is not None:
            velocity = _clamp(velocity, params.min_vel, params.max_vel)
            force = _clamp(force, params.min_force, params.max_force)
        self._command_params[index] = (velocity, force)
        return velocity, force

    def move_grippers_to_width(
        self, width: float, *, sides: list[str] | None = None
    ) -> dict[str, Any]:
        """Move every initialized gripper and make this the current Open width."""

        requested = float(width)
        if not math.isfinite(requested) or requested < 0:
            raise ValueError("Gripper width must be finite and nonnegative")
        self._default_gripper_width_m = requested
        selected = set(sides) if sides is not None else None
        moved: dict[str, float] = {}
        errors: dict[str, str] = {}
        for index, cfg in enumerate(self._configs):
            if cfg is None or cfg.follower != "gripper":
                continue
            gripper = self._grippers.get(index)
            params = self._gripper_params.get(index)
            side = self._sides[index]
            if selected is not None and side not in selected:
                continue
            if gripper is None or params is None:
                errors[side] = "Gripper is not initialized"
                continue
            target = _clamp(requested, params.min_width, params.max_width)
            if target != requested:
                warn(
                    f"Clamped gripper width for {side}",
                    f"requested={requested:.4f} clamped={target:.4f} "
                    f"range=[{params.min_width:.4f}, {params.max_width:.4f}]",
                )
            try:
                velocity, force = self._move_params_for(index)
                gripper.Move(target, velocity, force)
                moved[side] = target
                # A manual target may be between open and closed. Force the mirror
                # loop to reassert its state on the next teleop start/tick.
                self._last_gripper_target.pop(index, None)
            except Exception as exc:  # pragma: no cover - hardware specific
                message = describe_exception(exc)
                warn(f"Failed to tune gripper width for {side}", message)
                errors[side] = message
        return {"widths": moved, "errors": errors}

    def set_default_gripper_width(self, width: float | None) -> None:
        """Refresh the Open target from the persisted global configuration."""

        if width is not None and (not math.isfinite(width) or width < 0):
            raise ValueError("Gripper width must be finite and nonnegative")
        self._default_gripper_width_m = width
        self._last_gripper_target.clear()

    @staticmethod
    def _follower_is_idle(follower: Any) -> bool:
        # Tool.Switch() is only valid in IDLE control mode. Read the follower's
        # current mode and compare against Mode.IDLE; if the mode can't be read
        # (no flexivrdk / fake), assume not-IDLE so the switch is skipped rather
        # than risking a mode-mismatch on a running teleop loop.
        mode_reader = getattr(follower, "mode", None)
        if Mode is None or not callable(mode_reader):
            return False
        try:
            return mode_reader() == Mode.IDLE
        except Exception:  # pragma: no cover - hardware specific
            return False

    def _setup_gripper(
        self,
        index: int,
        cfg: EndEffectorSideConfig,
        *,
        initialize: bool = True,
    ) -> None:
        # Enable the gripper as a device, switch its tool so the gripper's mass
        # is accounted for in gravity compensation/TCP, optionally trigger the
        # mechanical Init() cycle, and read its params. Cached session adoption
        # performs every connection-local step but skips Init().
        if Gripper is None:
            raise RuntimeError("flexivrdk is not available; cannot control gripper")
        _, follower = self._controller.instances(index)
        if index not in self._grippers:
            gripper = Gripper(follower)
            gripper.Enable(cfg.gripper_model)
            self._grippers[index] = gripper
        if Tool is not None and self._follower_is_idle(follower):
            Tool(follower).Switch(cfg.gripper_model)
        if initialize:
            self._grippers[index].Init()
        self._gripper_params[index] = self._grippers[index].params()

    def gripper_states_by_index(self) -> dict[int, dict[str, float]]:
        """Measured width/force for every enabled gripper, keyed by pair index.

        Used by recording to fold a gripper's live state into the follower's
        per-arm telemetry. Only sides whose follower is a gripper and whose
        gripper has been enabled (initialize_grippers) and reports states are
        included; a side without a working gripper is simply absent so the
        recorder skips it.
        """
        states_by_index: dict[int, dict[str, float]] = {}
        for index, cfg in enumerate(self._configs):
            if cfg is None or cfg.follower != "gripper":
                continue
            gripper = self._grippers.get(index)
            if gripper is None:
                continue
            try:  # pragma: no cover - hardware specific
                states = gripper.states()
                states_by_index[index] = {
                    "width": float(states.width),
                    "force": float(states.force),
                }
            except Exception:  # pragma: no cover - hardware specific
                continue
        return states_by_index

    def gripper_snapshot(self) -> dict[str, Any]:
        """Per-side gripper parameters/state, for sides with an enabled gripper.

        Keyed by arm side; present only once initialize_grippers() has enabled
        it, so the UI can read the valid ranges and defaults straight from the
        hardware. Persists across teleop start/stop until disconnect.
        """
        snapshot: dict[str, Any] = {}
        for index, cfg in enumerate(self._configs):
            if cfg is None or cfg.follower != "gripper":
                continue
            params = self._gripper_params.get(index)
            gripper = self._grippers.get(index)
            if params is None or gripper is None:
                continue
            entry = {
                "model": params.name,
                "min_vel": params.min_vel,
                "max_vel": params.max_vel,
                "min_force": params.min_force,
                "max_force": params.max_force,
                "min_width": params.min_width,
                "max_width": params.max_width,
                "takeover_pending": index in self._takeover_pending,
            }
            try:  # pragma: no cover - hardware specific
                states = gripper.states()
                entry["width"] = states.width
                entry["is_moving"] = states.is_moving
            except Exception:  # pragma: no cover - hardware specific
                pass
            snapshot[self._sides[index]] = entry
        return snapshot
