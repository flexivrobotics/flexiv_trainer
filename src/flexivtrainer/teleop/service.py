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

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from typing import Callable

from flexivtrainer.config import (
    DEFAULT_HOME_POSTURE_DEG,
    AppSettings,
    EndEffectorSideConfig,
    TeleopRobotPair,
)
from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.teleop.end_effector import EndEffectorController

try:
    import flexivtdk
except (
    ImportError
):  # pragma: no cover - dependency availability is environment-specific
    flexivtdk = None

try:
    import flexivrdk
except ImportError:  # pragma: no cover - environment-specific
    flexivrdk = None

TransparentCartesianTeleopLAN = (
    getattr(flexivtdk, "TransparentCartesianTeleopLAN", None)
    if flexivtdk is not None
    else None
)

# Init()'s zero_ft_sensor parameter is a ZeroFTSensor enum (Enable/Disable).
ZeroFTSensor = (
    getattr(flexivtdk, "ZeroFTSensor", None) if flexivtdk is not None else None
)


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return {
            key: _serialize_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


@dataclass
class TeleopSnapshot:
    configured: bool
    available: bool
    initialized: bool
    started: bool
    stopped: bool
    engaged: bool
    can_home: bool
    fault: str | None
    error: str | None


class TeleopService:
    def __init__(
        self,
        settings: AppSettings,
        get_robot_pairs: Callable[[], list[TeleopRobotPair]] | None = None,
        get_active_sides: Callable[[], list[str]] | None = None,
        get_end_effector_config: (
            Callable[[], dict[str, EndEffectorSideConfig]] | None
        ) = None,
    ) -> None:
        self._settings = settings
        self._get_robot_pairs = get_robot_pairs or (lambda: settings.teleop_robot_pairs)
        # Active arm sides (in pair-index order) and the per-side end effector
        # config drive the optional end effector controller; defaults keep the
        # service usable without that wiring.
        self._get_active_sides = get_active_sides or (
            lambda: ["left_arm", "right_arm"]
        )
        self._get_end_effector_config = get_end_effector_config or (lambda: {})
        self._controller: Any | None = None
        self._error: str | None = None
        self._initialized = False
        # ``_started`` tracks whether the teleop control loop (Start()) is
        # running; ``_engaged`` tracks whether the pairs are currently engaged.
        # They are decoupled because the Stop button disengages without
        # stopping the loop.
        self._started = False
        self._engaged = False
        # Background mirror of leader digital inputs onto follower end effectors;
        # runs with the teleop control loop (started/stopped by Start/Stop), not
        # gated by engage -- it keeps mirroring across engage/disengage cycles.
        self._end_effectors: EndEffectorController | None = None

    def _configured_remote_serials(self) -> list[str]:
        serials: list[str] = []
        for pair in self._get_robot_pairs():
            serial = str(pair.follower_serial).strip()
            if serial:
                serials.append(serial)
        return serials

    def _coerce_numeric_vector(self, value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)):
            return None

        vector: list[float] = []
        for item in value:
            try:
                vector.append(float(item))
            except (TypeError, ValueError):
                return None
        return vector

    def _read_vector_field(self, source: Any, field: str) -> list[float] | None:
        # RobotStates/RobotActions are pybind11 structs without __dict__, so
        # their fields must be read by attribute rather than serialized en bloc.
        if isinstance(source, dict):
            raw = source.get(field)
        else:
            raw = getattr(source, field, None)
        return self._coerce_numeric_vector(_serialize_value(raw))

    def _follower_of(self, pair: Any) -> Any:
        # ``robot_states(idx)`` and ``instances(idx)`` both return the (leader,
        # follower) robot of the pair. These widgets/recordings use the follower
        # (the remote robot being teleoperated), so pick the second element.
        if isinstance(pair, (tuple, list)) and len(pair) >= 2:
            return pair[1]
        return pair

    def robot_data_snapshot(
        self,
        *,
        include_states: bool = True,
        include_actions: bool = True,
    ) -> dict[str, Any]:
        if self._controller is None:
            return {"robots": {}, "errors": {}}

        configured_serials = self._configured_remote_serials()
        if not configured_serials:
            return {"robots": {}, "errors": {}}

        # Read each pair explicitly by index via instances(idx), which returns
        # the (leader, follower) rdk::Robot handles, and take the follower (the
        # remote robot being teleoperated). The previous approach flattened
        # every handle into one list, so the follower telemetry could be
        # mislabeled (or a leader's ~0 wrench shown under a follower serial).
        instances_reader = getattr(self._controller, "instances", None)
        if not callable(instances_reader):
            return {"robots": {}, "errors": {}}

        snapshot_robots: dict[str, Any] = {}
        errors: dict[str, str] = {}

        # Live width/force for any follower configured as a gripper, keyed by the
        # same pair index used below. Folded into the follower payload so that
        # recording can append it to both observation.state and action. Read once
        # up front; absent for sides without an enabled gripper. Skip the read
        # entirely when neither states nor actions are requested so a bare
        # snapshot never touches the (hardware) gripper layer.
        gripper_states = (
            self._gripper_states_by_index()
            if (include_states or include_actions)
            else {}
        )

        for index, serial in enumerate(configured_serials):
            base_name = serial or f"robot_{index}"
            robot_name = base_name
            suffix = 1
            while robot_name in snapshot_robots:
                robot_name = f"{base_name}_{suffix}"
                suffix += 1

            payload: dict[str, Any] = {"connected": True}

            try:
                follower = self._follower_of(instances_reader(index))

                if include_states:
                    states_reader = getattr(follower, "states", None)
                    if callable(states_reader):
                        raw_states = states_reader()
                        states: dict[str, list[float]] = {}
                        for field in ("tcp_pose", "tcp_vel", "ext_wrench_in_world"):
                            vector = self._read_vector_field(raw_states, field)
                            if vector is not None:
                                states[field] = vector
                        if states:
                            payload["states"] = states

                if include_actions:
                    actions_reader = getattr(follower, "actions", None)
                    if callable(actions_reader):
                        raw_actions = actions_reader()
                        actions: dict[str, list[float]] = {}
                        for field in ("tcp_pose_d", "tcp_vel_d", "ext_wrench_d"):
                            vector = self._read_vector_field(raw_actions, field)
                            if vector is not None:
                                actions[field] = vector
                        if actions:
                            payload["actions"] = actions

                # A follower gripper's measured width/force feeds both the
                # observation and the action, so attach it whenever either is
                # requested; the recorder appends it to the matching vector(s).
                if include_states or include_actions:
                    gripper = gripper_states.get(index)
                    if gripper is not None:
                        payload["gripper"] = dict(gripper)
            except Exception as exc:  # pragma: no cover - hardware specific
                payload["connected"] = False
                payload["error"] = describe_exception(exc)
                errors[robot_name] = payload["error"]

            snapshot_robots[robot_name] = payload

        return {"robots": snapshot_robots, "errors": errors}

    def _gripper_states_by_index(self) -> dict[int, dict[str, float]]:
        # Live gripper width/force keyed by pair index, mirroring the index used
        # by instances(idx). Empty when no end effector controller exists or no
        # gripper is enabled, so the snapshot simply omits gripper telemetry.
        if self._end_effectors is None:
            return {}
        try:
            return self._end_effectors.gripper_states_by_index()
        except Exception:  # pragma: no cover - hardware specific
            return {}

    def initialize(self) -> TeleopSnapshot:
        robot_pairs_config = self._get_robot_pairs()
        if TransparentCartesianTeleopLAN is None:
            self._error = "flexivtdk is not importable in the selected environment"
            return self.snapshot()
        if not any(
            pair.leader_serial and pair.follower_serial for pair in robot_pairs_config
        ):
            self._error = "No teleoperation robot pairs are configured"
            return self.snapshot()
        if self._controller is None:
            robot_pairs = [
                (pair.leader_serial, pair.follower_serial)
                for pair in robot_pairs_config
                if pair.leader_serial and pair.follower_serial
            ]
            try:
                # Connect / service reset only establishes the connection. The
                # blocking Init() sequence (robot enable + F/T zeroing) is
                # deferred to the Start button, which always runs Init() then
                # Start().
                self._controller = TransparentCartesianTeleopLAN(
                    robot_pairs_sn=robot_pairs,
                    network_interface_whitelist=self._settings.network_interface_whitelist,
                )
                self._initialized = True
                self._started = False
                self._engaged = False
                self._error = None
                # On a successful connection, send the enable signal to every
                # connected robot. This is fire-and-forget: no operational/ready
                # check and no wait, just the enabling request for all robots.
                self._enable_all_robots()
            except Exception as exc:  # pragma: no cover - hardware specific
                self._error = describe_exception(exc)
                self._controller = None
        return self.snapshot()

    def _enable_all_robots(self) -> None:
        # Enable both robots of every configured pair via the underlying
        # rdk::Robot handles from instances(idx). Best-effort per robot so one
        # failed Enable() does not abort the rest or the connection.
        if self._controller is None:
            return
        instances_reader = getattr(self._controller, "instances", None)
        if not callable(instances_reader):
            return
        for idx in range(self._engageable_pair_count()):
            try:
                robots = instances_reader(idx)
            except Exception as exc:  # pragma: no cover - hardware specific
                warn(
                    f"Failed to read robot instances for pair {idx}",
                    describe_exception(exc),
                )
                continue
            for robot in robots if isinstance(robots, (tuple, list)) else (robots,):
                enable = getattr(robot, "Enable", None)
                if not callable(enable):
                    continue
                try:
                    enable()
                except Exception as exc:  # pragma: no cover - hardware specific
                    warn("Failed to enable robot", describe_exception(exc))

    def _engageable_pair_count(self) -> int:
        return sum(
            1
            for pair in self._get_robot_pairs()
            if pair.leader_serial and pair.follower_serial
        )

    def start(self, zero_ft_sensor: bool = True) -> TeleopSnapshot:
        # The Start button always runs Init() then Start(). Per the TDK contract
        # (see TransparentCartesianTeleopLAN::Start/Stop docs), restarting after
        # a Stop() requires calling Init() again first, so Init() is run on every
        # Start rather than only when the controller is first constructed. The
        # pairs stay disengaged by default; engaging is a separate action.
        #
        # zero_ft_sensor maps to Init()'s flag of the same name: when enabled the
        # force/torque sensors are zeroed during initialization (the robots must
        # be free of unexpected contact); when disabled the zeroing step is
        # skipped.
        self.initialize()
        if self._controller is None:
            return self.snapshot()
        try:
            init_method = getattr(self._controller, "Init", None)
            if callable(init_method):
                if ZeroFTSensor is not None:
                    flag = (
                        ZeroFTSensor.Enable
                        if zero_ft_sensor
                        else ZeroFTSensor.Disable
                    )
                    init_method(zero_ft_sensor=flag)
                else:
                    init_method()
            self._controller.Start()
            self._started = True
            self._engaged = False
            self._error = None
            # The end effector mirror thread runs while the teleop loop runs.
            # Reuse any controller built by the panel's Init button (with its
            # enabled grippers), or build a fresh one for digital-output-only
            # mirroring; then start the mirror thread. Gripper sides that were
            # not initialized are skipped per-tick (the panel shows a warning).
            self._ensure_end_effectors()
            if self._end_effectors is not None:
                self._end_effectors.start()
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
        return self.snapshot()

    def clear_fault(self, timeout_sec: int = 30) -> dict[str, Any]:
        # Try to clear minor/critical faults via the controller's ClearFault(),
        # which blocks until the fault clears or the timeout elapses. Returns a
        # status dict the route surfaces to the floating fault widget; the
        # ``cleared`` flag is re-derived from any_fault() so the UI does not have
        # to trust the per-robot return vector alone.
        if self._controller is None:
            return {"ok": False, "cleared": False, "error": "Teleoperation not connected"}

        clear_fault = getattr(self._controller, "ClearFault", None)
        if not callable(clear_fault):
            return {
                "ok": False,
                "cleared": False,
                "error": "Controller does not support ClearFault",
            }

        try:
            clear_fault(timeout_sec)
        except Exception as exc:  # pragma: no cover - hardware specific
            return {"ok": False, "cleared": False, "error": describe_exception(exc)}

        # _read_fault() returns a message while any robot is still faulted; a
        # None result means any_fault() now reports clear.
        cleared = self._read_fault() is None
        return {
            "ok": cleared,
            "cleared": cleared,
            "error": None if cleared else "Fault persists after ClearFault",
        }

    def _ensure_end_effectors(self) -> None:
        # Construct the end effector controller for the current config if one is
        # not already present. Does not enable grippers -- that is the gripper
        # panel's Init button's job (init_grippers). Safe to call repeatedly.
        if self._end_effectors is not None or self._controller is None:
            return
        try:
            self._end_effectors = EndEffectorController(
                self._controller,
                self._get_active_sides(),
                dict(self._get_end_effector_config() or {}),
            )
        except Exception:  # pragma: no cover - hardware specific
            self._end_effectors = None

    def init_grippers(self) -> dict[str, Any]:
        # Enable + tool-switch + Init + read params for every configured gripper.
        # Run from the panel's Init button after connect but before Start, while
        # the follower is IDLE (Tool.Switch() is IDLE-only). The enabled grippers
        # and their params persist across teleop start/stop until disconnect.
        if self._controller is None:
            return {
                "ok": False,
                "error": "Connect teleoperation before initializing grippers",
            }
        if self._started:
            return {
                "ok": False,
                "error": "Stop teleoperation before initializing grippers",
            }
        self._ensure_end_effectors()
        if self._end_effectors is None or not self._end_effectors.has_grippers():
            return {"ok": False, "error": "No grippers configured"}
        errors = self._end_effectors.initialize_grippers()
        snapshot = self._end_effectors.gripper_snapshot()
        if errors and not snapshot:
            return {"ok": False, "error": "; ".join(errors.values())}
        return {"ok": True, "gripper": snapshot, "errors": errors}

    def _teardown_end_effectors(self) -> None:
        if self._end_effectors is not None:
            self._end_effectors.shutdown()
            self._end_effectors = None

    def stop(self) -> TeleopSnapshot:
        # The Stop button stops the teleop control loop. Engagement is cleared
        # because Engage requires a running loop. The mirror thread is stopped
        # too, but the enabled grippers / params stay cached so the panel's
        # sliders survive a stop/start cycle (released only on disconnect).
        #
        # Stop must ALWAYS halt teleop, including when one or more robots are in
        # fault. The global Stop() raises std::runtime_error "if failed to stop
        # the robots" -- which a faulted robot can trigger -- so calling it alone
        # would abort before the operational pairs are stopped, leaving them
        # teleoperating. To stay robust we stop each pair individually via
        # StopWithIdx(idx) so a failure on a faulted pair does not prevent
        # stopping the rest, and we mark the loop stopped regardless of any error
        # (per the TDK contract a stopped/faulted loop requires Init()+Start() to
        # resume, so it is no longer the controlling process either way).
        if self._controller is None:
            return self.snapshot()

        # Stop the mirror thread first so it stops issuing follower commands;
        # best-effort so a mirror failure cannot block stopping the arms.
        try:
            if self._end_effectors is not None:
                self._end_effectors.stop()
        except Exception as exc:  # pragma: no cover - hardware specific
            warn("Failed to stop end effector mirror", describe_exception(exc))

        errors = self._stop_all_pairs()

        # Always treat the loop as halted: even a partial/failed Stop means
        # teleop is no longer reliably driving the arms, and the operator must
        # Init()+Start() again to resume. Leaving _started True would re-enable
        # the Stop button to no further effect and keep the arms shown as live.
        self._started = False
        self._engaged = False
        self._error = "; ".join(errors) if errors else None
        return self.snapshot()

    def _stop_all_pairs(self) -> list[str]:
        # Stop every configured pair, isolating failures so one faulted pair
        # cannot keep the others running. Prefer per-pair StopWithIdx(idx); fall
        # back to a single global Stop() when the controller predates it.
        controller = self._controller
        if controller is None:
            return []

        stop_with_idx = getattr(controller, "StopWithIdx", None)
        if callable(stop_with_idx):
            errors: list[str] = []
            for idx in range(self._engageable_pair_count()):
                try:
                    stop_with_idx(idx)
                except Exception as exc:  # pragma: no cover - hardware specific
                    message = describe_exception(exc)
                    warn(f"Failed to stop teleop pair {idx}", message)
                    errors.append(message)
            return errors

        try:
            controller.Stop()
        except Exception as exc:  # pragma: no cover - hardware specific
            message = describe_exception(exc)
            warn("Failed to stop teleop", message)
            return [message]
        return []

    def set_engaged(self, engaged: bool) -> TeleopSnapshot:
        # Engage/disengage every configured pair. Engage requires the teleop
        # control loop to be running (Start() called), so guard on it.
        if self._controller is None:
            self._error = "Teleoperation controller is not initialized"
            return self.snapshot()
        if not self._started:
            self._error = "Start teleoperation before engaging the robots"
            return self.snapshot()
        try:
            # Engage is per-pair (no engage-all variant), so apply the flag to
            # every configured pair by index.
            for idx in range(self._engageable_pair_count()):
                self._controller.Engage(idx, engaged)
            self._engaged = engaged
            self._error = None
            # The mirror thread is tied to teleop Start/Stop, not engage, so it
            # keeps running across engage cycles -- nothing to do here.
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
        return self.snapshot()

    def gripper_snapshot(self) -> dict[str, Any]:
        # Empty until the panel's Init enables grippers; persists until disconnect.
        if self._end_effectors is None:
            return {}
        return self._end_effectors.gripper_snapshot()

    def set_gripper_params(
        self, side: str, velocity: float, force: float
    ) -> dict[str, Any]:
        # Store this side's slider velocity/force; used by the mirror loop's
        # Move() calls. Allowed any time after init (including while the loop is
        # running, to tune it live) -- it only records values, it does not move
        # the gripper.
        if self._end_effectors is None:
            return {
                "ok": False,
                "error": "Initialize the gripper before setting its parameters",
            }
        try:
            velocity, force = self._end_effectors.set_command_params(
                side, velocity, force
            )
            return {"ok": True, "velocity": velocity, "force": force}
        except Exception as exc:  # pragma: no cover - hardware specific
            return {"ok": False, "error": describe_exception(exc)}

    def shutdown(self) -> None:
        if self._controller is None:
            return

        # Disconnect / service reset fully releases the end effectors (enabled
        # grippers + cached params), which a teleop Stop deliberately keeps.
        self._teardown_end_effectors()
        try:
            if self._started:
                self._controller.Stop()
        except Exception:  # pragma: no cover - hardware specific
            pass

        # TransparentCartesianTeleopLAN exposes no explicit close/disconnect
        # method; its C++ destructor performs teardown once the last reference
        # is dropped, so releasing the controller here is the cleanup.
        self._controller = None
        self._initialized = False
        self._started = False
        self._engaged = False
        self._error = None

    def reset_home(self) -> dict[str, Any]:
        if self._controller is None:
            return {"ok": False, "error": "Teleoperation controller is not initialized"}
        if self._started:
            # Homing requires the teleop control loop to be stopped. The UI gates
            # the Home button behind Stop, but guard here as well.
            return {
                "ok": False,
                "error": "Stop teleoperation before homing the robots",
            }
        if flexivrdk is None:
            return {"ok": False, "error": "flexivrdk is not importable"}

        instances_reader = getattr(self._controller, "instances", None)
        if not callable(instances_reader):
            return {
                "ok": False,
                "error": "Connected controller does not expose robot instances",
            }

        posture = self._home_posture()

        warnings: list[str] = []
        homed_any = False
        for idx in range(self._engageable_pair_count()):
            try:
                robots = instances_reader(idx)
            except Exception as exc:  # pragma: no cover - hardware specific
                warnings.append(f"Pair {idx}: {describe_exception(exc)}")
                continue
            for robot in robots if isinstance(robots, (tuple, list)) else (robots,):
                error = self._move_to_home(robot, posture)
                if error is None:
                    homed_any = True
                else:
                    warnings.append(error)

        if not homed_any:
            return {
                "ok": False,
                "error": warnings[0] if warnings else "No robots were homed",
            }
        return {"ok": True, "warnings": warnings}

    def _home_posture(self) -> list[float]:
        for pair in self._get_robot_pairs():
            if pair.leader_home_posture:
                return [float(value) for value in pair.leader_home_posture]
        return list(DEFAULT_HOME_POSTURE_DEG)

    def _move_to_home(
        self, robot: Any, posture: list[float], timeout_sec: float = 30.0
    ) -> str | None:
        # Drive one robot to `posture` (degrees) via the MoveJ primitive, then
        # block on reachedTarget. Returns None on success or an error string.
        switch_mode = getattr(robot, "SwitchMode", None)
        execute = getattr(robot, "ExecutePrimitive", None)
        states = getattr(robot, "primitive_states", None)
        if not (callable(switch_mode) and callable(execute) and callable(states)):
            return "Connected robot does not support primitive execution"
        try:
            switch_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
            execute("MoveJ", {"target": flexivrdk.JPos(posture)})
            deadline = time.monotonic() + timeout_sec
            while not states().get("reachedTarget"):
                if time.monotonic() >= deadline:
                    return "Timed out waiting for a robot to reach home"
                time.sleep(0.5)
        except Exception as exc:  # pragma: no cover - hardware specific
            return describe_exception(exc)
        return None

    def _read_fault(self) -> str | None:
        """Return a fault message if the controller reports a fault.

        ``TransparentCartesianTeleopLAN`` exposes fault state through the
        ``any_fault()`` method (not a ``fault`` attribute), so the method must
        be invoked rather than read as a value.
        """
        if self._controller is None:
            return None

        any_fault = getattr(self._controller, "any_fault", None)
        if not callable(any_fault):
            return None

        try:
            faulted = any_fault()
        except Exception:  # pragma: no cover - hardware specific
            return None

        if not faulted:
            return None

        # any_fault() is a bare bool; name the offending robots via fault(idx)
        # so the floating fault widget can show which arm tripped rather than a
        # generic message. fault(idx) returns (leader_faulted, follower_faulted)
        # for the pair at that index, matching the constructor's robot_pairs_sn
        # order. Fall back to the generic message if the detail lookup fails.
        try:
            pairs = [
                pair
                for pair in self._get_robot_pairs()
                if pair.leader_serial and pair.follower_serial
            ]
            fault_at = getattr(self._controller, "fault", None)
            faulted_serials: list[str] = []
            if callable(fault_at):
                for idx, pair in enumerate(pairs):
                    leader_fault, follower_fault = fault_at(idx)
                    if leader_fault:
                        faulted_serials.append(str(pair.leader_serial))
                    if follower_fault:
                        faulted_serials.append(str(pair.follower_serial))
            if faulted_serials:
                # "Fault occurred on robot [SN1]" / "... [SN1], [SN2], and [SN3]".
                bracketed = [f"[{sn}]" for sn in faulted_serials]
                if len(bracketed) == 1:
                    joined = bracketed[0]
                elif len(bracketed) == 2:
                    joined = f"{bracketed[0]} and {bracketed[1]}"
                else:
                    joined = ", ".join(bracketed[:-1]) + f", and {bracketed[-1]}"
                return f"Fault occurred on robot {joined}"
        except Exception:  # pragma: no cover - hardware specific
            pass

        return "Teleoperation fault detected"

    def snapshot(self) -> TeleopSnapshot:
        robot_pairs = self._get_robot_pairs()
        configured = any(
            pair.leader_serial and pair.follower_serial for pair in robot_pairs
        )
        # ``started`` reflects whether the teleop control loop is running;
        # ``engaged`` reflects whether the pairs are currently engaged. The Start
        # button controls the former, the Engage button the latter.
        started = self._started and self._controller is not None
        stopped = not started
        engaged = self._engaged and self._controller is not None
        fault: str | None = self._read_fault()
        # Home is offered whenever the robots are connected and the teleop
        # control loop is not running (HomeAll() throws if it is). It is
        # available right after Connect, before the first Start.
        can_home = (
            self._initialized and not started and self._controller is not None
        )
        return TeleopSnapshot(
            configured=configured,
            available=TransparentCartesianTeleopLAN is not None,
            initialized=self._initialized and self._controller is not None,
            started=started,
            stopped=stopped,
            engaged=engaged,
            can_home=can_home,
            fault=fault,
            error=self._error,
        )
