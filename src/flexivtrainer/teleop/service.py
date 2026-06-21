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

from dataclasses import dataclass
from typing import Any
from typing import Callable

from flexivtrainer.config import AppSettings, EndEffectorSideConfig, TeleopRobotPair
from flexivtrainer.observability import describe_exception
from flexivtrainer.teleop.end_effector import EndEffectorController

try:
    import flexivtdk
except (
    ImportError
):  # pragma: no cover - dependency availability is environment-specific
    flexivtdk = None

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
        # runs only while the pairs are engaged.
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
            except Exception as exc:  # pragma: no cover - hardware specific
                payload["connected"] = False
                payload["error"] = describe_exception(exc)
                errors[robot_name] = payload["error"]

            snapshot_robots[robot_name] = payload

        return {"robots": snapshot_robots, "errors": errors}

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
            except Exception as exc:  # pragma: no cover - hardware specific
                self._error = describe_exception(exc)
                self._controller = None
        return self.snapshot()

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
            # Keep any controller built by the Gripper Control Start button (with
            # its enabled grippers and running mirror thread) so it survives a
            # teleop (re)start; only build a fresh one if none exists yet.
            self._ensure_end_effectors()
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
        return self.snapshot()

    def _ensure_end_effectors(self) -> None:
        # Construct the end effector controller for the current config if one is
        # not already present. Does not enable grippers -- that is the gripper
        # panel's Start button's job (start_end_effectors). Safe to call repeatedly.
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

    def start_end_effectors(self, trigger_init: bool = False) -> dict[str, Any]:
        # Set up every configured gripper (enable + IDLE-only tool-switch +
        # optional Init + read params) and start the mirror thread for all
        # mirrored sides (gripper or digital output). Driven by the End Effector
        # Control panel's Start button and independent of teleop engage/disengage;
        # gripper.Move() does not change the arm's control mode, so the mirror
        # loop coexists with a running teleop loop. Tool.Switch() is skipped
        # unless the follower is IDLE (the controller guards this internally).
        if self._controller is None:
            return {
                "ok": False,
                "error": "Connect teleoperation before starting end effectors",
            }
        self._ensure_end_effectors()
        if self._end_effectors is None or not self._end_effectors.has_end_effectors():
            return {"ok": False, "error": "No end effectors configured"}
        errors = self._end_effectors.start(trigger_init)
        snapshot = self._end_effectors.gripper_snapshot()
        # Gripper setup can fail; digital-output sides never produce setup errors.
        if errors and not snapshot and self._end_effectors.has_grippers():
            return {"ok": False, "error": "; ".join(errors.values())}
        return {"ok": True, "gripper": snapshot, "errors": errors}

    def stop_end_effectors(self) -> dict[str, Any]:
        # Stop the mirror thread and release gripper state (panel's Stop button).
        if self._end_effectors is None:
            return {"ok": True}
        self._end_effectors.stop()
        return {"ok": True}

    def end_effectors_running(self) -> bool:
        return self._end_effectors is not None and self._end_effectors.is_running()

    def _teardown_end_effectors(self) -> None:
        if self._end_effectors is not None:
            self._end_effectors.shutdown()
            self._end_effectors = None

    def stop(self) -> TeleopSnapshot:
        # The Stop button stops the teleop control loop. Engagement is cleared
        # because Engage requires a running loop.
        if self._controller is None:
            return self.snapshot()
        try:
            self._teardown_end_effectors()
            self._controller.Stop()
            self._started = False
            self._engaged = False
            self._error = None
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
        return self.snapshot()

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
            # Gripper control is independent of engage/disengage now: the mirror
            # thread is owned by the Gripper Control panel's Start/Stop buttons
            # and keeps running across engage cycles.
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
        return self.snapshot()

    def gripper_snapshot(self) -> dict[str, Any]:
        # Empty until teleop is started (grippers are enabled in start()).
        if self._end_effectors is None:
            return {}
        return self._end_effectors.gripper_snapshot()

    def _manual_blocked(self, side: str) -> str | None:
        # Manual control of an effector is only allowed when the mirror thread
        # does not own this side: a side with a leader trigger is mirrored while
        # end-effector control is running, and a competing command would fight
        # the leader-driven one. A side without a leader trigger is never
        # mirrored, so it stays manually controllable.
        if self._end_effectors is None:
            return "Start end effector control before controlling it"
        if self._end_effectors.is_running() and self._end_effectors.is_mirrored(side):
            return "Stop end effector control to manually control this side"
        return None

    def command_gripper(
        self, side: str, action: str, velocity: float, force: float
    ) -> dict[str, Any]:
        blocked = self._manual_blocked(side)
        if blocked is not None:
            return {"ok": False, "error": blocked}
        try:
            self._end_effectors.command_gripper(side, action, velocity, force)
            return {"ok": True}
        except Exception as exc:  # pragma: no cover - hardware specific
            return {"ok": False, "error": describe_exception(exc)}

    def command_digital_output(self, side: str, high: bool) -> dict[str, Any]:
        # Manually drive a digital-output side's configured port high/low, gated
        # the same way as manual gripper control.
        blocked = self._manual_blocked(side)
        if blocked is not None:
            return {"ok": False, "error": blocked}
        try:
            self._end_effectors.command_digital_output(side, high)
            return {"ok": True}
        except Exception as exc:  # pragma: no cover - hardware specific
            return {"ok": False, "error": describe_exception(exc)}

    def set_gripper_params(
        self, side: str, velocity: float, force: float
    ) -> dict[str, Any]:
        # Store the panel's velocity/force for this side; used by both manual
        # control and the engaged mirror loop. Allowed any time after init
        # (including while engaged, to tune the mirror live) -- it only records
        # values, it does not move the gripper.
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

        # Disconnect / service reset is the only path that fully stops the
        # teleop control loop (Stop()); the Stop button merely disengages.
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
            # HomeAll() throws if the teleop control loop is running. The UI
            # gates the Home button behind Stop, but guard here as well.
            return {
                "ok": False,
                "error": "Stop teleoperation before homing the robots",
            }

        home_all = getattr(self._controller, "HomeAll", None)
        if not callable(home_all):
            return {
                "ok": False,
                "error": "Connected controller does not support HomeAll()",
            }

        try:
            # HomeAll() is blocking and moves every connected robot to its home
            # posture simultaneously, so no per-robot wait is needed.
            home_all()
        except Exception as exc:  # pragma: no cover - hardware specific
            return {"ok": False, "error": describe_exception(exc)}

        return {"ok": True, "warnings": []}

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

        return "Teleoperation fault detected" if faulted else None

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
