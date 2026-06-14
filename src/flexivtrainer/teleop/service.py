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

from flexivtrainer.config import AppSettings, TeleopRobotPair
from flexivtrainer.observability import describe_exception

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
    ) -> None:
        self._settings = settings
        self._get_robot_pairs = get_robot_pairs or (lambda: settings.teleop_robot_pairs)
        self._controller: Any | None = None
        self._error: str | None = None
        self._initialized = False
        # ``_started`` tracks whether the teleop control loop (Start()) is
        # running; ``_engaged`` tracks whether the pairs are currently engaged.
        # They are decoupled because the Stop button disengages without
        # stopping the loop.
        self._started = False
        self._engaged = False

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

    def start(self) -> TeleopSnapshot:
        # The Start button always runs Init() then Start(). Per the TDK contract
        # (see TransparentCartesianTeleopLAN::Start/Stop docs), restarting after
        # a Stop() requires calling Init() again first, so Init() is run on every
        # Start rather than only when the controller is first constructed. The
        # pairs stay disengaged by default; engaging is a separate action.
        self.initialize()
        if self._controller is None:
            return self.snapshot()
        try:
            init_method = getattr(self._controller, "Init", None)
            if callable(init_method):
                init_method()
            self._controller.Start()
            self._started = True
            self._engaged = False
            self._error = None
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
        return self.snapshot()

    def stop(self) -> TeleopSnapshot:
        # The Stop button stops the teleop control loop. Engagement is cleared
        # because Engage requires a running loop.
        if self._controller is None:
            return self.snapshot()
        try:
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
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
        return self.snapshot()

    def shutdown(self) -> None:
        if self._controller is None:
            return

        # Disconnect / service reset is the only path that fully stops the
        # teleop control loop (Stop()); the Stop button merely disengages.
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
