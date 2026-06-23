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

from types import SimpleNamespace

from flexivtrainer.config import AppSettings, StorageConfig, TeleopRobotPair
from flexivtrainer.teleop.service import TeleopService


class FakeRobot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    def ExecutePrimitive(self, primitive: str, params: dict | None = None) -> None:
        self.calls.append((primitive, params))

    def connected(self) -> bool:
        return True

    def states(self) -> dict[str, list[float]]:
        return {
            "tcp_pose": [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            "tcp_vel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "ext_wrench_in_world": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }

    def actions(self) -> dict[str, list[float]]:
        return {
            "tcp_pose_d": [10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0],
            "tcp_vel_d": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            "ext_wrench_d": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        }

    def primitive_states(self) -> dict[str, bool]:
        return {"reachedTarget": True}


class FakeController:
    """Mimics the TDK ``instances(idx)`` accessor, which returns the
    (leader, follower) rdk::Robot handles of the pair."""

    def __init__(self, robots: tuple[FakeRobot, ...]) -> None:
        # Treat each robot as the follower of its pair.
        self._robots = robots
        self.home_all_calls = 0

    def instances(self, idx: int):
        follower = self._robots[idx]
        return (follower, follower)

    def HomeAll(self) -> None:
        self.home_all_calls += 1


class _StructStates:
    """No ``__dict__`` — mimics a pybind11 ``RobotStates`` struct."""

    __slots__ = ("tcp_pose", "tcp_vel", "ext_wrench_in_world")

    def __init__(self) -> None:
        self.tcp_pose = [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0]
        self.tcp_vel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.ext_wrench_in_world = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


class _StructActions:
    __slots__ = ("tcp_pose_d", "tcp_vel_d", "ext_wrench_d")

    def __init__(self) -> None:
        self.tcp_pose_d = [10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0]
        self.tcp_vel_d = [10.1, 10.2, 10.3, 10.4, 10.5, 10.6]
        self.ext_wrench_d = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]


class FakeStructRobot(FakeRobot):
    """Returns struct-style state/action objects instead of dicts."""

    def states(self) -> _StructStates:  # type: ignore[override]
        return _StructStates()

    def actions(self) -> _StructActions:  # type: ignore[override]
        return _StructActions()


class FakeTeleopController:
    """Mimics the relevant TDK ``TransparentCartesianTeleopLAN`` surface.

    ``stopped`` and ``fault``/``any_fault`` are *methods* on the real
    controller, so reading them as attributes returns a (truthy) bound method.
    """

    def __init__(self) -> None:
        self._faulted = False
        self.init_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        # Per-pair indices passed to StopWithIdx(); the service prefers this over
        # the global Stop() so a faulted pair cannot abort stopping the rest.
        self.stop_idx_calls: list[int] = []
        # Pair indices whose StopWithIdx() should raise, simulating a faulted
        # robot whose stop fails while the operational pairs still stop.
        self.stop_idx_raises: set[int] = set()
        self.engage_calls: list[tuple[int, bool]] = []
        self.init_zero_ft_sensor: object = None
        self.clear_fault_calls: list[int] = []
        # When True, ClearFault() succeeds at clearing the fault; when False it
        # runs but leaves the controller faulted (the "fault persists" branch).
        self.clear_fault_succeeds = True

    def Init(self, zero_ft_sensor: object = None) -> None:
        self.init_calls += 1
        self.init_zero_ft_sensor = zero_ft_sensor

    def Start(self) -> None:
        self.start_calls += 1

    def Stop(self) -> None:
        self.stop_calls += 1

    def StopWithIdx(self, idx: int) -> None:
        if idx in self.stop_idx_raises:
            raise RuntimeError(f"failed to stop pair {idx}")
        self.stop_idx_calls.append(idx)

    def Engage(self, idx: int, engaged: bool) -> None:
        self.engage_calls.append((idx, engaged))

    def stopped(self, index: int = 0) -> bool:
        return True

    def fault(self, index: int = 0) -> bool:
        return self._faulted

    def any_fault(self) -> bool:
        return self._faulted

    def ClearFault(self, timeout_sec: int = 30) -> None:
        self.clear_fault_calls.append(timeout_sec)
        if self.clear_fault_succeeds:
            self._faulted = False


class FakeControllerWithoutClearFault:
    """Older controller surface that does not expose ClearFault()."""

    def any_fault(self) -> bool:
        return True


def test_snapshot_reports_not_started_after_initialize_only(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    # Simulate a connected-but-not-started controller.
    service._controller = FakeTeleopController()
    service._initialized = True

    snapshot = service.snapshot()

    assert snapshot.initialized is True
    assert snapshot.started is False
    assert snapshot.stopped is True


def test_snapshot_does_not_report_spurious_fault_from_method(tmp_path) -> None:
    # Regression: the controller exposes ``fault``/``any_fault`` as methods.
    # Reading ``fault`` as an attribute used to yield a truthy bound method and
    # report a permanent fault, which kept the Start button disabled.
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    service._controller = FakeTeleopController()
    service._initialized = True

    assert service.snapshot().fault is None


def test_snapshot_reports_fault_when_any_fault_is_true(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    controller = FakeTeleopController()
    controller._faulted = True
    service._controller = controller
    service._initialized = True

    assert service.snapshot().fault is not None


def test_clear_fault_clears_when_controller_recovers(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    controller = FakeTeleopController()
    controller._faulted = True
    service._controller = controller

    result = service.clear_fault()

    assert controller.clear_fault_calls == [30]
    assert result == {"ok": True, "cleared": True, "error": None}
    # Re-derived from any_fault(), which now reports clear.
    assert service.snapshot().fault is None


def test_clear_fault_reports_fault_persists_when_still_faulted(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    controller = FakeTeleopController()
    controller._faulted = True
    controller.clear_fault_succeeds = False
    service._controller = controller

    result = service.clear_fault()

    assert controller.clear_fault_calls == [30]
    assert result["ok"] is False
    assert result["cleared"] is False
    assert result["error"] == "Fault persists after ClearFault"


def test_clear_fault_errors_when_controller_unsupported(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    service._controller = FakeControllerWithoutClearFault()

    result = service.clear_fault()

    assert result["ok"] is False
    assert result["cleared"] is False
    assert result["error"] == "Controller does not support ClearFault"


def test_clear_fault_errors_when_not_connected(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))

    result = service.clear_fault()

    assert result["ok"] is False
    assert result["cleared"] is False
    assert result["error"] == "Teleoperation not connected"


def test_start_then_stop_tracks_started_flag(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    service._controller = FakeTeleopController()
    service._initialized = True

    started = service.start()
    assert started.started is True
    assert started.stopped is False

    stopped = service.stop()
    assert stopped.started is False
    assert stopped.stopped is True


def _configured_service(tmp_path) -> TeleopService:
    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    pairs = [
        TeleopRobotPair(leader_serial="LEADER_A", follower_serial="FOLLOWER_A"),
        TeleopRobotPair(leader_serial="LEADER_B", follower_serial="FOLLOWER_B"),
    ]
    service = TeleopService(settings, get_robot_pairs=lambda: pairs)
    service._controller = FakeTeleopController()
    service._initialized = True
    return service


def test_start_and_stop_do_not_engage_pairs(tmp_path) -> None:
    # Start only runs the control loop; Stop only stops it. Neither touches
    # engagement, which is a separate action.
    service = _configured_service(tmp_path)
    controller = service._controller

    started = service.start()
    assert controller.start_calls == 1
    assert started.started is True
    assert started.engaged is False

    stopped = service.stop()
    # Two configured pairs -> per-pair StopWithIdx for each, not global Stop().
    assert controller.stop_idx_calls == [0, 1]
    assert controller.stop_calls == 0
    assert controller.engage_calls == []
    assert stopped.started is False
    assert stopped.engaged is False


def test_stop_halts_operational_pairs_when_one_pair_fails(tmp_path) -> None:
    # Regression: with one arm in fault, the global Stop() raises and used to
    # abort before the operational pairs were stopped, leaving them in teleop.
    # Stopping per-pair must isolate the faulted pair so the rest still stop,
    # and the loop must report stopped regardless.
    service = _configured_service(tmp_path)
    controller = service._controller
    service.start()

    # Pair 0 (faulted) fails to stop; pair 1 (operational) must still stop.
    controller.stop_idx_raises = {0}

    stopped = service.stop()

    assert controller.stop_idx_calls == [1]
    assert stopped.started is False
    assert stopped.engaged is False
    # The failure is surfaced as an error, but the loop is still marked stopped.
    assert stopped.error is not None


def test_stop_marks_loop_stopped_even_when_all_pairs_fail(tmp_path) -> None:
    # Even if every pair's stop fails, teleop must be treated as halted: the TDK
    # contract requires Init()+Start() to resume, so the loop is no longer the
    # controlling process and the Stop button must not keep re-arming uselessly.
    service = _configured_service(tmp_path)
    controller = service._controller
    service.start()
    controller.stop_idx_raises = {0, 1}

    stopped = service.stop()

    assert stopped.started is False
    assert stopped.stopped is True
    assert stopped.error is not None


def test_start_always_calls_init_before_start(tmp_path) -> None:
    # Restarting after a Stop must re-run Init() before Start(), so every Start
    # pairs an Init() with the Start().
    service = _configured_service(tmp_path)
    controller = service._controller

    service.start()
    assert (controller.init_calls, controller.start_calls) == (1, 1)

    service.stop()
    service.start()
    assert (controller.init_calls, controller.start_calls) == (2, 2)


def test_start_passes_zero_ft_sensor_flag_to_init(tmp_path) -> None:
    # The Start button's "Zero force sensors on start" checkbox maps to Init()'s
    # zero_ft_sensor flag (Enable when checked, Disable when unchecked).
    from flexivtrainer.teleop import service as teleop_service

    service = _configured_service(tmp_path)
    controller = service._controller

    service.start(zero_ft_sensor=True)
    if teleop_service.ZeroFTSensor is not None:
        assert controller.init_zero_ft_sensor == teleop_service.ZeroFTSensor.Enable
    else:
        # Without the TDK bindings available, Init() is called argument-free.
        assert controller.init_zero_ft_sensor is None

    service.stop()
    service.start(zero_ft_sensor=False)
    if teleop_service.ZeroFTSensor is not None:
        assert controller.init_zero_ft_sensor == teleop_service.ZeroFTSensor.Disable
    else:
        assert controller.init_zero_ft_sensor is None


def test_engage_then_disengage_toggles_every_pair(tmp_path) -> None:
    service = _configured_service(tmp_path)
    controller = service._controller

    service.start()

    engaged = service.set_engaged(True)
    assert controller.engage_calls == [(0, True), (1, True)]
    assert engaged.engaged is True
    # Engaging must not restart or stop the control loop.
    assert controller.start_calls == 1
    assert controller.stop_calls == 0

    disengaged = service.set_engaged(False)
    assert controller.engage_calls[-2:] == [(0, False), (1, False)]
    assert disengaged.engaged is False


class _FakeGripper:
    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.moves: list[tuple[float, float, float]] = []

    def Enable(self, name: str) -> None:
        self.enabled_name = name

    def Init(self) -> None:
        pass

    def params(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            name="fake",
            min_width=0.0,
            max_width=0.1,
            min_vel=0.01,
            max_vel=0.5,
            min_force=1.0,
            max_force=50.0,
        )

    def Move(self, width: float, velocity: float, force_limit: float) -> None:
        self.moves.append((width, velocity, force_limit))


class _FakeTool:
    def __init__(self, robot: object) -> None:
        self.robot = robot

    def Switch(self, name: str) -> None:
        pass


class FakeEngageController(FakeTeleopController):
    """Adds the DI/instance surface the end effector controller polls."""

    def __init__(self) -> None:
        super().__init__()
        self.follower = FakeRobot()
        self.follower.digital_outputs = {}

        def _set_outputs(outputs: dict) -> None:
            self.follower.digital_outputs.update(outputs)

        self.follower.SetDigitalOutputs = _set_outputs  # type: ignore[attr-defined]

    def digital_inputs(self, idx: int):
        return ([False] * 18, [False] * 18)

    def instances(self, idx: int):
        return (object(), self.follower)


def test_mirror_thread_tied_to_teleop_start_stop(tmp_path) -> None:
    from flexivtrainer.config import EndEffectorSideConfig

    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    pairs = [TeleopRobotPair(leader_serial="LEADER_A", follower_serial="FOLLOWER_A")]
    configs = {
        "left_arm": EndEffectorSideConfig(
            leader="digital_input",
            follower="digital_output",
            follower_channel=2,
        )
    }
    service = TeleopService(
        settings,
        get_robot_pairs=lambda: pairs,
        get_active_sides=lambda: ["left_arm"],
        get_end_effector_config=lambda: configs,
    )
    service._controller = FakeEngageController()
    service._initialized = True

    # Teleop Start runs the mirror thread; it is independent of engage.
    service.start()
    assert service._end_effectors is not None
    assert service._end_effectors.is_running() is True

    service.set_engaged(True)
    assert service._end_effectors.is_running() is True
    service.set_engaged(False)
    assert service._end_effectors.is_running() is True

    # Teleop Stop stops the thread but keeps the controller (params cached).
    service.stop()
    assert service._end_effectors is not None
    assert service._end_effectors.is_running() is False

    # A re-Start runs the thread again on the same controller.
    service.start()
    assert service._end_effectors.is_running() is True

    # Disconnect fully tears the end effectors down.
    service.shutdown()
    assert service._end_effectors is None


def test_init_grippers_gated_and_cached(tmp_path, monkeypatch) -> None:
    import flexivtrainer.teleop.end_effector as ee
    from flexivtrainer.config import EndEffectorSideConfig

    # Lightweight fakes so gripper setup doesn't reach real RDK hardware.
    monkeypatch.setattr(ee, "Gripper", _FakeGripper)
    monkeypatch.setattr(ee, "Tool", _FakeTool)
    monkeypatch.setattr(ee, "Mode", None)  # skip the IDLE-only tool switch

    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    pairs = [TeleopRobotPair(leader_serial="LEADER_A", follower_serial="FOLLOWER_A")]
    configs = {
        "single_arm": EndEffectorSideConfig(
            leader="digital_input", follower="gripper"
        )
    }
    service = TeleopService(
        settings,
        get_robot_pairs=lambda: pairs,
        get_active_sides=lambda: ["single_arm"],
        get_end_effector_config=lambda: configs,
    )
    service._controller = FakeEngageController()
    service._initialized = True

    # Init enables the gripper and exposes its params.
    result = service.init_grippers()
    assert result["ok"] is True
    assert "single_arm" in service.gripper_snapshot()

    # Params stay cached across a teleop start/stop cycle.
    service.start()
    assert service._end_effectors.is_running() is True
    service.stop()
    assert "single_arm" in service.gripper_snapshot()

    # Init is rejected while teleop is started (Tool.Switch is IDLE-only).
    service.start()
    blocked = service.init_grippers()
    assert blocked["ok"] is False
    assert "Stop teleoperation" in blocked["error"]
    service.stop()


def test_engage_requires_started_control_loop(tmp_path) -> None:
    service = _configured_service(tmp_path)
    controller = service._controller

    snapshot = service.set_engaged(True)

    assert controller.engage_calls == []
    assert snapshot.engaged is False
    assert snapshot.error is not None


def test_shutdown_stops_the_control_loop(tmp_path) -> None:
    # Disconnect / service reset go through shutdown(), which is the only path
    # that tears down the controller.
    service = _configured_service(tmp_path)
    controller = service._controller

    service.start()
    service.shutdown()

    assert controller.stop_calls == 1
    assert service._controller is None


def test_can_home_when_connected_and_not_running(tmp_path) -> None:
    service = _configured_service(tmp_path)

    # Connected but not started: homing is available right after Connect.
    assert service.snapshot().can_home is True

    # Running: homing is disabled.
    assert service.start().can_home is False

    # Stopped again: homing is available.
    assert service.stop().can_home is True


def test_reset_home_calls_home_all_when_stopped(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    controller = FakeController((FakeRobot(), FakeRobot()))
    service._controller = controller

    result = service.reset_home()

    assert result == {"ok": True, "warnings": []}
    assert controller.home_all_calls == 1


def test_reset_home_is_blocked_while_teleop_running(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    controller = FakeController((FakeRobot(),))
    service._controller = controller
    service._started = True

    result = service.reset_home()

    assert result["ok"] is False
    assert controller.home_all_calls == 0


def test_reset_home_errors_when_home_all_unsupported(tmp_path) -> None:
    service = TeleopService(AppSettings(storage=StorageConfig(root=tmp_path)))
    service._controller = SimpleNamespace()

    result = service.reset_home()

    assert result["ok"] is False
    assert "HomeAll" in str(result["error"])


def test_robot_data_snapshot_uses_instance_states_and_actions(tmp_path) -> None:
    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    pairs = [
        TeleopRobotPair(leader_serial="LEADER_A", follower_serial="FOLLOWER_A"),
        TeleopRobotPair(leader_serial="LEADER_B", follower_serial="FOLLOWER_B"),
    ]
    service = TeleopService(settings, get_robot_pairs=lambda: pairs)
    service._controller = FakeController((FakeRobot(), FakeRobot()))

    snapshot = service.robot_data_snapshot()

    assert set(snapshot["robots"]) == {"FOLLOWER_A", "FOLLOWER_B"}
    first = snapshot["robots"]["FOLLOWER_A"]
    assert first["connected"] is True
    assert first["states"]["tcp_pose"] == [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    assert first["actions"]["tcp_pose_d"] == [10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0]


def test_robot_data_snapshot_folds_in_gripper_states(tmp_path) -> None:
    # A follower configured as a gripper contributes its measured width/force to
    # the snapshot, keyed by pair index, for recording into state and action.
    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    pairs = [
        TeleopRobotPair(leader_serial="LEADER_A", follower_serial="FOLLOWER_A"),
        TeleopRobotPair(leader_serial="LEADER_B", follower_serial="FOLLOWER_B"),
    ]
    service = TeleopService(settings, get_robot_pairs=lambda: pairs)
    service._controller = FakeController((FakeRobot(), FakeRobot()))
    service._end_effectors = SimpleNamespace(
        gripper_states_by_index=lambda: {0: {"width": 0.03, "force": -2.0}}
    )

    snapshot = service.robot_data_snapshot()

    # Index 0 (FOLLOWER_A) gets gripper telemetry; index 1 does not.
    assert snapshot["robots"]["FOLLOWER_A"]["gripper"] == {
        "width": 0.03,
        "force": -2.0,
    }
    assert "gripper" not in snapshot["robots"]["FOLLOWER_B"]


def test_robot_data_snapshot_skips_gripper_read_without_states_or_actions(
    tmp_path,
) -> None:
    # A bare snapshot (neither states nor actions requested) must not touch the
    # gripper layer, which on real hardware would issue a gripper.states() read.
    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    pairs = [TeleopRobotPair(leader_serial="LEADER_A", follower_serial="FOLLOWER_A")]
    service = TeleopService(settings, get_robot_pairs=lambda: pairs)
    service._controller = FakeController((FakeRobot(),))

    calls = {"count": 0}

    def _gripper_states():
        calls["count"] += 1
        return {0: {"width": 0.03, "force": -2.0}}

    service._end_effectors = SimpleNamespace(
        gripper_states_by_index=_gripper_states
    )

    snapshot = service.robot_data_snapshot(
        include_states=False, include_actions=False
    )

    assert calls["count"] == 0
    assert "gripper" not in snapshot["robots"]["FOLLOWER_A"]


def test_robot_data_snapshot_reads_struct_states_without_dict(tmp_path) -> None:
    # RobotStates/RobotActions from flexivrdk are pybind11 structs with no
    # __dict__; the snapshot must read their fields by attribute.
    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    pairs = [TeleopRobotPair(leader_serial="LEADER_A", follower_serial="FOLLOWER_A")]
    service = TeleopService(settings, get_robot_pairs=lambda: pairs)
    service._controller = FakeController((FakeStructRobot(),))

    snapshot = service.robot_data_snapshot()

    entry = snapshot["robots"]["FOLLOWER_A"]
    assert entry["states"]["ext_wrench_in_world"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert entry["actions"]["ext_wrench_d"] == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
