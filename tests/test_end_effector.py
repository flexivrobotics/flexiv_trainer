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

import flexivtrainer.teleop.end_effector as ee
from flexivtrainer.config import EndEffectorSideConfig


class FakeMode:
    IDLE = "IDLE"


class FakeFollower:
    def __init__(self, mode: str = FakeMode.IDLE) -> None:
        self.digital_outputs: dict[int, bool] = {}
        self._mode = mode

    def SetDigitalOutputs(self, outputs: dict[int, bool]) -> None:
        self.digital_outputs.update(outputs)

    def mode(self) -> str:
        return self._mode


class FakeGripper:
    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.enabled_name: str | None = None
        self.moves: list[tuple[float, float, float]] = []

    def Enable(self, name: str) -> None:
        self.enabled_name = name

    def Init(self) -> None:
        self.init_count = getattr(self, "init_count", 0) + 1

    def params(self) -> SimpleNamespace:
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

    def states(self) -> SimpleNamespace:
        return SimpleNamespace(width=0.042, force=-3.5, is_moving=False)


class FakeTool:
    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.switched_to: str | None = None

    def Switch(self, name: str) -> None:
        self.switched_to = name


class FakeTDK:
    """Mimics the TDK ``digital_inputs(idx)`` / ``instances(idx)`` surface."""

    def __init__(self, follower: FakeFollower, leader_ports: list[bool]) -> None:
        self._follower = follower
        self._leader_ports = leader_ports

    def digital_inputs(self, idx: int):
        return (list(self._leader_ports), [False] * 18)

    def instances(self, idx: int):
        return (object(), self._follower)


def _di_config(**overrides) -> EndEffectorSideConfig:
    base = dict(
        leader="digital_input",
        leader_channel=0,
        leader_activating_state="high",
        follower="digital_output",
        follower_channel=1,
        follower_activated_state="high",
    )
    base.update(overrides)
    return EndEffectorSideConfig(**base)


def test_has_work_only_when_leader_and_follower_configured() -> None:
    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)

    # Leader none -> nothing to read from.
    none_leader = ee.EndEffectorController(
        tdk, ["left_arm"], {"left_arm": EndEffectorSideConfig(follower="gripper")}
    )
    assert none_leader.has_work() is False

    # Follower none -> nothing to drive.
    none_follower = ee.EndEffectorController(
        tdk,
        ["left_arm"],
        {"left_arm": EndEffectorSideConfig(leader="digital_input")},
    )
    assert none_follower.has_work() is False

    configured = ee.EndEffectorController(tdk, ["left_arm"], {"left_arm": _di_config()})
    assert configured.has_work() is True


def test_digital_output_mirrors_leader_high_activating() -> None:
    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    config = {"left_arm": _di_config(leader_channel=2, follower_channel=5)}
    ctl = ee.EndEffectorController(tdk, ["left_arm"], config)
    cfg = ctl._configs[0]

    # Not triggered -> port driven low (activated state is high).
    ctl._tick(0, cfg)
    assert follower.digital_outputs == {5: False}

    # Leader DI[2] high -> triggered -> port high.
    tdk._leader_ports[2] = True
    ctl._tick(0, cfg)
    assert follower.digital_outputs == {5: True}


def test_digital_output_respects_low_activating_and_activated_states() -> None:
    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    cfg = _di_config(
        leader_channel=0,
        leader_activating_state="low",
        follower_channel=3,
        follower_activated_state="low",
    )
    ctl = ee.EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})

    # DI low + activating "low" => triggered; activated state "low" => port low.
    tdk._leader_ports[0] = False
    ctl._tick(0, ctl._configs[0])
    assert follower.digital_outputs == {3: False}

    # DI high + activating "low" => not triggered => opposite of activated => high.
    tdk._leader_ports[0] = True
    ctl._tick(0, ctl._configs[0])
    assert follower.digital_outputs == {3: True}


def test_digital_output_only_writes_on_change() -> None:
    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    ctl = ee.EndEffectorController(tdk, ["left_arm"], {"left_arm": _di_config()})

    writes: list[dict[int, bool]] = []
    follower.SetDigitalOutputs = lambda d: writes.append(dict(d))  # type: ignore[assignment]

    ctl._tick(0, ctl._configs[0])
    ctl._tick(0, ctl._configs[0])  # unchanged -> no second write
    assert len(writes) == 1


def test_setup_enables_switches_and_inits_gripper_when_idle(monkeypatch) -> None:
    # Setup must enable the gripper, switch the tool (for gravity compensation)
    # while the robot is IDLE, and trigger the gripper's own Init() when asked.
    tools: list[FakeTool] = []

    def _make_tool(robot: object) -> FakeTool:
        tool = FakeTool(robot)
        tools.append(tool)
        return tool

    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", _make_tool)
    monkeypatch.setattr(ee, "Mode", FakeMode)

    follower = FakeFollower(mode=FakeMode.IDLE)
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(
        leader="digital_input", follower="gripper", gripper_model="Flexiv-GN01"
    )
    ctl = ee.EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})

    ctl._setup_gripper(0, ctl._configs[0])
    assert ctl._grippers[0].enabled_name == "Flexiv-GN01"
    assert tools[-1].switched_to == "Flexiv-GN01"
    assert getattr(ctl._grippers[0], "init_count", 0) == 1


def test_setup_skips_tool_switch_when_not_idle(monkeypatch) -> None:
    # Tool.Switch() is IDLE-only; the IDLE guard skips it when the follower is
    # not IDLE so it can't trip a control-mode mismatch, while the gripper is
    # still enabled and initialized. (The panel gates Init to teleop-not-started,
    # so this guard is a safety net.)
    tools: list[FakeTool] = []

    def _make_tool(robot: object) -> FakeTool:
        tool = FakeTool(robot)
        tools.append(tool)
        return tool

    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", _make_tool)
    monkeypatch.setattr(ee, "Mode", FakeMode)

    follower = FakeFollower(mode="RT_CARTESIAN")  # not IDLE
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(
        leader="digital_input", follower="gripper", gripper_model="Flexiv-GN01"
    )
    ctl = ee.EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})

    ctl._setup_gripper(0, ctl._configs[0])
    assert ctl._grippers[0].enabled_name == "Flexiv-GN01"
    assert tools == []  # switch skipped: follower not IDLE
    assert getattr(ctl._grippers[0], "init_count", 0) == 1  # Init still triggered


def test_uninitialized_gripper_is_skipped_not_errored(monkeypatch) -> None:
    # Without Init, no gripper is enabled, so the mirror tick skips it silently
    # (the panel shows a warning) rather than raising and stopping the loop.
    monkeypatch.setattr(ee, "Gripper", FakeGripper)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(
        leader="digital_input", follower="gripper", gripper_activated_state="close"
    )
    ctl = ee.EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})

    # Should not raise, and no gripper was enabled to command.
    ctl._tick(0, ctl._configs[0])
    assert ctl._grippers == {}


def test_gripper_enabled_without_params_is_skipped_not_errored(monkeypatch) -> None:
    # If setup partially succeeds (Enable() populates _grippers but params()
    # throws, leaving _gripper_params empty), the mirror tick must skip the
    # gripper instead of raising KeyError on _gripper_params[index] every tick.
    monkeypatch.setattr(ee, "Gripper", FakeGripper)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(
        leader="digital_input", follower="gripper", gripper_activated_state="close"
    )
    ctl = ee.EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})

    # Simulate the partial state: a gripper object exists, but no params cached.
    ctl._grippers[0] = FakeGripper(object())
    assert 0 not in ctl._gripper_params

    # Should not raise, and no Move() was issued (nothing to command safely).
    ctl._tick(0, ctl._configs[0])
    assert ctl._grippers[0].moves == []


def test_gripper_moves_to_activated_state(monkeypatch) -> None:
    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", FakeTool)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(
        leader="digital_input",
        leader_channel=0,
        leader_activating_state="high",
        follower="gripper",
        gripper_model="Flexiv-GN01",
        gripper_activated_state="close",
    )
    ctl = ee.EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})
    ctl._setup_gripper(0, ctl._configs[0])

    # Idle: not triggered, activated state "close" -> open (max width).
    ctl._tick(0, ctl._configs[0])
    gripper = ctl._grippers[0]
    assert gripper.enabled_name == "Flexiv-GN01"
    assert gripper.moves[-1][0] == 0.1  # max_width (open)

    # Trigger -> close (min width). Velocity/force clamped into params range.
    tdk._leader_ports[0] = True
    ctl._tick(0, ctl._configs[0])
    assert gripper.moves[-1][0] == 0.0  # min_width (close)
    # No panel value set yet -> defaults derived from the gripper's own params:
    # max velocity, and DEFAULT_FORCE_FRACTION of max force.
    assert gripper.moves[-1][1] == 0.5  # max_vel
    assert gripper.moves[-1][2] == (
        50.0 * ee.EndEffectorController.DEFAULT_FORCE_FRACTION
    )

    # No state change -> no extra move.
    moves_before = len(gripper.moves)
    ctl._tick(0, ctl._configs[0])
    assert len(gripper.moves) == moves_before


def test_side_index_maps_to_pair_index() -> None:
    # The second side ("right_arm") must drive pair index 1.
    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    tdk._leader_ports[4] = True
    ctl = ee.EndEffectorController(
        tdk,
        ["left_arm", "right_arm"],
        {"right_arm": _di_config(leader_channel=4, follower_channel=7)},
    )

    assert ctl._configs[0] is None
    assert ctl._configs[1] is not None
    ctl._tick(1, ctl._configs[1])
    assert follower.digital_outputs == {7: True}


def test_gripper_snapshot_exposes_params_after_prepare(monkeypatch) -> None:
    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", FakeTool)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    # Gripper follower with NO leader trigger: still set up and reported.
    cfg = EndEffectorSideConfig(follower="gripper", gripper_model="Flexiv-GN01")
    ctl = ee.EndEffectorController(tdk, ["single_arm"], {"single_arm": cfg})

    # Nothing before prepare().
    assert ctl.gripper_snapshot() == {}
    assert ctl.has_work() is False  # no leader -> no mirror thread work

    ctl._setup_gripper(0, ctl._configs[0])
    snap = ctl.gripper_snapshot()
    assert set(snap) == {"single_arm"}
    entry = snap["single_arm"]
    assert entry["model"] == "fake"
    assert entry["max_vel"] == 0.5
    assert entry["max_force"] == 50.0
    assert entry["max_width"] == 0.1


def test_gripper_states_by_index_reports_width_and_force(monkeypatch) -> None:
    # Recording reads each enabled gripper's measured width/force keyed by pair
    # index; a side with no gripper (or no Init) is simply absent.
    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", FakeTool)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(follower="gripper", gripper_model="Flexiv-GN01")
    ctl = ee.EndEffectorController(
        tdk, ["left_arm", "right_arm"], {"left_arm": cfg}
    )

    # Nothing before the gripper is enabled.
    assert ctl.gripper_states_by_index() == {}

    ctl._setup_gripper(0, ctl._configs[0])
    states = ctl.gripper_states_by_index()
    # Only the configured/enabled left side (index 0) reports; right side absent.
    assert states == {0: {"width": 0.042, "force": -3.5}}


def test_command_params_apply_to_mirror_loop(monkeypatch) -> None:
    # The panel sliders' velocity/force (set_command_params) must drive the
    # mirror loop's Move() calls.
    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", FakeTool)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    # Gripper mirrored from a leader DI.
    cfg = EndEffectorSideConfig(
        leader="digital_input",
        leader_channel=0,
        follower="gripper",
        gripper_activated_state="close",
    )
    ctl = ee.EndEffectorController(tdk, ["single_arm"], {"single_arm": cfg})
    ctl._setup_gripper(0, ctl._configs[0])
    gripper = ctl._grippers[0]

    # Panel sets velocity/force (clamped into [0.01,0.5] / [1,50]).
    stored = ctl.set_command_params("single_arm", velocity=0.3, force=12.0)
    assert stored == (0.3, 12.0)

    # Mirror tick: leader triggered -> close, using the panel velocity/force.
    tdk._leader_ports[0] = True
    ctl._tick(0, ctl._configs[0])
    assert gripper.moves[-1] == (0.0, 0.3, 12.0)  # min_width, panel vel/force


def test_single_arm_uses_pair_index_zero() -> None:
    # Single mode: the lone "single_arm" side maps to pair index 0.
    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    tdk._leader_ports[1] = True
    ctl = ee.EndEffectorController(
        tdk,
        ["single_arm"],
        {"single_arm": _di_config(leader_channel=1, follower_channel=9)},
    )

    assert ctl.has_work() is True
    assert len(ctl._configs) == 1
    ctl._tick(0, ctl._configs[0])
    assert follower.digital_outputs == {9: True}
