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

from flexivtrainer.config import EndEffectorSideConfig
from flexivtrainer.teleop.end_effector import EndEffectorController


class FakeFollower:
    def __init__(self) -> None:
        self.digital_outputs: dict[int, bool] = {}

    def SetDigitalOutputs(self, outputs: dict[int, bool]) -> None:
        self.digital_outputs.update(outputs)


class FakeGripper:
    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.enabled_name: str | None = None
        self.moves: list[tuple[float, float, float]] = []

    def Enable(self, name: str) -> None:
        self.enabled_name = name

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
    none_leader = EndEffectorController(
        tdk, ["left_arm"], {"left_arm": EndEffectorSideConfig(follower="gripper")}
    )
    assert none_leader.has_work() is False

    # Follower none -> nothing to drive.
    none_follower = EndEffectorController(
        tdk,
        ["left_arm"],
        {"left_arm": EndEffectorSideConfig(leader="digital_input")},
    )
    assert none_follower.has_work() is False

    configured = EndEffectorController(tdk, ["left_arm"], {"left_arm": _di_config()})
    assert configured.has_work() is True


def test_digital_output_mirrors_leader_high_activating() -> None:
    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    config = {"left_arm": _di_config(leader_channel=2, follower_channel=5)}
    ctl = EndEffectorController(tdk, ["left_arm"], config)
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
    ctl = EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})

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
    ctl = EndEffectorController(tdk, ["left_arm"], {"left_arm": _di_config()})

    writes: list[dict[int, bool]] = []
    follower.SetDigitalOutputs = lambda d: writes.append(dict(d))  # type: ignore[assignment]

    ctl._tick(0, ctl._configs[0])
    ctl._tick(0, ctl._configs[0])  # unchanged -> no second write
    assert len(writes) == 1


def test_prepare_enables_gripper_and_switches_tool(monkeypatch) -> None:
    # The gripper must be enabled and the tool switched (for gravity
    # compensation) during prepare() -- while the robot is IDLE -- not lazily
    # during the engaged mirror loop.
    import flexivtrainer.teleop.end_effector as ee

    tools: list[FakeTool] = []

    def _make_tool(robot: object) -> FakeTool:
        tool = FakeTool(robot)
        tools.append(tool)
        return tool

    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", _make_tool)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(
        leader="digital_input", follower="gripper", gripper_model="Flexiv-GN01"
    )
    ctl = EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})

    ctl.initialize_grippers()
    assert ctl._grippers[0].enabled_name == "Flexiv-GN01"
    assert tools[-1].switched_to == "Flexiv-GN01"


def test_gripper_requires_prepare_before_moving(monkeypatch) -> None:
    # Without prepare(), no gripper is enabled, so driving it raises rather than
    # silently doing nothing.
    import flexivtrainer.teleop.end_effector as ee

    monkeypatch.setattr(ee, "Gripper", FakeGripper)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(
        leader="digital_input", follower="gripper", gripper_activated_state="close"
    )
    ctl = EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})

    try:
        ctl._tick(0, ctl._configs[0])
    except RuntimeError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("expected RuntimeError when gripper not prepared")


def test_gripper_moves_to_activated_state(monkeypatch) -> None:
    import flexivtrainer.teleop.end_effector as ee

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
    ctl = EndEffectorController(tdk, ["left_arm"], {"left_arm": cfg})
    ctl.initialize_grippers()

    # Idle: not triggered, activated state "close" -> open (max width).
    ctl._tick(0, ctl._configs[0])
    gripper = ctl._grippers[0]
    assert gripper.enabled_name == "Flexiv-GN01"
    assert gripper.moves[-1][0] == 0.1  # max_width (open)

    # Trigger -> close (min width). Velocity/force clamped into params range.
    tdk._leader_ports[0] = True
    ctl._tick(0, ctl._configs[0])
    assert gripper.moves[-1][0] == 0.0  # min_width (close)
    # Velocity/force are the controller defaults, clamped into the params range.
    assert gripper.moves[-1][1] == EndEffectorController.GRIPPER_VELOCITY
    assert gripper.moves[-1][2] == EndEffectorController.GRIPPER_FORCE

    # No state change -> no extra move.
    moves_before = len(gripper.moves)
    ctl._tick(0, ctl._configs[0])
    assert len(gripper.moves) == moves_before


def test_side_index_maps_to_pair_index() -> None:
    # The second side ("right_arm") must drive pair index 1.
    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    tdk._leader_ports[4] = True
    ctl = EndEffectorController(
        tdk,
        ["left_arm", "right_arm"],
        {"right_arm": _di_config(leader_channel=4, follower_channel=7)},
    )

    assert ctl._configs[0] is None
    assert ctl._configs[1] is not None
    ctl._tick(1, ctl._configs[1])
    assert follower.digital_outputs == {7: True}


def test_gripper_snapshot_exposes_params_after_prepare(monkeypatch) -> None:
    import flexivtrainer.teleop.end_effector as ee

    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", FakeTool)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    # Gripper follower with NO leader trigger: still set up and reported.
    cfg = EndEffectorSideConfig(follower="gripper", gripper_model="Flexiv-GN01")
    ctl = EndEffectorController(tdk, ["single_arm"], {"single_arm": cfg})

    # Nothing before prepare().
    assert ctl.gripper_snapshot() == {}
    assert ctl.has_work() is False  # no leader -> no mirror thread work

    ctl.initialize_grippers()
    snap = ctl.gripper_snapshot()
    assert set(snap) == {"single_arm"}
    entry = snap["single_arm"]
    assert entry["model"] == "fake"
    assert entry["max_vel"] == 0.5
    assert entry["max_force"] == 50.0
    assert entry["max_width"] == 0.1


def test_command_gripper_moves_and_clamps(monkeypatch) -> None:
    import flexivtrainer.teleop.end_effector as ee

    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", FakeTool)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(follower="gripper", gripper_model="Flexiv-GN01")
    ctl = EndEffectorController(tdk, ["single_arm"], {"single_arm": cfg})
    ctl.initialize_grippers()
    gripper = ctl._grippers[0]

    # Open at an in-range velocity/force.
    ctl.command_gripper("single_arm", "open", velocity=0.3, force=10.0)
    assert gripper.moves[-1] == (0.1, 0.3, 10.0)  # max_width

    # Close, with out-of-range inputs clamped to the params range.
    ctl.command_gripper("single_arm", "close", velocity=99.0, force=99.0)
    assert gripper.moves[-1] == (0.0, 0.5, 50.0)  # min_width, clamped vel/force


def test_command_gripper_rejects_unknown_side_and_action(monkeypatch) -> None:
    import flexivtrainer.teleop.end_effector as ee

    monkeypatch.setattr(ee, "Gripper", FakeGripper)
    monkeypatch.setattr(ee, "Tool", FakeTool)

    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    cfg = EndEffectorSideConfig(follower="gripper")
    ctl = EndEffectorController(tdk, ["single_arm"], {"single_arm": cfg})
    ctl.initialize_grippers()

    for side, action in [("left_arm", "open"), ("single_arm", "wiggle")]:
        try:
            ctl.command_gripper(side, action, velocity=0.1, force=1.0)
        except (ValueError, RuntimeError):
            pass
        else:  # pragma: no cover - failure path
            raise AssertionError(f"expected error for side={side} action={action}")


def test_single_arm_uses_pair_index_zero() -> None:
    # Single mode: the lone "single_arm" side maps to pair index 0.
    follower = FakeFollower()
    tdk = FakeTDK(follower, [False] * 18)
    tdk._leader_ports[1] = True
    ctl = EndEffectorController(
        tdk,
        ["single_arm"],
        {"single_arm": _di_config(leader_channel=1, follower_channel=9)},
    )

    assert ctl.has_work() is True
    assert len(ctl._configs) == 1
    ctl._tick(0, ctl._configs[0])
    assert follower.digital_outputs == {9: True}
