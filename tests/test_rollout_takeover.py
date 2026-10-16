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

"""Taking a running rollout over with the leader arm, and handing it back.

The loop tests drive the real RolloutService and TeleopService against a fake
TDK controller and fake rdk robots that write every call into one shared log,
so the ORDER of what reaches the hardware is what is asserted: the policy stops
before teleoperation starts, the force sensors are not zeroed, nothing planned
before a takeover is sent after it, and a failed switch leaves the robots
holding.
"""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_rollout_evaluation import _wait_for_verdict_slot
from test_rollout_service import (
    _cameras,
    _fake_loader,
    _FakePolicy,
    _FakeRobot,
    _settings,
    _single_arm_pairs,
)

import flexivtrainer.teleop.end_effector as ee
import flexivtrainer.teleop.service as teleop_module
from flexivtrainer.config import EndEffectorSideConfig, TeleopRobotPair
from flexivtrainer.evaluation.outcome import RolloutOutcome, read_outcome
from flexivtrainer.rollout.executors.gripper import GripperExecutor
from flexivtrainer.rollout.executors.waypoint import WaypointExecutor
from flexivtrainer.rollout.recorder import TICKS_FILE
from flexivtrainer.rollout.safety import SafetyEnvelope, SafetyStop
from flexivtrainer.rollout.service import RolloutService
from flexivtrainer.rollout.takeover import (
    LeaderTrigger,
    TakeoverRefused,
    operator_action,
)
from flexivtrainer.teleop.service import LENT_MESSAGE, TeleopService

# --- fakes ----------------------------------------------------------------------------

BASE_X = 0.4
STEP_X = 1e-4


def _call_of(pose) -> int:
    """Which policy call planned this pose (the fake policy encodes it in x)."""
    return round((float(pose[0]) - BASE_X) / STEP_X)


class _Arm(_FakeRobot):
    """A fake rdk::Robot that writes what it is told into the shared log."""

    def __init__(self, serial: str, log: list) -> None:
        super().__init__(serial)
        self.log = log
        self.commanded = {
            "tcp_pose_d": [0.5, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0],
            "tcp_vel_d": [0.01, 0.02, 0.03, 0.0, 0.0, 0.0],
            "ext_wrench_d": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        self.inputs = [False] * 16
        self.wrench: list[float] | None = None
        self.fail_switch = 0

    def SendCartesianMotionForce(self, pose, *args, **kwargs):  # noqa: N802
        super().SendCartesianMotionForce(pose, *args, **kwargs)
        self.log.append(("send", _call_of(pose)))

    def SwitchMode(self, mode) -> None:  # noqa: N802
        if self.fail_switch:
            self.fail_switch -= 1
            raise RuntimeError("mode switch refused")
        super().SwitchMode(mode)
        self.log.append(("SwitchMode", self.serial, mode))

    def ExecutePrimitive(self, name, input_params) -> None:  # noqa: N802
        super().ExecutePrimitive(name, input_params)
        self.log.append(("primitive", self.serial, name))

    def Stop(self) -> None:  # noqa: N802
        self.log.append(("robot.Stop", self.serial))

    def SetCartesianImpedance(self, values) -> None:  # noqa: N802
        self.log.append(("SetCartesianImpedance", self.serial, list(values)))

    def SetMaxContactWrench(self, values) -> None:  # noqa: N802
        self.log.append(("SetMaxContactWrench", self.serial, list(values)))

    def states(self):
        states = super().states()
        if self.wrench is not None:
            states.ext_wrench_in_world = list(self.wrench)
        return states

    def actions(self):
        return {key: list(value) for key, value in self.commanded.items()}

    def digital_inputs(self):
        return list(self.inputs)


class _TDK:
    """The TransparentCartesianTeleopLAN surface the trainer uses.

    Every call is logged before it acts; `delay` makes one slow and `fail`
    makes one raise. Engage raises unless the loop runs, as the real one does.
    """

    def __init__(self, leader: _Arm, follower: _Arm, log: list) -> None:
        self.leader = leader
        self.follower = follower
        self.log = log
        self.running = False
        self.delay: dict[str, float] = {}
        self.fail: set[str] = set()

    def _call(self, name: str, *args) -> None:
        self.log.append(("tdk", name, *args))
        time.sleep(self.delay.get(name, 0.0))
        if name in self.fail:
            raise RuntimeError(f"{name} failed")

    def instances(self, idx: int):
        return (self.leader, self.follower)

    def digital_inputs(self, idx: int):
        return (self.leader.digital_inputs(), [False] * 16)

    def Init(self, zero_ft_sensor=None) -> None:  # noqa: N802
        self._call("Init", zero_ft_sensor)
        self.running = False

    def SetFollowerMaxContactWrench(self, idx, wrench) -> None:  # noqa: N802
        self._call("SetFollowerMaxContactWrench", idx, list(wrench))

    def Start(self) -> None:  # noqa: N802
        self._call("Start")
        self.running = True

    def Stop(self) -> None:  # noqa: N802
        self._call("Stop")
        self.running = False

    def StopWithIdx(self, idx: int) -> None:  # noqa: N802
        self._call("StopWithIdx", idx)
        self.running = False

    def Engage(self, idx: int, engaged: bool) -> None:  # noqa: N802
        if not self.running:
            raise RuntimeError("Engage needs the teleop loop running")
        self._call("Engage", idx, engaged)

    def any_fault(self) -> bool:
        return False

    def fault(self, idx: int):
        return (False, False)


class _CountingPolicy(_FakePolicy):
    """Every call plans a different x, so a command names the call it came from.

    `forced` is set by the fake inference for a call that plans afresh. With
    `block_next_plan`, the next such call holds until `release` is set, like a
    slow inference; `blocked_call` names it, and its plan is moved by
    `blocked_offset_m` in y, as if the operator had moved the arm meanwhile.
    """

    def __init__(self, log: list) -> None:
        super().__init__([BASE_X, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0] + [0.0] * 12)
        self.log = log
        self.calls = 0
        self.forced = False
        self.block_next_plan = False
        self.blocked_call: int | None = None
        self.blocked_offset_m = 0.0
        self.blocked = threading.Event()
        self.release = threading.Event()

    def reset(self) -> None:
        super().reset()
        self.log.append(("policy.reset",))

    def select_action(self, batch):
        self.calls += 1
        call = self.calls
        self.log.append(("policy.select", call))
        if self.block_next_plan and self.forced:
            self.block_next_plan = False
            self.blocked_call = call
            self.blocked.set()
            assert self.release.wait(timeout=5.0)
        action = list(self._action)
        action[0] = BASE_X + call * STEP_X
        if call == self.blocked_call:
            action[1] += self.blocked_offset_m
        return np.asarray(action, dtype=np.float64)


ZERO_FLAGS = SimpleNamespace(Enable="zero-on", Disable="zero-off")
GRIPPER_TRIGGER = EndEffectorSideConfig(
    leader="digital_input",
    leader_channel=0,
    leader_activating_state="high",
    follower="gripper",
    gripper_model="fake-gripper",
    gripper_activated_state="close",
)


@pytest.fixture
def make_rig(tmp_path, monkeypatch):
    """A rollout service, a connected teleop service and the fake hardware."""
    monkeypatch.setattr(teleop_module, "ZeroFTSensor", ZERO_FLAGS)

    def predict(obs, pol, dev, pre, post, **kwargs):
        # Like LeRobot's queue: only a forced tick plans afresh; the others
        # keep the schedule the last plan installed.
        pol.forced = bool(kwargs.get("force_refresh"))
        return np.tile(pol.select_action(obs), (8, 1)), pol.forced

    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._predict_action_chunk", predict
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(
            NRT_PRIMITIVE_EXECUTION="prim", NRT_CARTESIAN_MOTION_FORCE="cmf"
        ),
    )
    services: list[RolloutService] = []

    def make(
        *,
        configure=None,
        end_effector=None,
        teleop_up: bool = True,
        teleop_pairs=_single_arm_pairs,
    ) -> SimpleNamespace:
        log: list = []
        settings = _settings(tmp_path)
        settings.ensure_storage()
        # A fresh plan every second tick, eight waypoints long: a plan always
        # has waypoints left to send when a takeover comes.
        settings.policies.act.rollout.replan_steps = 2
        if configure is not None:
            configure(settings)
        leader, follower = _Arm("L1", log), _Arm("F1", log)
        tdk = _TDK(leader, follower, log)
        configs = dict(end_effector or {})
        teleop = TeleopService(
            settings,
            get_robot_pairs=teleop_pairs,
            get_active_sides=lambda: ["single_arm"],
            get_end_effector_config=lambda: configs,
        )
        if teleop_up:
            teleop._controller = tdk
            teleop._initialized = True
        fresh: list[str] = []

        def factory(serial: str) -> _Arm:
            fresh.append(serial)
            return _Arm(serial, log)

        policy = _CountingPolicy(log)
        service = RolloutService(
            settings,
            _cameras(),
            teleop,
            _single_arm_pairs,
            lambda: ["single_arm"],
            get_end_effector_config=lambda: configs,
            policy_loader=_fake_loader(policy),
            robot_factory=factory,
            resolve_device=lambda configured: "cpu",
        )
        services.append(service)
        return SimpleNamespace(
            log=log,
            tdk=tdk,
            teleop=teleop,
            service=service,
            policy=policy,
            leader=leader,
            follower=follower,
            fresh=fresh,
            checkpoint=str(tmp_path / "ckpt"),
        )

    (tmp_path / "ckpt").mkdir()
    yield make
    for service in services:
        service.stop()


def _wait(predicate, what: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def _takeover(rig) -> dict:
    return rig.service.status()["takeover"]


def _wait_state(rig, state: str, timeout: float = 5.0) -> None:
    _wait(lambda: _takeover(rig)["state"] == state, f"state {state}", timeout)


def _sends(log, start: int = 0) -> list[int]:
    return [entry[1] for entry in log[start:] if entry[0] == "send"]


def _index(log, *prefix, start: int = 0) -> int:
    for index in range(start, len(log)):
        if log[index][: len(prefix)] == prefix:
            return index
    raise AssertionError(f"{prefix} is not in the log after {start}: {log[start:]}")


def _start(rig, **kwargs) -> None:
    rig.service.start(rig.checkpoint, **kwargs)
    _wait(lambda: _takeover(rig)["action"] == "take_over", "the policy's first plan")
    _wait(lambda: len(_sends(rig.log)) >= 2, "policy commands")


def _take_over(rig) -> int:
    """Take over and wait for the operator; returns the log index it began at."""
    mark = len(rig.log)
    rig.service.take_over()
    _wait_state(rig, "operator")
    return mark


def _hand_back(rig) -> int:
    mark = len(rig.log)
    rig.service.hand_back()
    _wait_state(rig, "policy")
    return mark


# --- borrowing the robots -------------------------------------------------------------


def test_the_rollout_borrows_the_teleop_robots_and_opens_no_connection(
    make_rig,
) -> None:
    rig = make_rig()
    rig.teleop._started = True  # teleoperation was running when Start was pressed
    _start(rig)

    assert rig.fresh == [], "a fresh RDK connection was opened"
    stop = _index(rig.log, "tdk", "StopWithIdx", 0)
    assert stop < _index(rig.log, "send"), "teleop must stop before the policy moves"
    assert rig.follower.commands, "the policy drives the TDK's own follower handle"
    snapshot = rig.teleop.snapshot()
    assert snapshot.initialized and snapshot.lent and not snapshot.started

    rig.service.stop()
    _wait(lambda: not rig.teleop.snapshot().lent, "the robots to be returned")
    snapshot = rig.teleop.snapshot()
    assert snapshot.initialized, "teleoperation stays connected after the rollout"
    assert not snapshot.started


def test_while_lent_teleop_refuses_everything_that_would_move_the_robots(
    make_rig,
) -> None:
    rig = make_rig()
    lease = rig.teleop.lend_to_rollout(["F1"])

    assert rig.teleop.start().error == LENT_MESSAGE
    assert not [entry for entry in rig.log if entry[:2] == ("tdk", "Start")]
    assert rig.teleop.set_engaged(True).error == LENT_MESSAGE
    assert rig.teleop.reset_home() == {"ok": False, "error": LENT_MESSAGE}
    assert rig.teleop.match_leader_to_follower()["error"] == LENT_MESSAGE
    assert rig.teleop.init_grippers()["error"] == LENT_MESSAGE
    assert rig.teleop.set_gripper_width(0.05)["error"] == LENT_MESSAGE
    with pytest.raises(RuntimeError, match="rollout is using the robots"):
        rig.teleop.shutdown()
    assert rig.teleop.snapshot().can_home is False
    stops = len(rig.log)
    rig.teleop.stop()
    assert rig.log[stops:] == [], "no teleop loop runs, so Stop must not reach it"

    rig.teleop.release_lease(lease)
    assert rig.teleop.snapshot().lent is False
    assert rig.teleop.start().started


def test_without_teleop_the_rollout_connects_as_before_and_cannot_take_over(
    make_rig,
) -> None:
    rig = make_rig(teleop_up=False)
    started = rig.service.start(rig.checkpoint)
    assert started["takeover"]["available"] is False
    _wait(lambda: len(_sends(rig.log)) >= 1, "policy commands")

    assert rig.fresh == ["F1"]
    assert not [entry for entry in rig.log if entry[0] == "tdk"]
    takeover = _takeover(rig)
    assert takeover["available"] is False and takeover["action"] is None
    assert "Teleoperation was not connected" in takeover["reason"]
    with pytest.raises(TakeoverRefused, match="Teleoperation was not connected"):
        rig.service.take_over()


def test_a_follower_teleop_does_not_hold_falls_back_to_disconnecting(
    make_rig,
) -> None:
    rig = make_rig(
        teleop_pairs=lambda: [TeleopRobotPair(leader_serial="L1", follower_serial="F9")]
    )
    rig.service.start(rig.checkpoint)
    _wait(lambda: len(_sends(rig.log)) >= 1, "policy commands")

    assert rig.fresh == ["F1"]
    assert rig.teleop.snapshot().initialized is False, "the old path disconnects"
    assert (
        "follower F1 is not part of a teleoperation pair" in (_takeover(rig)["reason"])
    )


# --- taking over ----------------------------------------------------------------------


def _with_wrench(settings) -> None:
    settings.rollout.max_contact_wrench = (30.0, 30.0, 30.0, 5.0, 5.0, 5.0)


def test_taking_over_stops_the_policy_before_teleop_starts_without_zeroing(
    make_rig,
) -> None:
    rig = make_rig(configure=_with_wrench)
    # A slow Init: a policy that kept sending would show up during it.
    rig.tdk.delay["Init"] = 0.3
    _start(rig)

    started = time.monotonic()
    status = rig.service.take_over()
    assert time.monotonic() - started < 0.2, "the request must not wait for Init"
    assert status["takeover"]["state"] == "switching_to_operator"
    assert status["takeover"]["control"] == "switching"
    _wait_state(rig, "operator")

    log = list(rig.log)
    init = _index(log, "tdk", "Init")
    assert log[init] == ("tdk", "Init", "zero-off"), "F/T must not be zeroed mid-trial"
    assert _sends(log, init) == [], "a policy command reached the arm after Init"
    wrench = _index(log, "tdk", "SetFollowerMaxContactWrench")
    start = _index(log, "tdk", "Start")
    engage = _index(log, "tdk", "Engage", 0, True)
    assert init < wrench < start < engage
    assert log[wrench] == (
        "tdk",
        "SetFollowerMaxContactWrench",
        0,
        [30.0, 30.0, 30.0, 5.0, 5.0, 5.0],
    )
    assert rig.teleop.snapshot().engaged and rig.teleop.snapshot().started
    takeover = _takeover(rig)
    assert takeover["control"] == "operator"
    assert takeover["last_switch"]["to"] == "operator"
    assert takeover["last_switch"]["s"] >= 0.3
    assert takeover["action"] == "hand_back"


def test_no_policy_command_is_sent_while_the_operator_has_control(make_rig) -> None:
    rig = make_rig()
    _start(rig)
    mark = _take_over(rig)
    calls = rig.policy.calls
    sends = len(_sends(rig.log))

    time.sleep(0.4)

    assert len(_sends(rig.log)) == sends
    assert _sends(rig.log, mark + 1) == []
    assert rig.policy.calls == calls, "the policy must not plan under the operator"
    assert rig.service.status()["status"] == "running"


def test_handing_back_resets_the_policy_and_sends_nothing_planned_before(
    make_rig,
) -> None:
    rig = make_rig()
    _start(rig)
    _take_over(rig)
    planned_before = rig.policy.calls
    mark = _hand_back(rig)
    _wait(lambda: len(_sends(rig.log, mark)) >= 3, "policy commands after hand-back")

    log = list(rig.log)
    disengage = _index(log, "tdk", "Engage", 0, False, start=mark)
    stop = _index(log, "tdk", "StopWithIdx", 0, start=mark)
    reclaim = _index(log, "SwitchMode", "F1", "cmf", start=mark)
    reset = _index(log, "policy.reset", start=mark)
    first_plan = _index(log, "policy.select", start=mark)
    first_send = _index(log, "send", start=mark)
    assert disengage < stop < reclaim < reset < first_plan < first_send
    assert rig.policy.reset_count == 2
    assert all(call > planned_before for call in _sends(log, mark))
    assert not [entry for entry in log[mark:] if entry[:1] == ("primitive",)], (
        "the F/T sensor must not be zeroed when the rollout takes the arm back"
    )
    assert rig.teleop.snapshot().started is False
    takeover = _takeover(rig)
    assert takeover["control"] == "policy"
    assert takeover["last_switch"]["to"] == "policy"


def test_a_plan_computed_across_a_takeover_is_never_sent(make_rig) -> None:
    # The fake arm measures ~3.4 m from every plan; a stale plan 10 m off would
    # trip the reference check if anything still read it as the next command.
    rig = make_rig(
        configure=lambda settings: setattr(
            settings.rollout, "max_reference_offset_m", 5.0
        )
    )
    _start(rig)
    rig.policy.blocked_offset_m = 10.0
    rig.policy.block_next_plan = True
    assert rig.policy.blocked.wait(timeout=5.0)
    stale = rig.policy.blocked_call

    _take_over(rig)
    mark = _hand_back(rig)
    rig.policy.release.set()
    _wait(lambda: len(_sends(rig.log, mark)) >= 3, "policy commands after hand-back")

    assert stale not in _sends(rig.log), "a plan from before the takeover was sent"
    assert all(call > stale for call in _sends(rig.log, mark))
    assert rig.service.status()["error"] is None


@pytest.mark.parametrize(
    "step", ["Init", "SetFollowerMaxContactWrench", "Start", "Engage"]
)
def test_a_failed_takeover_leaves_the_robots_holding_and_the_policy_stopped(
    make_rig, step
) -> None:
    rig = make_rig(configure=_with_wrench)
    rig.tdk.fail.add(step)
    _start(rig)
    mark = len(rig.log)
    rig.service.take_over()
    _wait_state(rig, "error")

    takeover = _takeover(rig)
    assert f"{step} failed" in takeover["error"]
    assert takeover["control"] == "switching"
    assert takeover["action"] == "hand_back", "the operator can retry the hand-back"
    assert not rig.tdk.running, "the teleop loop must not be left running"
    if step in ("Start", "Engage"):
        _index(rig.log, "tdk", "StopWithIdx", 0, start=mark)
    calls, sends = rig.policy.calls, len(_sends(rig.log))
    time.sleep(0.3)
    assert rig.policy.calls == calls and len(_sends(rig.log)) == sends
    assert rig.service.status()["status"] == "running"

    rig.tdk.fail.clear()
    retry = _hand_back(rig)
    _wait(lambda: len(_sends(rig.log, retry)) >= 2, "the policy after a retry")
    assert rig.policy.reset_count == 2


@pytest.mark.parametrize("step", ["Engage", "StopWithIdx", "reclaim"])
def test_a_failed_hand_back_does_not_resume_the_policy(make_rig, step) -> None:
    rig = make_rig()
    _start(rig)
    _take_over(rig)
    if step == "reclaim":
        rig.follower.fail_switch = 1
    else:
        rig.tdk.fail.add(step)
    mark = len(rig.log)
    rig.service.hand_back()
    _wait_state(rig, "error")

    takeover = _takeover(rig)
    assert takeover["error"]
    assert rig.teleop.snapshot().started is False
    if step != "StopWithIdx":
        assert not rig.tdk.running
    calls, sends = rig.policy.calls, len(_sends(rig.log))
    time.sleep(0.3)
    assert rig.policy.calls == calls and len(_sends(rig.log)) == sends
    assert _sends(rig.log, mark) == []
    assert rig.policy.reset_count == 1

    rig.tdk.fail.clear()
    retry = _hand_back(rig)
    _wait(lambda: len(_sends(rig.log, retry)) >= 2, "the policy after a retry")
    assert rig.policy.reset_count == 2


# --- safety ---------------------------------------------------------------------------


def test_the_stale_plan_watchdog_rests_while_the_operator_has_control(
    make_rig,
) -> None:
    rig = make_rig(
        configure=lambda settings: setattr(settings.rollout, "stale_plan_s", 0.25)
    )
    _start(rig)
    _take_over(rig)
    time.sleep(0.5)
    assert rig.service.status()["status"] == "running"

    mark = _hand_back(rig)
    _wait(lambda: len(_sends(rig.log, mark)) >= 3, "policy commands after hand-back")
    status = rig.service.status()
    assert status["status"] == "running", status["error"]
    assert status["error"] is None


def test_a_safety_stop_under_the_operator_disengages_and_holds(make_rig) -> None:
    rig = make_rig(
        configure=lambda settings: setattr(settings.rollout, "max_force_n", 100.0)
    )
    _start(rig)
    _take_over(rig)
    mark = len(rig.log)
    rig.follower.wrench = [500.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    _wait(lambda: rig.service.status()["status"] == "failed", "the safety stop")

    assert "Safety stop (force)" in rig.service.status()["error"]
    disengage = _index(rig.log, "tdk", "Engage", 0, False, start=mark)
    assert _index(rig.log, "tdk", "StopWithIdx", 0, start=disengage)
    assert not rig.tdk.running
    _wait(lambda: not rig.teleop.snapshot().lent, "the robots to be returned")
    assert rig.teleop.snapshot().initialized
    with pytest.raises(TakeoverRefused, match="No rollout is running"):
        rig.service.hand_back()


def test_the_workspace_box_is_checked_where_the_operator_puts_the_arm() -> None:
    envelope = SafetyEnvelope(workspace_min_m=(0, 0, 0), workspace_max_m=(1, 1, 1))
    envelope.check_measured_workspace([[0.5, 0.5, 0.5, 1, 0, 0, 0]])
    with pytest.raises(SafetyStop, match=r"arm 0 measures z=1\.2"):
        envelope.check_measured_workspace([[0.5, 0.5, 1.2, 1, 0, 0, 0]])
    SafetyEnvelope().check_measured_workspace([[9.0, 9.0, 9.0]])


# --- recording ------------------------------------------------------------------------


def test_a_recorded_takeover_marks_its_ticks_and_times_its_switches(
    make_rig,
) -> None:
    pytest.importorskip("lerobot")
    rig = make_rig()
    _start(rig, record=True)
    _take_over(rig)
    time.sleep(0.3)
    mark = _hand_back(rig)
    _wait(lambda: len(_sends(rig.log, mark)) >= 3, "policy commands after hand-back")
    rig.service.stop()
    status = _wait_for_verdict_slot(rig.service)

    episode_dir = (
        rig.service._settings.storage.rollouts_root
        / "adhoc"
        / status["awaiting_verdict"]["episode"]
    )
    ticks = [
        json.loads(line) for line in (episode_dir / TICKS_FILE).read_text().splitlines()
    ]
    assert all(tick["intervention"] == (tick["control"] != "policy") for tick in ticks)
    controls = [tick["control"] for tick in ticks]
    assert "operator" in controls and controls[0] == "policy"
    assert controls[-1] == "policy"
    # Under the operator the action is what teleoperation records: the
    # follower's commanded pose, twist and wrench.
    commanded = rig.follower.commanded
    expected = commanded["tcp_pose_d"] + commanded["tcp_vel_d"]
    expected += commanded["ext_wrench_d"]
    operator_ticks = [tick for tick in ticks if tick["control"] == "operator"]
    assert all(tick["command"] == pytest.approx(expected) for tick in operator_ticks)
    switches = [event for tick in ticks for event in tick.get("switches", [])]
    assert [event["to"] for event in switches] == ["operator", "policy"]

    outcome = read_outcome(episode_dir)
    assert outcome.dropped_frames == 0
    interventions = outcome.interventions
    assert interventions["count"] == 1
    assert interventions["ticks"] == sum(tick["intervention"] for tick in ticks)
    assert interventions["seconds"] >= 0.3
    assert [event["to"] for event in interventions["switch_s"]] == [
        "operator",
        "policy",
    ]
    assert all(event["s"] >= 0 for event in interventions["switch_s"])


def test_an_outcome_written_before_takeovers_existed_still_reads() -> None:
    outcome = RolloutOutcome.from_dict(
        {
            "episode": "e1",
            "checkpoint": "c",
            "task": None,
            "started_at": "2026-10-16T10:00:00",
            "schema_version": RolloutOutcome(
                episode="", checkpoint="", task=None, started_at=""
            ).schema_version,
        }
    )
    assert outcome.interventions == {}


def test_the_operator_action_uses_the_teleop_definition() -> None:
    robot = _Arm("F1", [])
    names = [
        f"single_arm.{name}"
        for name in (
            "tcp_pose.x tcp_pose.y tcp_pose.z tcp_rotation_6d.r1_x "
            "tcp_rotation_6d.r1_y tcp_rotation_6d.r1_z tcp_rotation_6d.r2_x "
            "tcp_rotation_6d.r2_y tcp_rotation_6d.r2_z tcp_twist.vx tcp_twist.vy "
            "tcp_twist.vz tcp_twist.wx tcp_twist.wy tcp_twist.wz "
            "gripper.target_width"
        ).split()
    ]
    fallback = [9.0] * len(names)
    action = operator_action([robot], ["single_arm"], names, {}, fallback)
    # Identity rotation in rotation-6D is the first two columns of I.
    assert action[:9] == pytest.approx([0.5, 0.1, 0.2, 1, 0, 0, 0, 1, 0])
    assert action[9:15] == pytest.approx(robot.commanded["tcp_vel_d"])
    assert action[15] == 9.0, "no leader gripper command yet keeps the last target"
    with_gripper = operator_action(
        [robot], ["single_arm"], names, {"single_arm": 0.02}, fallback
    )
    assert with_gripper[15] == pytest.approx(0.02)


# --- the leader trigger ---------------------------------------------------------------


def _trigger(inputs, *, active_high=True):
    now = [0.0]
    presses: list[float] = []
    trigger = LeaderTrigger(
        lambda: inputs,
        3,
        lambda: presses.append(now[0]),
        active_high=active_high,
        debounce_s=0.05,
        clock=lambda: now[0],
    )

    def at(t: float, level: bool | None = None) -> bool:
        now[0] = t
        if level is not None:
            inputs[3] = level if active_high else not level
        return trigger.poll()

    return at, presses


def test_the_trigger_fires_once_per_press_after_the_debounce() -> None:
    at, presses = _trigger([False] * 16)
    assert not at(0.0)  # the resting level
    assert not at(0.01, True)
    assert not at(0.04)
    assert at(0.07)
    assert not at(0.2), "holding the button is one press"
    assert not at(0.21, False)
    assert not at(0.3), "releasing is not a press"
    # Bounce shorter than the debounce restarts it.
    assert not at(0.31, True)
    assert not at(0.33, False)
    assert not at(0.35, True)
    assert not at(0.38)
    assert at(0.41)
    assert presses == [0.07, 0.41]


def test_a_button_held_when_polling_starts_does_not_fire() -> None:
    inputs = [False] * 16
    inputs[3] = True
    at, presses = _trigger(inputs)
    at(0.0)
    at(0.5)
    assert presses == []
    at(0.6, False)
    at(0.7)
    at(0.71, True)
    at(0.8)
    assert presses == [0.8]


def test_an_active_low_trigger_fires_on_the_falling_level() -> None:
    at, presses = _trigger([True] * 16, active_high=False)
    at(0.0)
    at(0.01, True)  # pressed: the input goes low
    at(0.1)
    assert presses == [0.1]


def test_the_leader_trigger_toggles_control(make_rig) -> None:
    def configure(settings) -> None:
        settings.rollout.takeover_trigger_channel = 3
        settings.rollout.takeover_trigger_debounce_s = 0.02

    rig = make_rig(configure=configure)
    _start(rig)
    assert _takeover(rig)["trigger_channel"] == 3

    rig.leader.inputs[3] = True
    _wait_state(rig, "operator")
    rig.leader.inputs[3] = False
    time.sleep(0.1)
    rig.leader.inputs[3] = True
    _wait_state(rig, "policy")


def test_a_trigger_channel_the_gripper_uses_is_refused(make_rig) -> None:
    rig = make_rig(
        configure=lambda settings: setattr(
            settings.rollout, "takeover_trigger_channel", 0
        ),
        end_effector={"single_arm": GRIPPER_TRIGGER},
    )
    with pytest.raises(RuntimeError, match="gripper trigger"):
        rig.service.start(rig.checkpoint)
    assert rig.teleop.snapshot().lent is False
    assert not [entry for entry in rig.log if entry[0] == "send"]


# --- the end effectors ----------------------------------------------------------------


class _Gripper:
    def __init__(self, robot) -> None:
        self.moves: list[float] = []

    def Enable(self, name: str) -> None:  # noqa: N802
        pass

    def Init(self) -> None:  # noqa: N802
        pass

    def params(self):
        return SimpleNamespace(
            name="fake",
            min_width=0.0,
            max_width=0.1,
            min_vel=0.01,
            max_vel=0.5,
            min_force=1.0,
            max_force=50.0,
        )

    def Move(self, width, velocity, force) -> None:  # noqa: N802
        self.moves.append(float(width))

    def states(self):
        return SimpleNamespace(width=0.01, force=-3.0, is_moving=False)


def test_the_jaws_keep_their_state_until_the_leader_trigger_moves(
    make_rig, monkeypatch
) -> None:
    grippers: list[_Gripper] = []
    monkeypatch.setattr(
        ee, "Gripper", lambda robot: grippers.append(_Gripper(robot)) or grippers[-1]
    )
    rig = make_rig(end_effector={"single_arm": GRIPPER_TRIGGER})
    assert rig.teleop.init_grippers()["ok"]
    gripper = grippers[0]
    assert gripper.moves == []

    # The policy holds a part; the leader trigger rests (which mirrors as Open).
    lease = rig.teleop.lend_to_rollout(["F1"])
    rig.teleop.resume_for_takeover(lease)
    time.sleep(0.25)  # several mirror ticks
    assert gripper.moves == [], "the takeover opened the jaws"

    rig.leader.inputs[0] = True
    _wait(lambda: gripper.moves == [0.0], "the operator's Close")
    rig.teleop.return_from_takeover(lease)
    rig.teleop.release_lease(lease)

    # A plain teleop restart after the rollout holds the jaws too.
    rig.leader.inputs[0] = False
    assert rig.teleop.start().started
    time.sleep(0.25)
    assert gripper.moves == [0.0]
    rig.teleop.stop()


# --- the executors --------------------------------------------------------------------


def _pose_action(x: float) -> list[float]:
    return [x, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0] + [0.0] * 12


def _waypoint_executor(robot) -> WaypointExecutor:
    layout = [
        {
            "side": "single_arm",
            "pose": slice(0, 7),
            "rotation": None,
            "twist": slice(7, 13),
            "wrench": slice(13, 19),
        }
    ]
    return WaypointExecutor(
        [robot], layout, threading.Event(), (0.25, 0.6, 1.0, 2.5), action_dim=19
    )


def test_a_paused_executor_drops_its_plan_and_refuses_a_stale_one() -> None:
    robot = _Arm("F1", [])
    executor = _waypoint_executor(robot)
    now = time.monotonic()
    epoch = executor.epoch
    assert executor.replace_waypoints([_pose_action(0.4)], [now + 10], now, epoch=epoch)
    assert executor.scheduled_count == 1

    executor.pause()
    assert executor.paused and executor.scheduled_count == 0
    assert not executor.replace_waypoints([_pose_action(0.4)], [now + 10], now)
    executor.resume()
    assert executor.scheduled_count == 0
    assert not executor.replace_waypoints(
        [_pose_action(0.4)], [now + 10], now, epoch=epoch
    ), "a plan computed before the pause must not be installed"
    assert executor.replace_waypoints(
        [_pose_action(0.4)], [now + 10], now, epoch=executor.epoch
    )


def test_a_paused_executor_sends_nothing() -> None:
    robot = _Arm("F1", [])
    executor = _waypoint_executor(robot)
    executor.start()
    try:
        now = time.monotonic()
        executor.replace_waypoints(
            [_pose_action(0.4)] * 3, [now + 0.05, now + 0.1, now + 0.15], now
        )
        executor.pause()
        time.sleep(0.3)
        assert robot.commands == []
    finally:
        executor._stop_event.set()
        executor.join()


def test_a_paused_gripper_executor_sends_nothing_and_resends_after() -> None:
    moves: list[float] = []

    class Gripper(_Gripper):
        def Move(self, width, velocity, force) -> None:  # noqa: N802
            moves.append(float(width))

    robot = SimpleNamespace(mode=lambda: "IDLE")
    executor = GripperExecutor(
        [robot],
        ["single_arm"],
        {"single_arm": {"follower": "gripper", "gripper_model": "fake"}},
        ["single_arm"],
        gripper_factory=Gripper,
        tool_factory=lambda robot: SimpleNamespace(Switch=lambda model: None),
        idle_mode="IDLE",
    )
    executor.initialize()
    executor.submit({"single_arm": 0.05})
    executor._send_pending()
    assert moves == [0.05]

    executor.pause()
    executor.submit({"single_arm": 0.02})
    executor._send_pending()
    assert moves == [0.05]
    assert executor.measured_states()["single_arm"]["width"] == 0.01

    executor.resume()
    # The operator may have moved the jaws: the same target goes out again.
    executor.submit({"single_arm": 0.05})
    executor._send_pending()
    assert moves == [0.05, 0.05]


# --- the API --------------------------------------------------------------------------


def test_the_api_refuses_a_switch_that_does_not_apply(make_rig) -> None:
    from fastapi.testclient import TestClient

    from flexivtrainer.api.app import create_app
    from flexivtrainer.runtime.manager import get_runtime_manager

    rig = make_rig()
    app = create_app()
    app.dependency_overrides[get_runtime_manager] = lambda: SimpleNamespace(
        rollout=rig.service
    )
    client = TestClient(app)

    idle = client.post("/rollout/takeover")
    assert idle.status_code == 409 and idle.json()["detail"] == "No rollout is running"
    assert client.post("/rollout/hand-back").status_code == 409

    _start(rig)
    policy = client.post("/rollout/hand-back")
    assert policy.status_code == 409
    assert policy.json()["detail"] == "The policy already has control"
    taken = client.post("/rollout/takeover")
    assert taken.status_code == 200
    assert taken.json()["takeover"]["state"] in ("switching_to_operator", "operator")
    _wait_state(rig, "operator")
    again = client.post("/rollout/takeover")
    assert again.status_code == 409
    assert again.json()["detail"] == "The operator already has control"
    assert client.post("/rollout/hand-back").status_code == 200
    _wait_state(rig, "policy")
