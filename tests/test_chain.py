"""Chains of skills and primitives, joined by checks, with the retry rule."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_step_rollout import (
    DOWN,
    START,
    _Arm,
    _rotation,
    _service,
    _skill,
    rdk,  # noqa: F401 - a fixture
)
from waits import WAIT_S

from flexivtrainer.api.app import create_app
from flexivtrainer.evaluation.checks import (
    GripperWidth,
    SettledInBox,
    parse_checks,
    run_checks,
)
from flexivtrainer.practice.space import Sample
from flexivtrainer.rollout.chain import ChainPlan, ChainRunner, Outcome
from flexivtrainer.rollout.executors.gripper import GripperExecutor
from flexivtrainer.rollout.live_checks import LiveChecks
from flexivtrainer.rollout.primitives import (
    Move,
    MoveBlocked,
    MoveInterrupted,
    Primitive,
    move_jaws,
    move_tool,
)
from flexivtrainer.runtime.gripper_session import GripperInitializationRegistry
from flexivtrainer.runtime.manager import get_runtime_manager

# --- the plan -------------------------------------------------------------------

BOX = {"kind": "settled_in_box", "box_min_m": [0, 0, 0], "box_max_m": [1, 1, 1]}


def _plan(steps, checks=None, **extra):
    return ChainPlan.from_dict(
        {"steps": steps, "checks": checks or {"ready": BOX, "home": BOX}, **extra}
    )


def test_a_plan_names_its_steps_and_their_checks():
    plan = _plan(
        [
            {"name": "approach", "skill": "skills/a", "end": "ready", "max_s": 30},
            {
                "name": "insert",
                "skill": "skills/b",
                "start": "ready",
                "end": "home",
                "tries": 3,
                "back_off": {
                    "moves": [{"tool_m": [0, 0, -0.02]}],
                    "max_force_n": 25,
                },
            },
            {"name": "lift", "primitive": {"moves": [{"base_m": [0, 0, 0.05]}]}},
        ]
    )
    approach, insert, lift = plan.steps
    assert approach.max_s == 30.0 and approach.tries == 1
    assert insert.back_off == Primitive(
        moves=(Move(tool_m=(0.0, 0.0, -0.02)),), max_force_n=25
    )
    assert lift.primitive.moves[0].base_m == (0.0, 0.0, 0.05)
    assert insert.start == ("ready",) and insert.end == ("home",)
    # An end check is scored on every tick, and ends its rollout.
    (check,) = plan.live(("home",)).values()
    assert check.live and check.end_on_pass
    assert plan.watch_s(("ready",)) == pytest.approx(1.0 + 0.5)


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([{"name": "a", "skill": "s"}], "needs an end check"),
        ([{"name": "a", "skill": "s", "end": "nope"}], "not one of the plan's"),
        (
            [{"name": "a", "skill": "s", "end": "ready", "primitive": {}}],
            "one skill or one primitive",
        ),
        ([{"name": "a", "primitive": {}, "tries": 0}], "tries must be 1 or more"),
        ([{"name": "a", "primitive": {}, "wait": 1}], "unknown step settings"),
        (
            [
                {
                    "name": "a",
                    "primitive": {
                        "moves": [{"tool_m": [0, 0, 1], "base_m": [0, 0, 1]}]
                    },
                }
            ],
            "one of tool_m, base_m, a box or gripper_m",
        ),
        (
            [{"name": "a", "primitive": {"moves": [{"box_min_m": [0, 0, 0]}]}}],
            "both box_min_m and box_max_m",
        ),
        (
            [
                {
                    "name": "a",
                    "primitive": {"moves": [{"gripper_m": 0, "rotation": DOWN}]},
                }
            ],
            "does not turn the tool",
        ),
        (
            [
                {
                    "name": "a",
                    "primitive": {
                        "moves": [{"base_m": [0, 0, 1], "gripper_force_n": 5}]
                    },
                }
            ],
            "belongs to a jaw move",
        ),
        ([{"name": "a", "primitive": {}, "end": ["ready", "ready"]}], "twice"),
        ([{"name": "a", "primitive": {}, "end": 3}], "a check's name, or a list"),
        ([{"name": "a", "primitive": {}}, {"name": "a", "primitive": {}}], "repeat"),
        ([], "at least one step"),
    ],
)
def test_a_plan_that_cannot_run_is_refused(steps, message):
    with pytest.raises(ValueError, match=message):
        _plan(steps)


# --- moves ------------------------------------------------------------------------


def _sample(position=START[:3], wrench=(0.0,) * 6):
    return Sample.from_driver([*position, *DOWN], [0.0] * 6, list(wrench))


def test_a_move_goes_by_an_offset_in_the_tool_or_base_frame_or_into_a_box():
    here = _sample()
    # The tool points down: backing off along its z goes up in the base frame.
    position, rotation = Move(tool_m=(0, 0, -0.02)).target(here)
    np.testing.assert_allclose(position, [0.4, 0.0, 0.32], atol=1e-12)
    np.testing.assert_allclose(rotation, here.rotation)
    position, _ = Move(base_m=(0.01, 0, 0)).target(here)
    np.testing.assert_allclose(position, [0.41, 0.0, 0.3], atol=1e-12)
    # Into a box: its point nearest the tool, turned as the move says.
    move = Move(box_min_m=(0.3, 0.0, 0.35), box_max_m=(0.5, 0.1, 0.4), rotation=DOWN)
    position, rotation = move.target(here)
    np.testing.assert_allclose(position, [0.4, 0.0, 0.35], atol=1e-12)
    np.testing.assert_allclose(rotation, _rotation(START), atol=1e-12)


def _mover(*, wrench=(0.0,) * 6, accept=True, follows=True):
    """read, send and what was sent, for an arm that goes where it is sent."""
    sent = []
    where = {"p": np.asarray(START[:3])}

    def read():
        return _sample(where["p"], wrench)

    def send(pose, limits):
        sent.append((list(pose), limits))
        if follows:
            where["p"] = np.asarray(pose[:3])
        return accept

    return read, send, sent


def _move(read, send, **options):
    values = {
        "speed_m_s": 0.02,
        "tolerance_m": 0.001,
        "tolerance_deg": 0.5,
        "timeout_s": 5.0,
        "what": "backed off",
    }
    move_tool(
        np.asarray([0.4, 0.0, 0.32]),
        _rotation(START),
        read=read,
        send=send,
        stop=threading.Event(),
        **{**values, **options},
    )


def test_a_move_is_sent_once_and_slowly_and_read_until_it_is_there():
    read, send, sent = _mover()
    _move(read, send)
    ((pose, limits),) = sent
    np.testing.assert_allclose(pose[:3], [0.4, 0.0, 0.32])
    assert limits[0] == 0.02


def test_a_move_that_meets_more_force_than_its_limit_holds_and_gives_up():
    read, send, sent = _mover(wrench=(0.0, 0.0, 30.0, 0, 0, 0), follows=False)
    with pytest.raises(MoveBlocked, match="30.0 N .* backed off, over the 20 N"):
        _move(read, send, max_force_n=20.0)
    # The target, then a hold where the arm was measured.
    assert len(sent) == 2
    np.testing.assert_allclose(sent[1][0][:3], START[:3])


def test_a_move_refused_by_the_arm_or_too_slow_fails():
    read, send, _ = _mover(accept=False)
    with pytest.raises(MoveInterrupted, match="taken while it backed off"):
        _move(read, send)
    read, send, _ = _mover(follows=False)
    with pytest.raises(RuntimeError, match=r"did not reach the target within 0.05 s"):
        _move(read, send, timeout_s=0.05)


# --- box checks scored live --------------------------------------------------------


def _snapshot(position, velocity=(0.0,) * 6):
    pose = [*position, *DOWN]
    states = {"tcp_pose": pose, "tcp_vel": list(velocity)}
    return {"robots": {"robot_0": {"states": states}}}


def test_a_box_check_passes_live_once_the_tool_rests_inside_long_enough():
    check = SettledInBox(
        box_min_m=(0.39, -0.01, 0.29),
        box_max_m=(0.41, 0.01, 0.31),
        hold_s=0.2,
        end_on_pass=True,
    )
    live = LiveChecks({"ready": check}, None, state_names=None, loader=lambda p: None)
    live.start()
    assert not live.needs_frames

    def feed(t, position, velocity=(0.0,) * 6):
        live.submit(
            t=t,
            state=[],
            images={},
            snapshot=_snapshot(position, velocity),
            pose_format="r",
        )

    feed(0.0, START[:3])
    feed(0.1, START[:3], velocity=(0.05, 0, 0, 0, 0, 0))  # inside, but moving
    feed(0.2, START[:3])
    assert live.ended() is None and live.values() == {"ready": 1.0}
    feed(0.4, START[:3])
    assert live.ended() == "ready" and live.all_passed()
    assert live.summary()["checks"]["ready"]["first_at_s"] == 0.2
    feed(0.5, [0.5, 0.0, 0.3])
    assert live.values() == {"ready": 0.0}
    live.reset()
    assert live.ended() is None
    live.stop()


def test_a_session_check_of_either_kind_can_be_scored_live():
    checks = parse_checks({"ready": {**BOX, "live": True, "end_on_pass": True}})
    assert checks["ready"].scored_live
    with pytest.raises(ValueError, match="end_on_pass must be true or false"):
        parse_checks({"ready": {**BOX, "end_on_pass": "yes"}})


def _live(checks, **options):
    live = LiveChecks(checks, None, state_names=None, loader=lambda p: None, **options)
    live.start()
    return live


def _jaw_snapshot(width):
    snapshot = _snapshot(START[:3])
    if width is not None:
        snapshot["robots"]["robot_0"]["gripper"] = {"width": width, "force": 0.0}
    return snapshot


def test_a_gripper_width_check_passes_once_the_jaws_stay_in_their_band():
    holding = GripperWidth(min_m=0.010, max_m=0.014, hold_s=0.1, end_on_pass=True)
    ticks = [
        {"t": t, "measured": [{"gripper": {"width": w}}]}
        for t, w in [(0.0, 0.05), (0.1, 0.012), (0.2, 0.0121), (0.3, 0.011)]
    ]
    result = run_checks({"holding": holding}, ticks)["holding"]
    assert result["passed"] and result["first_at_s"] == 0.1
    # Where the jaws were not read, it cannot be scored.
    unread = [{"t": 0.0, "measured": [{}]}]
    assert run_checks({"holding": holding}, unread)["holding"] == {
        "kind": "gripper_width",
        "error": "the jaws were not read",
    }
    live = _live({"holding": holding})
    assert not live.needs_frames and live.reads_jaws
    for t, width in [(0.0, 0.05), (0.1, 0.012), (0.25, 0.012)]:
        live.submit(
            t=t, state=[], images={}, snapshot=_jaw_snapshot(width), pose_format="r"
        )
    assert live.ended() == "holding"
    live.reset()
    live.submit(
        t=0.0, state=[], images={}, snapshot=_jaw_snapshot(None), pose_format="r"
    )
    assert live.summary()["checks"]["holding"]["error"] == (
        "the jaws are not read in this rollout"
    )
    live.stop()


def test_a_step_with_several_end_checks_ends_once_all_have_passed():
    near = SettledInBox(
        box_min_m=(0.3, -1, 0), box_max_m=(0.5, 1, 1), hold_s=0.01, end_on_pass=True
    )
    open_ = GripperWidth(min_m=0.07, max_m=0.1, hold_s=0.0, end_on_pass=True)
    live = _live({"near": near, "open": open_}, end_when_all=True)
    live.submit(
        t=0.0, state=[], images={}, snapshot=_jaw_snapshot(0.01), pose_format="r"
    )
    assert live.ended() is None
    live.submit(
        t=0.1, state=[], images={}, snapshot=_jaw_snapshot(0.08), pose_format="r"
    )
    assert live.ended() == "near, open"
    live.stop()


# --- the jaws ---------------------------------------------------------------------


class _Jaws:
    """A gripper whose jaws reach their target at once, unless a part between
    them stops them closing; `creep` keeps them moving."""

    def __init__(self, *, part_m=None, creep=False):
        self.width = 0.05
        self.part_m = part_m
        self.creep = creep
        self.moves = []
        self.inits = 0

    def Enable(self, model):  # noqa: N802 - RDK API names
        self.model = model

    def Init(self):  # noqa: N802
        self.inits += 1

    def params(self):
        return SimpleNamespace(
            min_width=0.0,
            max_width=0.1,
            min_vel=0.01,
            max_vel=0.2,
            min_force=5.0,
            max_force=80.0,
        )

    def Move(self, width, velocity, force):  # noqa: N802
        self.moves.append((width, velocity, force))
        stop = self.part_m if self.part_m is not None and width < self.part_m else width
        self.width = stop

    def states(self):
        if self.creep:
            self.width -= 0.001
        return SimpleNamespace(width=self.width, force=0.0, is_moving=self.creep)


class _Tool:
    def Switch(self, model):  # noqa: N802
        pass


class _IdleArm(_Arm):
    """The fake follower, with the mode a gripper is prepared in."""

    def __init__(self):
        super().__init__()
        self.current_mode = "IDLE"

    def mode(self):
        return self.current_mode

    def SwitchMode(self, mode):  # noqa: N802
        super().SwitchMode(mode)
        self.current_mode = mode

    def Stop(self):  # noqa: N802
        self.current_mode = "IDLE"


GRIPPER = {"single_arm": {"follower": "gripper", "gripper_model": "Flexiv-GN01"}}


def _executor(jaws):
    executor = GripperExecutor(
        [_IdleArm()],
        ["single_arm"],
        GRIPPER,
        ["single_arm"],
        gripper_factory=lambda robot: jaws,
        tool_factory=lambda robot: _Tool(),
        idle_mode="IDLE",
    )
    executor.initialize()
    return executor


def test_the_jaws_move_once_and_are_read_until_they_stop():
    jaws = _Jaws(part_m=0.012)
    executor = _executor(jaws)
    stop = threading.Event()
    # Open: at the width at once, with the gripper's own speed and a quarter
    # of its force.
    assert move_jaws(
        executor, "single_arm", Move(gripper_m=0.08), stop=stop, timeout_s=5
    ) == pytest.approx(0.08)
    assert jaws.moves == [(0.08, 0.2, 20.0)]
    # Close on a part: they stop at its width, still, and are taken as done.
    started = time.monotonic()
    width = move_jaws(
        executor,
        "single_arm",
        Move(gripper_m=0.0, gripper_speed_m_s=0.05, gripper_force_n=200),
        stop=stop,
        timeout_s=5,
    )
    assert width == pytest.approx(0.012)
    assert time.monotonic() - started >= 0.3
    # The force is held to the gripper's range.
    assert jaws.moves[-1] == (0.0, 0.05, 80.0)


def test_jaws_that_never_stop_fail_their_move():
    executor = _executor(_Jaws(creep=True))
    with pytest.raises(RuntimeError, match="the jaws did not stop within 0.1 s"):
        move_jaws(
            executor,
            "single_arm",
            Move(gripper_m=0.0),
            stop=threading.Event(),
            timeout_s=0.1,
        )


# --- the retry rule, with stand-in rollouts -------------------------------------------


class _Rig:
    """Stand-in rollouts: each answers from a script of outcomes."""

    def __init__(self, skills=(), primitives=()):
        self.skills = list(skills)
        self.primitives = list(primitives)
        self.calls = []

    def run_skill(self, step, checks):
        self.calls.append(("skill", step.name, sorted(checks)))
        return self.skills.pop(0)

    def run_primitive(self, primitive, checks, watch_s, step):
        what = "check" if primitive is None else "primitive"
        self.calls.append((what, step.name, sorted(checks)))
        return self.primitives.pop(0) if self.primitives else _ok("done")


def _ok(reason="check passed"):
    return Outcome(True, reason, seconds=1.0)


def _failed(reason="time limit", **extra):
    return Outcome(False, reason, seconds=1.0, **extra)


INSERT = {
    "name": "insert",
    "skill": "skills/insert",
    "start": "ready",
    "end": "home",
    "tries": 3,
    "back_off": {"moves": [{"tool_m": [0, 0, -0.02]}], "max_force_n": 25},
}


def _chain(tmp_path, steps, rig):
    logs = []
    chain = ChainRunner(
        _plan(steps),
        run_skill=rig.run_skill,
        run_primitive=rig.run_primitive,
        record_path=tmp_path / "chains" / "run.json",
        append_log=lambda *entry: logs.append(entry),
    )
    chain.start()
    return chain, logs


def _until(condition, what):
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.005)


def _record(tmp_path):
    return json.loads((tmp_path / "chains" / "run.json").read_text())


def test_a_failed_try_backs_off_to_the_start_and_runs_the_step_again(tmp_path):
    rig = _Rig(skills=[_failed(), _failed(), _ok()])
    chain, _ = _chain(tmp_path, [INSERT], rig)
    chain.join(WAIT_S)
    assert chain.status()["state"] == "done"
    assert rig.calls == [
        ("check", "insert", ["ready"]),
        ("skill", "insert", ["home"]),
        # The back-off ends by checking the step's start again.
        ("primitive", "insert", ["ready"]),
        ("skill", "insert", ["home"]),
        ("primitive", "insert", ["ready"]),
        ("skill", "insert", ["home"]),
    ]
    record = _record(tmp_path)
    assert record["result"] == "done"
    assert [e["event"] for e in record["events"]] == [
        "start check",
        "try",
        "back-off",
        "try",
        "back-off",
        "try",
    ]
    assert record["events"][1]["outcome"] == {
        "passed": False,
        "reason": "time limit",
        "seconds": 1.0,
    }


def test_after_its_tries_the_chain_waits_for_the_operator_and_retries(tmp_path):
    rig = _Rig(skills=[_failed()] * 3 + [_ok()])
    chain, _ = _chain(tmp_path, [INSERT, {"name": "lift", "primitive": {}}], rig)
    _until(lambda: chain.status()["waiting_for_operator"], "the operator is called")
    assert chain.status()["waiting_for_operator"] == "insert: 3 tries failed"
    assert chain.status()["state"] == "waiting for the operator"
    # No back-off after the last try: the arm stays as the operator finds it.
    assert [c[0] for c in rig.calls].count("primitive") == 2
    chain.go_on("retry")
    chain.join(WAIT_S)
    assert chain.status()["state"] == "done"
    # Retried from its start check, with its tries counted afresh.
    assert rig.calls[-3:] == [
        ("check", "insert", ["ready"]),
        ("skill", "insert", ["home"]),
        ("primitive", "lift", []),
    ]


def test_a_back_off_that_meets_its_force_limit_calls_the_operator_at_once(tmp_path):
    blocked = _failed("blocked", blocked={"force_n": 31.5, "limit_n": 25.0})
    rig = _Rig(skills=[_failed()], primitives=[_ok("checks passed"), blocked])
    chain, _ = _chain(tmp_path, [INSERT], rig)
    _until(lambda: chain.status()["waiting_for_operator"], "the operator is called")
    assert chain.status()["waiting_for_operator"] == (
        "insert: the back-off met 31.5 N, over its 25 N limit"
    )
    assert [c[0] for c in rig.calls] == ["check", "skill", "primitive"]
    chain.stop()
    chain.join(WAIT_S)
    assert chain.status()["state"] == "stopped"
    assert _record(tmp_path)["result"] == "stopped"


def test_a_start_check_that_does_not_pass_calls_the_operator_who_may_skip(tmp_path):
    rig = _Rig(primitives=[_failed("check failed")])
    chain, _ = _chain(tmp_path, [INSERT, {"name": "lift", "primitive": {}}], rig)
    _until(lambda: chain.status()["waiting_for_operator"], "the operator is called")
    assert (
        "its start check 'ready' did not pass" in chain.status()["waiting_for_operator"]
    )
    with pytest.raises(ValueError, match="retry"):
        chain.go_on("later")
    chain.go_on("skip")
    chain.join(WAIT_S)
    assert chain.status()["state"] == "done"
    assert rig.calls[-1] == ("primitive", "lift", [])
    events = [e["event"] for e in _record(tmp_path)["events"]]
    assert events[-3:] == ["operator went on", "skipped by the operator", "try"]


def test_a_try_the_safety_envelope_stopped_is_not_backed_off(tmp_path):
    stopped = _failed("safety", safety_stop={"code": "force", "detail": "120 N"})
    rig = _Rig(skills=[stopped])
    chain, _ = _chain(tmp_path, [INSERT], rig)
    _until(lambda: chain.status()["waiting_for_operator"], "the operator is called")
    assert chain.status()["waiting_for_operator"] == (
        "insert: try 1 stopped: the safety envelope stopped it: 120 N"
    )
    assert [c[0] for c in rig.calls] == ["check", "skill"]
    chain.stop()
    chain.join(WAIT_S)


def test_a_record_the_disk_will_not_replace_does_not_stop_the_chain(
    tmp_path, monkeypatch
):
    # Windows refuses to replace a file someone has open.
    def refuse(self, target):
        raise PermissionError("in use")

    monkeypatch.setattr("flexivtrainer.rollout.chain._WRITE_TRIES", 2)
    monkeypatch.setattr(Path, "replace", refuse)
    chain, _ = _chain(tmp_path, [INSERT], _Rig(skills=[_ok()]))
    chain.join(WAIT_S)
    assert chain.status()["state"] == "done"


# --- on the fake robot, through the rollout service ---------------------------------


def _ready(**options):
    """The tool resting 3-10 mm along base x from the start."""
    return {
        "kind": "settled_in_box",
        "box_min_m": [0.403, -0.01, 0.29],
        "box_max_m": [0.41, 0.01, 0.33],
        "hold_s": 0.02,
        **options,
    }


def _run_chain(service, plan):
    service.start_chain(ChainPlan.from_dict(plan))
    chain = service._chain
    return chain


def test_a_chain_runs_its_steps_as_rollouts_on_the_robot(tmp_path, rdk):  # noqa: F811
    arm = _Arm()
    service, _ = _service(tmp_path, arm)
    skill = _skill(tmp_path)
    chain = _run_chain(
        service,
        {
            "steps": [
                # 1 mm along tool x a step, until it rests in the box.
                {"name": "slide", "skill": skill, "end": "ready", "max_s": 10},
                {
                    "name": "lift",
                    "primitive": {"moves": [{"tool_m": [0, 0, -0.02]}]},
                    "start": "ready",
                    "end": "high",
                },
            ],
            "checks": {
                "ready": _ready(),
                "high": {**_ready(), "box_min_m": [0.403, -0.01, 0.315]},
            },
        },
    )
    chain.join(WAIT_S)
    status = service.status()
    assert status["chain"]["state"] == "done", status["logs"][-5:]
    assert status["status"] == "idle"
    # The skill stopped once the tool was in the box: about 3 steps out.
    assert 0.403 <= arm.pose[0] <= 0.405
    # The lift backed off 2 cm along the tool's z, which points down.
    assert arm.pose[2] == pytest.approx(0.32)
    record = json.loads(Path(status["chain"]["record"]).read_text())
    events = [
        (e["step"], e["event"], e.get("outcome", {}).get("reason"))
        for e in record["events"]
    ]
    assert events == [
        ("slide", "try", "check ready passed"),
        ("lift", "start check", "checks passed"),
        ("lift", "try", "checks passed"),
    ]
    # The force axis the skill set was cleared before the primitive moved.
    axes = [call for call in arm.calls if call[0] == "axes"]
    assert axes[-1][1] == [False] * 6


def test_a_skill_out_of_time_backs_off_and_a_jam_calls_the_operator(tmp_path, rdk):  # noqa: F811
    arm = _Arm()
    # The tool feels 30 N: a part caught on the way back.
    arm.wrench = [0.0, 0.0, 30.0, 0.0, 0.0, 0.0]
    service, _ = _service(tmp_path, arm)
    skill = _skill(tmp_path)
    chain = _run_chain(
        service,
        {
            "steps": [
                {
                    "name": "insert",
                    "skill": skill,
                    "end": "far",
                    "max_s": 0.3,
                    "tries": 2,
                    "back_off": {
                        "moves": [{"tool_m": [0, 0, -0.01]}],
                        "max_force_n": 20,
                    },
                }
            ],
            "checks": {
                "far": {**_ready(), "box_min_m": [0.9, 0, 0], "box_max_m": [1, 1, 1]}
            },
        },
    )
    _until(
        lambda: service.status()["chain"]["waiting_for_operator"],
        "the operator is called",
    )
    status = service.status()
    assert status["chain"]["waiting_for_operator"] == (
        "insert: the back-off met 30.0 N, over its 20 N limit"
    )
    assert status["status"] == "running"
    # Nothing else may take the arm meanwhile.
    with pytest.raises(RuntimeError, match="A chain is running"):
        service.start(skill)
    record = json.loads(Path(status["chain"]["record"]).read_text())
    assert record["events"][0]["outcome"]["reason"] == "time limit"
    assert record["events"][1]["outcome"]["blocked"] == {
        "force_n": 30.0,
        "limit_n": 20.0,
    }
    service.stop()
    chain.join(WAIT_S)
    assert service.status()["chain"]["state"] == "stopped"
    assert service.status()["status"] == "idle"


def test_a_chain_picks_and_releases_with_the_jaws(tmp_path, rdk, monkeypatch):  # noqa: F811
    jaws = _Jaws(part_m=0.012)
    monkeypatch.setattr(
        "flexivtrainer.rollout.executors.gripper.flexivrdk",
        SimpleNamespace(
            Gripper=lambda robot: jaws,
            Tool=lambda robot: _Tool(),
            Mode=SimpleNamespace(IDLE="IDLE"),
        ),
    )
    monkeypatch.setattr(GripperExecutor, "INIT_SETTLE_S", 0.01)
    arm = _IdleArm()
    service, _ = _service(tmp_path, arm)
    service._get_end_effector_config = lambda: GRIPPER
    service._gripper_initialization_registry = GripperInitializationRegistry()
    jaw_check = {"kind": "gripper_width", "hold_s": 0.02}
    chain = _run_chain(
        service,
        {
            "steps": [
                {
                    "name": "pick",
                    "primitive": {
                        "moves": [
                            {"gripper_m": 0.08},
                            {"base_m": [0, 0, -0.02]},
                            {"gripper_m": 0, "gripper_force_n": 30},
                            {"base_m": [0, 0, 0.02]},
                        ]
                    },
                    "end": "holding",
                },
                {
                    "name": "release",
                    "primitive": {
                        "moves": [{"gripper_m": 0.08}, {"base_m": [0, 0, 0.03]}]
                    },
                    "end": ["open", "clear"],
                },
            ],
            "checks": {
                "holding": {**jaw_check, "min_m": 0.010, "max_m": 0.014},
                "open": {**jaw_check, "min_m": 0.07, "max_m": 0.1},
                "clear": {**_ready(), "box_min_m": [0.3, -0.1, 0.325]},
            },
        },
    )
    chain.join(WAIT_S)
    status = service.status()
    assert status["chain"]["state"] == "done", status["logs"][-5:]
    assert [m[0] for m in jaws.moves] == [0.08, 0.0, 0.08]
    assert jaws.moves[1][2] == 30.0
    # Initialised once in the session, though each step prepared the gripper.
    assert jaws.inits == 1
    assert arm.pose[2] == pytest.approx(0.33)
    record = json.loads(Path(status["chain"]["record"]).read_text())
    assert [e["outcome"]["reason"] for e in record["events"]] == ["checks passed"] * 2


def test_an_end_check_the_skill_cannot_score_calls_the_operator_at_once(
    tmp_path,
    rdk,  # noqa: F811
):
    # A step skill does not read the jaws, so their check cannot pass.
    service, _ = _service(tmp_path, _Arm())
    service._get_end_effector_config = lambda: GRIPPER
    chain = _run_chain(
        service,
        {
            "steps": [
                {
                    "name": "grip",
                    "skill": _skill(tmp_path),
                    "end": "holding",
                    "max_s": 0.2,
                    "tries": 3,
                }
            ],
            "checks": {
                "holding": {"kind": "gripper_width", "min_m": 0.01, "max_m": 0.014}
            },
        },
    )
    _until(
        lambda: service.status()["chain"]["waiting_for_operator"],
        "the operator is called",
    )
    assert service.status()["chain"]["waiting_for_operator"] == (
        "grip: try 1 stopped: check 'holding': the jaws are not read in this rollout"
    )
    service.stop()
    chain.join(WAIT_S)


def test_a_chain_that_needs_the_jaws_needs_a_gripper(tmp_path, rdk):  # noqa: F811
    service, _ = _service(tmp_path, _Arm())
    with pytest.raises(RuntimeError, match="no follower gripper is configured"):
        _run_chain(
            service,
            {
                "steps": [
                    {"name": "open", "primitive": {"moves": [{"gripper_m": 0.08}]}}
                ]
            },
        )


def test_a_chain_refuses_a_skill_that_is_not_there(tmp_path, rdk):  # noqa: F811
    service, _ = _service(tmp_path, _Arm())
    with pytest.raises(RuntimeError, match="'a': checkpoint not found"):
        _run_chain(
            service,
            {
                "steps": [{"name": "a", "skill": "nowhere", "end": "ready"}],
                "checks": {"ready": _ready()},
            },
        )


# --- the routes -----------------------------------------------------------------------


def test_the_routes_start_a_chain_and_pass_on_the_operators_answer():
    seen = []

    def start_chain(plan):
        seen.append(plan)
        return {"status": "running"}

    def chain_continue(action):
        seen.append(action)
        raise RuntimeError("No chain is running")

    app = create_app()
    rollout = SimpleNamespace(start_chain=start_chain, chain_continue=chain_continue)
    app.dependency_overrides[get_runtime_manager] = lambda: SimpleNamespace(
        rollout=rollout
    )
    client = TestClient(app)
    plan = {
        "steps": [{"name": "lift", "primitive": {"moves": [{"base_m": [0, 0, 1]}]}}]
    }
    assert client.post("/rollout/chain", json={"plan": plan}).status_code == 200
    assert isinstance(seen[0], ChainPlan) and seen[0].steps[0].name == "lift"
    bad = client.post("/rollout/chain", json={"plan": {"steps": []}})
    assert bad.status_code == 400 and "at least one step" in bad.json()["detail"]
    answer = client.post("/rollout/chain/continue", json={"action": "skip"})
    assert answer.status_code == 409 and seen[-1] == "skip"
    wrong = client.post("/rollout/chain/continue", json={"action": "later"})
    assert wrong.status_code == 422
