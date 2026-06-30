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

import time
from types import SimpleNamespace

import numpy as np
import pytest

from flexivtrainer.config import AppSettings, StorageConfig, TeleopRobotPair
from flexivtrainer.rollout.service import RolloutService


class _FakeRobotStates:
    def __init__(self, base: float) -> None:
        self.tcp_pose = [base + i for i in range(7)]
        self.tcp_vel = [base + 10 + i for i in range(6)]
        self.ext_wrench_in_world = [base + 20 + i for i in range(6)]


class _FakeRobot:
    """Records the Cartesian commands it receives; never faults by default."""

    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.enabled = False
        self.mode = None
        self.commands: list[tuple[list[float], list[float]]] = []
        self._fault = False

    def fault(self) -> bool:
        return self._fault

    def ClearFault(self) -> bool:  # noqa: N802 - RDK API name
        self._fault = False
        return True

    def Enable(self) -> None:  # noqa: N802
        self.enabled = True

    def operational(self) -> bool:
        return self.enabled

    def SwitchMode(self, mode) -> None:  # noqa: N802
        self.mode = mode

    def states(self):
        return _FakeRobotStates(base=1.0)

    def SendCartesianMotionForce(self, pose, wrench):  # noqa: N802
        self.commands.append((list(pose), list(wrench)))

    def Stop(self) -> None:  # noqa: N802
        pass


class _FakePolicy:
    """Returns a fixed action vector with side-prefixed names baked in below."""

    def __init__(self, action_vector: list[float]) -> None:
        self._action = action_vector
        self.batches: list[dict] = []

    def select_action(self, batch):
        self.batches.append(batch)
        return np.asarray(self._action, dtype=np.float32)


def _identity_processor(value):
    return value


def _fake_loader(policy):
    """A policy_loader returning the (policy, preprocessor, postprocessor) tuple.

    Tests exercise action dispatch, not normalization, so the processors are
    identity passthroughs; ``predict_action`` is patched per loop test below.
    """
    return lambda path, device: (policy, _identity_processor, _identity_processor)


def _settings(tmp_path) -> AppSettings:
    return AppSettings(storage=StorageConfig(root=tmp_path))


def _teleop(initialized: bool = False):
    return SimpleNamespace(
        snapshot=lambda: SimpleNamespace(initialized=initialized)
    )


def _cameras():
    # No image entries are exercised in these state-only tests; capture_frame
    # returns a frame missing an image so it is simply skipped.
    return SimpleNamespace(capture_frame=lambda name, **kwargs: {})


def _single_arm_pairs():
    return [TeleopRobotPair(leader_serial="L1", follower_serial="F1")]


def _checkpoint(tmp_path) -> str:
    """A path that exists, so ``start`` passes its checkpoint-exists guard."""
    path = tmp_path / "ckpt"
    path.mkdir()
    return str(path)


def _make_service(tmp_path, *, policy, robot):
    return RolloutService(
        _settings(tmp_path),
        _cameras(),
        _teleop(initialized=False),
        _single_arm_pairs,
        lambda: ["single_arm"],
        policy_loader=_fake_loader(policy),
        robot_factory=lambda serial: robot,
        resolve_device=lambda configured: "cpu",
    )


def test_start_refuses_when_teleop_initialized(tmp_path) -> None:
    service = RolloutService(
        _settings(tmp_path),
        _cameras(),
        _teleop(initialized=True),
        _single_arm_pairs,
        lambda: ["single_arm"],
        policy_loader=_fake_loader(_FakePolicy([])),
        robot_factory=_FakeRobot,
        resolve_device=lambda configured: "cpu",
    )
    with pytest.raises(RuntimeError, match="Stop teleoperation"):
        service.start("/tmp/ckpt")


def test_start_refuses_missing_checkpoint(tmp_path) -> None:
    service = _make_service(tmp_path, policy=_FakePolicy([]), robot=_FakeRobot("F1"))
    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        service.start(str(tmp_path / "does-not-exist"))


def test_start_refuses_without_follower_serial(tmp_path) -> None:
    service = RolloutService(
        _settings(tmp_path),
        _cameras(),
        _teleop(initialized=False),
        lambda: [TeleopRobotPair(leader_serial="L1", follower_serial="")],
        lambda: ["single_arm"],
        policy_loader=_fake_loader(_FakePolicy([])),
        robot_factory=_FakeRobot,
        resolve_device=lambda configured: "cpu",
    )
    with pytest.raises(RuntimeError, match="follower robot serial"):
        service.start(_checkpoint(tmp_path))


def test_action_to_list_handles_numpy_and_torch_like() -> None:
    assert RolloutService._action_to_list(np.array([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]

    class _TorchLike:
        def __init__(self, data):
            self._data = np.asarray(data)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._data

    assert RolloutService._action_to_list(_TorchLike([4.0, 5.0])) == [4.0, 5.0]


def test_plan_action_layout_locates_pose_and_wrench_runs(tmp_path) -> None:
    service = _make_service(tmp_path, policy=_FakePolicy([]), robot=_FakeRobot("F1"))
    # single_arm action: tcp_pose (7) -> tcp_twist (6) -> tcp_wrench (6).
    names = (
        [f"single_arm.tcp_pose.{a}" for a in "abcdefg"]
        + [f"single_arm.tcp_twist.{i}" for i in range(6)]
        + [f"single_arm.tcp_wrench.{i}" for i in range(6)]
    )
    layout = service._plan_action_layout(names, ["single_arm"])
    assert len(layout) == 1
    assert layout[0]["pose"] == slice(0, 7)
    assert layout[0]["wrench"] == slice(13, 19)


def test_dispatch_action_sends_pose_and_wrench_slices(tmp_path) -> None:
    robot = _FakeRobot("F1")
    service = _make_service(tmp_path, policy=_FakePolicy([]), robot=robot)
    action = list(range(19))  # pose=0..6, twist=7..12, wrench=13..18
    layout = [{"side": "single_arm", "pose": slice(0, 7), "wrench": slice(13, 19)}]
    service._dispatch_action(action, [robot], layout)
    (pose, wrench), = robot.commands
    # Position and wrench pass through unchanged; the quaternion (pose[3:7]) is
    # renormalized to unit length before commanding.
    assert pose[0:3] == [0, 1, 2]
    assert pytest.approx(sum(c * c for c in pose[3:7]) ** 0.5) == 1.0
    assert wrench == [13, 14, 15, 16, 17, 18]


def _run_one_tick(service: RolloutService, robot: _FakeRobot, checkpoint: str) -> None:
    """Start the loop and stop it after at least one command is sent."""
    service.start(checkpoint)
    deadline = time.monotonic() + 2.0
    while not robot.commands and time.monotonic() < deadline:
        time.sleep(0.01)
    service.stop()


def test_running_loop_sends_commands_and_stops(tmp_path, monkeypatch) -> None:
    # 19-D single-arm action (pose+twist+wrench) so pose/wrench slices exist.
    action = [float(i) for i in range(19)]
    policy = _FakePolicy(action)
    robot = _FakeRobot("F1")
    service = RolloutService(
        _settings(tmp_path),
        _cameras(),
        _teleop(initialized=False),
        _single_arm_pairs,
        lambda: ["single_arm"],
        policy_loader=_fake_loader(policy),
        robot_factory=lambda serial: robot,
        resolve_device=lambda configured: "cpu",
    )
    # Inference runs through lerobot's predict_action (which needs torch/lerobot);
    # patch the module wrapper to call the fake policy directly so the test stays
    # hermetic and still exercises the real dispatch/slicing path.
    monkeypatch.setattr(
        "flexivtrainer.rollout.service._predict_action",
        lambda obs, pol, dev, pre, post: pol.select_action(obs),
    )
    # Patch the RDK mode lookup so no real flexivrdk import is needed.
    monkeypatch.setattr(
        "flexivtrainer.rollout.service._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    _run_one_tick(service, robot, _checkpoint(tmp_path))

    assert robot.enabled
    assert robot.mode == "cmf"
    assert robot.commands, "expected at least one Cartesian command"
    pose, wrench = robot.commands[0]
    assert pose[0:3] == action[0:3]
    assert pytest.approx(sum(c * c for c in pose[3:7]) ** 0.5) == 1.0
    assert wrench == action[13:19]
    assert service.status()["status"] in {"idle", "stopped"}


def test_fault_aborts_loop_and_records_error(tmp_path, monkeypatch) -> None:
    policy = _FakePolicy([float(i) for i in range(19)])
    robot = _FakeRobot("F1")
    robot._fault = False
    service = RolloutService(
        _settings(tmp_path),
        _cameras(),
        _teleop(initialized=False),
        _single_arm_pairs,
        lambda: ["single_arm"],
        policy_loader=_fake_loader(policy),
        robot_factory=lambda serial: robot,
        resolve_device=lambda configured: "cpu",
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.service._predict_action",
        lambda obs, pol, dev, pre, post: pol.select_action(obs),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.service._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    service.start(_checkpoint(tmp_path))
    # Trip a fault; the loop checks fault() each tick and must abort.
    robot._fault = True
    deadline = time.monotonic() + 2.0
    while service.status()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    status = service.status()
    assert status["status"] == "failed"
    assert "Fault" in (status["error"] or "")
    service.stop()
