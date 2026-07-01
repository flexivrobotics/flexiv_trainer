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

import threading
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

    def SendCartesianMotionForce(  # noqa: N802
        self,
        pose,
        wrench=(),
        velocity=(),
        max_linear_vel=0.5,
        max_angular_vel=1.0,
        max_linear_acc=2.0,
        max_angular_acc=5.0,
    ):
        self.commands.append((list(pose), list(wrench), list(velocity)))
        self.motion_limits = (
            max_linear_vel, max_angular_vel, max_linear_acc, max_angular_acc
        )

    def Stop(self) -> None:  # noqa: N802
        pass


class _FakePolicy:
    """Returns a fixed action vector with side-prefixed names baked in below."""

    def __init__(self, action_vector: list[float]) -> None:
        self._action = action_vector
        self.batches: list[dict] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

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


def test_actions_to_lists_handles_chunk_and_single() -> None:
    # Bare 1-D action -> single-element outer list.
    assert RolloutService._actions_to_lists(
        np.array([1.0, 2.0, 3.0])
    ) == [[1.0, 2.0, 3.0]]
    # 2-D chunk -> one inner list per step.
    assert RolloutService._actions_to_lists(
        np.array([[1.0, 2.0], [3.0, 4.0]])
    ) == [[1.0, 2.0], [3.0, 4.0]]

    class _TorchLike:
        def __init__(self, data):
            self._data = np.asarray(data)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._data

    assert RolloutService._actions_to_lists(_TorchLike([[4.0, 5.0]])) == [[4.0, 5.0]]


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
    assert layout[0]["twist"] == slice(7, 13)
    assert layout[0]["wrench"] == slice(13, 19)


def test_diffusion_scheduler_override_swaps_to_ddim(tmp_path) -> None:
    pytest.importorskip("diffusers")
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    # A diffusion-policy stand-in: only the attributes the override touches.
    ddpm = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon",
    )
    policy = SimpleNamespace(
        diffusion=SimpleNamespace(noise_scheduler=ddpm, num_inference_steps=100)
    )
    service = _make_service(tmp_path, policy=_FakePolicy([]), robot=_FakeRobot("F1"))
    # Request a DDIM swap explicitly and confirm the override applies it.
    service._settings.rollout.diffusion.scheduler = "DDIM"
    service._settings.rollout.diffusion.inference_steps = 10
    service._apply_diffusion_scheduler_override(policy)
    assert isinstance(policy.diffusion.noise_scheduler, DDIMScheduler)
    assert policy.diffusion.num_inference_steps == 10
    # The trained schedule is preserved -- only the sampler family changed.
    assert policy.diffusion.noise_scheduler.config.num_train_timesteps == 100


def test_diffusion_scheduler_override_noop_when_disabled(tmp_path) -> None:
    policy = SimpleNamespace(
        diffusion=SimpleNamespace(noise_scheduler=object(), num_inference_steps=100)
    )
    settings = _settings(tmp_path)
    settings.rollout.diffusion.scheduler = ""
    service = RolloutService(
        settings, _cameras(), _teleop(initialized=False),
        _single_arm_pairs, lambda: ["single_arm"],
        policy_loader=_fake_loader(_FakePolicy([])),
        robot_factory=_FakeRobot, resolve_device=lambda configured: "cpu",
    )
    sentinel = policy.diffusion.noise_scheduler
    service._apply_diffusion_scheduler_override(policy)
    # "" leaves the checkpoint's own scheduler and step count untouched.
    assert policy.diffusion.noise_scheduler is sentinel
    assert policy.diffusion.num_inference_steps == 100


def _run_one_tick(service: RolloutService, robot: _FakeRobot, checkpoint: str) -> None:
    """Start the loop and stop it after at least one command is sent."""
    service.start(checkpoint)
    deadline = time.monotonic() + 2.0
    while not robot.commands and time.monotonic() < deadline:
        time.sleep(0.01)
    service.stop()


def test_rollout_loop_streams_commands_and_stops(tmp_path, monkeypatch) -> None:
    # The rollout's only send path: a high-rate sender thread streams interpolated
    # poses to the robot. Verify the loop runs, enables + switches the robot,
    # streams commands toward the policy pose, and shuts down cleanly (no hang).
    action = [float(i) for i in range(19)]
    policy = _FakePolicy(action)
    robot = _FakeRobot("F1")
    settings = _settings(tmp_path)
    settings.rollout.sender_hz = 200
    service = RolloutService(
        settings,
        _cameras(),
        _teleop(initialized=False),
        _single_arm_pairs,
        lambda: ["single_arm"],
        policy_loader=_fake_loader(policy),
        robot_factory=lambda serial: robot,
        resolve_device=lambda configured: "cpu",
    )
    # Inference runs through lerobot's predict_action (needs torch/lerobot); patch
    # the wrapper to call the fake policy directly so the test stays hermetic.
    monkeypatch.setattr(
        "flexivtrainer.rollout.service._predict_action_chunk",
        lambda obs, pol, dev, pre, post: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
    )
    # Patch the RDK mode lookup so no real flexivrdk import is needed.
    monkeypatch.setattr(
        "flexivtrainer.rollout.service._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    _run_one_tick(service, robot, _checkpoint(tmp_path))

    assert policy.reset_count == 1
    assert robot.enabled
    assert robot.mode == "cmf"
    assert robot.commands, "expected at least one streamed Cartesian command"
    pose, _wrench, _velocity = robot.commands[0]
    # The interpolator seeds from the measured pose ([1,2,3]) and eases toward the
    # policy command ([0,1,2]); the first streamed x lies within that range and
    # the quaternion stays unit-norm.
    assert min(action[0], 1.0) - 1e-6 <= pose[0] <= max(action[0], 1.0) + 1e-6
    assert pytest.approx(sum(c * c for c in pose[3:7]) ** 0.5) == 1.0
    # The configured hardware speed/accel caps are passed to the robot.
    cfg = service._settings.rollout
    assert robot.motion_limits == (
        cfg.max_linear_vel, cfg.max_angular_vel,
        cfg.max_linear_acc, cfg.max_angular_acc,
    )
    # Clean shutdown: status settled and the sender thread no longer running.
    assert service.status()["status"] in {"idle", "stopped"}
    assert not any(
        t.name == "rollout-sender" and t.is_alive() for t in threading.enumerate()
    )


def test_log_step_reports_expected_and_actual_frequency(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setattr(
        "flexivtrainer.rollout.service._predict_action_chunk",
        lambda obs, pol, dev, pre, post: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.service._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    _run_one_tick(service, robot, _checkpoint(tmp_path))

    expected_hz = service._settings.rollout.planner_hz
    logs = service.status()["logs"]
    # An obs row is logged on step 0 (0 % log_every == 0) carrying the expected
    # frequency and a measured actual frequency, e.g. "freq=123.4/30.0Hz".
    assert any(f"/{float(expected_hz):.1f}Hz" in line for line in logs)


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
        "flexivtrainer.rollout.service._predict_action_chunk",
        lambda obs, pol, dev, pre, post: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
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
