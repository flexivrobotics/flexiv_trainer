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

import json
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from flexivtrainer.config import AppSettings, StorageConfig, TeleopRobotPair
from flexivtrainer.policies import act as act_policy
from flexivtrainer.policies import diffusion as diffusion_policy
from flexivtrainer.policies import dit as dit_policy
from flexivtrainer.rollout import hardware, observations
from flexivtrainer.rollout.checkpoint import (
    _checkpoint_policy_type,
    _checkpoint_requires_task,
    _checkpoint_target_hz,
    checkpoint_image_resolutions,
)
from flexivtrainer.rollout.executors.waypoint import WaypointExecutor
from flexivtrainer.rollout.hardware import _zero_ft_sensor
from flexivtrainer.rollout.runners.bspline import BSplineRunner
from flexivtrainer.rollout.runners.waypoint import WaypointRunner
from flexivtrainer.rollout.service import RolloutService, _describe_rollout_overrides


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
        self.mode_history: list = []
        self.primitives: list = []
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
        self.mode_history.append(mode)

    def ExecutePrimitive(self, name, input_params) -> None:  # noqa: N802
        self.primitives.append((name, input_params))

    def primitive_states(self) -> dict:
        return {"reachedTarget": 1}

    def busy(self) -> bool:
        return False

    def states(self):
        return _FakeRobotStates(base=1.0)

    def SendCartesianMotionForce(  # noqa: N802
        self,
        pose,
        wrench=(),
        *args,
        velocity=(),
        max_linear_vel=0.5,
        max_angular_vel=1.0,
        max_linear_acc=2.0,
        max_angular_acc=5.0,
    ):
        if len(args) == 5:
            (
                velocity,
                max_linear_vel,
                max_angular_vel,
                max_linear_acc,
                max_angular_acc,
            ) = args
        elif len(args) == 4:
            max_linear_vel, max_angular_vel, max_linear_acc, max_angular_acc = args
            velocity = ()
        elif args:
            raise TypeError(f"unexpected SendCartesianMotionForce args: {args!r}")
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


def _checkpoint_with_dataset_fps(tmp_path, fps: int = 10) -> str:
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"fps": fps}), encoding="utf-8")

    model = tmp_path / "ckpt" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({"type": "diffusion"}), encoding="utf-8"
    )
    (model / "train_config.json").write_text(
        json.dumps({"dataset": {"root": str(dataset)}}),
        encoding="utf-8",
    )
    return str(tmp_path / "ckpt")


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


def test_zero_ft_sensor_runs_primitive_before_force_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(
            NRT_PRIMITIVE_EXECUTION="prim",
            NRT_CARTESIAN_MOTION_FORCE="cmf",
        ),
    )
    service = _make_service(tmp_path, policy=_FakePolicy([]), robot=_FakeRobot("F1"))
    robot = hardware.connect_robot(
        service._robot_factory,
        "F1",
        service._stop_event,
        prepare_motion=service._prepare_motion,
    )

    assert robot.primitives == [("ZeroFTSensor", {})]
    assert robot.mode_history == ["prim", "cmf"]


def test_connect_robot_without_primitive_support(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(
            NRT_PRIMITIVE_EXECUTION="prim",
            NRT_CARTESIAN_MOTION_FORCE="cmf",
        ),
    )
    class _NoPrimitiveRobot(_FakeRobot):
        ExecutePrimitive = None  # firmware/stub lacking the primitive

    robot = _NoPrimitiveRobot("F1")
    service = _make_service(tmp_path, policy=_FakePolicy([]), robot=robot)
    connected = hardware.connect_robot(
        service._robot_factory,
        "F1",
        service._stop_event,
        prepare_motion=service._prepare_motion,
    )

    assert connected.mode_history == ["cmf"]


def test_zero_ft_sensor_returns_false_without_primitive(monkeypatch) -> None:
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_PRIMITIVE_EXECUTION="prim"),
    )

    class _NoPrimitiveRobot(_FakeRobot):
        ExecutePrimitive = None

    assert not _zero_ft_sensor(_NoPrimitiveRobot("F1"), threading.Event())


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
    assert WaypointRunner._actions_to_lists(
        np.array([1.0, 2.0, 3.0])
    ) == [[1.0, 2.0, 3.0]]
    # 2-D chunk -> one inner list per step.
    assert WaypointRunner._actions_to_lists(
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

    assert WaypointRunner._actions_to_lists(_TorchLike([[4.0, 5.0]])) == [[4.0, 5.0]]


def test_diffusion_scheduler_override_swaps_to_ddim(tmp_path) -> None:
    pytest.importorskip("diffusers")
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    # A diffusion-policy stand-in: only the attributes the override touches.
    ddpm = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    policy = SimpleNamespace(
        diffusion=SimpleNamespace(noise_scheduler=ddpm, num_inference_steps=100)
    )
    # Request a DDIM swap explicitly and confirm the override applies it.
    rollout_cfg = _settings(tmp_path).policies.diffusion.rollout
    rollout_cfg.noise_scheduler_type = "DDIM"
    rollout_cfg.num_denoise_steps = 10
    assert diffusion_policy.apply_rollout_overrides(policy, rollout_cfg)
    assert isinstance(policy.diffusion.noise_scheduler, DDIMScheduler)
    assert policy.diffusion.num_inference_steps == 10
    # The trained schedule is preserved -- only the sampler family changed.
    assert policy.diffusion.noise_scheduler.config.num_train_timesteps == 100


def test_diffusion_scheduler_override_noop_when_disabled(tmp_path) -> None:
    policy = SimpleNamespace(
        diffusion=SimpleNamespace(noise_scheduler=object(), num_inference_steps=100)
    )
    settings = _settings(tmp_path)
    settings.policies.diffusion.rollout.noise_scheduler_type = ""
    sentinel = policy.diffusion.noise_scheduler
    assert not diffusion_policy.apply_rollout_overrides(
        policy, settings.policies.diffusion.rollout
    )
    # "" leaves the checkpoint's own scheduler and step count untouched.
    assert policy.diffusion.noise_scheduler is sentinel
    assert policy.diffusion.num_inference_steps == 100


def test_dit_scheduler_override_swaps_to_ddim(tmp_path) -> None:
    pytest.importorskip("diffusers")
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    ddpm = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    # A DiT stand-in: config.objective string + objective module attributes.
    policy = SimpleNamespace(
        config=SimpleNamespace(objective="diffusion"),
        objective=SimpleNamespace(noise_scheduler=ddpm, num_inference_steps=100),
    )
    rollout_cfg = _settings(tmp_path).policies.multi_task_dit.rollout
    rollout_cfg.noise_scheduler_type = "DDIM"
    rollout_cfg.num_denoise_steps = 10
    assert dit_policy.apply_rollout_overrides(policy, rollout_cfg)
    assert isinstance(policy.objective.noise_scheduler, DDIMScheduler)
    assert policy.objective.num_inference_steps == 10
    assert policy.objective.noise_scheduler.config.num_train_timesteps == 100


def test_dit_scheduler_override_skips_flow_matching(tmp_path) -> None:
    sentinel = object()
    policy = SimpleNamespace(
        config=SimpleNamespace(objective="flow_matching"),
        objective=SimpleNamespace(noise_scheduler=sentinel, num_inference_steps=100),
    )
    rollout_cfg = _settings(tmp_path).policies.multi_task_dit.rollout
    rollout_cfg.noise_scheduler_type = "DDIM"
    assert not dit_policy.apply_rollout_overrides(policy, rollout_cfg)
    assert policy.objective.noise_scheduler is sentinel
    assert policy.objective.num_inference_steps == 100


def test_rollout_for_multi_task_dit_returns_dit_config(tmp_path) -> None:
    rollout_cfg = _settings(tmp_path).policies.rollout_for("multi_task_dit")
    assert isinstance(rollout_cfg, dit_policy.RolloutConfig)
    assert rollout_cfg.noise_scheduler_type == "DDIM"
    assert rollout_cfg.num_denoise_steps == 10


def test_checkpoint_target_hz_reads_training_dataset_fps(tmp_path) -> None:
    checkpoint = _checkpoint_with_dataset_fps(tmp_path, fps=12)

    assert _checkpoint_target_hz(checkpoint) == 12.0


def _checkpoint_of_type(tmp_path, policy_type: str) -> str:
    model = tmp_path / "ckpt" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({"type": policy_type}), encoding="utf-8"
    )
    return str(tmp_path / "ckpt")


def test_checkpoint_policy_type_and_requires_task(tmp_path) -> None:
    vla = _checkpoint_of_type(tmp_path / "a", "multi_task_dit")
    assert _checkpoint_policy_type(vla) == "multi_task_dit"
    assert _checkpoint_requires_task(vla) is True

    non_vla = _checkpoint_of_type(tmp_path / "b", "diffusion")
    assert _checkpoint_policy_type(non_vla) == "diffusion"
    assert _checkpoint_requires_task(non_vla) is False

    # Unknown/missing type defaults to requiring a task (box stays available).
    bare = tmp_path / "c"
    bare.mkdir()
    assert _checkpoint_requires_task(str(bare)) is True


def _checkpoint_with_image_shapes(tmp_path, shapes: dict[str, list[int]]) -> str:
    model = tmp_path / "ckpt" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    f"observation.images.{name}": {"type": "VISUAL", "shape": shape}
                    for name, shape in shapes.items()
                }
                | {"observation.state": {"type": "STATE", "shape": [38]}},
            }
        ),
        encoding="utf-8",
    )
    return str(tmp_path / "ckpt")


def test_checkpoint_image_resolutions_reads_channels_first_shapes(tmp_path) -> None:
    checkpoint = _checkpoint_with_image_shapes(
        tmp_path, {"ego": [3, 240, 320], "left_wrist": [3, 480, 640]}
    )

    assert checkpoint_image_resolutions(checkpoint) == {
        "ego": (240, 320),
        "left_wrist": (480, 640),
    }


def test_checkpoint_image_resolutions_skips_channels_last_shapes(tmp_path) -> None:
    # (height, width, channels) must not be misread as (240, 320) here.
    checkpoint = _checkpoint_with_image_shapes(tmp_path, {"ego": [240, 320, 3]})

    assert checkpoint_image_resolutions(checkpoint) == {}


def _cameras_returning(images: dict[str, np.ndarray]):
    return SimpleNamespace(
        capture_frame=lambda name, **kwargs: {"image": images.get(name)}
    )


def test_grab_images_downscales_to_checkpoint_resolution() -> None:
    # BGR red so the resize is checked alongside the RGB flip.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 2] = 255

    images = observations.grab_images(
        _cameras_returning({"ego": frame}), ["ego"], {"ego": (240, 320)}
    )

    assert images["ego"].shape == (240, 320, 3)
    assert images["ego"][0, 0].tolist() == [255, 0, 0]


def test_grab_images_passes_matching_frames_through() -> None:
    frame = np.arange(240 * 320 * 3, dtype=np.uint8).reshape(240, 320, 3)

    images = observations.grab_images(
        _cameras_returning({"ego": frame}), ["ego"], {"ego": (240, 320)}
    )

    assert np.array_equal(images["ego"], frame[:, :, ::-1])


def test_grab_images_rejects_aspect_ratio_mismatch() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="aspect ratio"):
        observations.grab_images(
            _cameras_returning({"ego": frame}), ["ego"], {"ego": (240, 240)}
        )


def _bspline_action_names(
    rows: int = 16,
    side: str = "single_arm",
    *,
    gripper: bool = False,
) -> list[str]:
    channels = [
        "knot",
        *(f"{side}.tcp_pose.{axis}" for axis in ("x", "y", "z")),
        *(
            f"{side}.tcp_rotation_6d.{axis}"
            for axis in ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")
        ),
    ]
    if gripper:
        channels.append(f"{side}.gripper.width")
    return [
        f"bspline.row_{row:02d}.{channel}"
        for row in range(rows)
        for channel in channels
    ]


class _FakeBSplinePolicy:
    def __init__(self, action: np.ndarray, *, knot_rate_hz: float | None) -> None:
        self.config = SimpleNamespace(
            type="bspline_diffusion",
            action_feature_names=_bspline_action_names(),
            horizon=16,
            spline_degree=3,
            knot_rate_hz=knot_rate_hz,
        )
        self.action = action.reshape(1, 1, -1)
        self.observations: list[dict] = []

    def reset(self) -> None:
        self.observations.clear()

    def enqueue_observation(self, batch: dict) -> None:
        self.observations.append(batch)

    def predict_action_chunk(self) -> np.ndarray:
        return self.action.copy()


def _constant_bspline_action(*, end_time: float = 9.0) -> np.ndarray:
    matrix = np.zeros((16, 10), dtype=np.float64)
    matrix[:, 0] = np.concatenate(
        [
            np.zeros(4),
            np.linspace(end_time / 9, end_time * 8 / 9, 8),
            np.full(4, end_time),
        ]
    )
    matrix[:, 1:4] = [0.4, -0.1, 0.3]
    matrix[:, 4:10] = [1, 0, 0, 0, 1, 0]
    return matrix.reshape(-1)


def test_bspline_missing_timing_fails_before_robot_initialization(tmp_path) -> None:
    checkpoint = _checkpoint_of_type(tmp_path, "bspline_diffusion")
    initialized = []
    policy = _FakeBSplinePolicy(
        _constant_bspline_action(), knot_rate_hz=None
    )
    service = RolloutService(
        _settings(tmp_path),
        _cameras(),
        _teleop(initialized=False),
        _single_arm_pairs,
        lambda: ["single_arm"],
        policy_loader=lambda path, device: (
            initialized.append("policy")
            or (policy, _identity_processor, _identity_processor)
        ),
        robot_factory=lambda serial: initialized.append("robot"),
        resolve_device=lambda configured: "cpu",
    )

    with pytest.raises(RuntimeError, match="no knot_rate_hz"):
        service.start(checkpoint)

    assert initialized == ["policy"]


def test_bspline_malformed_layout_fails_before_robot_initialization(
    tmp_path,
) -> None:
    checkpoint = _checkpoint_of_type(tmp_path, "bspline_diffusion")
    initialized = []
    policy = _FakeBSplinePolicy(_constant_bspline_action(), knot_rate_hz=10)
    policy.config.action_feature_names[0] = "action.0"
    service = RolloutService(
        _settings(tmp_path),
        _cameras(),
        _teleop(initialized=False),
        _single_arm_pairs,
        lambda: ["single_arm"],
        policy_loader=_fake_loader(policy),
        robot_factory=lambda serial: initialized.append("robot"),
        resolve_device=lambda configured: "cpu",
    )

    with pytest.raises(RuntimeError, match="Malformed B-spline"):
        service.start(checkpoint)

    assert initialized == []


def test_bspline_gripper_contract_fails_before_robot_initialization(
    tmp_path,
) -> None:
    checkpoint = _checkpoint_of_type(tmp_path, "bspline_diffusion")
    initialized = []
    policy = _FakeBSplinePolicy(_constant_bspline_action(), knot_rate_hz=10)
    policy.config.action_feature_names = _bspline_action_names(gripper=True)
    service = RolloutService(
        _settings(tmp_path),
        _cameras(),
        _teleop(initialized=False),
        _single_arm_pairs,
        lambda: ["single_arm"],
        get_end_effector_config=lambda: {
            "single_arm": {"follower": "none"}
        },
        policy_loader=_fake_loader(policy),
        robot_factory=lambda serial: initialized.append("robot"),
        resolve_device=lambda configured: "cpu",
    )

    with pytest.raises(RuntimeError, match="no follower gripper"):
        service.start(checkpoint)

    assert initialized == []


def test_robot_snapshot_includes_measured_gripper_telemetry(tmp_path) -> None:
    _make_service(tmp_path, policy=_FakePolicy([]), robot=_FakeRobot("F1"))

    snapshot = observations.read_robot_snapshot(
        [_FakeRobot("F1")],
        {"single_arm": {"width": 0.04, "force": -2.0}},
        ["single_arm"],
    )

    assert snapshot["robots"]["robot_0"]["gripper"] == {
        "width": 0.04,
        "force": -2.0,
    }


def test_stop_releases_robot_when_gripper_shutdown_fails(tmp_path) -> None:
    robot = _FakeRobot("F1")
    service = _make_service(tmp_path, policy=_FakePolicy([]), robot=robot)

    class FailingStop:
        def stop(self) -> None:
            raise RuntimeError("worker stuck")

    runner = object.__new__(BSplineRunner)
    runner._robots = [robot]
    runner._thread = None
    runner._bspline_executor = None
    runner._gripper_executor = FailingStop()
    runner._stop_robots = hardware.stop_robots

    service._running = True
    service._robots = [robot]
    service._runner = runner

    status = service.stop()

    assert service._robots == []
    assert status["status"] == "failed"
    assert "worker stuck" in status["error"]


def _run_one_tick(service: RolloutService, robot: _FakeRobot, checkpoint: str) -> None:
    """Start the loop and stop it after at least one command is sent."""
    service.start(checkpoint)
    deadline = time.monotonic() + 2.0
    while not robot.commands and time.monotonic() < deadline:
        time.sleep(0.01)
    service.stop()


def test_bspline_rollout_decodes_before_cartesian_dispatch(
    tmp_path, monkeypatch
) -> None:
    policy = _FakeBSplinePolicy(_constant_bspline_action(), knot_rate_hz=10)
    robot = _FakeRobot("F1")
    service = _make_service(tmp_path, policy=policy, robot=robot)
    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._prepare_policy_observation",
        lambda observation, device, preprocessor, **kwargs: observation,
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(
            NRT_PRIMITIVE_EXECUTION="prim",
            NRT_CARTESIAN_MOTION_FORCE="cmf",
        ),
    )

    _run_one_tick(
        service,
        robot,
        _checkpoint_of_type(tmp_path, "bspline_diffusion"),
    )

    assert policy.observations
    assert robot.commands
    pose, wrench, velocity = robot.commands[0]
    assert pose == pytest.approx([0.4, -0.1, 0.3, 1, 0, 0, 0])
    assert wrench == [0.0] * 6
    assert velocity == [0.0] * 6
    assert len(pose) == 7
    metrics = service.status()["metrics"]
    assert metrics
    assert set(metrics[-1]) >= {
        "send_hz",
        "missed_deadlines",
        "spline_remaining_s",
        "infer_ms",
        "alignment_error",
        "handoff_warnings",
    }


def test_bspline_observations_continue_during_slow_inference(
    tmp_path, monkeypatch
) -> None:
    inference_started = threading.Event()
    inference_release = threading.Event()

    class SlowPolicy(_FakeBSplinePolicy):
        def __init__(self) -> None:
            super().__init__(
                _constant_bspline_action(end_time=2.0),
                knot_rate_hz=10,
            )
            self.inference_count = 0

        def predict_action_chunk(self) -> np.ndarray:
            self.inference_count += 1
            if self.inference_count == 2:
                inference_started.set()
                assert inference_release.wait(timeout=2.0)
            return super().predict_action_chunk()

    policy = SlowPolicy()
    robot = _FakeRobot("F1")
    service = _make_service(tmp_path, policy=policy, robot=robot)
    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._prepare_policy_observation",
        lambda observation, device, preprocessor, **kwargs: observation,
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(
            NRT_PRIMITIVE_EXECUTION="prim",
            NRT_CARTESIAN_MOTION_FORCE="cmf",
        ),
    )

    service.start(_checkpoint_of_type(tmp_path, "bspline_diffusion"))
    assert inference_started.wait(timeout=2.0)
    observations_before = len(policy.observations)
    commands_before = len(robot.commands)
    time.sleep(0.22)

    assert len(policy.observations) >= observations_before + 2
    assert len(robot.commands) > commands_before
    inference_release.set()
    service.stop()


def test_rollout_loop_streams_commands_and_stops(tmp_path, monkeypatch) -> None:
    # The rollout's only send path: a dispatcher thread sends each waypoint once
    # at its target time. Verify the loop runs, enables + switches the robot,
    # sends the raw policy waypoint, and shuts down cleanly (no hang).
    action = [float(i) for i in range(19)]
    policy = _FakePolicy(action)
    robot = _FakeRobot("F1")
    settings = _settings(tmp_path)
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
        "flexivtrainer.rollout.observations._predict_action_chunk",
        lambda obs, pol, dev, pre, post, **kwargs: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
    )
    # Patch the RDK mode lookup so no real flexivrdk import is needed.
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    _run_one_tick(service, robot, _checkpoint(tmp_path))

    assert policy.reset_count == 1
    assert robot.enabled
    assert robot.mode == "cmf"
    assert robot.commands, "expected at least one dispatched Cartesian command"
    pose, wrench, velocity = robot.commands[0]
    # The dispatcher sends the raw waypoint: the action's pose slice with a
    # unit-norm quaternion, its twist slice as velocity, its wrench slice as-is.
    assert pose[0] == pytest.approx(action[0])
    assert pytest.approx(sum(c * c for c in pose[3:7]) ** 0.5) == 1.0
    assert velocity == pytest.approx(action[7:13])
    assert wrench == pytest.approx(action[13:19])
    # The configured hardware speed/accel caps are passed to the robot.
    cfg = service._settings.rollout
    assert robot.motion_limits == (
        cfg.max_linear_vel, cfg.max_angular_vel,
        cfg.max_linear_acc, cfg.max_angular_acc,
    )
    # Clean shutdown: status settled and both rollout threads no longer running.
    assert service.status()["status"] in {"idle", "stopped"}
    assert not any(
        t.name in {"rollout-policy-planner", "rollout-waypoint-executor"}
        and t.is_alive()
        for t in threading.enumerate()
    )


def test_start_threads_task_into_prediction(tmp_path, monkeypatch) -> None:
    action = [float(i) for i in range(19)]
    policy = _FakePolicy(action)
    robot = _FakeRobot("F1")
    service = _make_service(tmp_path, policy=policy, robot=robot)
    tasks_seen: list = []

    def _fake_predict(obs, pol, dev, pre, post, **kwargs):
        tasks_seen.append(kwargs.get("task"))
        return np.tile(pol.select_action(obs), (8, 1)), True

    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._predict_action_chunk", _fake_predict
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    service.start(_checkpoint(tmp_path), task="pick up the cube")
    deadline = time.monotonic() + 2.0
    while not robot.commands and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service.status()["task"] == "pick up the cube"
    service.stop()

    assert tasks_seen and tasks_seen[0] == "pick up the cube"


def test_start_normalizes_blank_task_to_none(tmp_path, monkeypatch) -> None:
    action = [float(i) for i in range(19)]
    policy = _FakePolicy(action)
    robot = _FakeRobot("F1")
    service = _make_service(tmp_path, policy=policy, robot=robot)
    tasks_seen: list = []

    def _fake_predict(obs, pol, dev, pre, post, **kwargs):
        tasks_seen.append(kwargs.get("task"))
        return np.tile(pol.select_action(obs), (8, 1)), True

    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._predict_action_chunk", _fake_predict
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    service.start(_checkpoint(tmp_path), task="   ")
    deadline = time.monotonic() + 2.0
    while not robot.commands and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service.status()["task"] is None
    service.stop()

    assert tasks_seen and tasks_seen[0] is None


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
        "flexivtrainer.rollout.observations._predict_action_chunk",
        lambda obs, pol, dev, pre, post, **kwargs: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    _run_one_tick(service, robot, _checkpoint_with_dataset_fps(tmp_path, fps=12))

    status = service.status()
    logs = status["logs"]
    # An obs row is logged on step 0 (0 % log_every == 0) carrying the checkpoint
    # target frequency and a measured actual frequency, e.g. "freq=9.8/12.0Hz".
    assert any("/12.0Hz" in line for line in logs)
    assert any(
        "cmd_twist=[7.000, 8.000, 9.000, 10.000, 11.000, 12.000]" in line
        for line in logs
    )
    metrics = status["metrics"]
    assert isinstance(metrics, list) and metrics
    for sample in metrics:
        assert set(sample) >= {"t", "step", "hz", "infer_ms", "fresh"}
        assert "sched" not in sample
    assert any(sample["fresh"] is True for sample in metrics)
    assert status["target_hz"] == 12.0


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
        "flexivtrainer.rollout.observations._predict_action_chunk",
        lambda obs, pol, dev, pre, post, **kwargs: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
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


def test_overlapped_replan_forces_and_extends_committed_path(
    tmp_path, monkeypatch
) -> None:
    # The planner must force a fresh inference every replan_steps ticks, splice a
    # new chunk more than once, and always keep a committed path extending at
    # least replan_steps*dt past now so the dispatcher is never left dry.
    action = [float(i) for i in range(19)]
    policy = _FakePolicy(action)
    # Identify as diffusion so the per-family rollout config (replan_steps=4)
    # applies instead of the shared defaults.
    policy.config = SimpleNamespace(type="diffusion")
    robot = _FakeRobot("F1")
    settings = _settings(tmp_path)
    settings.policies.diffusion.rollout.replan_steps = 4
    service = _make_service(tmp_path, policy=policy, robot=robot)
    service._settings = settings

    forces: list[bool] = []
    schedules: list[float] = []
    real_replace = WaypointExecutor.replace_waypoints

    def _recording_replace(self, actions, target_times, now):
        real_replace(self, actions, target_times, now)
        schedules.append(self._waypoints[-1].target_time - now)

    monkeypatch.setattr(WaypointExecutor, "replace_waypoints", _recording_replace)

    def _fake_predict(obs, pol, dev, pre, post, **kwargs):
        force = bool(kwargs.get("force_refresh"))
        forces.append(force)
        return np.tile(pol.select_action(obs), (8, 1)), force

    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._predict_action_chunk", _fake_predict
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    service.start(_checkpoint(tmp_path))
    deadline = time.monotonic() + 3.0
    while len(forces) < 12 and time.monotonic() < deadline:
        time.sleep(0.01)
    service.stop()

    # The first tick forces (replan_steps unresolved), then every 4th tick after.
    assert forces[0] is True
    forced_ticks = [i for i, f in enumerate(forces) if f]
    assert 4 in forced_ticks and 8 in forced_ticks
    # A fresh chunk was spliced on more than one forced tick.
    assert len(schedules) >= 2
    dt = 1.0 / float(settings.rollout.action_dt_hz)
    # Each schedule leaves a committed horizon covering at least the replan gap.
    assert all(extent >= 4 * dt - 1e-6 for extent in schedules)


def test_n_action_steps_override_applies_clamps_and_skips(tmp_path) -> None:
    service = _make_service(tmp_path, policy=_FakePolicy([]), robot=_FakeRobot("F1"))
    rollout_cfg = service._settings.policies.diffusion.rollout

    def _policy():
        return SimpleNamespace(
            config=SimpleNamespace(n_action_steps=8, horizon=16, n_obs_steps=2)
        )

    # In-range value is applied verbatim.
    policy = _policy()
    rollout_cfg.n_action_steps = 12
    service._apply_n_action_steps(policy, rollout_cfg)
    assert policy.config.n_action_steps == 12

    # Above horizon - n_obs_steps + 1 (= 15) is clamped.
    policy = _policy()
    rollout_cfg.n_action_steps = 20
    service._apply_n_action_steps(policy, rollout_cfg)
    assert policy.config.n_action_steps == 15

    # 0 leaves the checkpoint default untouched.
    policy = _policy()
    rollout_cfg.n_action_steps = 0
    service._apply_n_action_steps(policy, rollout_cfg)
    assert policy.config.n_action_steps == 8

    # ACT has no horizon, so chunk_size is the bound.
    policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=1, chunk_size=60))
    rollout_cfg.n_action_steps = 80
    service._apply_n_action_steps(policy, rollout_cfg)
    assert policy.config.n_action_steps == 60


def test_act_rollout_overrides_disable_temporal_ensembling() -> None:
    # Set explicitly: the shipped default is an operator preference that flips.
    rollout_cfg = act_policy.RolloutConfig(disable_temporal_ensemble=True)
    policy = SimpleNamespace(
        config=SimpleNamespace(temporal_ensemble_coeff=0.01, n_action_steps=1),
        temporal_ensembler=object(),
    )

    assert act_policy.apply_rollout_overrides(policy, rollout_cfg) is True
    assert policy.config.temporal_ensemble_coeff is None
    assert policy.temporal_ensembler is None

    # Already-disabled checkpoints report no override, so the log stays honest.
    assert act_policy.apply_rollout_overrides(policy, rollout_cfg) is False


def test_act_rollout_overrides_respect_the_opt_out() -> None:
    rollout_cfg = act_policy.RolloutConfig(disable_temporal_ensemble=False)
    ensembler = object()
    policy = SimpleNamespace(
        config=SimpleNamespace(temporal_ensemble_coeff=0.01, n_action_steps=1),
        temporal_ensembler=ensembler,
    )

    assert act_policy.apply_rollout_overrides(policy, rollout_cfg) is False
    assert policy.config.temporal_ensemble_coeff == 0.01
    assert policy.temporal_ensembler is ensembler


def test_wrench_sample_decimates_to_the_trend_chart_rate() -> None:
    snapshot = {
        "robots": {
            "robot_0": {"states": {"ext_wrench_in_world": [1, 2, 3, 0.1, 0.2, 0.3]}},
            "robot_1": {"states": {"ext_wrench_in_world": [4, 5, 6, 0.4, 0.5, 0.6]}},
        }
    }
    sides = ["left_arm", "right_arm"]
    out: list[dict] = []

    last = observations.sample_wrench(out.append, snapshot, sides, 100.0, 0.0)
    assert out[-1] == {
        "t": 100.0,
        "left_arm": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        "right_arm": [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
    }

    # The boundary must survive float error (0.1 - 0.0 == 0.09999... in practice).
    last = observations.sample_wrench(out.append, snapshot, sides, 100.05, last)
    assert len(out) == 1
    observations.sample_wrench(out.append, snapshot, sides, 100.10, last)
    assert len(out) == 2

    out.clear()
    stamp = 0.0
    for index in range(90):
        stamp = observations.sample_wrench(
            out.append, snapshot, sides, 1000.0 + index / 30.0, stamp
        )
    deltas = {
        round(out[i + 1]["t"] - out[i]["t"], 3) for i in range(len(out) - 1)
    }
    assert deltas == {0.1}


def test_wrench_sample_tolerates_absent_and_malformed_input() -> None:
    snapshot = {"robots": {"robot_0": {"states": {"ext_wrench_in_world": [1] * 6}}}}
    out: list[dict] = []

    # snapshot=None is a b-spline tick that skipped observation.
    assert observations.sample_wrench(None, snapshot, ["single_arm"], 1.0, 0.0) == 0.0
    assert observations.sample_wrench(out.append, None, ["single_arm"], 1.0, 0.0) == 0.0
    assert not out

    short = {"robots": {"robot_0": {"states": {"ext_wrench_in_world": [1, 2]}}}}
    observations.sample_wrench(out.append, short, ["single_arm"], 1.0, 0.0)
    assert out[-1] == {"t": 1.0}

    out.clear()
    observations.sample_wrench(out.append, snapshot, ["single_arm"], 2.0, 0.0)
    assert out[-1]["single_arm"] == [1.0] * 6


def test_rollout_status_exposes_wrench_and_sides(tmp_path) -> None:
    service = _make_service(
        tmp_path, policy=_FakePolicy([0.0]), robot=_FakeRobot("F1")
    )
    status = service.status()
    assert status["wrench"] == []
    assert status["sides"] == []

    service._sides = ["left_arm", "right_arm"]
    service._wrench.append({"t": 1.0, "left_arm": [1.0] * 6})
    status = service.status()
    assert status["sides"] == ["left_arm", "right_arm"]
    assert status["wrench"][-1]["left_arm"] == [1.0] * 6
    # The gauges read the newest sample; the rest only steady the auto-range.
    assert service._wrench.maxlen == 30


def test_policy_action_queue_handles_both_cache_layouts() -> None:
    from lerobot.utils.constants import ACTION

    queues = SimpleNamespace(_queues={ACTION: deque([1, 2])})
    assert list(observations._policy_action_queue(queues, ACTION)) == [1, 2]

    # ACT keeps a single _action_queue instead of a per-feature dict.
    act_like = SimpleNamespace(_action_queue=deque([3]))
    assert list(observations._policy_action_queue(act_like, ACTION)) == [3]

    assert observations._policy_action_queue(SimpleNamespace(), ACTION) is None


def test_rollout_for_selects_per_policy_config_and_loop_runs_for_act(
    tmp_path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    diffusion_rollout = settings.policies.rollout_for("diffusion")
    # A diffusion family exposes its own sampler knob; an unknown family falls
    # back to the shared config, which has none.
    assert hasattr(diffusion_rollout, "noise_scheduler_type")
    # ACT exposes its own ensembling knob but no sampler knob.
    act_rollout = settings.policies.rollout_for("act")
    assert hasattr(act_rollout, "disable_temporal_ensemble")
    assert not hasattr(act_rollout, "noise_scheduler_type")
    # An unknown family still falls back to the shared config.
    assert settings.policies.rollout_for("pi0").__class__.__name__ == (
        "SharedRolloutConfig"
    )

    # temporal_ensemble_coeff makes the override fire, exercising the
    # "overrides applied" log path -- it once assumed diffusion's sampler fields.
    action = [float(i) for i in range(19)]
    policy = _FakePolicy(action)
    policy.config = SimpleNamespace(
        type="act", temporal_ensemble_coeff=0.01, n_action_steps=1, chunk_size=60
    )
    policy.temporal_ensembler = object()
    robot = _FakeRobot("F1")
    service = _make_service(tmp_path, policy=policy, robot=robot)
    # This test guards the log path, not the shipped default.
    service._settings.policies.act.rollout.disable_temporal_ensemble = True
    act_rollout = service._settings.policies.rollout_for("act")
    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._predict_action_chunk",
        lambda obs, pol, dev, pre, post, **kwargs: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    _run_one_tick(service, robot, _checkpoint(tmp_path))
    assert robot.commands
    assert service.status()["status"] in {"idle", "stopped"}
    # The override must have been applied and reported, not crashed past.
    assert policy.config.temporal_ensemble_coeff is None
    assert policy.config.n_action_steps == act_rollout.n_action_steps
    assert any(
        "Rollout overrides applied" in entry for entry in service.status()["logs"]
    )


def test_describe_rollout_overrides_reports_each_family_shape() -> None:
    scheduler_cfg = SimpleNamespace(
        noise_scheduler_type="DDIM", num_denoise_steps=16
    )
    assert "scheduler=DDIM" in _describe_rollout_overrides(scheduler_cfg)

    act_cfg = act_policy.RolloutConfig(disable_temporal_ensemble=True)
    detail = _describe_rollout_overrides(act_cfg)
    assert "temporal ensembling disabled" in detail
    assert str(act_cfg.n_action_steps) in detail

    # With every ACT knob off, the generic detail is used.
    opted_out = act_policy.RolloutConfig(
        disable_temporal_ensemble=False, compile_model=False
    )
    assert _describe_rollout_overrides(opted_out) == "applied"

    # A family with neither knob must still produce a string, not raise.
    assert _describe_rollout_overrides(SimpleNamespace()) == "applied"


def test_env_var_plumbs_into_rollout_config(monkeypatch) -> None:
    monkeypatch.setenv(
        "FLEXIV_TRAINER_POLICIES__DIFFUSION__ROLLOUT__REPLAN_STEPS", "4"
    )
    settings = AppSettings()
    assert settings.policies.diffusion.rollout.replan_steps == 4


def test_max_steps_reports_timeout_stop_reason(tmp_path, monkeypatch) -> None:
    # The runner ends the run itself at max_steps; the service must publish that
    # reason instead of attributing the stop to the operator.
    policy = _FakePolicy([float(i) for i in range(19)])
    robot = _FakeRobot("F1")
    settings = _settings(tmp_path)
    settings.rollout.max_steps = 2
    service = RolloutService(
        settings,
        _cameras(),
        _teleop(initialized=False),
        lambda: [TeleopRobotPair(leader_serial="L1", follower_serial="F1")],
        lambda: ["single_arm"],
        policy_loader=_fake_loader(policy),
        robot_factory=lambda serial: robot,
        resolve_device=lambda configured: "cpu",
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._predict_action_chunk",
        lambda obs, pol, dev, pre, post, **kwargs: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    service.start(_checkpoint(tmp_path))
    deadline = time.monotonic() + 3.0
    while service.status()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    status = service.stop()

    assert status["stop_reason"] == "timeout"
    assert status["error"] is None
    assert any("reason=timeout steps=2" in line for line in status["logs"])


def test_rollout_start_clears_depth_alignment_leases(tmp_path, monkeypatch) -> None:
    # A depth preview already running would keep stealing the GIL from the
    # policy loop until its lease lapsed, so starting a rollout drops it.
    cleared: list[bool] = []
    policy = _FakePolicy([float(i) for i in range(19)])
    policy.config = SimpleNamespace(type="act")
    robot = _FakeRobot("F1")
    service = _make_service(tmp_path, policy=policy, robot=robot)
    service._cameras = SimpleNamespace(
        capture_frame=lambda name, **kwargs: {},
        clear_depth_alignment_leases=lambda: cleared.append(True),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.observations._predict_action_chunk",
        lambda obs, pol, dev, pre, post, **kwargs: (
            np.tile(pol.select_action(obs), (8, 1)),
            True,
        ),
    )
    monkeypatch.setattr(
        "flexivtrainer.rollout.hardware._rdk_mode",
        lambda: SimpleNamespace(NRT_CARTESIAN_MOTION_FORCE="cmf"),
    )

    _run_one_tick(service, robot, _checkpoint(tmp_path))

    assert cleared == [True]


def test_act_compile_override_replaces_only_the_model(monkeypatch) -> None:
    # Compile the tensor core, not ACTPolicy: select_action mutates a Python
    # deque and is not a pure graph.
    calls: list[tuple] = []
    monkeypatch.setattr(
        "torch.compile",
        lambda model, **kwargs: calls.append((model, kwargs)) or "COMPILED",
    )
    sentinel = object()
    policy = SimpleNamespace(config=SimpleNamespace(type="act"), model=sentinel)
    rollout_cfg = act_policy.RolloutConfig(compile_model=True)

    assert act_policy.apply_rollout_overrides(policy, rollout_cfg) is True
    assert policy.model == "COMPILED"
    assert calls[0][0] is sentinel
    assert calls[0][1] == {"mode": "reduce-overhead"}


def test_act_compile_failure_falls_back_to_eager(monkeypatch) -> None:
    def boom(model, **kwargs):
        raise RuntimeError("inductor unavailable")

    monkeypatch.setattr("torch.compile", boom)
    sentinel = object()
    policy = SimpleNamespace(config=SimpleNamespace(type="act"), model=sentinel)

    applied = act_policy.apply_rollout_overrides(
        policy, act_policy.RolloutConfig(compile_model=True)
    )

    # A missing accelerator must not abort the rollout.
    assert applied is False
    assert policy.model is sentinel


def test_describe_rollout_overrides_mentions_compilation() -> None:
    detail = _describe_rollout_overrides(act_policy.RolloutConfig(compile_model=True))
    assert "compiled" in detail

    both = _describe_rollout_overrides(
        act_policy.RolloutConfig(compile_model=True, disable_temporal_ensemble=True)
    )
    assert "compiled" in both and "ensembling disabled" in both


def test_playback_speed_scales_the_action_rate(tmp_path) -> None:
    service = RolloutService.__new__(RolloutService)
    service._lock = threading.Lock()
    service._logs = deque(maxlen=50)

    assert service._apply_playback_speed(
        30.0, SimpleNamespace(playback_speed=1.0)
    ) == 30.0
    assert not service._logs  # unscaled replay is not worth a warning

    assert service._apply_playback_speed(
        30.0, SimpleNamespace(playback_speed=2.0)
    ) == 60.0
    assert any("playback_speed=2" in str(entry) for entry in service._logs)

    # A family without the knob must fall through unscaled.
    assert service._apply_playback_speed(30.0, SimpleNamespace()) == 30.0
