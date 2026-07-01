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

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TeleopRobotPair(BaseModel):
    leader_serial: str = ""
    follower_serial: str = ""
    leader_home_posture: list[float] = Field(default_factory=list)
    follower_home_posture: list[float] = Field(default_factory=list)


class CameraConfig(BaseModel):
    name: str
    device_serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30


class CameraSerialConfig(BaseModel):
    """Persisted mapping of camera location name -> assigned device serial."""

    serials: dict[str, str] = Field(default_factory=dict)

    def normalized(self) -> CameraSerialConfig:
        return CameraSerialConfig(
            serials={
                str(name): str(serial).strip() for name, serial in self.serials.items()
            }
        )


class StorageConfig(BaseModel):
    root: Path = Path(".local")
    episodes_dirname: str = "episodes"
    staging_dirname: str = "staging"
    merged_dirname: str = "datasets"
    training_dirname: str = "training"
    cache_dirname: str = "cache"

    @property
    def episodes_root(self) -> Path:
        return self.root / self.episodes_dirname

    @property
    def staging_root(self) -> Path:
        return self.root / self.staging_dirname

    @property
    def merged_root(self) -> Path:
        return self.root / self.merged_dirname

    @property
    def training_root(self) -> Path:
        return self.root / self.training_dirname

    @property
    def cache_root(self) -> Path:
        return self.root / self.cache_dirname

    @property
    def runtime_config_path(self) -> Path:
        return self.root / "robot_serials.json"

    @property
    def camera_config_path(self) -> Path:
        return self.root / "camera_serials.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.merged_root.mkdir(parents=True, exist_ok=True)
        self.training_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)


class TrainingConfig(BaseModel):
    default_policy: str = "diffusion"
    # Device passed to lerobot via --policy.device. "auto" (default) resolves to
    # the best available device on this machine (cuda > mps > cpu) at train time,
    # so the trainer stays portable across platforms; set an explicit "cuda" /
    # "mps" / "cpu" to force one.
    default_device: str = "auto"
    save_frequency: int = 5_000


class RolloutConfig(BaseModel):
    loop_hz: int = Field(default=10, ge=1, le=120)
    max_steps: int = Field(default=0, ge=0)
    # Override a diffusion policy's denoising sampler at rollout load time. The
    # checkpoints train with DDPM/100 steps, which costs ~100 U-Net forwards per
    # action-chunk refill and stalls the control loop; DDIM reuses the same
    # weights but reaches the target in far fewer steps for much faster
    # inference. "" leaves the checkpoint's own scheduler/steps untouched.
    diffusion_scheduler: Literal["", "DDPM", "DDIM"] = "DDPM"
    diffusion_inference_steps: int = Field(default=100, ge=1, le=1000)
    # The policy's sparse action waypoints are interpolated into a continuous pose
    # spline streamed to the robot by a separate high-rate sender. interp_hz is
    # that sender's rate; SendCartesianMotionForce handles up to 1000 Hz, 100-200
    # is ideal.
    interp_hz: int = Field(default=200, ge=1, le=1000)
    max_linear_vel: float = Field(default=0.25, gt=0)  # m/s
    max_angular_vel: float = Field(default=0.6, gt=0)  # rad/s
    max_linear_acc: float = Field(default=1.0, gt=0)  # m/s^2
    max_angular_acc: float = Field(default=2.5, gt=0)  # rad/s^2


class EndEffectorSideConfig(BaseModel):
    """End effector selections for one arm side (leader + follower devices)."""

    leader: Literal["none", "digital_input"] = "none"
    leader_channel: int = Field(default=0, ge=0, le=15)
    leader_activating_state: Literal["high", "low"] = "high"
    follower: Literal["none", "digital_output", "gripper"] = "none"
    follower_channel: int = Field(default=0, ge=0, le=15)
    follower_activated_state: Literal["high", "low"] = "high"
    gripper_model: str = "Flexiv-GN01"
    gripper_activated_state: Literal["close", "open"] = "close"


class RobotSerialConfig(BaseModel):
    arm_mode: Literal["single", "dual"] = "dual"
    leader_robot_serials: list[str] = Field(default_factory=lambda: ["", ""])
    follower_robot_serials: list[str] = Field(default_factory=lambda: ["", ""])
    # Per-side end effector selections, keyed by arm side ("left_arm",
    # "right_arm", "single_arm"). Cached alongside the serials so selections
    # survive reloads.
    end_effector_config: dict[str, EndEffectorSideConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_prefixes(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "leader_robot_serials" not in payload and "local_robot_serials" in payload:
            payload["leader_robot_serials"] = payload.get("local_robot_serials")
        if (
            "follower_robot_serials" not in payload
            and "remote_robot_serials" in payload
        ):
            payload["follower_robot_serials"] = payload.get("remote_robot_serials")
        return payload

    def active_arm_count(self) -> int:
        return 1 if self.arm_mode == "single" else 2

    def active_sides(self) -> list[str]:
        if self.arm_mode == "single":
            return ["single_arm"]
        return ["left_arm", "right_arm"]

    def _normalize_serials(self, values: list[str]) -> list[str]:
        # Cache every provided serial (trimmed) rather than truncating to the
        # active arm count, so a serial entered for an arm that is inactive in
        # the current mode survives a single -> dual -> single round trip.
        # Always keep at least `count` slots so active sides have a slot to fill.
        count = self.active_arm_count()
        serials = [str(value).strip() for value in values]
        serials.extend([""] * (count - len(serials)))
        return serials

    def normalized(self) -> RobotSerialConfig:
        return RobotSerialConfig(
            arm_mode=self.arm_mode,
            leader_robot_serials=self._normalize_serials(self.leader_robot_serials),
            follower_robot_serials=self._normalize_serials(self.follower_robot_serials),
            # Preserve selections for every side (even ones not currently active),
            # so toggling between single/dual keeps cached choices.
            end_effector_config=dict(self.end_effector_config),
        )

    @classmethod
    def from_settings(cls, settings: AppSettings) -> RobotSerialConfig:
        return cls(
            leader_robot_serials=[
                pair.leader_serial for pair in settings.teleop_robot_pairs
            ],
            follower_robot_serials=[
                pair.follower_serial for pair in settings.teleop_robot_pairs
            ],
        ).normalized()


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FLEXIV_TRAINER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    public_base_url: str | None = None
    robot_type: str = "flexiv_rizon_dual"
    default_task: str = "Dual-arm Flexiv teleoperation demonstration"
    # Codec for the recorded camera MP4s. Default is software H.264 (libx264):
    # it is browser-decodable everywhere and encodes identically on Ubuntu/macOS/
    # Windows across x64/arm64/aarch64, so dataset previews play on every platform
    # (LeRobot's default 'libsvtav1'/AV1 has no hardware decode on many ARM boards
    # and won't play in the embedded webview). Set 'auto' to prefer a platform
    # hardware H.264 encoder (videotoolbox/nvenc/vaapi/qsv) with software fallback,
    # or name an explicit encoder. Resolved by resolve_recording_vcodec(); an
    # unavailable codec falls back to software 'h264' rather than failing.
    video_codec: str = "h264"
    network_interface_whitelist: list[str] = Field(default_factory=list)
    teleop_robot_pairs: list[TeleopRobotPair] = Field(default_factory=list)
    cameras: list[CameraConfig] = Field(
        default_factory=lambda: [
            CameraConfig(name="ego", fps=30, width=640, height=480),
            CameraConfig(name="left_wrist", fps=30, width=640, height=480),
            CameraConfig(name="right_wrist", fps=30, width=640, height=480),
            CameraConfig(name="wrist", fps=30, width=640, height=480),
        ]
    )
    storage: StorageConfig = Field(default_factory=StorageConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)

    @property
    def follower_robot_serials(self) -> list[str]:
        return [
            pair.follower_serial
            for pair in self.teleop_robot_pairs
            if pair.follower_serial
        ]

    def ensure_storage(self) -> None:
        self.storage.ensure()

    @property
    def ui_url(self) -> str:
        if self.public_base_url:
            return self.public_base_url.rstrip("/") + "/"

        host = self.host
        if host in {"0.0.0.0", "::", ""}:
            host = "127.0.0.1"
        return f"http://{host}:{self.port}/"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.ensure_storage()
    return settings
