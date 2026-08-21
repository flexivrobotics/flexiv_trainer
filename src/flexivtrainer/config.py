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

from flexivtrainer.policies import PolicyConfig

# Flexiv "Home" primitive default for the Rizon 4 (degrees, joints A1..A7).
DEFAULT_HOME_POSTURE_DEG: list[float] = [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0]


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
    # Streaming depth is cheap; the costly depth->color alignment only runs
    # while a consumer holds a reference (see RealSenseService).
    use_depth: bool = True


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
    def hub_cache_root(self) -> Path:
        # Inside the storage root on purpose: a downloaded checkpoint then passes
        # the same resolve_checkpoint_path validation as a local one.
        return self.cache_root / "hub"

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
        (self.hub_cache_root / "datasets").mkdir(parents=True, exist_ok=True)
        (self.hub_cache_root / "checkpoints").mkdir(parents=True, exist_ok=True)


# Read by the dataset patches in flexivtrainer.policies.lerobot_plugins. Only
# TrainingService sets it; unset means "leave the dataset alone", since the app
# imports those patches too and still needs depth for previews.
TRAIN_LOAD_DEPTH_ENV = "FLEXIVTRAINER_TRAIN_LOAD_DEPTH"


class HubConfig(BaseModel):
    """HuggingFace Hub access for training datasets and policy checkpoints."""

    enabled: bool = True
    # Only needed for private or gated repos. When unset, the HF_TOKEN /
    # HUGGING_FACE_HUB_TOKEN environment variables and the huggingface-cli login
    # cache are used instead.
    token: str | None = None
    # Pin every fetch to a branch, tag, or commit; None tracks the repo default.
    default_revision: str | None = None


class TrainingDefaultsConfig(BaseModel):
    default_policy: str = "diffusion"
    # Device passed to lerobot via --policy.device. "auto" (default) resolves to
    # the best available device on this machine (cuda > mps > cpu) at train time,
    # so the trainer stays portable across platforms; set an explicit "cuda" /
    # "mps" / "cpu" to force one.
    default_device: str = "auto"
    # LeRobot's reader decodes every video feature a dataset declares, not just
    # the ones the policy asked for, and lossless 12-bit HEVC depth dominates that
    # cost. Turn on once a policy consumes depth.
    load_depth: bool = False


class RolloutLoopConfig(BaseModel):
    # Fallback planner-loop tick rate used only when the checkpoint has no FPS
    # metadata; otherwise the loop ticks at the checkpoint's data rate.
    planner_hz: int = Field(default=10, ge=1, le=120)
    max_steps: int = Field(default=0, ge=0)
    # Time-spacing of poses within one predicted chunk = the training data rate.
    # A checkpoint property, not a loop rate; set to match the checkpoint.
    action_dt_hz: int = Field(default=10, ge=1, le=120)
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
    # Shared 7-DOF home posture (degrees) for the "Home All Robots" action.
    home_posture_deg: list[float] = Field(
        default_factory=lambda: list(DEFAULT_HOME_POSTURE_DEG)
    )
    # Shared gripper startup/open width in metres. None preserves each gripper's
    # hardware-defined maximum width and the legacy initialization behaviour.
    gripper_default_width_m: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    gripper_velocity_m_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    gripper_force_limit_n: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    # Cached recording checklist; empty means never saved, so defaults apply.
    recording_entries: list[str] = Field(default_factory=list)
    # Cached capture-resolution preset id; empty means never saved.
    record_resolution: str = ""

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

    def _normalize_home_posture(self) -> list[float]:
        # Coerce to exactly 7 floats; missing/bad joints fall back to defaults.
        posture: list[float] = []
        for index in range(7):
            try:
                posture.append(float(self.home_posture_deg[index]))
            except (IndexError, TypeError, ValueError):
                posture.append(DEFAULT_HOME_POSTURE_DEG[index])
        return posture

    def normalized(self) -> RobotSerialConfig:
        return RobotSerialConfig(
            arm_mode=self.arm_mode,
            leader_robot_serials=self._normalize_serials(self.leader_robot_serials),
            follower_robot_serials=self._normalize_serials(self.follower_robot_serials),
            # Preserve selections for every side (even ones not currently active),
            # so toggling between single/dual keeps cached choices.
            end_effector_config=dict(self.end_effector_config),
            home_posture_deg=self._normalize_home_posture(),
            gripper_default_width_m=self.gripper_default_width_m,
            gripper_velocity_m_s=self.gripper_velocity_m_s,
            gripper_force_limit_n=self.gripper_force_limit_n,
            # Entries for inactive sides are kept, as with the serials above.
            recording_entries=[
                str(entry) for entry in self.recording_entries if str(entry).strip()
            ],
            record_resolution=str(self.record_resolution).strip(),
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
    # Depth ceiling in meters for LeRobot's 12-bit depth quantization and for
    # colorized previews. The 2 m default focuses the available range on a
    # tabletop workspace; farther pixels are clamped.
    depth_max_m: float = Field(default=2.0, gt=0)
    # Per-side [q_w, q_x, q_y, q_z] seeding the sign of each episode's first
    # recorded TCP quaternion; see data/quaternion.py.
    recording_quaternion_reference: dict[str, list[float]] = Field(
        default_factory=dict
    )
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
    hub: HubConfig = Field(default_factory=HubConfig)
    training: TrainingDefaultsConfig = Field(default_factory=TrainingDefaultsConfig)
    policies: PolicyConfig = Field(default_factory=PolicyConfig)
    rollout: RolloutLoopConfig = Field(default_factory=RolloutLoopConfig)

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
