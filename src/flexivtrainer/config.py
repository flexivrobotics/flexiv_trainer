from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
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


class StorageConfig(BaseModel):
    root: Path = Path(".local")
    episodes_dirname: str = "episodes"
    staging_dirname: str = "staging"
    combined_dirname: str = "combined"
    training_dirname: str = "training"
    calibration_dirname: str = "calibration"
    cache_dirname: str = "cache"

    @property
    def episodes_root(self) -> Path:
        return self.root / self.episodes_dirname

    @property
    def staging_root(self) -> Path:
        return self.root / self.staging_dirname

    @property
    def combined_root(self) -> Path:
        return self.root / self.combined_dirname

    @property
    def training_root(self) -> Path:
        return self.root / self.training_dirname

    @property
    def calibration_root(self) -> Path:
        return self.root / self.calibration_dirname

    @property
    def cache_root(self) -> Path:
        return self.root / self.cache_dirname

    @property
    def runtime_config_path(self) -> Path:
        return self.root / "robot_serials.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.combined_root.mkdir(parents=True, exist_ok=True)
        self.training_root.mkdir(parents=True, exist_ok=True)
        self.calibration_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)


class TrainingConfig(BaseModel):
    default_policy: str = "diffusion"
    default_device: str = "cuda"
    save_frequency: int = 5_000


class RobotSerialConfig(BaseModel):
    local_robot_serials: list[str] = Field(default_factory=lambda: ["", ""])
    remote_robot_serials: list[str] = Field(default_factory=lambda: ["", ""])

    @staticmethod
    def _normalize_serials(values: list[str]) -> list[str]:
        serials = [str(value).strip() for value in values[:2]]
        serials.extend([""] * (2 - len(serials)))
        return serials

    def normalized(self) -> RobotSerialConfig:
        return RobotSerialConfig(
            local_robot_serials=self._normalize_serials(self.local_robot_serials),
            remote_robot_serials=self._normalize_serials(self.remote_robot_serials),
        )

    @classmethod
    def from_settings(cls, settings: AppSettings) -> RobotSerialConfig:
        return cls(
            local_robot_serials=[
                pair.leader_serial for pair in settings.teleop_robot_pairs[:2]
            ],
            remote_robot_serials=[
                pair.follower_serial for pair in settings.teleop_robot_pairs[:2]
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
    network_interface_whitelist: list[str] = Field(default_factory=list)
    teleop_robot_pairs: list[TeleopRobotPair] = Field(default_factory=list)
    cameras: list[CameraConfig] = Field(
        default_factory=lambda: [
            CameraConfig(name="ego", fps=30, width=848, height=480),
            CameraConfig(name="left_wrist", fps=30, width=640, height=480),
            CameraConfig(name="right_wrist", fps=30, width=640, height=480),
        ]
    )
    storage: StorageConfig = Field(default_factory=StorageConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @property
    def remote_robot_serials(self) -> list[str]:
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
