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
    default_device: str = "cuda"
    save_frequency: int = 5_000


class RobotSerialConfig(BaseModel):
    leader_robot_serials: list[str] = Field(default_factory=lambda: ["", ""])
    follower_robot_serials: list[str] = Field(default_factory=lambda: ["", ""])

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

    @staticmethod
    def _normalize_serials(values: list[str]) -> list[str]:
        serials = [str(value).strip() for value in values[:2]]
        serials.extend([""] * (2 - len(serials)))
        return serials

    def normalized(self) -> RobotSerialConfig:
        return RobotSerialConfig(
            leader_robot_serials=self._normalize_serials(self.leader_robot_serials),
            follower_robot_serials=self._normalize_serials(self.follower_robot_serials),
        )

    @classmethod
    def from_settings(cls, settings: AppSettings) -> RobotSerialConfig:
        return cls(
            leader_robot_serials=[
                pair.leader_serial for pair in settings.teleop_robot_pairs[:2]
            ],
            follower_robot_serials=[
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
            CameraConfig(name="ego", fps=30, width=640, height=480),
            CameraConfig(name="left_wrist", fps=30, width=640, height=480),
            CameraConfig(name="right_wrist", fps=30, width=640, height=480),
        ]
    )
    storage: StorageConfig = Field(default_factory=StorageConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

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
