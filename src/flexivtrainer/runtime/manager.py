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

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from flexivtrainer.cameras.realsense import RealSenseService
from flexivtrainer.config import (
    AppSettings,
    CameraSerialConfig,
    RobotSerialConfig,
    TeleopRobotPair,
    get_settings,
)
from flexivtrainer.ddk.service import DDKService
from flexivtrainer.jobs.train import TrainingService
from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.teleop.service import TeleopService


def _optional_dependency_error(feature: str, exc: ImportError) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{feature} is unavailable: {detail}"
    return f"{feature} is unavailable in the selected environment"


class _UnavailableRecordingService:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    def status(self) -> dict[str, Any]:
        return {
            "active": False,
            "awaiting_save": False,
            "frames_captured": 0,
            "episode_name": None,
            "fps": None,
            "error": self._reason,
            "save_in_progress": False,
            "save_progress": 0,
            "elapsed_s": 0.0,
        }

    def start(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(self._reason)

    def stop(self) -> dict[str, Any]:
        raise RuntimeError(self._reason)

    def save(self) -> dict[str, Any]:
        raise RuntimeError(self._reason)

    def discard(self) -> dict[str, Any]:
        raise RuntimeError(self._reason)

    def shutdown(self) -> None:
        return None


class RuntimeManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._robot_config = self._load_robot_config()
        self.teleop = TeleopService(settings, self.get_teleop_robot_pairs)
        self.ddk = DDKService(settings, self.get_remote_robot_serials)
        self.cameras = RealSenseService(settings)
        self._load_camera_config()
        try:
            from flexivtrainer.data.recording_service import RecordingService
        except ImportError as exc:
            self.recording = _UnavailableRecordingService(
                _optional_dependency_error("Episode recording", exc)
            )
        else:
            self.recording = RecordingService(settings, self.teleop, self.cameras)
        self.training = TrainingService(settings)

    def _load_robot_config(self) -> RobotSerialConfig:
        path = self.settings.storage.runtime_config_path
        if path.exists():
            loaded = RobotSerialConfig.model_validate_json(
                path.read_text(encoding="utf-8")
            ).normalized()
        else:
            loaded = RobotSerialConfig.from_settings(self.settings)
            path.write_text(loaded.model_dump_json(indent=2), encoding="utf-8")
        return loaded

    def _save_robot_config(self) -> None:
        self.settings.storage.runtime_config_path.write_text(
            self._robot_config.model_dump_json(indent=2), encoding="utf-8"
        )

    def robot_config_snapshot(self) -> dict[str, Any]:
        return self._robot_config.model_dump()

    def _load_camera_config(self) -> None:
        path = self.settings.storage.camera_config_path
        if path.exists():
            try:
                config = CameraSerialConfig.model_validate_json(
                    path.read_text(encoding="utf-8")
                ).normalized()
            except (ValueError, OSError):
                config = CameraSerialConfig()
            if config.serials:
                self.cameras.set_device_serials(config.serials, manual=False)
        else:
            self._save_camera_config()

    def _save_camera_config(self) -> None:
        serials = {
            name: (serial or "")
            for name, serial in self.cameras.configured_serials().items()
        }
        self.settings.storage.camera_config_path.write_text(
            CameraSerialConfig(serials=serials).model_dump_json(indent=2),
            encoding="utf-8",
        )

    def camera_config_snapshot(self) -> dict[str, Any]:
        return {
            "cameras": [
                {"name": name, "device_serial": serial or ""}
                for name, serial in self.cameras.configured_serials().items()
            ]
        }

    def update_camera_config(self, serials: dict[str, str]) -> dict[str, Any]:
        self.cameras.set_device_serials(serials, manual=True)
        self._save_camera_config()
        return {
            "camera_config": self.camera_config_snapshot(),
            "cameras": self.cameras.status(),
            "services": self.service_summary(),
        }

    def get_remote_robot_serials(self) -> list[str]:
        return [serial for serial in self._robot_config.remote_robot_serials if serial]

    def get_teleop_robot_pairs(self) -> list[TeleopRobotPair]:
        defaults = self.settings.teleop_robot_pairs
        pairs: list[TeleopRobotPair] = []
        for index in range(2):
            template = defaults[index] if index < len(defaults) else TeleopRobotPair()
            leader_serial = self._robot_config.local_robot_serials[index]
            follower_serial = self._robot_config.remote_robot_serials[index]
            pairs.append(
                TeleopRobotPair(
                    leader_serial=leader_serial,
                    follower_serial=follower_serial,
                    leader_home_posture=list(template.leader_home_posture),
                    follower_home_posture=list(template.follower_home_posture),
                )
            )
        return pairs

    def update_robot_config(self, payload: RobotSerialConfig) -> dict[str, Any]:
        normalized = payload.normalized()
        changed = normalized != self._robot_config
        self._robot_config = normalized
        self._save_robot_config()
        if changed:
            self.recording.shutdown()
            self.teleop.shutdown()
            self.ddk.shutdown()
        return {
            "robot_config": self.robot_config_snapshot(),
            "services": self.service_summary(),
        }

    def _service_message(
        self, errors: list[str], fallback: str | None = None
    ) -> str | None:
        if errors:
            return errors[0]
        return fallback

    def service_summary(self) -> dict[str, Any]:
        teleop_snapshot = self.teleop.snapshot()
        teleop_pair_count = sum(
            1
            for pair in self.get_teleop_robot_pairs()
            if pair.leader_serial and pair.follower_serial
        )
        teleop_errors = [
            item for item in [teleop_snapshot.error, teleop_snapshot.fault] if item
        ]
        if not teleop_snapshot.available:
            teleop_state = "Unavailable"
            teleop_tone = "error"
            teleop_detail = self._service_message(
                teleop_errors, "TDK is not available."
            )
        elif teleop_pair_count == 0:
            teleop_state = "Not configured"
            teleop_tone = "error"
            teleop_detail = "Enter two local and two remote robot serial numbers."
        elif teleop_snapshot.initialized:
            teleop_state = "Connected"
            teleop_tone = "ok"
            teleop_detail = (
                "Teleoperation running."
                if teleop_snapshot.started
                else "Service ready."
            )
        else:
            teleop_state = "Not connected"
            teleop_tone = "error" if teleop_errors else "neutral"
            teleop_detail = self._service_message(
                teleop_errors, "Press Connect to initialize TDK."
            )

        if not teleop_snapshot.available:
            robot_data_state = "Unavailable"
            robot_data_tone = "error"
            robot_data_detail = "TDK is not available."
        elif teleop_pair_count == 0:
            robot_data_state = "Not configured"
            robot_data_tone = "error"
            robot_data_detail = "Enter two local and two remote robot serial numbers."
        elif teleop_snapshot.initialized:
            robot_data_state = "Connected"
            robot_data_tone = "ok"
            robot_data_detail = "Robot states/actions are available through TDK."
        else:
            robot_data_state = "Not connected"
            robot_data_tone = "neutral"
            robot_data_detail = "Connect the teleoperation service first."

        camera_status = self.cameras.status()
        configured_camera_count = len(camera_status["cameras"])
        started_camera_count = sum(
            1 for camera in camera_status["cameras"].values() if camera.get("started")
        )
        camera_errors = [
            str(value) for value in camera_status["errors"].values() if value
        ]
        if not camera_status["available"]:
            camera_state = "Unavailable"
            camera_tone = "error"
            camera_detail = self._service_message(
                camera_errors, "RealSense is not available."
            )
        elif (
            configured_camera_count and started_camera_count == configured_camera_count
        ):
            camera_state = f"{started_camera_count}/{configured_camera_count} connected"
            camera_tone = "ok"
            camera_detail = "All camera feeds are active."
        elif started_camera_count > 0:
            camera_state = f"{started_camera_count}/{configured_camera_count} connected"
            camera_tone = "working"
            camera_detail = self._service_message(
                camera_errors, "Some camera feeds are active."
            )
        else:
            camera_state = f"0/{configured_camera_count} connected"
            camera_tone = "error"
            camera_detail = self._service_message(
                camera_errors, "Press Connect to start the camera feeds."
            )

        return {
            "teleop_service": {
                "label": "TELEOP SERVICE",
                "state": teleop_state,
                "detail": teleop_detail,
                "tone": teleop_tone,
            },
            "robot_data_service": {
                "label": "ROBOT DATA SERVICE",
                "state": robot_data_state,
                "detail": robot_data_detail,
                "tone": robot_data_tone,
            },
            "cameras": {
                "label": "CAMERAS",
                "state": camera_state,
                "detail": camera_detail,
                "tone": camera_tone,
            },
        }

    def control_service(self, service_name: str, action: str) -> dict[str, Any]:
        if action not in {"connect", "disconnect"}:
            raise ValueError(f"Unsupported action: {action}")

        service_name = service_name.lower()
        if service_name == "teleop":
            if action == "connect":
                result = self.teleop.initialize().__dict__
            else:
                result = self.teleop.shutdown() or {"disconnected": True}
        elif service_name == "ddk":
            result = (
                self.ddk.initialize()
                if action == "connect"
                else (self.ddk.shutdown() or {"disconnected": True})
            )
        elif service_name == "cameras":
            if action == "connect":
                result = self.cameras.start_streams()
                self._save_camera_config()
            else:
                result = self.cameras.stop_streams()
        else:
            raise ValueError(f"Unsupported service: {service_name}")

        return {"result": result, "services": self.service_summary()}

    def system_summary(self) -> dict[str, Any]:
        return {
            "backend": {
                "reachable": True,
                "host": self.settings.host,
                "port": self.settings.port,
                "ui_url": self.settings.ui_url,
            },
            "teleop": self.teleop.snapshot().__dict__,
            "robot_data": self.teleop.robot_data_snapshot(),
            "cameras": self.cameras.discover(),
            "storage": {
                "root": str(self.settings.storage.root),
                "episodes": str(self.settings.storage.episodes_root),
                "combined": str(self.settings.storage.combined_root),
                "training": str(self.settings.storage.training_root),
            },
            "robot_config": self.robot_config_snapshot(),
            "camera_config": self.camera_config_snapshot(),
            "services": self.service_summary(),
        }

    def shutdown(self) -> None:
        shutdown_steps = [
            ("Training service", self.training.shutdown),
            ("Recording service", self.recording.shutdown),
            ("Teleoperation service", self.teleop.shutdown),
            ("Camera service", self.cameras.stop_streams),
            ("DDK service", self.ddk.shutdown),
        ]
        for label, action in shutdown_steps:
            try:
                action()
            except Exception as exc:
                warn(f"{label} shutdown failed", describe_exception(exc))

    def bootstrap_teleop_module(self) -> dict[str, Any]:
        stages = []

        teleop = self.teleop.initialize().__dict__
        stages.append({"stage": "teleop", "progress": 34, "detail": teleop})

        robot_data = self.teleop.robot_data_snapshot()
        stages.append(
            {
                "stage": "robot_data",
                "progress": 67,
                "detail": robot_data,
            }
        )

        cameras = self.cameras.start_streams()
        self._save_camera_config()
        stages.append({"stage": "cameras", "progress": 100, "detail": cameras})

        robot_data_robots = robot_data.get("robots") or {}
        camera_states = cameras.get("cameras") or {}
        teleop_ready = (
            bool(teleop.get("available"))
            and bool(teleop.get("initialized"))
            and not teleop.get("error")
            and not teleop.get("fault")
        )
        robot_data_ready = (
            bool(robot_data_robots)
            and not robot_data.get("errors")
            and all(
                bool(robot.get("connected")) for robot in robot_data_robots.values()
            )
        )
        cameras_ready = (
            bool(cameras.get("available"))
            and bool(camera_states)
            and not cameras.get("errors")
            and all(bool(camera.get("started")) for camera in camera_states.values())
        )

        return {
            "ready": teleop_ready and robot_data_ready and cameras_ready,
            "stages": stages,
            "recording": self.recording.status(),
        }

    def bootstrap_training_module(self) -> dict[str, Any]:
        return {
            "ready": True,
            "stages": [
                {
                    "stage": "storage",
                    "progress": 50,
                    "detail": {
                        "episodes_root": str(self.settings.storage.episodes_root),
                        "combined_root": str(self.settings.storage.combined_root),
                    },
                },
                {
                    "stage": "policies",
                    "progress": 100,
                    "detail": self.training.list_policies(),
                },
            ],
        }

    def browse_path(
        self,
        path: Path | None = None,
        directories_only: bool = False,
        *,
        root_path: Path | None = None,
        annotate_episode_dirs: bool = False,
    ) -> dict[str, Any]:
        storage_root = self.settings.storage.root.expanduser().resolve()
        restricted_root = (root_path or storage_root).expanduser().resolve()
        target = (path or restricted_root).expanduser().resolve()

        if not restricted_root.is_relative_to(storage_root):
            raise ValueError(
                f"Access denied: root must be within storage root ({storage_root})"
            )
        if not target.is_relative_to(restricted_root):
            raise ValueError(
                f"Access denied: path must be within root ({restricted_root})"
            )
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")
        items = []
        for child in sorted(target.iterdir()):
            if directories_only and not child.is_dir():
                continue
            item = {
                "name": child.name,
                "path": str(child),
                "is_dir": child.is_dir(),
            }
            if annotate_episode_dirs and child.is_dir():
                item["is_valid_episode"] = (child / "episode.json").exists() or (
                    child / "combined.json"
                ).exists()
            items.append(item)
        return {
            "path": str(target),
            "root_path": str(restricted_root),
            "items": items,
        }

    def list_episode_datasets(self) -> list[dict[str, Any]]:
        episodes = []
        for root in sorted(self.settings.storage.episodes_root.iterdir()):
            manifest = root / "episode.json"
            if manifest.exists():
                episodes.append(
                    {
                        "name": root.name,
                        "path": str(root),
                    }
                )
        return episodes

    def combine_episodes(
        self,
        episode_paths: list[str],
        output_name: str,
        on_progress: Any | None = None,
    ) -> dict[str, Any]:
        try:
            from flexivtrainer.jobs.combine import combine_episode_datasets
        except ImportError as exc:
            raise RuntimeError(
                _optional_dependency_error("Dataset combination", exc)
            ) from exc

        storage_root = self.settings.storage.root.expanduser().resolve()
        roots = [Path(path).resolve() for path in episode_paths]
        for root in roots:
            if not str(root).startswith(str(storage_root)):
                raise ValueError(
                    f"Access denied: path must be within storage root ({storage_root})"
                )
        return combine_episode_datasets(
            roots, self.settings.storage.combined_root, output_name, on_progress
        )

    def preview_dataset(self, dataset_path: Path) -> dict[str, Any]:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                _optional_dependency_error("Dataset preview", exc)
            ) from exc

        storage_root = self.settings.storage.root.expanduser().resolve()
        dataset_path = dataset_path.resolve()
        if not str(dataset_path).startswith(str(storage_root)):
            raise ValueError(
                f"Access denied: path must be within storage root ({storage_root})"
            )
        manifest_path = dataset_path / "episode.json"
        if not manifest_path.exists():
            manifest_path = dataset_path / "combined.json"

        repo_id = f"local/{dataset_path.name}"
        if manifest_path.exists():
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            repo_id = manifest.get("repo_id", repo_id)

        dataset = LeRobotDataset(repo_id, root=dataset_path)
        camera_keys = [
            key
            for key, feature in dataset.features.items()
            if feature["dtype"] in {"image", "video"}
        ]
        _META_KEYS = {
            "index",
            "episode_index",
            "frame_index",
            "task_index",
            "timestamp",
        }
        numeric_keys = [
            key
            for key, feature in dataset.features.items()
            if feature["dtype"] not in {"image", "video"} and key not in _META_KEYS
        ]
        first_item = dataset.get_raw_item(0) if dataset.num_frames else {}
        return {
            "name": dataset_path.name,
            "path": str(dataset_path),
            "repo_id": repo_id,
            "fps": dataset.fps,
            "num_frames": dataset.num_frames,
            "num_episodes": dataset.num_episodes,
            "camera_keys": camera_keys,
            "numeric_keys": numeric_keys,
            "sample_task": (
                first_item.get("task") if isinstance(first_item, dict) else None
            ),
        }

    def dataset_series(self, dataset_path: Path) -> dict[str, Any]:
        """Return all numeric time-series data from a dataset for plotting."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                _optional_dependency_error("Dataset series", exc)
            ) from exc

        import json

        storage_root = self.settings.storage.root.expanduser().resolve()
        dataset_path = dataset_path.resolve()
        if not str(dataset_path).startswith(str(storage_root)):
            raise ValueError(
                f"Access denied: path must be within storage root ({storage_root})"
            )

        manifest_path = dataset_path / "episode.json"
        if not manifest_path.exists():
            manifest_path = dataset_path / "combined.json"

        repo_id = f"local/{dataset_path.name}"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            repo_id = manifest.get("repo_id", repo_id)

        dataset = LeRobotDataset(repo_id, root=dataset_path)

        # Identify numeric keys (excluding metadata)
        _META_KEYS = {
            "index",
            "episode_index",
            "frame_index",
            "task_index",
            "timestamp",
        }
        numeric_keys = [
            key
            for key, feat in dataset.features.items()
            if feat["dtype"] not in {"image", "video"} and key not in _META_KEYS
        ]

        # Read all numeric data
        import numpy as np
        import torch

        series: dict[str, list[float | None]] = {key: [] for key in numeric_keys}
        timestamps: list[float] = []

        for idx in range(dataset.num_frames):
            item = dataset.get_raw_item(idx)
            ts = item.get("timestamp")
            if isinstance(ts, torch.Tensor):
                ts = ts.item()
            timestamps.append(float(ts) if ts is not None else idx / dataset.fps)
            for key in numeric_keys:
                val = item.get(key)
                if val is None:
                    series[key].append(None)
                elif isinstance(val, torch.Tensor):
                    series[key].append(val.item())
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    series[key].append(float(val))
                else:
                    series[key].append(None)

        return {
            "path": str(dataset_path),
            "fps": dataset.fps,
            "num_frames": dataset.num_frames,
            "timestamps": timestamps,
            "series": series,
            "numeric_keys": numeric_keys,
        }

    def dataset_frame_image(
        self, dataset_path: Path, camera_key: str, frame_index: int
    ) -> bytes:
        """Return a single frame image as JPEG bytes."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                _optional_dependency_error("Dataset frame image", exc)
            ) from exc

        import io
        import json

        import numpy as np
        import torch
        from PIL import Image

        storage_root = self.settings.storage.root.expanduser().resolve()
        dataset_path = dataset_path.resolve()
        if not str(dataset_path).startswith(str(storage_root)):
            raise ValueError(
                f"Access denied: path must be within storage root ({storage_root})"
            )

        manifest_path = dataset_path / "episode.json"
        if not manifest_path.exists():
            manifest_path = dataset_path / "combined.json"

        repo_id = f"local/{dataset_path.name}"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            repo_id = manifest.get("repo_id", repo_id)

        dataset = LeRobotDataset(repo_id, root=dataset_path)

        if frame_index < 0 or frame_index >= dataset.num_frames:
            raise IndexError(
                f"Frame index {frame_index} out of range [0, {dataset.num_frames})"
            )

        if camera_key not in dataset.features:
            raise KeyError(f"Camera key '{camera_key}' not found in dataset")

        item = dataset.get_raw_item(frame_index)
        image_data = item.get(camera_key)
        if image_data is None:
            raise KeyError(
                f"No image data for key '{camera_key}' at frame {frame_index}"
            )

        # Convert tensor to numpy HWC
        if isinstance(image_data, torch.Tensor):
            image_data = image_data.numpy()
        if isinstance(image_data, np.ndarray):
            if image_data.ndim == 3 and image_data.shape[0] in (1, 3, 4):
                image_data = np.moveaxis(image_data, 0, -1)
            # Scale from [0, 1] float to [0, 255] uint8 if needed
            if image_data.dtype in (np.float32, np.float64):
                image_data = (image_data * 255).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(image_data)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()


@lru_cache(maxsize=1)
def get_runtime_manager() -> RuntimeManager:
    return RuntimeManager(get_settings())
