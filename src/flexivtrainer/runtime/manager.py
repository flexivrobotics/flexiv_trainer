from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from flexivtrainer.cameras.realsense import RealSenseService
from flexivtrainer.config import (
    AppSettings,
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
        try:
            from flexivtrainer.data.recording_service import RecordingService
        except ImportError as exc:
            self.recording = _UnavailableRecordingService(
                _optional_dependency_error("Episode recording", exc)
            )
        else:
            self.recording = RecordingService(settings, self.ddk, self.cameras)
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

        ddk_status = self.ddk.status()
        remote_count = len(self.get_remote_robot_serials())
        connected_count = sum(
            1 for robot in ddk_status["robots"].values() if robot.get("connected")
        )
        ddk_errors = [str(value) for value in ddk_status["errors"].values() if value]
        if not ddk_status["available"]:
            ddk_state = "Unavailable"
            ddk_tone = "error"
            ddk_detail = self._service_message(ddk_errors, "DDK is not available.")
        elif remote_count == 0:
            ddk_state = "Not configured"
            ddk_tone = "error"
            ddk_detail = "Enter two remote robot serial numbers."
        elif connected_count == remote_count and remote_count > 0:
            ddk_state = "Connected"
            ddk_tone = "ok"
            ddk_detail = f"{connected_count} data streams ready."
        else:
            ddk_state = "Not connected"
            ddk_tone = "error" if ddk_errors else "neutral"
            ddk_detail = self._service_message(
                ddk_errors, "Press Connect to initialize DDK."
            )

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
            camera_state = "Connected"
            camera_tone = "ok"
            camera_detail = f"{started_camera_count} feeds active."
        else:
            camera_state = "Not connected"
            camera_tone = "error" if camera_errors else "neutral"
            camera_detail = self._service_message(
                camera_errors, "Press Connect to start the camera feeds."
            )

        calibration_files = sorted(
            path.name for path in self.settings.storage.calibration_root.glob("*.json")
        )
        if calibration_files:
            calibration_state = "Available"
            calibration_tone = "ok"
            calibration_detail = f"{len(calibration_files)} calibration files found."
        else:
            calibration_state = "Missing"
            calibration_tone = "error"
            calibration_detail = "No calibration files found."

        return {
            "teleop_service": {
                "label": "TELEOP SERVICE",
                "state": teleop_state,
                "detail": teleop_detail,
                "tone": teleop_tone,
            },
            "robot_data_service": {
                "label": "ROBOT DATA SERVICE",
                "state": ddk_state,
                "detail": ddk_detail,
                "tone": ddk_tone,
            },
            "cameras": {
                "label": "CAMERAS",
                "state": camera_state,
                "detail": camera_detail,
                "tone": camera_tone,
            },
            "calibration": {
                "label": "CALIBRATION",
                "state": calibration_state,
                "detail": calibration_detail,
                "tone": calibration_tone,
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
            result = (
                self.cameras.start_streams()
                if action == "connect"
                else self.cameras.stop_streams()
            )
        else:
            raise ValueError(f"Unsupported service: {service_name}")

        return {"result": result, "services": self.service_summary()}

    def run_calibration(self, calibration_name: str) -> dict[str, Any]:
        labels = {
            "egocentric": "Egocentric calibration is not implemented in this build.",
            "in-hand": "In-hand calibration is not implemented in this build.",
        }
        if calibration_name not in labels:
            raise ValueError(f"Unsupported calibration action: {calibration_name}")
        return {
            "ok": False,
            "message": labels[calibration_name],
            "services": self.service_summary(),
        }

    def system_summary(self) -> dict[str, Any]:
        return {
            "backend": {
                "reachable": True,
                "host": self.settings.host,
                "port": self.settings.port,
                "ui_url": self.settings.ui_url,
            },
            "teleop": self.teleop.snapshot().__dict__,
            "ddk": self.ddk.status(),
            "cameras": self.cameras.discover(),
            "calibration": {
                "root": str(self.settings.storage.calibration_root),
                "available_files": sorted(
                    path.name
                    for path in self.settings.storage.calibration_root.glob("*.json")
                ),
            },
            "storage": {
                "root": str(self.settings.storage.root),
                "episodes": str(self.settings.storage.episodes_root),
                "combined": str(self.settings.storage.combined_root),
                "training": str(self.settings.storage.training_root),
            },
            "robot_config": self.robot_config_snapshot(),
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
        stages.append({"stage": "teleop", "progress": 25, "detail": teleop})

        ddk = self.ddk.initialize()
        stages.append({"stage": "ddk", "progress": 50, "detail": ddk})

        cameras = self.cameras.start_streams()
        stages.append({"stage": "cameras", "progress": 75, "detail": cameras})

        overlay = {
            "calibration_files": sorted(
                path.name
                for path in self.settings.storage.calibration_root.glob("*.json")
            )
        }
        stages.append({"stage": "overlay", "progress": 100, "detail": overlay})

        return {
            "ready": not teleop.get("error")
            and not ddk["errors"]
            and self.cameras.available(),
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
        self, path: Path | None = None, directories_only: bool = False
    ) -> dict[str, Any]:
        target = (path or self.settings.storage.root).expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")
        items = []
        for child in sorted(target.iterdir()):
            if directories_only and not child.is_dir():
                continue
            items.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": child.is_dir(),
                }
            )
        return {"path": str(target), "items": items}

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
        self, episode_paths: list[str], output_name: str
    ) -> dict[str, Any]:
        try:
            from flexivtrainer.jobs.combine import combine_episode_datasets
        except ImportError as exc:
            raise RuntimeError(
                _optional_dependency_error("Dataset combination", exc)
            ) from exc

        roots = [Path(path).resolve() for path in episode_paths]
        return combine_episode_datasets(
            roots, self.settings.storage.combined_root, output_name
        )

    def preview_dataset(self, dataset_path: Path) -> dict[str, Any]:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                _optional_dependency_error("Dataset preview", exc)
            ) from exc

        dataset_path = dataset_path.resolve()
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
        numeric_keys = [
            key
            for key, feature in dataset.features.items()
            if feature["dtype"] not in {"image", "video"}
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


@lru_cache(maxsize=1)
def get_runtime_manager() -> RuntimeManager:
    return RuntimeManager(get_settings())
