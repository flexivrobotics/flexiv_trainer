from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from flexivtrainer.cameras.realsense import RealSenseService
from flexivtrainer.config import AppSettings, get_settings
from flexivtrainer.ddk.service import DDKService
from flexivtrainer.jobs.train import TrainingService
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


class RuntimeManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.teleop = TeleopService(settings)
        self.ddk = DDKService(settings)
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
        }

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
