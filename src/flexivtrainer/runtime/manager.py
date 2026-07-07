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
from flexivtrainer.data.lerobot_io import active_camera_names
from flexivtrainer.jobs.train_policy import TrainingService
from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.teleop.service import TeleopService


def _optional_dependency_error(feature: str, exc: ImportError) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{feature} is unavailable: {detail}"
    return f"{feature} is unavailable in the selected environment"


def _entry_created_time(path: Path) -> float:
    # Creation time for the browser to sort episodes by. ``st_ctime`` is the
    # inode-change time on POSIX (close to creation for a freshly recorded,
    # never-modified episode) and the real creation time on Windows. Fall back
    # to 0.0 if the stat call fails (e.g. a race on deletion).
    try:
        return path.stat().st_ctime
    except OSError:
        return 0.0


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


class _UnavailableRolloutService:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    def status(self) -> dict[str, Any]:
        return {"status": "idle", "checkpoint_path": None, "error": self._reason}

    def start(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(self._reason)

    def stop(self) -> dict[str, Any]:
        return self.status()

    def shutdown(self) -> None:
        return None


class RuntimeManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._robot_config = self._load_robot_config()
        self.teleop = TeleopService(
            settings,
            self.get_teleop_robot_pairs,
            self.get_active_sides,
            self.get_end_effector_config,
        )
        self.cameras = RealSenseService(settings)
        self.cameras.set_active_locations(
            active_camera_names(self._robot_config.active_sides())
        )
        self._load_camera_config()
        try:
            from flexivtrainer.data.recording_service import RecordingService
        except ImportError as exc:
            self.recording = _UnavailableRecordingService(
                _optional_dependency_error("Episode recording", exc)
            )
        else:
            self.recording = RecordingService(
                settings, self.teleop, self.cameras, self.get_active_sides
            )
        self.training = TrainingService(settings)
        try:
            from flexivtrainer.rollout.service import RolloutService
        except ImportError as exc:
            self.rollout = _UnavailableRolloutService(
                _optional_dependency_error("Policy rollout", exc)
            )
        else:
            self.rollout = RolloutService(
                settings,
                self.cameras,
                self.teleop,
                self.get_teleop_robot_pairs,
                self.get_active_sides,
            )
        # Cache of constructed LeRobotDataset objects keyed by resolved path.
        # Building one parses metadata and the parquet index, which is far too
        # costly to repeat for every frame request during preview playback.
        self._dataset_cache: dict[str, tuple[int, Any]] = {}

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

    def get_active_sides(self) -> list[str]:
        return self._robot_config.active_sides()

    def get_end_effector_config(self) -> dict[str, Any]:
        return self._robot_config.end_effector_config

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
        active = active_camera_names(self.get_active_sides())
        configured = self.cameras.configured_serials()
        return {
            "cameras": [
                {"name": name, "device_serial": configured.get(name) or ""}
                for name in active
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

    def get_teleop_robot_pairs(self) -> list[TeleopRobotPair]:
        defaults = self.settings.teleop_robot_pairs
        pairs: list[TeleopRobotPair] = []
        for index in range(self._robot_config.active_arm_count()):
            template = defaults[index] if index < len(defaults) else TeleopRobotPair()
            leader_serial = self._robot_config.leader_robot_serials[index]
            follower_serial = self._robot_config.follower_robot_serials[index]
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
        self.cameras.set_active_locations(
            active_camera_names(self._robot_config.active_sides())
        )
        if changed:
            self.recording.shutdown()
            self.teleop.shutdown()
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
        arm_counts = self._robot_config.active_arm_count()
        serial_prompt = (
            f"Enter {arm_counts} leader and {arm_counts} follower robot serial numbers."
        )
        if not teleop_snapshot.available:
            teleop_state = "Unavailable"
            teleop_tone = "error"
            teleop_detail = self._service_message(
                teleop_errors, "TDK is not available."
            )
        elif teleop_pair_count == 0:
            teleop_state = "Not configured"
            teleop_tone = "error"
            teleop_detail = serial_prompt
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
            teleop_tone = "error"
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
            robot_data_detail = serial_prompt
        elif teleop_snapshot.initialized:
            robot_data_state = "Connected"
            robot_data_tone = "ok"
            robot_data_detail = "Service ready."
        else:
            robot_data_state = "Not connected"
            robot_data_tone = "error"
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
                "merged": str(self.settings.storage.merged_root),
                "training": str(self.settings.storage.training_root),
            },
            "robot_config": self.robot_config_snapshot(),
            "camera_config": self.camera_config_snapshot(),
            "services": self.service_summary(),
        }

    def shutdown(self) -> None:
        shutdown_steps = [
            ("Training service", self.training.shutdown),
            ("Rollout service", self.rollout.shutdown),
            ("Recording service", self.recording.shutdown),
            ("Teleoperation service", self.teleop.shutdown),
            ("Camera service", self.cameras.stop_streams),
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
                        "merged_root": str(self.settings.storage.merged_root),
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
            if annotate_episode_dirs and child.is_dir():
                # Recordings and merged datasets are both standard LeRobot
                # datasets, identified by their meta/info.json.
                is_episode = (child / "meta" / "info.json").exists()
                if not is_episode:
                    # Not an episode itself: treat it as a per-job folder and
                    # expand its episodes inline, tagged with the job name, so
                    # the episode picker shows a single grouped list rather than
                    # requiring the user to drill into each job folder.
                    job_episodes = self._expand_job_episodes(child)
                    if job_episodes:
                        items.extend(job_episodes)
                        continue
                items.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "is_dir": True,
                        "is_valid_episode": is_episode,
                        "job": None,
                        "created": _entry_created_time(child),
                    }
                )
                continue
            items.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": child.is_dir(),
                    "created": _entry_created_time(child),
                }
            )
        return {
            "path": str(target),
            "root_path": str(restricted_root),
            "items": items,
        }

    def _expand_job_episodes(self, job_dir: Path) -> list[dict[str, Any]]:
        # List the valid episodes inside a per-job folder, each tagged with the
        # job name so the picker can group them. Returns [] when the folder holds
        # no episodes (so the caller can fall back to showing it as-is).
        episodes: list[dict[str, Any]] = []
        for child in sorted(job_dir.iterdir()):
            if child.is_dir() and (child / "meta" / "info.json").exists():
                episodes.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "is_dir": True,
                        "is_valid_episode": True,
                        "job": job_dir.name,
                        "created": _entry_created_time(child),
                    }
                )
        return episodes

    def list_episode_datasets(self) -> list[dict[str, Any]]:
        # Episodes are filed under a per-job subfolder
        # (episodes/<job_name>/<episode>), so recurse one level into each job
        # folder. Any dataset still sitting directly under episodes/ (older flat
        # layout) is reported with no job so it stays visible.
        episodes_root = self.settings.storage.episodes_root
        if not episodes_root.exists():
            return []

        episodes: list[dict[str, Any]] = []
        for entry in sorted(episodes_root.iterdir()):
            if not entry.is_dir():
                continue
            if (entry / "meta" / "info.json").exists():
                # Flat, ungrouped episode directly under episodes/.
                episodes.append({"name": entry.name, "path": str(entry), "job": None})
                continue
            # Otherwise treat ``entry`` as a job folder and list its episodes.
            for child in sorted(entry.iterdir()):
                if child.is_dir() and (child / "meta" / "info.json").exists():
                    episodes.append(
                        {"name": child.name, "path": str(child), "job": entry.name}
                    )
        return episodes

    def merge_episodes(
        self,
        episode_paths: list[str],
        output_name: str,
        on_progress: Any | None = None,
    ) -> dict[str, Any]:
        try:
            from flexivtrainer.jobs.merge_episodes import merge_episode_datasets
        except ImportError as exc:
            raise RuntimeError(
                _optional_dependency_error("Dataset merge", exc)
            ) from exc

        storage_root = self.settings.storage.root.expanduser().resolve()
        roots = [Path(path).resolve() for path in episode_paths]
        for root in roots:
            if not str(root).startswith(str(storage_root)):
                raise ValueError(
                    f"Access denied: path must be within storage root ({storage_root})"
                )
        return merge_episode_datasets(
            roots, self.settings.storage.merged_root, output_name, on_progress
        )

    def _resolve_dataset_repo(self, dataset_path: Path) -> tuple[Path, str]:
        """Validate the path against the storage root and resolve its repo id."""
        storage_root = self.settings.storage.root.expanduser().resolve()
        dataset_path = dataset_path.resolve()
        if not str(dataset_path).startswith(str(storage_root)):
            raise ValueError(
                f"Access denied: path must be within storage root ({storage_root})"
            )
        # Recordings and merged datasets carry no extra manifest; their repo id
        # is local/<name>, exactly what the recorder/merge write into the dataset.
        repo_id = f"local/{dataset_path.name}"
        return dataset_path, repo_id

    def _load_dataset(self, dataset_path: Path) -> Any:
        """Return a cached LeRobotDataset for the given path.

        The cache is invalidated when the dataset directory's mtime changes, so
        a regenerated merged dataset under the same name is picked up.
        """
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                _optional_dependency_error("Dataset access", exc)
            ) from exc

        dataset_path, repo_id = self._resolve_dataset_repo(dataset_path)
        key = str(dataset_path)
        try:
            mtime = dataset_path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        cached = self._dataset_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        # LeRobot marks torchcodec unsupported on linux/aarch64. Explicitly
        # use the PyAV backend so frame preview decoding is stable here.
        dataset = LeRobotDataset(repo_id, root=dataset_path, video_backend="pyav")
        self._dataset_cache[key] = (mtime, dataset)
        return dataset

    _META_KEYS = {
        "index",
        "episode_index",
        "frame_index",
        "task_index",
        "timestamp",
    }

    def _numeric_channels(self, dataset: Any) -> list[tuple[str, str, int]]:
        """Plottable scalar channels as (series_key, feature_key, element_index).

        Grouped vector features (e.g. ``observation.state`` with per-axis
        ``names`` like ``left_arm.tcp_pose.x``) are expanded into one channel
        per element, keyed ``"<feature_key>.<name>"``. Legacy per-scalar
        features pass through unchanged so older datasets still plot.
        """
        channels: list[tuple[str, str, int]] = []
        for key, feature in dataset.features.items():
            if feature["dtype"] in {"image", "video"} or key in self._META_KEYS:
                continue
            shape = feature.get("shape") or (1,)
            size = int(shape[0]) if len(shape) else 1
            names = feature.get("names")
            if size <= 1:
                channels.append((key, key, 0))
                continue
            for i in range(size):
                axis = (
                    names[i]
                    if isinstance(names, (list, tuple)) and i < len(names)
                    else str(i)
                )
                channels.append((f"{key}.{axis}", key, i))
        return channels

    def _episode_list(self, dataset: Any) -> list[dict[str, int]]:
        """Per-episode summary [{index, num_frames}] for building an episode picker."""
        episodes = getattr(dataset.meta, "episodes", None)
        if episodes is None:
            return []
        try:
            indices = episodes["episode_index"]
            lengths = episodes["length"]
        except Exception:
            return []
        return [
            {"index": int(idx), "num_frames": int(length)}
            for idx, length in zip(indices, lengths, strict=False)
        ]

    def _episode_record(self, dataset: Any, episode_index: int) -> dict[str, Any]:
        """The meta/episodes row for one episode (frame range + per-camera windows)."""
        episodes = getattr(dataset.meta, "episodes", None)
        if episodes is None:
            raise IndexError("Dataset has no per-episode metadata")
        try:
            indices = [int(i) for i in episodes["episode_index"]]
            row_pos = indices.index(int(episode_index))
        except (ValueError, KeyError) as exc:
            raise IndexError(f"Episode {episode_index} not found") from exc
        return dict(episodes[row_pos])

    def preview_dataset(
        self, dataset_path: Path, episode_index: int | None = None
    ) -> dict[str, Any]:
        # Validate the path is within the storage root *before* attempting any
        # dataset load: the access-denied guard must run regardless of whether
        # the optional dataset dependencies are importable.
        dataset_path, repo_id = self._resolve_dataset_repo(dataset_path)
        dataset = self._load_dataset(dataset_path)
        camera_keys = [
            key
            for key, feature in dataset.features.items()
            if feature["dtype"] in {"image", "video"}
        ]
        numeric_keys = [
            series_key for series_key, _, _ in self._numeric_channels(dataset)
        ]
        result: dict[str, Any] = {
            "name": dataset_path.name,
            "path": str(dataset_path),
            "repo_id": repo_id,
            "fps": dataset.fps,
            "num_frames": dataset.num_frames,
            "num_episodes": dataset.num_episodes,
            "camera_keys": camera_keys,
            "numeric_keys": numeric_keys,
            "episodes": self._episode_list(dataset),
        }

        if episode_index is None:
            first_item = dataset.get_raw_item(0) if dataset.num_frames else {}
            result["sample_task"] = (
                first_item.get("task") if isinstance(first_item, dict) else None
            )
            return result

        # Scope the preview to a single episode within the dataset. The episode's
        # frames live in a shared (concatenated) MP4 per camera, so we expose the
        # video file + time window the browser should play instead of the whole feed.
        record = self._episode_record(dataset, episode_index)
        from_index = int(record["dataset_from_index"])
        to_index = int(record["dataset_to_index"])
        windows: dict[str, dict[str, Any]] = {}
        for key in camera_keys:
            chunk = record.get(f"videos/{key}/chunk_index")
            file = record.get(f"videos/{key}/file_index")
            if chunk is None or file is None:
                continue
            windows[key] = {
                "chunk_index": int(chunk),
                "file_index": int(file),
                "from_timestamp": float(record.get(f"videos/{key}/from_timestamp") or 0.0),
                "to_timestamp": float(record.get(f"videos/{key}/to_timestamp") or 0.0),
            }
        first_item = (
            dataset.get_raw_item(from_index) if to_index > from_index else {}
        )
        result.update(
            {
                "episode_index": int(episode_index),
                "num_frames": to_index - from_index,
                "video_windows": windows,
                "sample_task": (
                    first_item.get("task") if isinstance(first_item, dict) else None
                ),
            }
        )
        return result

    def dataset_series(
        self, dataset_path: Path, episode_index: int | None = None
    ) -> dict[str, Any]:
        """Return numeric time-series data for plotting.

        When ``episode_index`` is given, only that episode's frames are returned
        and timestamps are re-based to start at 0 so the plot x-axis is local.
        """
        dataset = self._load_dataset(dataset_path)
        dataset_path = dataset_path.resolve()

        import numpy as np
        import torch

        if episode_index is None:
            start, end = 0, dataset.num_frames
        else:
            record = self._episode_record(dataset, episode_index)
            start = int(record["dataset_from_index"])
            end = int(record["dataset_to_index"])

        channels = self._numeric_channels(dataset)
        numeric_keys = [series_key for series_key, _, _ in channels]
        series: dict[str, list[float | None]] = {key: [] for key in numeric_keys}
        timestamps: list[float] = []
        base_ts: float | None = None

        def _element(val: Any, element_index: int) -> float | None:
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                flat = val.reshape(-1)
                return (
                    flat[element_index].item() if element_index < flat.numel() else None
                )
            if isinstance(val, np.ndarray):
                flat = val.reshape(-1)
                return float(flat[element_index]) if element_index < flat.size else None
            if isinstance(val, (list, tuple)):
                return float(val[element_index]) if element_index < len(val) else None
            if element_index == 0 and isinstance(
                val, (int, float, np.integer, np.floating)
            ):
                return float(val)
            return None

        for idx in range(start, end):
            item = dataset.get_raw_item(idx)
            ts = item.get("timestamp")
            if isinstance(ts, torch.Tensor):
                ts = ts.item()
            ts = float(ts) if ts is not None else idx / dataset.fps
            if base_ts is None:
                base_ts = ts
            timestamps.append(ts - base_ts)
            for series_key, feature_key, element_index in channels:
                series[series_key].append(
                    _element(item.get(feature_key), element_index)
                )

        return {
            "path": str(dataset_path),
            "fps": dataset.fps,
            "num_frames": end - start,
            "timestamps": timestamps,
            "series": series,
            "numeric_keys": numeric_keys,
        }

    def dataset_frame_image(
        self,
        dataset_path: Path,
        camera_key: str,
        frame_index: int,
        episode_index: int | None = None,
    ) -> bytes:
        """Return a single frame image as JPEG bytes.

        When ``episode_index`` is given, ``frame_index`` is episode-local and is
        mapped to the dataset-global frame.
        """
        import io

        import numpy as np
        import torch
        from PIL import Image

        dataset = self._load_dataset(dataset_path)

        if episode_index is not None:
            record = self._episode_record(dataset, episode_index)
            start = int(record["dataset_from_index"])
            end = int(record["dataset_to_index"])
            if frame_index < 0 or frame_index >= (end - start):
                raise IndexError(
                    f"Frame index {frame_index} out of range [0, {end - start})"
                )
            frame_index = start + frame_index
        elif frame_index < 0 or frame_index >= dataset.num_frames:
            raise IndexError(
                f"Frame index {frame_index} out of range [0, {dataset.num_frames})"
            )

        if camera_key not in dataset.features:
            raise KeyError(f"Camera key '{camera_key}' not found in dataset")

        # Use __getitem__ rather than get_raw_item: the latter only returns the
        # parquet (numeric) columns, while video frames are decoded lazily on
        # item access. get_raw_item would yield no image for video features.
        item = dataset[frame_index]
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

    def dataset_video_path(
        self,
        dataset_path: Path,
        camera_key: str,
        chunk_index: int = 0,
        file_index: int = 0,
    ) -> Path:
        """Resolve the MP4 file for a camera feed so it can be streamed directly.

        Playing the encoded video natively in the browser is far smoother than
        decoding individual frames server-side. For a single-file episode the
        whole clip is one MP4 and ``time = frame_index / fps``.
        """
        dataset_path, _ = self._resolve_dataset_repo(dataset_path)
        relative = (
            f"videos/{camera_key}/" f"chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        video_path = (dataset_path / relative).resolve()
        # Guard against a camera_key that tries to escape the dataset directory.
        if not str(video_path).startswith(str(dataset_path)):
            raise ValueError("Access denied: video path escapes the dataset root")
        if not video_path.is_file():
            raise FileNotFoundError(f"No video file for '{camera_key}': {video_path}")
        return video_path


@lru_cache(maxsize=1)
def get_runtime_manager() -> RuntimeManager:
    return RuntimeManager(get_settings())
