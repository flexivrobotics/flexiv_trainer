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

import contextlib
import json
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from flexivtrainer.config import AppSettings
from flexivtrainer.data.lerobot_io import (
    build_features_from_sample,
    extract_recording_images,
    resolve_recording_entries,
    resolve_recording_image_names,
)


class RecordingService:
    def __init__(
        self,
        settings: AppSettings,
        ddk: Any,
        cameras: Any,
    ) -> None:
        self._settings = settings
        self._ddk = ddk
        self._cameras = cameras

        self._lock = threading.Lock()
        self._active = False
        self._awaiting_save = False
        self._frames_captured = 0
        self._episode_name: str | None = None
        self._fps: int | None = None
        self._task: str | None = None
        self._recording_entries: list[str] | None = None
        self._staging_path: Path | None = None
        self._dataset: Any = None
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: str | None = None
        self._save_in_progress = False
        self._save_progress = 0
        self._started_at_monotonic: float | None = None
        self._elapsed_s = 0.0

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._active and self._started_at_monotonic is not None:
                elapsed_s = max(0.0, time.monotonic() - self._started_at_monotonic)
            elif self._awaiting_save:
                elapsed_s = max(0.0, self._elapsed_s)
            else:
                elapsed_s = 0.0
            return {
                "active": self._active,
                "awaiting_save": self._awaiting_save,
                "frames_captured": self._frames_captured,
                "episode_name": self._episode_name,
                "fps": self._fps,
                "error": self._error,
                "save_in_progress": self._save_in_progress,
                "save_progress": self._save_progress,
                "elapsed_s": elapsed_s,
            }

    def start(
        self,
        task: str = "Dual-arm Flexiv teleoperation demonstration",
        fps: int | None = None,
        recording_entries: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._active:
                raise RuntimeError("Recording is already active")
            if self._awaiting_save:
                raise RuntimeError("Previous recording is awaiting save or discard")

        entries = resolve_recording_entries(recording_entries)
        target_fps = fps or 30
        episode_name, staging_path = self._create_staging_path()

        try:
            camera_names = resolve_recording_image_names(entries)
            self._ensure_camera_streams(camera_names)
            images = self._grab_images(camera_names, require_all=True, attempts=3)
            ddk_snapshot = self._ddk.snapshot(initialize=False)

            features, _, _ = build_features_from_sample(ddk_snapshot, images, entries)
            if not features:
                raise RuntimeError(
                    "No recording features resolved for the selected entries"
                )

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset.create(
                repo_id=f"local/{episode_name}",
                fps=target_fps,
                features=features,
                root=staging_path,
                robot_type=self._settings.robot_type,
                use_videos=True,
            )
        except Exception as exc:
            shutil.rmtree(staging_path, ignore_errors=True)
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Failed to create dataset: {exc}") from exc

        with self._lock:
            self._active = True
            self._awaiting_save = False
            self._frames_captured = 0
            self._episode_name = episode_name
            self._fps = target_fps
            self._task = task
            self._recording_entries = entries
            self._staging_path = staging_path
            self._dataset = dataset
            self._error = None
            self._stop_event.clear()
            self._save_in_progress = False
            self._save_progress = 0
            self._started_at_monotonic = time.monotonic()
            self._elapsed_s = 0.0

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="recording-capture"
        )
        self._capture_thread.start()

        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._active:
                raise RuntimeError("No active recording to stop")

        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None

        with self._lock:
            self._active = False
            self._awaiting_save = True
            if self._started_at_monotonic is not None:
                self._elapsed_s = max(
                    0.0, time.monotonic() - self._started_at_monotonic
                )
            self._started_at_monotonic = None

        return self.status()

    def save(self) -> dict[str, Any]:
        with self._lock:
            if not self._awaiting_save:
                raise RuntimeError("No recording awaiting save")
            if self._save_in_progress:
                raise RuntimeError("Save is already in progress")
            episode_name = self._episode_name
            staging_path = self._staging_path
            dataset = self._dataset
            task = self._task
            fps = self._fps
            frames_captured = self._frames_captured
            self._save_in_progress = True
            self._save_progress = 0
            self._error = None

        if episode_name is None or staging_path is None or dataset is None:
            with self._lock:
                self._save_in_progress = False
            raise RuntimeError("Recording state is inconsistent; cannot save")

        try:
            try:
                self._set_save_progress(5)
                self._run_with_terminal_progress(dataset.save_episode, base=5, span=40)
                self._set_save_progress(50)
                self._run_with_terminal_progress(dataset.finalize, base=50, span=45)
                self._set_save_progress(95)
            except Exception as exc:
                with self._lock:
                    self._error = f"Failed to finalize dataset: {exc}"
                raise RuntimeError(self._error) from exc

            manifest = {
                "repo_id": f"local/{episode_name}",
                "task": task,
                "fps": fps,
                "frames": frames_captured,
            }
            (staging_path / "episode.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

            episodes_root = self._settings.storage.episodes_root
            episodes_root.mkdir(parents=True, exist_ok=True)
            target_path = episodes_root / episode_name
            shutil.move(str(staging_path), str(target_path))

            with self._lock:
                self._awaiting_save = False
                self._dataset = None
                self._staging_path = None
                self._save_progress = 100
                self._started_at_monotonic = None
                self._elapsed_s = 0.0

            return {
                "episode_name": episode_name,
                "frames_captured": frames_captured,
                "path": str(target_path),
            }
        finally:
            with self._lock:
                self._save_in_progress = False

    def discard(self) -> dict[str, Any]:
        with self._lock:
            if not self._awaiting_save:
                raise RuntimeError("No recording awaiting discard")
            episode_name = self._episode_name
            staging_path = self._staging_path

        if staging_path and staging_path.exists():
            shutil.rmtree(staging_path, ignore_errors=True)

        with self._lock:
            self._awaiting_save = False
            self._dataset = None
            self._staging_path = None
            self._save_in_progress = False
            self._save_progress = 0
            self._started_at_monotonic = None
            self._elapsed_s = 0.0

        return {"episode_name": episode_name, "discarded": True}

    def shutdown(self) -> None:
        if self._active:
            self._stop_event.set()
            if self._capture_thread is not None:
                self._capture_thread.join(timeout=5.0)
                self._capture_thread = None
            with self._lock:
                self._active = False

        if self._awaiting_save and self._staging_path:
            shutil.rmtree(self._staging_path, ignore_errors=True)

        with self._lock:
            self._awaiting_save = False
            self._dataset = None
            self._staging_path = None
            self._episode_name = None
            self._frames_captured = 0
            self._error = None
            self._save_in_progress = False
            self._save_progress = 0
            self._started_at_monotonic = None
            self._elapsed_s = 0.0

    def _create_staging_path(self) -> tuple[str, Path]:
        base_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        staging_root = self._settings.storage.staging_root
        staging_root.mkdir(parents=True, exist_ok=True)

        for suffix in range(1000):
            episode_name = base_name if suffix == 0 else f"{base_name}_{suffix:03d}"
            candidate = staging_root / episode_name
            if not candidate.exists():
                return episode_name, candidate

        raise RuntimeError("Unable to allocate a unique staging directory")

    def _ensure_camera_streams(self, camera_names: list[str]) -> None:
        if not camera_names:
            return

        try:
            status = self._cameras.start_streams(camera_names=camera_names)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to start selected camera streams: {exc}"
            ) from exc

        cameras = status.get("cameras") if isinstance(status, dict) else None
        errors = status.get("errors") if isinstance(status, dict) else None

        unavailable: list[str] = []
        for camera_name in camera_names:
            camera_status = (
                cameras.get(camera_name) if isinstance(cameras, dict) else None
            )
            started = (
                bool(camera_status.get("started"))
                if isinstance(camera_status, dict)
                else False
            )
            if started:
                continue

            detail = None
            if isinstance(camera_status, dict):
                detail = camera_status.get("error")
            if not detail and isinstance(errors, dict):
                detail = errors.get(camera_name)
            unavailable.append(
                f"{camera_name}: {detail or 'camera stream is not started'}"
            )

        if unavailable:
            raise RuntimeError(
                "Selected camera stream is unavailable: " + "; ".join(unavailable)
            )

    def _grab_images(
        self,
        camera_names: list[str],
        *,
        require_all: bool,
        attempts: int,
        timeout_ms: int = 1_200,
        block: bool = True,
    ) -> dict[str, np.ndarray]:
        if not camera_names:
            return {}

        last_images: dict[str, np.ndarray] = {}
        last_errors: list[str] = []

        for attempt in range(max(attempts, 1)):
            images: dict[str, np.ndarray] = {}
            errors: list[str] = []

            for camera_name in camera_names:
                try:
                    frame = self._cameras.capture_frame(
                        camera_name,
                        block=block,
                        timeout_ms=max(1, int(timeout_ms)),
                        allow_cached=True,
                    )
                except Exception as exc:
                    errors.append(f"{camera_name}: {exc}")
                    continue

                image = frame.get("image") if isinstance(frame, dict) else None
                if image is None:
                    errors.append(f"{camera_name}: missing image payload")
                    continue
                images[camera_name] = np.asarray(image)

            if not require_all or all(name in images for name in camera_names):
                return images

            last_images = images
            last_errors = errors
            if attempt + 1 < attempts:
                time.sleep(0.08)

        if require_all:
            missing = [name for name in camera_names if name not in last_images]
            detail = "; ".join(last_errors) if last_errors else "no frame returned"
            raise RuntimeError(
                "No frame available for selected camera(s): "
                f"{', '.join(missing)}. {detail}"
            )

        return last_images

    def _capture_loop(self) -> None:
        interval = 1.0 / (self._fps or 30)
        entries = list(self._recording_entries or [])
        camera_names = resolve_recording_image_names(entries)
        includes_observation_values = any(
            entry in _OBSERVATION_ENTRY_SPECS for entry in entries
        )
        includes_action_values = any(entry in _ACTION_ENTRY_SPECS for entry in entries)
        requires_robot_values = includes_observation_values or includes_action_values
        capture_timeout_ms = max(10, int(interval * 1_000))

        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                images = self._grab_images(
                    camera_names,
                    require_all=False,
                    attempts=1,
                    timeout_ms=capture_timeout_ms,
                    block=False,
                )
                ddk_snapshot = (
                    self._ddk.snapshot(initialize=False)
                    if requires_robot_values
                    else {}
                )

                selected_images = extract_recording_images(images, entries)
                if len(selected_images) != len(camera_names):
                    missing = [
                        name for name in camera_names if name not in selected_images
                    ]
                    with self._lock:
                        self._error = (
                            "Missing camera frame(s) during capture: "
                            + ", ".join(missing)
                        )
                    continue

                frame: dict[str, Any] = {}

                for camera_name, image in selected_images.items():
                    frame[f"observation.images.{camera_name}"] = image

                if requires_robot_values:
                    obs_values, act_values = _extract_snapshot_values(
                        ddk_snapshot, entries
                    )
                    for key, value in obs_values.items():
                        frame[key] = np.array([value], dtype=np.float32)
                    for key, value in act_values.items():
                        frame[key] = np.array([value], dtype=np.float32)

                frame["task"] = self._task

                with self._lock:
                    dataset = self._dataset
                if dataset is not None:
                    dataset.add_frame(frame)
                    with self._lock:
                        self._frames_captured += 1

            except Exception as exc:
                with self._lock:
                    self._error = str(exc)

            elapsed = time.monotonic() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    def _set_save_progress(self, progress: int) -> None:
        clamped = max(0, min(int(progress), 100))
        with self._lock:
            if self._save_in_progress:
                self._save_progress = max(self._save_progress, clamped)
            else:
                self._save_progress = clamped

    def _run_with_terminal_progress(
        self,
        operation: Callable[[], Any],
        *,
        base: int,
        span: int,
    ) -> None:
        def on_percent(percent: int) -> None:
            mapped = base + int((max(0, min(percent, 100)) / 100.0) * span)
            self._set_save_progress(mapped)

        out_tee = _ProgressTee(sys.stdout, on_percent)
        err_tee = _ProgressTee(sys.stderr, on_percent)
        with contextlib.redirect_stdout(out_tee), contextlib.redirect_stderr(err_tee):
            operation()
        self._set_save_progress(base + span)


_TERMINAL_PERCENT_RE = re.compile(r"(\d{1,3})%")


class _ProgressTee:
    def __init__(self, stream: Any, on_percent: Callable[[int], None]) -> None:
        self._stream = stream
        self._on_percent = on_percent

    def write(self, chunk: str) -> int:
        text = str(chunk)
        written = self._stream.write(text)
        for match in _TERMINAL_PERCENT_RE.finditer(text):
            try:
                self._on_percent(int(match.group(1)))
            except Exception:
                continue
        return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        isatty = getattr(self._stream, "isatty", None)
        if callable(isatty):
            return bool(isatty())
        return False


_OBSERVATION_ENTRY_SPECS: dict[str, tuple[str, str]] = {
    "observation.state.tcp_pose": ("cartesian_state", "tcp_pose"),
    "observation.state.tcp_twist": ("cartesian_state", "tcp_vel"),
    "observation.state.tcp_wrench": ("cartesian_state", "ext_wrench_in_world"),
}

_ACTION_ENTRY_SPECS: dict[str, tuple[str, str]] = {
    "action.tcp_pose": ("cartesian_command", "tcp_pose_des"),
    "action.tcp_twist": ("cartesian_command", "tcp_vel_des"),
    "action.tcp_wrench": ("cartesian_command", "wrench_des_in_ctrl_frame"),
}


def _extract_snapshot_values(
    ddk_snapshot: dict[str, Any], entries: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract actual numeric values from a DDK snapshot, keyed by feature label."""
    robots = ddk_snapshot.get("robots") if isinstance(ddk_snapshot, dict) else None
    if not isinstance(robots, dict):
        return {}, {}

    observation_values: dict[str, float] = {}
    action_values: dict[str, float] = {}

    for robot_name, robot_payload in robots.items():
        if not isinstance(robot_payload, dict):
            continue

        for entry, (parent_key, source_key) in _OBSERVATION_ENTRY_SPECS.items():
            if entry not in entries:
                continue
            section = robot_payload.get(parent_key)
            values = section.get(source_key) if isinstance(section, dict) else None
            if isinstance(values, (list, tuple)):
                metric = entry.rsplit(".", 1)[-1]
                for i, v in enumerate(values):
                    observation_values[f"{robot_name}.state.{metric}.{i}"] = float(v)

        for entry, (parent_key, source_key) in _ACTION_ENTRY_SPECS.items():
            if entry not in entries:
                continue
            section = robot_payload.get(parent_key)
            values = section.get(source_key) if isinstance(section, dict) else None
            if isinstance(values, (list, tuple)):
                metric = entry.rsplit(".", 1)[-1]
                for i, v in enumerate(values):
                    action_values[f"{robot_name}.command.{metric}.{i}"] = float(v)

    return observation_values, action_values
