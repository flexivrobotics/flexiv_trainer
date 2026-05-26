from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flexiv_trainer.config import AppSettings

POLICY_CATALOG = {
    "diffusion": {
        "label": "Diffusion",
        "description": "Default policy for this project. Good general-purpose multimodal action modeling.",
    },
    "act": {
        "label": "ACT",
        "description": "Fast and lightweight baseline for single-task imitation learning.",
    },
    "smolvla": {
        "label": "SmolVLA",
        "description": "Small vision-language-action model for richer multitask behavior.",
    },
    "pi0": {
        "label": "pi0",
        "description": "Large VLA baseline that typically needs substantially more GPU memory.",
    },
}


@dataclass
class TrainingJob:
    job_id: str
    command: list[str]
    output_dir: Path
    process: subprocess.Popen[str] | None = None
    logs: list[str] = field(default_factory=list)
    status: str = "pending"
    return_code: int | None = None
    error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "command": self.command,
            "output_dir": str(self.output_dir),
            "status": self.status,
            "return_code": self.return_code,
            "error": self.error,
            "logs": self.logs[-200:],
            "progress": 100 if self.status == "completed" else 0,
        }


class TrainingService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._job: TrainingJob | None = None
        self._lock = threading.Lock()

    def list_policies(self) -> dict[str, Any]:
        return {
            "default": self._settings.training.default_policy,
            "policies": POLICY_CATALOG,
        }

    def _resolve_dataset(self, dataset_root: Path) -> tuple[str, Path]:
        combined_manifest = dataset_root / "combined.json"
        episode_manifest = dataset_root / "episode.json"
        manifest_path = (
            combined_manifest if combined_manifest.exists() else episode_manifest
        )
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return manifest["repo_id"], dataset_root
        return f"local/{dataset_root.name}", dataset_root

    def start(
        self,
        dataset_root: Path,
        output_dir: Path,
        policy_type: str,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise RuntimeError("A training job is already running")

            repo_id, resolved_root = self._resolve_dataset(dataset_root)
            output_dir.mkdir(parents=True, exist_ok=True)

            executable = shutil.which("lerobot-train")
            if executable is None:
                executable = sys.executable
                command = [
                    executable,
                    "-m",
                    "lerobot.scripts.train",
                ]
            else:
                command = [executable]

            command.extend(
                [
                    "--dataset.repo_id",
                    repo_id,
                    "--dataset.root",
                    str(resolved_root),
                    "--policy.type",
                    policy_type,
                    "--output_dir",
                    str(output_dir),
                    "--job_name",
                    output_dir.name,
                    "--save_freq",
                    str(self._settings.training.save_frequency),
                ]
            )
            if extra_args:
                command.extend(extra_args)

            job = TrainingJob(
                job_id=str(uuid.uuid4()),
                command=command,
                output_dir=output_dir,
                status="running",
            )
            job.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(output_dir.parent),
            )
            self._job = job
            threading.Thread(
                target=self._collect_logs, args=(job,), daemon=True
            ).start()
            return job.snapshot()

    def _collect_logs(self, job: TrainingJob) -> None:
        assert job.process is not None
        try:
            assert job.process.stdout is not None
            for line in job.process.stdout:
                job.logs.append(line.rstrip())
            job.return_code = job.process.wait()
            job.status = "completed" if job.return_code == 0 else "failed"
            if job.return_code != 0:
                job.error = f"Training exited with code {job.return_code}"
        except Exception as exc:  # pragma: no cover - subprocess specific
            job.status = "failed"
            job.error = str(exc)

    def status(self) -> dict[str, Any]:
        if self._job is None:
            return {"status": "idle", "logs": [], "progress": 0}
        return self._job.snapshot()
