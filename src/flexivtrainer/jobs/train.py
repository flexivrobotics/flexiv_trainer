from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flexivtrainer.config import AppSettings
from flexivtrainer.observability import (
    Pulse,
    error,
    format_elapsed,
    info,
    ok,
    print_command,
    section,
    stream,
)

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

TRACKER_KEY_MAP = {
    "step": "step",
    "smpl": "samples",
    "ep": "episodes",
    "epch": "epochs",
    "loss": "loss",
    "grdn": "grad_norm",
    "lr": "lr",
    "updt_s": "update_seconds",
    "data_s": "data_seconds",
}

TRACKER_TOKEN_PATTERN = re.compile(
    r"(?P<key>step|smpl|ep|epch|loss|grdn|lr|updt_s|data_s):(?P<value>[^\s]+)"
)
TOTAL_STEPS_PATTERN = re.compile(r"cfg\.steps=(?P<steps>\d+)")
DATASET_FRAMES_PATTERN = re.compile(r"dataset\.num_frames=(?P<frames>\d+)")
DATASET_EPISODES_PATTERN = re.compile(r"dataset\.num_episodes=(?P<episodes>\d+)")
EFFECTIVE_BATCH_SIZE_PATTERN = re.compile(
    r"effective batch size:\s*(?P<batch_size>\d+)", re.IGNORECASE
)
CHECKPOINT_STEP_PATTERN = re.compile(r"Checkpoint policy after step (?P<step>\d+)")
EVAL_STEP_PATTERN = re.compile(r"Eval policy at step (?P<step>\d+)")


@dataclass
class TrainingJob:
    job_id: str
    command: list[str]
    output_dir: Path
    dataset_root: Path
    policy_type: str
    process: subprocess.Popen[str] | None = None
    logs: list[str] = field(default_factory=list)
    status: str = "pending"
    return_code: int | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    metrics: dict[str, int | float] = field(default_factory=dict)
    total_steps: int | None = None
    dataset_num_frames: int | None = None
    dataset_num_episodes: int | None = None
    effective_batch_size: int | None = None
    last_checkpoint_step: int | None = None
    last_eval_step: int | None = None
    last_event: str | None = None
    pulse: Pulse | None = field(default=None, repr=False, compare=False)

    def snapshot(self) -> dict[str, Any]:
        current_step = self.metrics.get("step")
        progress = 100 if self.status == "completed" else 0
        if self.total_steps and current_step is not None and self.status != "completed":
            progress = min(
                99, max(0, int((float(current_step) / self.total_steps) * 100))
            )

        return {
            "job_id": self.job_id,
            "command": self.command,
            "output_dir": str(self.output_dir),
            "dataset_root": str(self.dataset_root),
            "policy_type": self.policy_type,
            "status": self.status,
            "return_code": self.return_code,
            "error": self.error,
            "logs": self.logs[-200:],
            "elapsed": format_elapsed(time.monotonic() - self.started_at),
            "log_lines": len(self.logs),
            "progress": progress,
            "metrics": self.metrics,
            "total_steps": self.total_steps,
            "dataset_num_frames": self.dataset_num_frames,
            "dataset_num_episodes": self.dataset_num_episodes,
            "effective_batch_size": self.effective_batch_size,
            "last_checkpoint_step": self.last_checkpoint_step,
            "last_eval_step": self.last_eval_step,
            "last_event": self.last_event,
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

    @staticmethod
    def _parse_compact_number(raw: str) -> int | float:
        suffix_scale = {
            "K": 1_000,
            "M": 1_000_000,
            "B": 1_000_000_000,
        }

        token = raw.strip()
        if token and token[-1].upper() in suffix_scale:
            value = float(token[:-1]) * suffix_scale[token[-1].upper()]
        else:
            value = float(token)
        return int(value) if value.is_integer() else value

    def _update_job_from_log(self, job: TrainingJob, line: str) -> None:
        tracker_matches = {
            match.group("key"): match.group("value")
            for match in TRACKER_TOKEN_PATTERN.finditer(line)
        }
        if tracker_matches and "step" in tracker_matches:
            parsed_metrics: dict[str, int | float] = {}
            for raw_key, raw_value in tracker_matches.items():
                parsed_metrics[TRACKER_KEY_MAP[raw_key]] = self._parse_compact_number(
                    raw_value
                )
            job.metrics.update(parsed_metrics)
            job.last_event = "training_metrics"

        if total_steps_match := TOTAL_STEPS_PATTERN.search(line):
            job.total_steps = int(total_steps_match.group("steps"))
            job.last_event = "config_loaded"

        if dataset_frames_match := DATASET_FRAMES_PATTERN.search(line):
            job.dataset_num_frames = int(dataset_frames_match.group("frames"))

        if dataset_episodes_match := DATASET_EPISODES_PATTERN.search(line):
            job.dataset_num_episodes = int(dataset_episodes_match.group("episodes"))

        if batch_size_match := EFFECTIVE_BATCH_SIZE_PATTERN.search(line):
            job.effective_batch_size = int(batch_size_match.group("batch_size"))
            job.last_event = "training_started"

        if checkpoint_match := CHECKPOINT_STEP_PATTERN.search(line):
            job.last_checkpoint_step = int(checkpoint_match.group("step"))
            job.last_event = "checkpoint_saved"

        if eval_match := EVAL_STEP_PATTERN.search(line):
            job.last_eval_step = int(eval_match.group("step"))
            job.last_event = "evaluation_running"

        if "End of training" in line:
            job.last_event = "training_finished"

    @staticmethod
    def _pulse_detail(job: TrainingJob) -> str:
        parts = [
            f"job_id={job.job_id}",
            f"elapsed={format_elapsed(time.monotonic() - job.started_at)}",
        ]
        if (step := job.metrics.get("step")) is not None:
            total = job.total_steps if job.total_steps is not None else "?"
            parts.append(f"step={int(step)}/{total}")
        if (loss := job.metrics.get("loss")) is not None:
            parts.append(f"loss={float(loss):.3f}")
        if (grad_norm := job.metrics.get("grad_norm")) is not None:
            parts.append(f"grdn={float(grad_norm):.3f}")
        if (lr := job.metrics.get("lr")) is not None:
            parts.append(f"lr={float(lr):.2e}")
        parts.append(f"lines={len(job.logs)}")
        return " ".join(parts)

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
            section(
                "Training Session",
                f"policy={policy_type} dataset={resolved_root.name} output={output_dir.name}",
                style="bright_magenta",
            )
            info("Training dataset resolved", f"repo_id={repo_id} root={resolved_root}")

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

            print_command("Training command", command)

            job = TrainingJob(
                job_id=str(uuid.uuid4()),
                command=command,
                output_dir=output_dir,
                dataset_root=resolved_root,
                policy_type=policy_type,
                status="running",
            )
            job.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(output_dir.parent),
            )
            job.pulse = Pulse(
                "Training job running",
                detail_factory=lambda: self._pulse_detail(job),
                interval_seconds=5.0,
            ).start()
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
                text = line.rstrip()
                if not text:
                    continue
                job.logs.append(text)
                self._update_job_from_log(job, text)
                stream("TRAIN", text, detail=f"job_id={job.job_id}")
            job.return_code = job.process.wait()
            job.status = "completed" if job.return_code == 0 else "failed"
            if job.return_code != 0:
                job.error = f"Training exited with code {job.return_code}"
                if job.pulse is not None:
                    job.pulse.stop(
                        level="ERROR",
                        message="Training job failed",
                        detail=(
                            f"job_id={job.job_id} elapsed={format_elapsed(time.monotonic() - job.started_at)} "
                            f"code={job.return_code}"
                        ),
                    )
                    job.pulse = None
                section(
                    "Training Failed",
                    f"job_id={job.job_id} output={job.output_dir}",
                    style="red",
                )
                error("Training process exited non-zero", job.error)
                return

            if job.pulse is not None:
                job.pulse.stop(
                    level="OK",
                    message="Training job completed",
                    detail=(
                        f"job_id={job.job_id} elapsed={format_elapsed(time.monotonic() - job.started_at)} "
                        f"output={job.output_dir}"
                    ),
                )
                job.pulse = None
            section(
                "Training Complete",
                f"job_id={job.job_id} output={job.output_dir}",
                style="green",
            )
            ok("Training artifacts ready", f"output={job.output_dir}")
        except Exception as exc:  # pragma: no cover - subprocess specific
            job.status = "failed"
            job.error = str(exc)
            if job.pulse is not None:
                job.pulse.stop(
                    level="ERROR",
                    message="Training job failed",
                    detail=(
                        f"job_id={job.job_id} elapsed={format_elapsed(time.monotonic() - job.started_at)} "
                        f"error={job.error}"
                    ),
                )
                job.pulse = None
            section("Training Failed", f"job_id={job.job_id}", style="red")
            error("Training log collection failed", job.error)

    def status(self) -> dict[str, Any]:
        if self._job is None:
            return {"status": "idle", "logs": [], "progress": 0}
        return self._job.snapshot()
