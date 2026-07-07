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
    warn,
)
from flexivtrainer.policies import training_field_schema

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

TRAINING_DEVICE_ORDER = ("auto", "cuda", "mps", "cpu")

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
UI_LOG_PREFIX = "@@TRAIN_LOG@@"


def _stream_level(text: str) -> str:
    lowered = text.lower()
    if any(
        token in lowered
        for token in ("traceback", "exception", " fatal", "failed", "error")
    ):
        return "ERROR"
    if any(token in lowered for token in ("warning", "warn", "deprecated")):
        return "WARN"
    return "INFO"


def _encode_ui_log(
    level: str,
    source: str,
    message: str,
    detail: str | None = None,
) -> str:
    payload = {
        "level": level,
        "source": source,
        "message": message,
        "detail": detail or "",
    }
    return f"{UI_LOG_PREFIX}{json.dumps(payload, separators=(',', ':'))}"


def resolve_training_device(configured: str) -> str:
    """Concrete --policy.device value for lerobot.

    ``"auto"`` (or empty) detects the best available device on this machine so
    the trainer is portable across platforms; an explicit value is passed
    through unchanged. Passing a concrete device also avoids lerobot's
    "Device 'None' is not available. Switching to ..." auto-detect log line.
    """
    if configured and configured.lower() != "auto":
        return configured
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # torch missing/unimportable -> let CPU handle it
        pass
    return "cpu"


def _set_process_tree_suspended(pid: int, suspend: bool) -> None:
    """Suspend (SIGSTOP) or resume (SIGCONT) a process and all its children.

    Uses psutil so it works on Linux/macOS/Windows. The whole tree is covered so
    lerobot's dataloader worker processes are frozen too, not just the trainer's
    main process. Missing/exited processes are skipped.
    """
    import psutil

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for proc in [root, *root.children(recursive=True)]:
        try:
            proc.suspend() if suspend else proc.resume()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


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
    paused: bool = False
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
            "paused": self.paused,
        }


class TrainingService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._job: TrainingJob | None = None
        self._lock = threading.Lock()
        # Availability of cuda/mps/cpu is fixed for the process lifetime, but
        # probing it the first time pays a one-off ``import torch`` plus CUDA
        # context init that can take tens of seconds on a cold start. Cache the
        # probe so the first UI request doesn't hang, and let startup warm it up
        # in the background (see ``warm_up_devices``).
        self._device_probe: list[dict[str, Any]] | None = None
        self._device_probe_lock = threading.Lock()

    def list_policies(self) -> dict[str, Any]:
        # Attach each policy's UI form schema (derived from its TrainingConfig) so
        # the frontend can render config inputs without duplicating defaults.
        policies = {
            key: {**entry, "fields": training_field_schema(key)}
            for key, entry in POLICY_CATALOG.items()
        }
        return {
            "default": self._settings.training.default_policy,
            "policies": policies,
            "device": self._settings.training.default_device,
        }

    def _probe_devices(self) -> list[dict[str, Any]]:
        """Probe cuda/mps/cpu availability (the slow, ``import torch`` part).

        This is the expensive half of device evaluation: the first ``import
        torch`` plus CUDA initialization. The result is cached by
        :meth:`_get_device_probe` because availability cannot change during the
        process lifetime.
        """
        results: list[dict[str, Any]] = []

        try:
            import torch
        except Exception as exc:
            torch = None
            import_error = str(exc).strip() or type(exc).__name__
        else:
            import_error = ""

        for device in TRAINING_DEVICE_ORDER:
            if device == "auto":
                # Resolved against current settings in evaluate_devices().
                continue

            available = False
            detail = ""
            try:
                if torch is None:
                    raise RuntimeError(import_error or "PyTorch is unavailable")

                if device == "cuda":
                    available = bool(torch.cuda.is_available())
                    if available:
                        tensor = torch.zeros(1, device="cuda")
                        detail = torch.cuda.get_device_name(tensor.device)
                    else:
                        detail = "CUDA is not available"
                elif device == "mps":
                    backend = getattr(torch.backends, "mps", None)
                    available = bool(backend and backend.is_available())
                    if available:
                        torch.zeros(1, device="mps")
                        detail = "Apple Metal backend is available"
                    else:
                        detail = "MPS is not available"
                elif device == "cpu":
                    torch.zeros(1, device="cpu")
                    available = True
                    detail = "CPU is available"
            except Exception as exc:
                available = False
                detail = str(exc).strip() or type(exc).__name__

            results.append(
                {
                    "name": device,
                    "available": available,
                    "detail": detail,
                }
            )

        return results

    def _get_device_probe(self, *, force: bool = False) -> list[dict[str, Any]]:
        with self._device_probe_lock:
            if self._device_probe is None or force:
                self._device_probe = self._probe_devices()
            return self._device_probe

    def warm_up_devices(self) -> None:
        """Pre-compute the device probe so the first UI request is instant.

        Intended to run in a background thread at server startup. The heavy
        ``import torch`` and CUDA init happen here instead of inside the first
        ``GET /training/devices`` request, which previously left the device list
        empty for tens of seconds.
        """
        try:
            self._get_device_probe()
        except Exception as exc:  # never let warmup crash startup
            warn(
                "Training device warmup failed",
                str(exc).strip() or type(exc).__name__,
            )

    def evaluate_devices(self, *, force: bool = False) -> dict[str, Any]:
        probe = self._get_device_probe(force=force)
        # Resolve "auto" after the probe so torch/CUDA are already warm.
        resolved_auto = resolve_training_device(self._settings.training.default_device)
        devices = [
            {
                "name": "auto",
                "available": True,
                "resolved": resolved_auto,
                "detail": f"Resolves to {resolved_auto}",
            },
            *probe,
        ]
        return {
            "configured": self._settings.training.default_device,
            "resolved": resolved_auto,
            "devices": devices,
        }

    def set_default_device(self, device: str) -> dict[str, Any]:
        normalized = (device or "auto").strip().lower()
        if normalized not in TRAINING_DEVICE_ORDER:
            raise RuntimeError(f"Unsupported training device: {device}")
        self._settings.training.default_device = normalized
        return self.evaluate_devices()

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
    def _append_ui_log(
        job: TrainingJob,
        level: str,
        source: str,
        message: str,
        detail: str | None = None,
    ) -> None:
        job.logs.append(_encode_ui_log(level, source, message, detail))

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
        # Recordings and merged datasets are standard LeRobot datasets whose
        # repo id is local/<name> (what the recorder/merge write).
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
            # lerobot-train creates output_dir itself and refuses to run if it
            # already exists (resume is False), so only ensure the parent here.
            output_dir.parent.mkdir(parents=True, exist_ok=True)
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
                    # Concrete device resolved for this machine (cuda/mps/cpu),
                    # so lerobot uses the GPU without its "Device None" auto-
                    # detect line and stays portable across platforms.
                    "--policy.device",
                    resolve_training_device(self._settings.training.default_device),
                    # Local trainer: never push checkpoints to the HF Hub.
                    # Without this, lerobot requires a policy.repo_id and aborts.
                    "--policy.push_to_hub",
                    "false",
                    "--output_dir",
                    str(output_dir),
                    "--job_name",
                    output_dir.name,
                    "--save_freq",
                    str(self._settings.training.save_frequency),
                ]
            )
            # Per-policy knobs (incl. the diffusion sampler baked into the
            # checkpoint) come from the Web UI form as --policy.* flags via
            # extra_args; training_field_schema() is the single source of the
            # form's fields, flags and defaults.
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
                self._append_ui_log(
                    job,
                    _stream_level(text),
                    "TRAIN",
                    text,
                    f"job_id={job.job_id}",
                )
                self._update_job_from_log(job, text)
                stream("TRAIN", text, detail=f"job_id={job.job_id}")
            job.return_code = job.process.wait()
            if job.status == "stopped":
                return
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
                self._append_ui_log(
                    job,
                    "ERROR",
                    "TRAIN",
                    "Training process exited non-zero",
                    job.error,
                )
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
            self._append_ui_log(
                job,
                "OK",
                "TRAIN",
                "Training artifacts ready",
                f"output={job.output_dir}",
            )
        except Exception as exc:  # pragma: no cover - subprocess specific
            if job.status == "stopped":
                return
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
            self._append_ui_log(
                job,
                "ERROR",
                "TRAIN",
                "Training log collection failed",
                job.error,
            )

    def shutdown(self) -> None:
        with self._lock:
            job = self._job

        if job is None:
            return

        if job.process is not None and job.process.poll() is None:
            info("Stopping training process", f"job_id={job.job_id}")
            job.status = "stopped"
            job.error = "Server shutdown"
            # Resume first if paused — a SIGSTOP'd process won't act on SIGTERM.
            if job.paused:
                _set_process_tree_suspended(job.process.pid, suspend=False)
                job.paused = False
            job.process.terminate()
            try:
                job.process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                job.process.kill()
                job.process.wait(timeout=1.0)
            job.return_code = job.process.returncode

        if job.pulse is not None:
            job.pulse.stop(
                level="WARN",
                message="Training job stopped",
                detail=f"job_id={job.job_id} reason=server shutdown",
            )
            job.pulse = None

    def pause(self) -> dict[str, Any]:
        """Suspend the running training process (and its children)."""
        with self._lock:
            job = self._job
            if job is None or job.status != "running" or job.process is None:
                raise RuntimeError("No running training job to pause")
            if not job.paused:
                _set_process_tree_suspended(job.process.pid, suspend=True)
                job.paused = True
                info("Training job paused", f"job_id={job.job_id}")
                self._append_ui_log(
                    job,
                    "OK",
                    "TRAIN",
                    "Training job paused",
                    f"job_id={job.job_id}",
                )
            return job.snapshot()

    def resume(self) -> dict[str, Any]:
        """Resume a previously paused training process."""
        with self._lock:
            job = self._job
            if job is None or job.status != "running" or job.process is None:
                raise RuntimeError("No training job to resume")
            if job.paused:
                _set_process_tree_suspended(job.process.pid, suspend=False)
                job.paused = False
                info("Training job resumed", f"job_id={job.job_id}")
                self._append_ui_log(
                    job,
                    "OK",
                    "TRAIN",
                    "Training job resumed",
                    f"job_id={job.job_id}",
                )
            return job.snapshot()

    def stop(self) -> dict[str, Any]:
        """Terminate the running training process and keep its final snapshot."""
        with self._lock:
            job = self._job
            if job is None or job.status != "running" or job.process is None:
                raise RuntimeError("No running training job to stop")

            info("Stopping training process", f"job_id={job.job_id}")
            job.status = "stopped"
            job.error = None
            job.last_event = "training_stopped"
            if job.paused:
                _set_process_tree_suspended(job.process.pid, suspend=False)
                job.paused = False
            job.process.terminate()
            try:
                job.process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                job.process.kill()
                job.process.wait(timeout=1.0)
            job.return_code = job.process.returncode

            if job.pulse is not None:
                job.pulse.stop(
                    level="WARN",
                    message="Training job stopped",
                    detail=(
                        f"job_id={job.job_id} elapsed={format_elapsed(time.monotonic() - job.started_at)} "
                        "reason=user request"
                    ),
                )
                job.pulse = None
            self._append_ui_log(
                job,
                "WARN",
                "TRAIN",
                "Training job stopped",
                "reason=user request",
            )
            self._append_ui_log(
                job,
                "OK",
                "TRAIN",
                "Training job stopped",
                f"job_id={job.job_id}",
            )
            return job.snapshot()

    def status(self) -> dict[str, Any]:
        if self._job is None:
            return {"status": "idle", "logs": [], "progress": 0}
        return self._job.snapshot()
