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

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

# psutil arrives transitively via lerobot, which is not installed in the
# deps-light CI env. Skip this module rather than fail collection when it's gone.
psutil = pytest.importorskip("psutil")

from flexivtrainer.config import AppSettings, StorageConfig  # noqa: E402
from flexivtrainer.jobs import train_policy  # noqa: E402
from flexivtrainer.jobs.train_policy import TrainingJob, TrainingService  # noqa: E402


def make_service(tmp_path: Path) -> TrainingService:
    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    return TrainingService(settings)


def make_job(tmp_path: Path) -> TrainingJob:
    return TrainingJob(
        job_id="job-1",
        command=["lerobot-train"],
        output_dir=tmp_path / "output",
        dataset_root=tmp_path / "dataset",
        policy_type="diffusion",
    )


def make_dataset(tmp_path: Path, *, state_size: int = 7) -> Path:
    root = tmp_path / "datasets" / "fine_tune_data"
    (root / "meta").mkdir(parents=True)
    info = {
        "features": {
            "observation.images.ego": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_size],
                "names": [f"joint_{i}" for i in range(state_size)],
            },
            "action": {
                "dtype": "float32",
                "shape": [state_size],
                "names": [f"joint_{i}" for i in range(state_size)],
            },
        }
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


def make_checkpoint(tmp_path: Path, *, policy_type: str = "diffusion") -> Path:
    step = tmp_path / "training" / "source_run" / "checkpoints" / "005000"
    model = step / "pretrained_model"
    model.mkdir(parents=True)
    config = {
        "type": policy_type,
        "input_features": {
            "observation.images.ego": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [7]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "optimizer_lr": 3e-5,
        "vision_encoder_lr_multiplier": 0.2,
        "freeze_vision_encoder": True,
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    return step


class _FakeProcess:
    stdout: list[str] = []
    returncode = None
    pid = 12345

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None


class _NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


class _FakePulse:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return self

    def stop(self, *args, **kwargs):
        pass


def test_parse_compact_number_supports_suffixes_and_floats(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    assert service._parse_compact_number("100K") == 100_000
    assert service._parse_compact_number("2.5M") == 2_500_000
    assert service._parse_compact_number("1.0e-04") == 1.0e-04


def test_update_job_from_log_parses_common_lerobot_lines(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    job = make_job(tmp_path)

    lines = [
        "cfg.steps=1000 (1K)",
        "dataset.num_frames=50000 (50K)",
        "dataset.num_episodes=125",
        "Start offline training on a fixed dataset, with effective batch size: 64",
        "step:500 smpl:32000 ep:10 epch:0.5 loss:0.234 grdn:1.111 lr:1.0e-04 updt_s:0.120 data_s:0.030",
        "Checkpoint policy after step 500",
        "Eval policy at step 500",
        "End of training",
    ]

    for line in lines:
        service._update_job_from_log(job, line)

    snapshot = job.snapshot()

    assert snapshot["total_steps"] == 1000
    assert snapshot["dataset_num_frames"] == 50_000
    assert snapshot["dataset_num_episodes"] == 125
    assert snapshot["effective_batch_size"] == 64
    assert snapshot["last_checkpoint_step"] == 500
    assert snapshot["last_eval_step"] == 500
    assert snapshot["last_event"] == "training_finished"
    assert snapshot["metrics"] == {
        "step": 500,
        "samples": 32_000,
        "episodes": 10,
        "epochs": 0.5,
        "loss": 0.234,
        "grad_norm": 1.111,
        "lr": 1.0e-04,
        "update_seconds": 0.12,
        "data_seconds": 0.03,
    }
    assert snapshot["progress"] == 50


def test_pause_resume_without_running_job_raises(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with pytest.raises(RuntimeError):
        service.pause()
    with pytest.raises(RuntimeError):
        service.resume()


def test_pause_resume_suspends_and_continues_process(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        job = make_job(tmp_path)
        job.process = proc
        job.status = "running"
        service._job = job

        snap = service.pause()
        assert snap["paused"] is True
        time.sleep(0.3)
        assert psutil.Process(proc.pid).status() == psutil.STATUS_STOPPED

        # Pausing again is idempotent.
        assert service.pause()["paused"] is True

        snap = service.resume()
        assert snap["paused"] is False
        time.sleep(0.3)
        assert psutil.Process(proc.pid).status() != psutil.STATUS_STOPPED
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_evaluate_devices_caching(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    # The first call should probe and cache.
    res1 = service.evaluate_devices()
    assert "devices" in res1
    assert len(res1["devices"]) > 0
    # The cache should be populated now.
    assert service._device_probe is not None
    cached_probe = service._device_probe

    # The second call should return cached results.
    res2 = service.evaluate_devices()
    assert res2["devices"][1:] == cached_probe


def test_inspect_checkpoint_accepts_step_and_model_dirs(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    step = make_checkpoint(tmp_path)

    from_step = service.inspect_checkpoint(step)
    from_model = service.inspect_checkpoint(step / "pretrained_model")

    assert from_step["checkpoint_path"] == str(step.resolve())
    assert from_step["model_path"] == str((step / "pretrained_model").resolve())
    assert from_model["checkpoint_path"] == from_step["checkpoint_path"]
    assert from_step["policy_type"] == "diffusion"
    fields = {field["name"]: field for field in from_step["fields"]}
    assert set(fields) == {
        "batch_size",
        "epochs",
        "save_freq",
        "log_freq",
        "num_workers",
        "optimizer_lr",
    }
    assert fields["optimizer_lr"]["default"] == 3e-5


@pytest.mark.parametrize(
    ("policy_type", "policy_fields"),
    [
        ("diffusion", {"optimizer_lr"}),
        ("act", {"optimizer_lr"}),
        ("smolvla", {"optimizer_lr", "freeze_vision_encoder"}),
        ("pi0", {"optimizer_lr"}),
        (
            "multi_task_dit",
            {"optimizer_lr", "vision_encoder_lr_multiplier"},
        ),
    ],
)
def test_fine_tune_schema_for_supported_policy_families(
    tmp_path: Path, policy_type: str, policy_fields: set[str]
) -> None:
    service = make_service(tmp_path)
    checkpoint = make_checkpoint(tmp_path, policy_type=policy_type)

    info = service.inspect_checkpoint(checkpoint)

    names = {field["name"] for field in info["fields"]}
    assert names == _FINE_TUNE_COMMON_NAMES | policy_fields


_FINE_TUNE_COMMON_NAMES = {
    "batch_size",
    "epochs",
    "save_freq",
    "log_freq",
    "num_workers",
}


def test_inspect_checkpoint_rejects_unsupported_and_escaped_paths(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    step = make_checkpoint(tmp_path, policy_type="unknown_policy")
    with pytest.raises(ValueError, match="Unsupported checkpoint policy"):
        service.inspect_checkpoint(step)

    outside = tmp_path.parent / f"{tmp_path.name}-outside-checkpoint"
    outside.mkdir()
    with pytest.raises(ValueError, match="Access denied"):
        service.inspect_checkpoint(outside)


def test_inspect_checkpoint_requires_config_and_weights(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    step = tmp_path / "training" / "run" / "checkpoints" / "001000"
    step.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="config.json"):
        service.inspect_checkpoint(step)

    model = step / "pretrained_model"
    model.mkdir()
    (model / "config.json").write_text('{"type": "act"}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        service.inspect_checkpoint(step)


def test_training_dataset_and_output_paths_are_restricted(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    dataset = make_dataset(tmp_path)
    assert service._resolve_dataset(dataset)[1] == dataset.resolve()

    with pytest.raises(ValueError, match="Access denied"):
        service._resolve_dataset(tmp_path / "episodes" / "recording")
    with pytest.raises(ValueError, match="Access denied"):
        service._resolve_output_dir(tmp_path.parent / "outside-training")


def test_inspect_checkpoint_rejects_symlink_escape(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    outside_root = tmp_path.parent / f"{tmp_path.name}-outside-training"
    outside_step = make_checkpoint(outside_root)
    training_root = tmp_path / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    link = training_root / "escaped"
    try:
        link.symlink_to(outside_step, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="Access denied"):
        service.inspect_checkpoint(link)


def test_fine_tune_dataset_feature_validation(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    checkpoint = service.inspect_checkpoint(make_checkpoint(tmp_path))
    compatible = make_dataset(tmp_path)

    service._validate_checkpoint_dataset(checkpoint, compatible)

    incompatible = make_dataset(tmp_path / "bad", state_size=8)
    with pytest.raises(ValueError, match="observation.state"):
        service._validate_checkpoint_dataset(checkpoint, incompatible)

    camera_mismatch = make_dataset(tmp_path / "camera")
    info_path = camera_mismatch / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.ego"]["shape"] = [240, 320, 3]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="observation.images.ego"):
        service._validate_checkpoint_dataset(checkpoint, camera_mismatch)

    missing_action = make_dataset(tmp_path / "missing")
    info_path = missing_action / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    del info["features"]["action"]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="missing action"):
        service._validate_checkpoint_dataset(checkpoint, missing_action)


def test_fine_tune_extra_args_are_whitelisted_and_normalized(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    checkpoint = service.inspect_checkpoint(make_checkpoint(tmp_path))

    args = service._fine_tune_extra_args(
        ["--batch_size", "32", "--policy.optimizer_lr", "1e-5"],
        checkpoint["fields"],
    )

    assert args == ["--batch_size", "32", "--policy.optimizer_lr=1e-5"]
    with pytest.raises(ValueError, match="not allowed"):
        service._fine_tune_extra_args(["--policy.horizon", "16"], checkpoint["fields"])


def test_start_fine_tune_builds_new_pretrained_run_command(
    tmp_path: Path, monkeypatch
) -> None:
    service = make_service(tmp_path)
    dataset = make_dataset(tmp_path)
    checkpoint = make_checkpoint(tmp_path)
    output = tmp_path / "training" / "fine_tuned_run"
    monkeypatch.setattr(train_policy.shutil, "which", lambda _: "/bin/lerobot-train")
    monkeypatch.setattr(
        train_policy.subprocess, "Popen", lambda *a, **kw: _FakeProcess()
    )
    monkeypatch.setattr(train_policy.threading, "Thread", _NoopThread)
    monkeypatch.setattr(train_policy, "Pulse", _FakePulse)
    monkeypatch.setattr(train_policy, "resolve_training_device", lambda _: "cpu")

    snapshot = service.start(
        dataset_root=dataset,
        output_dir=output,
        policy_type="act",  # ignored in favor of checkpoint metadata
        training_mode="fine_tune",
        checkpoint_path=checkpoint,
        extra_args=["--batch_size", "32", "--policy.optimizer_lr", "1e-5"],
    )

    command = snapshot["command"]
    assert snapshot["training_mode"] == "fine_tune"
    assert snapshot["policy_type"] == "diffusion"
    assert snapshot["checkpoint_path"] == str(checkpoint.resolve())
    assert f"--policy.path={(checkpoint / 'pretrained_model').resolve()}" in command
    assert "--policy.device=cpu" in command
    assert "--policy.push_to_hub=false" in command
    assert "--policy.optimizer_lr=1e-5" in command
    assert "--policy.type" not in command
    assert "--resume" not in command
    assert command[command.index("--dataset.root") + 1] == str(dataset.resolve())
    assert command[command.index("--output_dir") + 1] == str(output.resolve())


def test_start_new_policy_keeps_existing_policy_arguments(
    tmp_path: Path, monkeypatch
) -> None:
    service = make_service(tmp_path)
    dataset = make_dataset(tmp_path)
    output = tmp_path / "training" / "new_run"
    monkeypatch.setattr(train_policy.shutil, "which", lambda _: "/bin/lerobot-train")
    monkeypatch.setattr(
        train_policy.subprocess, "Popen", lambda *a, **kw: _FakeProcess()
    )
    monkeypatch.setattr(train_policy.threading, "Thread", _NoopThread)
    monkeypatch.setattr(train_policy, "Pulse", _FakePulse)
    monkeypatch.setattr(train_policy, "resolve_training_device", lambda _: "cpu")

    snapshot = service.start(
        dataset_root=dataset,
        output_dir=output,
        policy_type="act",
        extra_args=["--batch_size", "16"],
    )

    command = snapshot["command"]
    assert snapshot["training_mode"] == "new"
    assert snapshot["checkpoint_path"] is None
    assert command[command.index("--policy.type") + 1] == "act"
    assert command[command.index("--policy.device") + 1] == "cpu"
    assert "--policy.path" not in " ".join(command)
