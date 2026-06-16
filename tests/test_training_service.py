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

from pathlib import Path

from flexivtrainer.config import AppSettings, StorageConfig
from flexivtrainer.jobs.train_policy import TrainingJob, TrainingService


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
