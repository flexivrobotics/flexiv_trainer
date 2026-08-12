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

"""End-to-end wiring for Hub datasets and checkpoints (no network access)."""

import json
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

from flexivtrainer.config import (  # noqa: E402
    AppSettings,
    HubConfig,
    StorageConfig,
)
from flexivtrainer.data.hub import (  # noqa: E402
    ACTION_NAMES_FILENAME,
    HubRef,
    parse_hub_ref,
)
from flexivtrainer.jobs.train_policy import (  # noqa: E402
    TrainingJob,
    TrainingService,
)
from flexivtrainer.rollout.checkpoint import (  # noqa: E402
    checkpoint_action_names,
    resolve_hub_checkpoint,
)


def make_settings(tmp_path: Path, **hub_kwargs) -> AppSettings:
    storage = StorageConfig(root=tmp_path)
    storage.ensure()
    return AppSettings(storage=storage, hub=HubConfig(**hub_kwargs))


def write_dataset_meta(root: Path, action_names: list[str]) -> Path:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [len(action_names)],
                        "names": list(action_names),
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [len(action_names)],
                        "names": list(action_names),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def stub_dataset_fetch(monkeypatch, action_names: list[str]) -> list[HubRef]:
    """Make fetch_dataset_metadata materialize a fake meta/ tree locally."""
    seen: list[HubRef] = []

    def fake_metadata(repo_id, root=None, revision=None, force_cache_sync=False):
        seen.append(HubRef(repo_id, revision))
        write_dataset_meta(Path(root), action_names)
        return object()

    monkeypatch.setattr(
        "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", fake_metadata
    )
    return seen


def stub_checkpoint_fetch(
    monkeypatch,
    *,
    action_names: list[str] | None = None,
    train_config: dict | None = None,
    with_weights: bool = True,
) -> None:
    def fake_snapshot(repo_id, **kwargs):
        local = Path(kwargs["local_dir"])
        local.mkdir(parents=True, exist_ok=True)
        (local / "config.json").write_text(
            json.dumps({"type": "act", "output_features": {"action": {"shape": [26]}}}),
            encoding="utf-8",
        )
        if with_weights:
            (local / "model.safetensors").write_bytes(b"weights")
        if action_names is not None:
            (local / ACTION_NAMES_FILENAME).write_text(
                json.dumps({"action_names": action_names}), encoding="utf-8"
            )
        if train_config is not None:
            (local / "train_config.json").write_text(
                json.dumps(train_config), encoding="utf-8"
            )
        return str(local)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)


class TestTrainingCommandShape:
    def test_hub_dataset_passes_repo_id_and_omits_root(self, tmp_path, monkeypatch):
        settings = make_settings(tmp_path)
        service = TrainingService(settings)
        stub_dataset_fetch(monkeypatch, ["a", "b"])
        ref = parse_hub_ref("lerobot/pusht")
        _, resolved = service._resolve_hub_dataset(ref)

        command = service._build_training_command(
            resolved_root=resolved,
            output_dir=tmp_path / "training" / "run",
            policy_type="diffusion",
            extra_args=[],
            checkpoint_info=None,
            bspline_contract=None,
            hub_ref=ref,
        )

        assert "--dataset.repo_id" in command
        assert command[command.index("--dataset.repo_id") + 1] == "lerobot/pusht"
        # The local cache holds only meta/, so passing it as the root would look
        # like a truncated dataset to LeRobot.
        assert "--dataset.root" not in command

    def test_local_dataset_still_passes_root(self, tmp_path, monkeypatch):
        settings = make_settings(tmp_path)
        service = TrainingService(settings)
        root = write_dataset_meta(tmp_path / "datasets" / "local_ds", ["a", "b"])

        command = service._build_training_command(
            resolved_root=root,
            output_dir=tmp_path / "training" / "run",
            policy_type="diffusion",
            extra_args=[],
            checkpoint_info=None,
            bspline_contract=None,
            hub_ref=None,
        )

        assert command[command.index("--dataset.repo_id") + 1] == "local/local_ds"
        assert "--dataset.root" in command

    def test_revision_is_forwarded(self, tmp_path, monkeypatch):
        settings = make_settings(tmp_path)
        service = TrainingService(settings)
        stub_dataset_fetch(monkeypatch, ["a"])
        ref = parse_hub_ref("lerobot/pusht", "v2.1")
        _, resolved = service._resolve_hub_dataset(ref)

        command = service._build_training_command(
            resolved_root=resolved,
            output_dir=tmp_path / "training" / "run",
            policy_type="diffusion",
            extra_args=[],
            checkpoint_info=None,
            bspline_contract=None,
            hub_ref=ref,
        )
        assert "--dataset.revision=v2.1" in command


class TestResolveHubDataset:
    def test_materializes_inside_hub_cache(self, tmp_path, monkeypatch):
        settings = make_settings(tmp_path)
        service = TrainingService(settings)
        stub_dataset_fetch(monkeypatch, ["a", "b"])

        repo_id, resolved = service._resolve_hub_dataset(parse_hub_ref("acme/demo"))

        assert repo_id == "acme/demo"
        assert resolved.is_relative_to(settings.storage.hub_cache_root.resolve())
        assert (resolved / "meta" / "info.json").is_file()

    def test_directory_name_has_no_separator(self, tmp_path, monkeypatch):
        # resolved_root.name is reused to build the B-spline output directory.
        settings = make_settings(tmp_path)
        service = TrainingService(settings)
        stub_dataset_fetch(monkeypatch, ["a"])
        _, resolved = service._resolve_hub_dataset(parse_hub_ref("acme/demo"))
        assert "/" not in resolved.name


class TestTrainingEnv:
    def test_token_injected_when_configured(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        service = TrainingService(make_settings(tmp_path, token="secret"))
        assert service._training_env()["HF_TOKEN"] == "secret"

    def test_no_token_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        service = TrainingService(make_settings(tmp_path))
        assert "HF_TOKEN" not in service._training_env()


class TestStartValidation:
    def test_hub_source_requires_repo_id(self, tmp_path):
        service = TrainingService(make_settings(tmp_path))
        with pytest.raises(ValueError, match="dataset_repo_id is required"):
            service.start(
                None,
                tmp_path / "training" / "out",
                "diffusion",
                dataset_source="hub",
            )

    def test_local_source_requires_path(self, tmp_path):
        service = TrainingService(make_settings(tmp_path))
        with pytest.raises(ValueError, match="dataset_path is required"):
            service.start(None, tmp_path / "training" / "out", "diffusion")

    def test_unknown_source_rejected(self, tmp_path):
        service = TrainingService(make_settings(tmp_path))
        with pytest.raises(ValueError, match="Unsupported dataset source"):
            service.start(
                None,
                tmp_path / "training" / "out",
                "diffusion",
                dataset_source="ftp",
            )

    def test_bspline_hub_dataset_is_refused(self, tmp_path, monkeypatch):
        # Conversion is a subprocess over parquet files that a meta-only fetch
        # never materializes, so this must fail loudly rather than half-run.
        service = TrainingService(make_settings(tmp_path))
        stub_dataset_fetch(monkeypatch, ["a", "b"])
        with pytest.raises(ValueError, match="cannot convert a Hub dataset"):
            service.start(
                None,
                tmp_path / "training" / "out",
                "bspline_diffusion",
                dataset_source="hub",
                dataset_repo_id="acme/demo",
            )


class TestActionNamesSidecar:
    def test_written_beside_each_checkpoint(self, tmp_path):
        output = tmp_path / "training" / "run"
        model_dir = output / "checkpoints" / "000100" / "pretrained_model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"w")
        job = TrainingJob(
            job_id="j",
            command=["x"],
            output_dir=output,
            dataset_root=tmp_path,
            policy_type="act",
            action_names=["left_arm.tcp_pose.x", "left_arm.gripper.target_width"],
        )

        TrainingService._sync_action_names(job)

        payload = json.loads((model_dir / ACTION_NAMES_FILENAME).read_text())
        assert payload["action_names"] == [
            "left_arm.tcp_pose.x",
            "left_arm.gripper.target_width",
        ]

    def test_noop_without_names(self, tmp_path):
        output = tmp_path / "training" / "run"
        model_dir = output / "checkpoints" / "000100" / "pretrained_model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"w")
        job = TrainingJob(
            job_id="j",
            command=["x"],
            output_dir=output,
            dataset_root=tmp_path,
            policy_type="act",
        )
        TrainingService._sync_action_names(job)
        assert not (model_dir / ACTION_NAMES_FILENAME).exists()

    def test_dataset_action_names_read(self, tmp_path):
        service = TrainingService(make_settings(tmp_path))
        root = write_dataset_meta(tmp_path / "datasets" / "d", ["x", "y", "z"])
        assert service._dataset_action_names(root) == ["x", "y", "z"]

    def test_duplicate_names_rejected(self, tmp_path):
        service = TrainingService(make_settings(tmp_path))
        root = write_dataset_meta(tmp_path / "datasets" / "d", ["x", "x"])
        assert service._dataset_action_names(root) is None


class TestHubCheckpointRollout:
    def test_resolves_through_existing_path_validator(self, tmp_path, monkeypatch):
        # The cache is inside the storage root, so the unmodified
        # resolve_checkpoint_path accepts it and still guards the content.
        settings = make_settings(tmp_path)
        stub_checkpoint_fetch(monkeypatch)
        resolved = resolve_hub_checkpoint("acme/policy", None, settings)
        assert resolved.is_relative_to(settings.storage.root.resolve())
        assert (resolved / "config.json").is_file()

    def test_sidecar_recovers_gripper_action_names(self, tmp_path, monkeypatch):
        # Width 20 is gripper-bearing, so canonical inference would raise; the
        # sidecar is what makes a Hub checkpoint usable.
        settings = make_settings(tmp_path)
        names = [f"axis_{i}" for i in range(20)]
        stub_checkpoint_fetch(monkeypatch, action_names=names)
        resolved = resolve_hub_checkpoint("acme/policy", None, settings)
        assert (
            checkpoint_action_names(
                str(resolved), settings.storage.root, settings=settings
            )
            == names
        )

    def test_override_takes_priority(self, tmp_path, monkeypatch):
        settings = make_settings(tmp_path)
        stub_checkpoint_fetch(monkeypatch, action_names=["a", "b"])
        resolved = resolve_hub_checkpoint("acme/policy", None, settings)
        assert checkpoint_action_names(
            str(resolved),
            settings.storage.root,
            settings=settings,
            override=["c", "d"],
        ) == ["c", "d"]

    def test_falls_back_to_hub_training_dataset(self, tmp_path, monkeypatch):
        settings = make_settings(tmp_path)
        names = [f"axis_{i}" for i in range(20)]
        stub_checkpoint_fetch(
            monkeypatch,
            train_config={"dataset": {"repo_id": "acme/train-data"}},
        )
        stub_dataset_fetch(monkeypatch, names)
        resolved = resolve_hub_checkpoint("acme/policy", None, settings)
        assert (
            checkpoint_action_names(
                str(resolved), settings.storage.root, settings=settings
            )
            == names
        )

    def test_synthetic_local_repo_id_is_not_fetched(self, tmp_path, monkeypatch):
        settings = make_settings(tmp_path)
        stub_checkpoint_fetch(
            monkeypatch,
            train_config={"dataset": {"repo_id": "local/actions"}},
        )
        seen = stub_dataset_fetch(monkeypatch, ["a"])
        resolved = resolve_hub_checkpoint("acme/policy", None, settings)
        assert (
            checkpoint_action_names(
                str(resolved), settings.storage.root, settings=settings
            )
            is None
        )
        assert seen == [], "local/* must never be treated as a Hub repo"

    def test_returns_none_when_every_tier_misses(self, tmp_path, monkeypatch):
        settings = make_settings(tmp_path)
        stub_checkpoint_fetch(monkeypatch)
        resolved = resolve_hub_checkpoint("acme/policy", None, settings)
        assert (
            checkpoint_action_names(
                str(resolved), settings.storage.root, settings=settings
            )
            is None
        )


class TestRequestModelValidation:
    """The discriminator is what keeps repo ids and paths on separate rails."""

    def test_training_legacy_local_request_still_valid(self):
        from flexivtrainer.api.routes.training import StartTrainingRequest

        request = StartTrainingRequest(
            dataset_path="/data/ds", output_dir="/out"
        )
        assert request.dataset_source == "local"
        assert request.dataset_repo_id is None

    def test_training_hub_requires_repo_id(self):
        from flexivtrainer.api.routes.training import StartTrainingRequest

        with pytest.raises(ValueError, match="dataset_repo_id is required"):
            StartTrainingRequest(dataset_source="hub", output_dir="/out")

    def test_training_rejects_both_path_and_repo_id(self):
        from flexivtrainer.api.routes.training import StartTrainingRequest

        with pytest.raises(ValueError, match="dataset_path is not allowed"):
            StartTrainingRequest(
                dataset_source="hub",
                dataset_repo_id="acme/demo",
                dataset_path="/data/ds",
                output_dir="/out",
            )

    def test_training_local_requires_path(self):
        from flexivtrainer.api.routes.training import StartTrainingRequest

        with pytest.raises(ValueError, match="dataset_path is required"):
            StartTrainingRequest(output_dir="/out")

    def test_training_hub_checkpoint_requires_repo_id(self):
        from flexivtrainer.api.routes.training import StartTrainingRequest

        with pytest.raises(ValueError, match="checkpoint_repo_id is required"):
            StartTrainingRequest(
                dataset_path="/data/ds",
                output_dir="/out",
                training_mode="fine_tune",
                checkpoint_source="hub",
            )

    def test_rollout_legacy_local_request_still_valid(self):
        from flexivtrainer.api.routes.rollout import StartRolloutRequest

        request = StartRolloutRequest(checkpoint_path="/ckpt")
        assert request.source == "local"

    def test_rollout_hub_requires_repo_id(self):
        from flexivtrainer.api.routes.rollout import StartRolloutRequest

        with pytest.raises(ValueError, match="repo_id is required"):
            StartRolloutRequest(source="hub")

    def test_rollout_rejects_both_path_and_repo_id(self):
        from flexivtrainer.api.routes.rollout import StartRolloutRequest

        with pytest.raises(ValueError, match="checkpoint_path is not allowed"):
            StartRolloutRequest(
                source="hub", repo_id="acme/policy", checkpoint_path="/ckpt"
            )

    def test_rollout_local_requires_path(self):
        from flexivtrainer.api.routes.rollout import StartRolloutRequest

        with pytest.raises(ValueError, match="checkpoint_path is required"):
            StartRolloutRequest()


class TestHubFineTuneCheckpoint:
    def test_requires_weights(self, tmp_path, monkeypatch):
        service = TrainingService(make_settings(tmp_path))
        stub_checkpoint_fetch(monkeypatch, with_weights=False)
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            service._resolve_hub_checkpoint(parse_hub_ref("acme/policy"))

    def test_resolves_when_complete(self, tmp_path, monkeypatch):
        service = TrainingService(make_settings(tmp_path))
        stub_checkpoint_fetch(monkeypatch)
        checkpoint_dir, model_dir = service._resolve_hub_checkpoint(
            parse_hub_ref("acme/policy")
        )
        assert (model_dir / "config.json").is_file()
        assert (model_dir / "model.safetensors").is_file()
        assert checkpoint_dir == model_dir
