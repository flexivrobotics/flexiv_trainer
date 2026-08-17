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


class TestLayoutWarning:
    """The warning must state a real blocker, not a hypothetical about grippers."""

    def test_silent_when_names_are_known(self):
        from flexivtrainer.api.routes.rollout import _layout_ok, _layout_warning
        from flexivtrainer.rollout.executors.waypoint import canonical_action_names

        # Width 14 is not canonical, but recorded names are not bound to that.
        names = canonical_action_names(13, ["single_arm"]) + [
            "single_arm.gripper.target_width"
        ]
        assert _layout_warning(names, 14, ["single_arm"]) is None

        message = _layout_ok(names, 14, ["single_arm"])
        assert message is not None
        assert "Policy loaded successfully" in message
        assert "canonical" not in message

    def test_silent_when_width_is_inferable(self):
        from flexivtrainer.api.routes.rollout import _layout_warning

        assert _layout_warning(None, 13, ["single_arm"]) is None
        assert _layout_warning(None, 26, ["left_arm", "right_arm"]) is None

    def test_confirms_inferable_width(self):
        from flexivtrainer.api.routes.rollout import _layout_ok

        message = _layout_ok(None, 26, ["left_arm", "right_arm"])
        assert message is not None
        assert "action width is 26" in message
        assert "for 2 arms are 26 or 38" in message
        assert "Arm mode matches: dual-arm" in message

        assert _layout_ok(["a"] * 26, 26, ["left_arm", "right_arm"]) is None
        assert _layout_ok(None, 26, ["single_arm"]) is None

    def test_warns_on_recorded_arm_count_mismatch(self):
        from flexivtrainer.api.routes.rollout import _layout_ok, _layout_warning
        from flexivtrainer.rollout.executors.waypoint import canonical_action_names

        dual = canonical_action_names(26, ["left_arm", "right_arm"])
        message = _layout_warning(dual, 26, ["single_arm"])
        assert message is not None
        assert "Arm mode mismatch: set dual-arm mode" in message
        assert _layout_ok(dual, 26, ["single_arm"]) is None

    def test_warns_on_arm_count_mismatch(self):
        from flexivtrainer.api.routes.rollout import _layout_warning

        # A dual-arm policy under single-arm config: the real cause is the arm
        # count, so the message must name it rather than blaming a gripper.
        message = _layout_warning(None, 26, ["single_arm"])
        assert message is not None
        assert "action width is 26" in message
        assert "for 1 arm are 13 or 19" in message
        assert "Arm mode mismatch: set dual-arm mode" in message
        assert "gripper" not in message.lower()

    def test_warns_on_unknown_gripper_width(self):
        from flexivtrainer.api.routes.rollout import _layout_warning

        message = _layout_warning(None, 15, ["single_arm"])
        assert message is not None
        assert "action_names" in message

    def test_tolerates_missing_inputs(self):
        from flexivtrainer.api.routes.rollout import _layout_warning

        assert _layout_warning(None, None, ["single_arm"]) is None
        assert _layout_warning(None, 26, []) is None


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


class TestSessionHubToken:
    """An operator-supplied token lives in memory only, and outranks the rest."""

    def teardown_method(self):
        from flexivtrainer.data.hub import set_session_token

        set_session_token(None)

    def test_session_token_overrides_configured_and_env(self, monkeypatch):
        from flexivtrainer.config import AppSettings
        from flexivtrainer.data.hub import (
            has_session_token,
            hub_token,
            set_session_token,
        )

        settings = AppSettings()
        settings.hub.token = "configured-token"
        monkeypatch.setenv("HF_TOKEN", "env-token")

        assert hub_token(settings) == "configured-token"
        assert has_session_token() is False

        # A stale configured token must not beat the operator's typed one.
        set_session_token("  typed-token  ")
        assert has_session_token() is True
        assert hub_token(settings) == "typed-token"

        set_session_token("")
        assert has_session_token() is False
        assert hub_token(settings) == "configured-token"

    def test_session_token_is_never_persisted(self):
        from flexivtrainer.config import AppSettings
        from flexivtrainer.data.hub import set_session_token

        settings = AppSettings()
        set_session_token("secret")
        # Lives beside the settings object, so dumping config cannot leak it.
        assert settings.hub.token is None
        assert "secret" not in settings.model_dump_json()

    def test_session_token_reaches_the_training_subprocess(self, monkeypatch):
        from flexivtrainer.config import AppSettings
        from flexivtrainer.data.hub import set_session_token
        from flexivtrainer.jobs.train_policy import TrainingService

        # The subprocess downloads the bulk dataset, so a stale HF_TOKEN there
        # would fail the run even after the operator supplied a working one.
        monkeypatch.setenv("HF_TOKEN", "stale-env-token")
        service = TrainingService(AppSettings())
        assert service._training_env()["HF_TOKEN"] == "stale-env-token"

        set_session_token("typed-token")
        assert service._training_env()["HF_TOKEN"] == "typed-token"


def _pose_twist_names(prefix: str = "single_arm") -> list[str]:
    return [
        f"{prefix}.tcp_pose.{axis}"
        for axis in ("x", "y", "z", "q_w", "q_x", "q_y", "q_z")
    ] + [f"{prefix}.tcp_twist.{axis}" for axis in ("vx", "vy", "vz", "wx", "wy", "wz")]


class TestInspectHubDataset:
    """Load-time pre-flight: same checks and same words as Start."""

    def test_reports_facts_and_a_confirmation(self, tmp_path, monkeypatch):
        service = TrainingService(make_settings(tmp_path))
        stub_dataset_fetch(monkeypatch, _pose_twist_names())

        result = service.inspect_hub_dataset("acme/demo")

        assert result["repo_id"] == "acme/demo"
        assert result["action_dim"] == 13
        assert result["fps"] == 30
        assert result["dataset_warning"] is None
        assert "loaded successfully" in result["dataset_ok"]

    def test_tolerates_missing_counts(self, tmp_path, monkeypatch):
        # The fixture writes no total_episodes/total_frames; reporting 0 would
        # read as an empty dataset rather than as absent metadata.
        service = TrainingService(make_settings(tmp_path))
        stub_dataset_fetch(monkeypatch, _pose_twist_names())

        result = service.inspect_hub_dataset("acme/demo")

        assert result["num_episodes"] is None
        assert result["num_frames"] is None
        assert "0 episodes" not in result["dataset_ok"]
        assert "records no episode or frame counts" in result["dataset_ok"]

    def test_missing_gripper_metadata_is_trainable_but_noted(
        self, tmp_path, monkeypatch
    ):
        # Training only feeds tensors to a policy, so an unusable gripper schema
        # must not block it. Rollout refuses such a checkpoint on its own.
        service = TrainingService(make_settings(tmp_path))
        names = [*_pose_twist_names(), "single_arm.gripper.target_width"]
        stub_dataset_fetch(monkeypatch, names)

        result = service.inspect_hub_dataset("acme/demo")

        assert result["dataset_warning"] is None
        assert "gripper command metadata" in result["dataset_ok"]
        assert "cannot be rolled out" in result["dataset_ok"]
        # The Hub cache directory is an opaque digest; it must not reach the UI.
        assert str(tmp_path) not in result["dataset_ok"]

    def test_unnamed_action_axes_are_trainable(self, tmp_path, monkeypatch):
        # LeRobot's grouped {"motors": [...]} form: no arm layout needed.
        service = TrainingService(make_settings(tmp_path))
        stub_dataset_fetch(monkeypatch, _pose_twist_names())
        root = service._resolve_hub_dataset(parse_hub_ref("acme/demo"))[1]
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["features"]["action"]["names"] = {"motors": ["motor_0", "motor_1"]}
        info_path.write_text(json.dumps(info), encoding="utf-8")

        result = service.inspect_hub_dataset("acme/demo")

        assert result["dataset_warning"] is None
        assert "cannot be rolled out" in result["dataset_ok"]

    @pytest.mark.parametrize(
        "names",
        [
            pytest.param(["motor_0", "motor_1"], id="arbitrary"),
            pytest.param(
                ["a.gripper.target_width"], id="target_width_without_metadata"
            ),
            pytest.param(["a.gripper.close"], id="boolean_gripper"),
            pytest.param(
                ["a.gripper.target_width", "b.gripper.width"], id="mixed_conventions"
            ),
        ],
    )
    def test_start_and_load_both_accept_any_action_schema(
        self, tmp_path, monkeypatch, names
    ):
        # Anti-drift: Start and Load share _dataset_preflight. Asserted on that
        # call rather than by running start(), which spawns a real subprocess.
        service = TrainingService(make_settings(tmp_path))
        stub_dataset_fetch(monkeypatch, names)
        _, root = service._resolve_hub_dataset(parse_hub_ref("acme/demo"))

        assert service._dataset_preflight(root).gripper_command_metadata is None
        assert service.inspect_hub_dataset("acme/demo")["dataset_warning"] is None

    def test_session_token_reaches_the_metadata_fetch(self, tmp_path, monkeypatch):
        # LeRobotDatasetMetadata takes no token argument, so the environment is
        # the only route for a UI-typed token.
        import os

        from flexivtrainer.data.hub import set_session_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        seen: list[str | None] = []

        def fake_metadata(repo_id, root=None, revision=None, force_cache_sync=False):
            seen.append(os.environ.get("HF_TOKEN"))
            write_dataset_meta(Path(root), _pose_twist_names())
            return object()

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", fake_metadata
        )
        service = TrainingService(make_settings(tmp_path))
        set_session_token("typed-token")
        try:
            service.inspect_hub_dataset("acme/demo")
        finally:
            set_session_token(None)

        assert seen == ["typed-token"]
        # Restored, so one fetch cannot leak a credential into later work.
        assert os.environ.get("HF_TOKEN") is None
