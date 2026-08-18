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

"""Tests for HuggingFace Hub identity, validation, and local materialization."""

import json
from pathlib import Path

import pytest

from flexivtrainer.config import AppSettings, HubConfig, StorageConfig
from flexivtrainer.data import hub
from flexivtrainer.data.hub import (
    HubAuthError,
    HubError,
    HubNotFoundError,
    HubRef,
    HubUnavailableError,
    describe_hub_error,
    fetch_checkpoint_snapshot,
    fetch_dataset_metadata,
    hub_cache_dir,
    hub_token,
    is_hub_repo_id,
    parse_hub_ref,
    sanitize_repo_id,
    validate_repo_id,
    validate_revision,
)

HOSTILE_REPO_IDS = [
    "../etc",
    "../../etc/passwd",
    "/absolute/path",
    "a/b/c",
    "",
    "   ",
    "a/",
    "/b",
    ".hidden/x",
    "a b/c",
    "a\nb/c",
    "a/b\x00c",
    "-lead/x",
    "x/-lead",
    "a/..",
    "../a/b",
]


def make_settings(tmp_path: Path, **hub_kwargs) -> AppSettings:
    storage = StorageConfig(root=tmp_path)
    storage.ensure()
    return AppSettings(storage=storage, hub=HubConfig(**hub_kwargs))


class TestRepoIdValidation:
    @pytest.mark.parametrize("repo_id", HOSTILE_REPO_IDS)
    def test_hostile_repo_ids_rejected(self, repo_id: str) -> None:
        with pytest.raises(ValueError):
            validate_repo_id(repo_id)

    @pytest.mark.parametrize(
        "repo_id", ["lerobot/pusht", "acme/demo-v2", "a/b", "Org.Name/model_1"]
    )
    def test_valid_repo_ids_accepted(self, repo_id: str) -> None:
        assert validate_repo_id(repo_id) == repo_id

    def test_repo_id_is_stripped(self) -> None:
        assert validate_repo_id("  lerobot/pusht  ") == "lerobot/pusht"

    def test_non_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_repo_id(None)

    @pytest.mark.parametrize("revision", ["../x", "a..b", "", "  "])
    def test_bad_or_empty_revisions(self, revision: str) -> None:
        if ".." in revision:
            with pytest.raises(ValueError):
                validate_revision(revision)
        else:
            assert validate_revision(revision) is None

    def test_valid_revisions(self) -> None:
        assert validate_revision("main") == "main"
        assert validate_revision("v2.1") == "v2.1"
        assert validate_revision(None) is None

    def test_parse_hub_ref(self) -> None:
        ref = parse_hub_ref("lerobot/pusht", "main")
        assert ref == HubRef("lerobot/pusht", "main")

    def test_is_hub_repo_id_excludes_synthetic_local(self) -> None:
        # This app synthesizes local/<name> for its own datasets; those must not
        # be mistaken for something fetchable from the Hub.
        assert is_hub_repo_id("local/merged_20260810") is False
        assert is_hub_repo_id("lerobot/pusht") is True
        assert is_hub_repo_id("../etc") is False
        assert is_hub_repo_id(None) is False


class TestSanitizeRepoId:
    def test_no_separator_survives(self) -> None:
        slug = sanitize_repo_id("lerobot/pusht")
        assert "/" not in slug and "\\" not in slug
        assert slug.startswith("lerobot__pusht-")

    def test_deterministic(self) -> None:
        assert sanitize_repo_id("lerobot/pusht") == sanitize_repo_id("lerobot/pusht")

    def test_revisions_get_distinct_directories(self) -> None:
        base = sanitize_repo_id("lerobot/pusht")
        pinned = sanitize_repo_id("lerobot/pusht", "v2")
        other = sanitize_repo_id("lerobot/pusht", "v3")
        assert len({base, pinned, other}) == 3

    def test_distinct_repos_distinct_slugs(self) -> None:
        assert sanitize_repo_id("a/model") != sanitize_repo_id("b/model")

    def test_length_bounded(self) -> None:
        slug = sanitize_repo_id("o" * 200 + "/" + "n" * 200)
        assert len(slug) <= 90

    def test_long_names_stay_distinct_despite_truncation(self) -> None:
        first = sanitize_repo_id("owner/" + "n" * 200 + "a")
        second = sanitize_repo_id("owner/" + "n" * 200 + "b")
        assert first != second

    @pytest.mark.parametrize("repo_id", HOSTILE_REPO_IDS)
    def test_hostile_input_never_sanitized_into_a_path(self, repo_id: str) -> None:
        with pytest.raises(ValueError):
            sanitize_repo_id(repo_id)


class TestHubCacheDir:
    def test_inside_storage_root(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        target = hub_cache_dir(settings, "datasets", parse_hub_ref("lerobot/pusht"))
        # Inside the storage root by design, so resolve_checkpoint_path accepts
        # downloaded checkpoints without any relaxation.
        assert target.is_relative_to(tmp_path.resolve())
        assert target.is_relative_to(settings.storage.hub_cache_root.resolve())

    def test_datasets_and_checkpoints_are_separate(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        ref = parse_hub_ref("lerobot/pusht")
        assert hub_cache_dir(settings, "datasets", ref) != hub_cache_dir(
            settings, "checkpoints", ref
        )

    def test_unknown_kind_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            hub_cache_dir(make_settings(tmp_path), "bogus", parse_hub_ref("a/b"))

    @pytest.mark.parametrize("repo_id", HOSTILE_REPO_IDS)
    def test_hostile_ids_never_escape(self, tmp_path: Path, repo_id: str) -> None:
        settings = make_settings(tmp_path)
        with pytest.raises(ValueError):
            hub_cache_dir(settings, "datasets", HubRef(repo_id))


class TestHubToken:
    def test_config_token_wins(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "from-env")
        assert hub_token(make_settings(tmp_path, token="from-config")) == "from-config"

    def test_falls_back_to_environment(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "from-env")
        assert hub_token(make_settings(tmp_path)) == "from-env"

    def test_none_when_unset(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert hub_token(make_settings(tmp_path)) is None


class TestErrorMapping:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("RepositoryNotFoundError", HubNotFoundError),
            ("RevisionNotFoundError", HubNotFoundError),
            ("GatedRepoError", HubAuthError),
            ("LocalEntryNotFoundError", HubUnavailableError),
        ],
    )
    def test_named_exceptions_classified(self, name, expected) -> None:
        exc = type(name, (Exception,), {})("boom")
        assert isinstance(hub._classify_hub_error(exc), expected)

    def test_http_status_classified(self) -> None:
        class Response:
            status_code = 403

        exc = Exception("denied")
        exc.response = Response()
        assert isinstance(hub._classify_hub_error(exc), HubAuthError)

    def test_connection_error_is_unavailable(self) -> None:
        assert isinstance(
            hub._classify_hub_error(ConnectionError("no route")), HubUnavailableError
        )

    def test_messages_are_actionable(self) -> None:
        exc = type("RepositoryNotFoundError", (Exception,), {})("x")
        assert "token" in describe_hub_error(exc).lower()

    def test_self_raising_str_does_not_break_reporting(self) -> None:
        # HfHubHTTPError.__str__ dereferences response.headers and can raise.
        # The diagnostic path must never fail, or it buries the real cause.
        class Nasty(Exception):
            def __str__(self) -> str:
                raise RuntimeError("boom")

        assert "Nasty" in describe_hub_error(Nasty())
        assert isinstance(hub._classify_hub_error(Nasty()), HubError)

    def test_self_raising_response_property(self) -> None:
        class NastyResponse(Exception):
            @property
            def response(self):
                raise RuntimeError("boom")

        assert hub._status_code(NastyResponse()) is None

    def test_untagged_dataset_is_recognised(self) -> None:
        # LeRobot raises RevisionNotFoundError without the keyword-only
        # 'response' huggingface_hub>=1.0 requires, so the real cause arrives
        # as an unrelated TypeError.
        exc = TypeError(
            "HfHubHTTPError.__init__() missing 1 required "
            "keyword-only argument: 'response'"
        )
        message = describe_hub_error(exc)
        assert "codebase-version tag" in message
        assert "create_tag" in message
        assert isinstance(
            hub._classify_hub_error(exc), hub.HubDatasetNotTaggedError
        )

    def test_untagged_detected_from_lerobot_message(self) -> None:
        exc = RuntimeError("Your dataset must be tagged with a codebase version.")
        assert "codebase-version tag" in describe_hub_error(exc)


def write_fake_dataset_meta(root: Path) -> None:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 30, "features": {}}), encoding="utf-8"
    )


class TestFetchDatasetMetadata:
    def test_downloads_then_caches(self, tmp_path, monkeypatch) -> None:
        settings = make_settings(tmp_path)
        calls = []

        def fake_metadata(repo_id, root=None, revision=None, force_cache_sync=False):
            calls.append(repo_id)
            write_fake_dataset_meta(Path(root))
            return object()

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", fake_metadata
        )
        ref = parse_hub_ref("lerobot/pusht")
        first = fetch_dataset_metadata(settings, ref)
        second = fetch_dataset_metadata(settings, ref)
        assert first == second
        assert (first / "meta" / "info.json").is_file()
        assert len(calls) == 1, "second call should hit the cache"

    def test_force_redownloads(self, tmp_path, monkeypatch) -> None:
        settings = make_settings(tmp_path)
        calls = []

        def fake_metadata(repo_id, root=None, revision=None, force_cache_sync=False):
            calls.append(repo_id)
            write_fake_dataset_meta(Path(root))
            return object()

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", fake_metadata
        )
        ref = parse_hub_ref("lerobot/pusht")
        fetch_dataset_metadata(settings, ref)
        fetch_dataset_metadata(settings, ref, force=True)
        assert len(calls) == 2

    def test_failed_download_leaves_no_completion_marker(
        self, tmp_path, monkeypatch
    ) -> None:
        settings = make_settings(tmp_path)

        def boom(repo_id, root=None, revision=None, force_cache_sync=False):
            Path(root).mkdir(parents=True, exist_ok=True)
            (Path(root) / "partial.bin").write_bytes(b"half")
            raise type("RepositoryNotFoundError", (Exception,), {})("missing")

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", boom
        )
        ref = parse_hub_ref("lerobot/pusht")
        with pytest.raises(HubNotFoundError):
            fetch_dataset_metadata(settings, ref)

        target = hub_cache_dir(settings, "datasets", ref)
        # A partial download must never be mistaken for a usable cache entry.
        assert not (target / hub.HUB_MARKER_FILENAME).exists()

    def test_missing_info_json_is_an_error(self, tmp_path, monkeypatch) -> None:
        settings = make_settings(tmp_path)
        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata",
            lambda *a, **k: object(),
        )
        with pytest.raises(HubError, match="meta/info.json"):
            fetch_dataset_metadata(settings, parse_hub_ref("lerobot/pusht"))

    def test_disabled_hub_refuses(self, tmp_path) -> None:
        settings = make_settings(tmp_path, enabled=False)
        with pytest.raises(HubError, match="disabled"):
            fetch_dataset_metadata(settings, parse_hub_ref("lerobot/pusht"))


class TestFetchCheckpointSnapshot:
    def test_downloads_then_caches(self, tmp_path, monkeypatch) -> None:
        settings = make_settings(tmp_path)
        calls = []

        def fake_snapshot(repo_id, **kwargs):
            calls.append(repo_id)
            local = Path(kwargs["local_dir"])
            local.mkdir(parents=True, exist_ok=True)
            (local / "config.json").write_text('{"type": "act"}', encoding="utf-8")
            return str(local)

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
        ref = parse_hub_ref("acme/policy")
        first = fetch_checkpoint_snapshot(settings, ref)
        fetch_checkpoint_snapshot(settings, ref)
        assert (first / "config.json").is_file()
        assert len(calls) == 1

    def test_accepts_nested_pretrained_model_layout(
        self, tmp_path, monkeypatch
    ) -> None:
        settings = make_settings(tmp_path)

        def fake_snapshot(repo_id, **kwargs):
            nested = Path(kwargs["local_dir"]) / "pretrained_model"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "config.json").write_text('{"type": "act"}', encoding="utf-8")
            return str(kwargs["local_dir"])

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
        target = fetch_checkpoint_snapshot(settings, parse_hub_ref("acme/policy"))
        assert (target / "pretrained_model" / "config.json").is_file()

    def test_missing_config_is_an_error(self, tmp_path, monkeypatch) -> None:
        settings = make_settings(tmp_path)

        def fake_snapshot(repo_id, **kwargs):
            Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
            return str(kwargs["local_dir"])

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
        with pytest.raises(HubError, match="config.json"):
            fetch_checkpoint_snapshot(settings, parse_hub_ref("acme/policy"))

    def test_auth_failure_classified(self, tmp_path, monkeypatch) -> None:
        settings = make_settings(tmp_path)

        def boom(repo_id, **kwargs):
            raise type("GatedRepoError", (Exception,), {})("gated")

        monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
        with pytest.raises(HubAuthError):
            fetch_checkpoint_snapshot(settings, parse_hub_ref("acme/policy"))

    def test_token_is_passed_through(self, tmp_path, monkeypatch) -> None:
        settings = make_settings(tmp_path, token="secret-token")
        seen = {}

        def fake_snapshot(repo_id, **kwargs):
            seen.update(kwargs)
            local = Path(kwargs["local_dir"])
            local.mkdir(parents=True, exist_ok=True)
            (local / "config.json").write_text("{}", encoding="utf-8")
            return str(local)

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
        fetch_checkpoint_snapshot(settings, parse_hub_ref("acme/policy"))
        assert seen["token"] == "secret-token"
        assert seen["repo_type"] == "model"
