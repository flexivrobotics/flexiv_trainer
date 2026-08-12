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

"""HuggingFace Hub identity, validation, and local materialization.

Hub identifiers never travel on the same input as filesystem paths: callers pass
an explicit ``source`` discriminator plus a repo id, so a repo id can never reach
a path resolver and a path can never reach a downloader.

Everything is materialized under ``.local/cache/hub``, which is *inside* the
storage root. A downloaded checkpoint is therefore validated by the existing
``resolve_checkpoint_path``, so hub content gets the same segment-walk and
symlink-escape checks as any local checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from flexivtrainer.config import AppSettings

HUB_CACHE_DIRNAME = "hub"
HUB_MARKER_FILENAME = ".flexivtrainer_hub.json"
ACTION_NAMES_FILENAME = "action_names.json"

# ``owner/name``: exactly one slash, each segment starting alphanumeric. This is
# the security boundary, applied before any part of the string touches a path.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_SLUG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")
_SLUG_MAX_CHARS = 80

HubKind = Literal["datasets", "checkpoints"]

# Serializes concurrent fetches of the same repo so two requests cannot race into
# the same local_dir.
_FETCH_LOCKS: dict[str, threading.Lock] = {}
_FETCH_LOCKS_GUARD = threading.Lock()


class HubError(RuntimeError):
    """Base for Hub failures, so routes can map them to HTTP status codes."""


class HubNotFoundError(HubError):
    pass


class HubAuthError(HubError):
    pass


class HubUnavailableError(HubError):
    pass


@dataclass(frozen=True)
class HubRef:
    repo_id: str
    revision: str | None = None

    def __str__(self) -> str:
        if self.revision is None:
            return self.repo_id
        return f"{self.repo_id}@{self.revision}"


def validate_repo_id(repo_id: Any) -> str:
    """Accept only a well-formed ``owner/name``; reject anything path-shaped."""
    if not isinstance(repo_id, str):
        raise ValueError("Hub repo id must be a string")
    candidate = repo_id.strip()
    if not candidate:
        raise ValueError("Hub repo id must not be empty")
    if not _REPO_ID_RE.match(candidate):
        raise ValueError(
            f"Invalid Hub repo id: {repo_id!r}. Expected 'owner/name' using "
            "letters, digits, '.', '_', or '-'."
        )
    # _REPO_ID_RE already forbids these, but a traversal slipping through would be
    # a filesystem escape rather than a validation nit, so assert it directly.
    if any(part in {"", ".", ".."} for part in candidate.split("/")):
        raise ValueError(f"Invalid Hub repo id: {repo_id!r}")
    return candidate


def validate_revision(revision: Any) -> str | None:
    """Accept a branch, tag, or commit sha; ``None`` means the default revision."""
    if revision is None:
        return None
    if not isinstance(revision, str):
        raise ValueError("Hub revision must be a string")
    candidate = revision.strip()
    if not candidate:
        return None
    if not _REVISION_RE.match(candidate) or ".." in candidate:
        raise ValueError(f"Invalid Hub revision: {revision!r}")
    return candidate


def parse_hub_ref(repo_id: Any, revision: Any = None) -> HubRef:
    return HubRef(validate_repo_id(repo_id), validate_revision(revision))


def sanitize_repo_id(repo_id: str, revision: str | None = None) -> str:
    """Build a single filesystem-safe path segment for a Hub reference.

    ``lerobot/pusht`` becomes ``lerobot__pusht-3f2a9c11``. The digest keeps
    truncated slugs distinct and gives each revision its own directory, so
    pinning a tag never clobbers the default-branch copy. The result contains no
    separator, which matters because a dataset directory name is reused to build
    the B-spline conversion output directory.
    """
    ref = parse_hub_ref(repo_id, revision)
    owner, name = ref.repo_id.split("/", 1)
    slug = _SLUG_UNSAFE_RE.sub("-", f"{owner}__{name}")[:_SLUG_MAX_CHARS]
    digest = hashlib.sha256(
        f"{ref.repo_id}\n{ref.revision or ''}".encode()
    ).hexdigest()[:8]
    return f"{slug}-{digest}"


def hub_cache_root(settings: AppSettings) -> Path:
    return settings.storage.cache_root / HUB_CACHE_DIRNAME


def hub_cache_dir(settings: AppSettings, kind: HubKind, ref: HubRef) -> Path:
    """Resolve the cache directory for a reference, refusing to escape the root."""
    if kind not in {"datasets", "checkpoints"}:
        raise ValueError(f"Unsupported hub cache kind: {kind!r}")
    root = hub_cache_root(settings).expanduser().resolve()
    target = (root / kind / sanitize_repo_id(ref.repo_id, ref.revision)).resolve()
    # sanitize_repo_id cannot emit a separator, so this is a backstop rather than
    # the primary defense.
    if not target.is_relative_to(root):
        raise ValueError(f"Access denied: hub cache path escapes {root}")
    return target


def hub_token(settings: AppSettings | None = None) -> str | None:
    """Token for private/gated repos, or ``None`` to use anonymous access.

    Falls through to the ``huggingface-cli login`` cache when nothing is set.
    """
    configured = getattr(getattr(settings, "hub", None), "token", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _safe_str(exc: BaseException) -> str:
    """Stringify an exception that may raise from its own ``__str__``.

    ``HfHubHTTPError.__str__`` dereferences ``response.headers``, which blows up
    when the error was constructed without a response. Reporting that secondary
    failure would bury the original cause.
    """
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return type(exc).__name__
    return text or type(exc).__name__


def _status_code(exc: BaseException) -> int | None:
    """HTTP status behind an exception, tolerating attributes that raise."""
    try:
        return getattr(getattr(exc, "response", None), "status_code", None)
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return None


def _describe_untagged_dataset(exc: BaseException) -> str | None:
    """Detect LeRobot's "dataset has no version tag" failure.

    LeRobot requires a codebase-version git tag on dataset repos. When one is
    missing it raises RevisionNotFoundError, but on huggingface_hub>=1.0 that
    constructor needs a keyword-only ``response`` it does not pass, so the real
    cause surfaces as an unrelated TypeError. Recognise both shapes.
    """
    text = _safe_str(exc)
    chain: list[BaseException] = []
    cursor: BaseException | None = exc
    while cursor is not None and len(chain) < 6:
        chain.append(cursor)
        cursor = cursor.__cause__ or cursor.__context__
    blames_tagging = any(
        "tagged with a codebase version" in _safe_str(item) for item in chain
    )
    hf_constructor_bug = isinstance(exc, TypeError) and (
        "HfHubHTTPError.__init__" in text or "RevisionNotFound" in text
    )
    if not blames_tagging and not hf_constructor_bug:
        return None
    return (
        "This Hub dataset has no codebase-version tag, which LeRobot requires. "
        "Tag the repo to match the codebase_version in its meta/info.json, for "
        'example: HfApi().create_tag("<owner>/<name>", tag="v3.0", '
        'repo_type="dataset")'
    )


def describe_hub_error(exc: BaseException) -> str:
    """Human-readable, actionable message for a Hub failure."""
    untagged = _describe_untagged_dataset(exc)
    if untagged is not None:
        return untagged
    name = type(exc).__name__
    if name == "RepositoryNotFoundError":
        return (
            "Hub repository not found, or it is private and no token is "
            "configured. Set FLEXIV_TRAINER_HUB__TOKEN or HF_TOKEN."
        )
    if name == "RevisionNotFoundError":
        return "Hub revision not found."
    if name == "GatedRepoError":
        return (
            "Hub repository is gated. Accept its terms on huggingface.co and "
            "configure an access token."
        )
    if name in {"LocalEntryNotFoundError", "OfflineModeIsEnabled"}:
        return "Cannot reach the HuggingFace Hub (offline or network unavailable)."
    status = _status_code(exc)
    if status in {401, 403}:
        return (
            "Hub authentication failed. Set FLEXIV_TRAINER_HUB__TOKEN or HF_TOKEN "
            "with access to this repository."
        )
    if status == 404:
        return "Hub repository or revision not found."
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        return "Not enough disk space to download from the HuggingFace Hub."
    return f"HuggingFace Hub request failed: {_safe_str(exc)}"


class HubDatasetNotTaggedError(HubError):
    """The repo exists but carries no codebase-version tag LeRobot can use."""


def _classify_hub_error(exc: BaseException) -> HubError:
    name = type(exc).__name__
    message = describe_hub_error(exc)
    status = _status_code(exc)
    if _describe_untagged_dataset(exc) is not None:
        # The repo is reachable; the operator has to tag it. That is a bad
        # request, not an upstream gateway failure.
        return HubDatasetNotTaggedError(message)
    if name in {"RepositoryNotFoundError", "RevisionNotFoundError"} or status == 404:
        return HubNotFoundError(message)
    if name == "GatedRepoError" or status in {401, 403}:
        return HubAuthError(message)
    if name in {"LocalEntryNotFoundError", "OfflineModeIsEnabled"} or isinstance(
        exc, ConnectionError | TimeoutError
    ):
        return HubUnavailableError(message)
    return HubError(message)


def _fetch_lock(key: str) -> threading.Lock:
    with _FETCH_LOCKS_GUARD:
        return _FETCH_LOCKS.setdefault(key, threading.Lock())


def _read_marker(target: Path) -> dict[str, Any]:
    try:
        payload = json.loads((target / HUB_MARKER_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_marker(target: Path, ref: HubRef, **extra: Any) -> None:
    payload = {
        "repo_id": ref.repo_id,
        "revision": ref.revision,
        "complete": True,
        **extra,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / HUB_MARKER_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _require_hub_enabled(settings: AppSettings) -> None:
    if not getattr(getattr(settings, "hub", None), "enabled", True):
        raise HubError("HuggingFace Hub access is disabled by configuration")


def fetch_dataset_metadata(
    settings: AppSettings, ref: HubRef, *, force: bool = False
) -> Path:
    """Download only a dataset's ``meta/`` directory and return its local root.

    LeRobot pulls with ``allow_patterns="meta/"``, so this stays small even for
    datasets with hundreds of gigabytes of video. The returned root has the same
    on-disk shape the training pre-flight already reads, so every existing
    ``meta/info.json`` / ``meta/bspline.json`` / ``meta/gripper_command.json``
    read works unchanged. Bulk data is left to the training subprocess.
    """
    _require_hub_enabled(settings)
    target = hub_cache_dir(settings, "datasets", ref)
    with _fetch_lock(str(target)):
        marker = _read_marker(target)
        if (
            not force
            and marker.get("complete")
            and (target / "meta" / "info.json").is_file()
        ):
            return target

        from lerobot.datasets.lerobot_dataset import (  # noqa: PLC0415
            LeRobotDatasetMetadata,
        )

        target.mkdir(parents=True, exist_ok=True)
        try:
            LeRobotDatasetMetadata(
                ref.repo_id, root=target, revision=ref.revision, force_cache_sync=force
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed HubError
            # Deliberately leave any partial content in place (snapshot_download
            # resumes) but never mark it complete.
            raise _classify_hub_error(exc) from exc

        if not (target / "meta" / "info.json").is_file():
            raise HubError(f"Hub dataset has no meta/info.json: {ref}")
        _write_marker(target, ref, meta_only=True)
        return target


def fetch_checkpoint_snapshot(
    settings: AppSettings, ref: HubRef, *, force: bool = False
) -> Path:
    """Download a policy checkpoint and return its local directory.

    Checkpoints are small relative to datasets and every metadata helper reads
    them by path, so a full snapshot keeps all of those helpers working unchanged.
    """
    _require_hub_enabled(settings)
    target = hub_cache_dir(settings, "checkpoints", ref)
    with _fetch_lock(str(target)):
        marker = _read_marker(target)
        if not force and marker.get("complete") and _has_model_config(target):
            return target

        from huggingface_hub import snapshot_download  # noqa: PLC0415

        target.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                ref.repo_id,
                repo_type="model",
                revision=ref.revision,
                local_dir=target,
                token=hub_token(settings),
                force_download=force,
                # Keep optimizer state, wandb logs, and training scratch out of
                # the cache; the policy loader only needs config + weights.
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.md",
                    "pretrained_model/*",
                ],
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed HubError
            raise _classify_hub_error(exc) from exc

        if not _has_model_config(target):
            raise HubError(
                f"Hub checkpoint has no config.json (looked in the repo root and "
                f"pretrained_model/): {ref}"
            )
        _write_marker(target, ref)
        return target


def _has_model_config(target: Path) -> bool:
    """Mirror ``_checkpoint_model_dir``: config may be at the root or nested."""
    return (target / "config.json").is_file() or (
        target / "pretrained_model" / "config.json"
    ).is_file()


def is_hub_repo_id(value: Any) -> bool:
    """True when a string looks like a real Hub id rather than a local label.

    This app synthesizes ``local/<name>`` repo ids for its own datasets, so those
    must not be mistaken for something fetchable.
    """
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate or candidate.startswith("local/"):
        return False
    try:
        validate_repo_id(candidate)
    except ValueError:
        return False
    return True
