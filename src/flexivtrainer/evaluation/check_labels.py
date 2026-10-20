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

"""Which frames of a recording show a check's state: `meta/check_labels.json`.

A learned check (learned_check.py) is trained from labelled frames. Labelling
every frame is too slow, and what such a check judges is usually a state that,
once reached, stays: a part seated, a latch closed, a lid shut. So a label is
ONE NUMBER PER EPISODE, the first frame at which the state holds:

    {
      "format_version": 1,
      "checks": {
        "<check name>": {"<episode index>": <first frame>, "<episode>": null}
      }
    }

Every frame from the first one to the end is a positive example and every
frame before it a negative, except a short margin just before it, which is left
out: nobody can say to the frame when a part became seated. `null` means the
state never holds in that episode, so all its frames are negatives -- the most
useful ones when the episode is a near miss. Episodes without a label are not
used.

A recorded rollout that has an operator verdict can stand in for a label
(`verdict_targets`), when the verdict judges the check's state: "failure" means
the state was never reached, "success" that it held at the end.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .outcome import RolloutOutcome

__all__ = [
    "LABELS_FILE",
    "IGNORED",
    "describe",
    "episode_lengths",
    "frame_targets",
    "labels_for",
    "merge_labels",
    "read_labels",
    "remove_label",
    "verdict_targets",
    "write_label",
]

LABELS_FILE = Path("meta") / "check_labels.json"
FORMAT_VERSION = 1
#: A frame's target when it is left out of training.
IGNORED = -1

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(
            f"check name {name!r} must be 1-64 letters, digits, '.', '_' or '-'"
        )
    return name


def read_labels(root: Path) -> dict[str, dict[int, int | None]]:
    """Every check's labels in one dataset: check -> episode -> first frame."""
    path = Path(root) / LABELS_FILE
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a format {FORMAT_VERSION} labels file")
    checks = data.get("checks")
    if not isinstance(checks, dict):
        raise ValueError(f"{path} has no 'checks' object")
    labels: dict[str, dict[int, int | None]] = {}
    for name, episodes in checks.items():
        if not isinstance(episodes, dict):
            raise ValueError(f"{path}: check {name!r} must map episodes to frames")
        labels[_check_name(name)] = {
            int(episode): None if first is None else int(first)
            for episode, first in episodes.items()
        }
    return labels


def labels_for(root: Path, check: str) -> dict[int, int | None]:
    """One check's labels in one dataset: episode -> first frame, or None."""
    return read_labels(root).get(_check_name(check), {})


def _write(root: Path, labels: dict[str, dict[int, int | None]]) -> Path:
    """Write atomically: a labels file half-written by a crash would read as
    malformed, and every label in it would be lost."""
    path = Path(root) / LABELS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "checks": {
            name: {str(e): episodes[e] for e in sorted(episodes)}
            for name, episodes in sorted(labels.items())
            if episodes
        },
    }
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".check-labels-", suffix=".json"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def episode_lengths(root: Path) -> dict[int, int]:
    """Frames per episode, from the dataset's `meta/episodes` files."""
    import pandas as pd  # noqa: PLC0415

    files = sorted((Path(root) / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root} has no meta/episodes files")
    lengths: dict[int, int] = {}
    for file in files:
        table = pd.read_parquet(file, columns=["episode_index", "length"])
        for episode, length in zip(table["episode_index"], table["length"]):
            lengths[int(episode)] = int(length)
    return lengths


def write_label(root: Path, check: str, episode: int, first_frame: int | None) -> Path:
    """Label one episode: the first frame at which `check`'s state holds, or
    None when it never does. Refused for an episode or frame the dataset does
    not have."""
    _check_name(check)
    lengths = episode_lengths(root)
    if episode not in lengths:
        raise ValueError(f"episode {episode} is not in {root}")
    if first_frame is not None and not 0 <= first_frame < lengths[episode]:
        raise ValueError(
            f"frame {first_frame} is outside episode {episode} "
            f"(0..{lengths[episode] - 1})"
        )
    labels = read_labels(root)
    labels.setdefault(check, {})[int(episode)] = first_frame
    return _write(root, labels)


def remove_label(root: Path, check: str, episode: int) -> bool:
    """Remove one episode's label; False when it had none."""
    labels = read_labels(root)
    removed = labels.get(_check_name(check), {}).pop(int(episode), "absent")
    if removed == "absent":
        return False
    _write(root, labels)
    return True


def frame_targets(length: int, first_frame: int | None, margin: int) -> np.ndarray:
    """Per frame: 1 (the state holds), 0 (it does not) or IGNORED.

    The `margin` frames just before `first_frame` are ignored.
    """
    targets = np.zeros(length, dtype=np.int8)
    if first_frame is None:
        return targets
    first = min(max(int(first_frame), 0), length)
    targets[max(first - max(int(margin), 0), 0) : first] = IGNORED
    targets[first:] = 1
    return targets


def verdict_targets(
    length: int, outcome: RolloutOutcome | None, tail: int
) -> np.ndarray | None:
    """Per-frame targets from a recorded rollout's verdict, or None without one.

    "failure": every frame is a negative. "success": the last `tail` frames
    are positives and the rest are ignored, because the verdict says the state
    held at the end, not when it began. "unknown" gives nothing.
    """
    verdict = outcome.verdict if outcome is not None else None
    if verdict is None or verdict.outcome == "unknown":
        return None
    if verdict.outcome == "failure":
        return np.zeros(length, dtype=np.int8)
    targets = np.full(length, IGNORED, dtype=np.int8)
    targets[max(length - max(int(tail), 1), 0) :] = 1
    return targets


def merge_labels(
    roots: Sequence[Path], target: Path, episodes: Sequence[int] | None = None
) -> Path | None:
    """Carry the sources' labels into a dataset merged from them, in order.

    `episodes` is each source's episode count; by default it is read from the
    sources. Source i's episode e becomes episode e plus the episodes of the
    sources before it, which is how LeRobot's merge numbers them. Returns None
    when no source has a label.
    """
    sources = [read_labels(root) for root in roots]
    if not any(episodes for labels in sources for episodes in labels.values()):
        return None
    counts = (
        list(episodes)
        if episodes is not None
        else [len(episode_lengths(root)) for root in roots]
    )
    merged: dict[str, dict[int, int | None]] = {}
    offset = 0
    for labels, count in zip(sources, counts, strict=True):
        for name, labelled in labels.items():
            for episode, first in labelled.items():
                merged.setdefault(name, {})[offset + episode] = first
        offset += count
    return _write(target, merged)


def describe(labels: dict[int, int | None]) -> dict[str, Any]:
    """How many episodes a check has labelled, and how many reach its state."""
    return {
        "episodes": len(labels),
        "reached": sum(1 for first in labels.values() if first is not None),
        "never": sum(1 for first in labels.values() if first is None),
    }
