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

"""A check that learns its answer from labelled frames.

Some yes/no questions are not settled by where the tool is: whether a latch has
closed, whether a part sits flush, whether a cap is on. A person answers them
from the cameras and the force reading, and a small classifier can learn to
from a few labelled episodes (`check_labels.py`).

    python -m flexivtrainer.evaluation.learned_check label DATASET \
        --check seated --episode 3 --first-frame 412
    python -m flexivtrainer.evaluation.learned_check train --check seated \
        --data DATASET --camera wrist_1 --state single_arm.tcp_wrench.fz \
        --out checks/seated
    python -m flexivtrainer.evaluation.learned_check score DATASET \
        --model checks/seated --episode 5

A trained check is a folder (`check.json` + `head.safetensors`) that a session
plan names like any other check, its path absolute or relative to the storage
root:

    "checks": {"seated": {"kind": "learned", "model": "checks/seated", "hold_s": 0.2}}

It passes when its probability stays at or above the threshold for `hold_s`.

THE MODEL. Each camera frame is cut to its crop box, resized, passed through a
frozen image network (ResNet-18 trained on ImageNet) and pooled to a small grid
of features. The chosen `observation.state` columns are standardised. A small
head, the only part trained, maps both to the probability that the state holds.
Training the head alone needs only a few thousand frames and minutes; the price
is image features that were not tuned to the task, which a tight crop offsets.
The `pixels` encoder (the crop averaged over a grid) is for simple scenes and
for tests.

HELD-OUT EPISODES, NOT FRAMES. Neighbouring frames of one episode are near
copies, so a split by frame reports the training episodes back. Training holds
out whole episodes and reports, for each, whether the check would have passed
it. That is the number to read: a check that passes a near miss is worse than
none, because whatever runs next starts from a failure. The held-out episodes
are not used to choose anything, so the report is an honest estimate -- of
labelled episodes like these, not yet of live trials. Live trust is earned the
same way as any check's: agreement with the verdicts of a session.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .check_labels import (
    IGNORED,
    describe,
    episode_lengths,
    frame_targets,
    labels_for,
    read_labels,
    remove_label,
    verdict_targets,
    write_label,
)
from .checks import held_runs
from .outcome import read_outcome

__all__ = [
    "CHECK_FILE",
    "ENCODERS",
    "CheckInputs",
    "DatasetReader",
    "LabelledEpisode",
    "LearnedCheckModel",
    "TrainingOptions",
    "collect_episodes",
    "judge",
    "load_model",
    "score_episode",
    "train_check",
    "main",
]

CHECK_FILE = "check.json"
HEAD_FILE = "head.safetensors"
FORMAT_VERSION = 1
ENCODERS = ("resnet18", "pixels")
IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
#: Frames decoded and encoded at once, per camera.
_CHUNK = 128
_SWEEP = tuple(round(0.1 * i, 1) for i in range(1, 10))


# --- what a check looks at -------------------------------------------------


@dataclass(frozen=True)
class CheckInputs:
    """What a learned check looks at.

    `cameras` are camera names (`wrist_1` for `observation.images.wrist_1`);
    `state` are column names of `observation.state`, such as a force axis.
    Each camera frame is cut to its `crops` box, given as fractions of the
    frame (x0, y0, x1, y1), then resized to `image_size` (height, width) and
    pooled to a `grid` x `grid` of features.
    """

    cameras: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    image_size: tuple[int, int] = (224, 224)
    crops: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    encoder: str = "resnet18"
    grid: int = 2

    def __post_init__(self) -> None:
        if not self.cameras and not self.state:
            raise ValueError(
                "a learned check needs at least one camera or state column"
            )
        for names, what in ((self.cameras, "camera"), (self.state, "state column")):
            if len(set(names)) != len(names):
                raise ValueError(f"a {what} is listed twice: {list(names)}")
        if len(self.image_size) != 2 or min(self.image_size) < 16:
            raise ValueError("image_size must be (height, width), each at least 16")
        if self.encoder not in ENCODERS:
            raise ValueError(f"encoder {self.encoder!r} is not one of {ENCODERS}")
        if not 1 <= self.grid <= 7:
            raise ValueError("grid must be 1 to 7")
        for camera, box in self.crops.items():
            if camera not in self.cameras:
                raise ValueError(f"crop for {camera!r}, which is not an input camera")
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError(f"crop for {camera!r} must be four numbers")
            x0, y0, x1, y1 = box
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    f"crop for {camera!r} must be fractions with x0 < x1 and y0 < y1"
                )

    @property
    def feature_dim(self) -> int:
        """Features per camera frame."""
        channels = 512 if self.encoder == "resnet18" else 3
        return channels * self.grid * self.grid

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameras": list(self.cameras),
            "state": list(self.state),
            "image_size": list(self.image_size),
            "crops": {name: list(box) for name, box in self.crops.items()},
            "encoder": self.encoder,
            "grid": self.grid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckInputs:
        known = {"cameras", "state", "image_size", "crops", "encoder", "grid"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown input keys {unknown}; known are {sorted(known)}")
        return cls(
            cameras=tuple(str(c) for c in data.get("cameras", ())),
            state=tuple(str(s) for s in data.get("state", ())),
            image_size=tuple(int(v) for v in data.get("image_size", (224, 224))),
            crops={
                str(name): tuple(float(v) for v in box)
                for name, box in (data.get("crops") or {}).items()
            },
            encoder=str(data.get("encoder", "resnet18")),
            grid=int(data.get("grid", 2)),
        )


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_ENCODER_CACHE: dict[tuple[str, int, str], nn.Module] = {}


def _encoder(name: str, grid: int, device: torch.device) -> nn.Module:
    """The frozen image network, loaded once per process and device."""
    key = (name, grid, str(device))
    if key not in _ENCODER_CACHE:
        if name == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18  # noqa: PLC0415

            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            trunk: list[nn.Module] = list(net.children())[:-2]  # drop pool and fc
        else:
            trunk = []
        module = nn.Sequential(*trunk, nn.AdaptiveAvgPool2d(grid), nn.Flatten())
        _ENCODER_CACHE[key] = module.eval().requires_grad_(False).to(device)
    return _ENCODER_CACHE[key]


def _as_nchw(frames: Any) -> torch.Tensor:
    """Frames as an (N, 3, H, W) uint8 tensor, from NCHW or NHWC."""
    tensor = torch.as_tensor(
        np.asarray(frames) if not torch.is_tensor(frames) else frames
    )
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(
            f"expected a batch of RGB frames, got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[1] != 3:
        raise ValueError(f"expected RGB frames, got shape {tuple(tensor.shape)}")
    return tensor


def _prepare(
    frames: Any,
    box: tuple[float, float, float, float] | None,
    inputs: CheckInputs,
    device: torch.device,
) -> torch.Tensor:
    """Crop, resize and scale one camera's frames for the encoder."""
    images = _as_nchw(frames).to(device)
    if box is not None:
        height, width = images.shape[-2:]
        x0, x1 = (min(int(round(v * width)), width) for v in (box[0], box[2]))
        y0, y1 = (min(int(round(v * height)), height) for v in (box[1], box[3]))
        images = images[..., y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)]
    images = images.float().div_(255.0)
    images = nn.functional.interpolate(
        images,
        size=tuple(inputs.image_size),
        mode="bilinear",
        antialias=True,
        align_corners=False,
    )
    if inputs.encoder == "resnet18":
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


@torch.inference_mode()
def _image_features(
    frames: Any, camera: str, inputs: CheckInputs, device: torch.device
) -> torch.Tensor:
    encoder = _encoder(inputs.encoder, inputs.grid, device)
    return encoder(_prepare(frames, inputs.crops.get(camera), inputs, device))


# --- the trained part --------------------------------------------------------


class _Head(nn.Module):
    """Per-input projections, then a small MLP to one logit."""

    def __init__(
        self,
        cameras: int,
        image_dim: int,
        state_dim: int,
        width: int = 128,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        def project(dim: int) -> nn.Module:
            return nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width), nn.GELU())

        self.images = nn.ModuleList(project(image_dim) for _ in range(cameras))
        self.state = project(state_dim) if state_dim else None
        inputs = width * (cameras + (1 if state_dim else 0))
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(inputs, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, images: list[torch.Tensor], state: torch.Tensor | None
    ) -> torch.Tensor:
        parts = [project(x) for project, x in zip(self.images, images, strict=True)]
        if self.state is not None and state is not None:
            parts.append(self.state(state))
        return self.out(torch.cat(parts, dim=-1)).squeeze(-1)


@dataclass
class LearnedCheckModel:
    """A trained check: its inputs, its head, and how to standardise its state."""

    name: str
    inputs: CheckInputs
    head: _Head
    state_mean: np.ndarray
    state_std: np.ndarray
    threshold: float = 0.5
    trained: dict[str, Any] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def _standardise(self, state: Any) -> torch.Tensor | None:
        if not self.inputs.state:
            return None
        values = np.asarray(state, dtype=np.float32).reshape(-1, len(self.inputs.state))
        values = np.clip((values - self.state_mean) / self.state_std, -10.0, 10.0)
        return torch.as_tensor(values, device=self.device)

    @torch.inference_mode()
    def probabilities(
        self, images: dict[str, Any] | None = None, state: Any = None
    ) -> np.ndarray:
        """P(the state holds) per frame. `images` maps each input camera to a
        batch of RGB uint8 frames (NHWC or NCHW); `state` is (N, columns) in
        the order of `inputs.state`."""
        missing = [c for c in self.inputs.cameras if c not in (images or {})]
        if missing:
            raise ValueError(f"no frames for camera {missing}")
        features = [
            _image_features(images[camera], camera, self.inputs, self.device)
            for camera in self.inputs.cameras
        ]
        return self._from_features(features, self._standardise(state))

    @torch.inference_mode()
    def _from_features(
        self, features: list[torch.Tensor], state: torch.Tensor | None
    ) -> np.ndarray:
        self.head.eval()
        logits = self.head([f.to(self.device).float() for f in features], state)
        return torch.sigmoid(logits).float().cpu().numpy()

    def save(self, directory: Path) -> Path:
        from safetensors.torch import save_file  # noqa: PLC0415

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in self.head.state_dict().items()
            },
            str(directory / HEAD_FILE),
        )
        description = {
            "format_version": FORMAT_VERSION,
            "check": self.name,
            "inputs": self.inputs.to_dict(),
            "state_mean": [float(v) for v in self.state_mean],
            "state_std": [float(v) for v in self.state_std],
            "threshold": float(self.threshold),
            "trained": self.trained,
        }
        path = directory / CHECK_FILE
        path.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path, device: str | None = None) -> LearnedCheckModel:
        from safetensors.torch import load_file  # noqa: PLC0415

        directory = Path(directory)
        path = directory / CHECK_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"no learned check at {directory} (no {CHECK_FILE})"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path} is not a format {FORMAT_VERSION} learned check")
        inputs = CheckInputs.from_dict(data["inputs"])
        target = _device(device)
        head = _Head(len(inputs.cameras), inputs.feature_dim, len(inputs.state))
        head.load_state_dict(load_file(str(directory / HEAD_FILE)))
        return cls(
            name=str(data["check"]),
            inputs=inputs,
            head=head.to(target).eval(),
            state_mean=np.asarray(data["state_mean"], dtype=np.float32),
            state_std=np.asarray(data["state_std"], dtype=np.float32),
            threshold=float(data.get("threshold", 0.5)),
            trained=dict(data.get("trained") or {}),
            device=target,
        )


# --- reading recordings ------------------------------------------------------


class DatasetReader:
    """The camera frames and state columns of one LeRobot dataset."""

    def __init__(self, root: Path) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        self.root = Path(root).expanduser().resolve()
        # PyAV, as the dataset preview uses: torchcodec does not load everywhere.
        self.dataset = LeRobotDataset(
            f"local/{self.root.name}", root=self.root, video_backend="pyav"
        )
        self.fps = float(self.dataset.fps)
        features = self.dataset.features
        state = features.get(STATE_KEY) or {}
        names = state.get("names")
        self.state_names: list[str] = list(names) if isinstance(names, list) else []
        self.cameras = [
            key[len(IMAGE_PREFIX) :]
            for key in self.dataset.meta.video_keys
            if key.startswith(IMAGE_PREFIX)
            and not (features[key].get("info") or {}).get("is_depth_map")
        ]
        episodes = self.dataset.meta.episodes
        self._rows = {int(e): row for row, e in enumerate(episodes["episode_index"])}

    def require(self, inputs: CheckInputs) -> None:
        """Refuse a dataset that lacks one of the check's inputs, naming it."""
        missing = [c for c in inputs.cameras if c not in self.cameras]
        if missing:
            raise ValueError(
                f"{self.root.name} has no RGB camera {missing}; it has {self.cameras}"
            )
        absent = [s for s in inputs.state if s not in self.state_names]
        if absent:
            raise ValueError(f"{self.root.name} has no state column {absent}")

    @property
    def episodes(self) -> list[int]:
        return sorted(self._rows)

    def _record(self, episode: int) -> dict[str, Any]:
        if episode not in self._rows:
            raise IndexError(f"episode {episode} is not in {self.root}")
        return dict(self.dataset.meta.episodes[self._rows[episode]])

    def columns(self, episode: int, keys: Sequence[str]) -> dict[str, np.ndarray]:
        """Whole columns of one episode, such as `timestamp` or `action`."""
        record = self._record(episode)
        rows = self.dataset.hf_dataset.select(
            range(int(record["dataset_from_index"]), int(record["dataset_to_index"]))
        ).with_format("numpy")
        return {key: np.asarray(rows[key]) for key in keys}

    def timestamps_and_state(
        self, episode: int, columns: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Every frame's timestamp, and the named state columns."""
        keys = ["timestamp", STATE_KEY] if columns else ["timestamp"]
        rows = self.columns(episode, keys)
        times = rows["timestamp"].astype(np.float64).reshape(-1)
        if not columns:
            return times, np.zeros((len(times), 0), dtype=np.float32)
        index = [self.state_names.index(c) for c in columns]
        state = rows[STATE_KEY].astype(np.float32)[:, index]
        return times, state

    def frames(
        self, episode: int, camera: str, times: np.ndarray
    ) -> Iterator[torch.Tensor]:
        """The camera's frames at the given episode times, `_CHUNK` at a time."""
        from lerobot.datasets.video_utils import decode_video_frames  # noqa: PLC0415

        key = IMAGE_PREFIX + camera
        record = self._record(episode)
        start = float(record[f"videos/{key}/from_timestamp"])
        path = self.root / self.dataset.meta.get_video_file_path(episode, key)
        for i in range(0, len(times), _CHUNK):
            query = [start + float(t) for t in times[i : i + _CHUNK]]
            yield decode_video_frames(
                path, query, self.dataset.tolerance_s, backend="pyav", return_uint8=True
            )


def _encode(
    reader: DatasetReader,
    episode: int,
    indices: np.ndarray | None,
    inputs: CheckInputs,
    device: torch.device,
) -> tuple[list[torch.Tensor], np.ndarray, np.ndarray]:
    """Image features (float16, on `device`), raw state and times of the
    frames at `indices`, or of every frame."""
    times, state = reader.timestamps_and_state(episode, inputs.state)
    if indices is not None:
        times, state = times[indices], state[indices]
    features = []
    for camera in inputs.cameras:
        parts = [
            _image_features(chunk, camera, inputs, device).half()
            for chunk in reader.frames(episode, camera, times)
        ]
        features.append(
            torch.cat(parts)
            if parts
            else torch.zeros((0, inputs.feature_dim), device=device)
        )
    return features, state, times


def score_episode(
    model: LearnedCheckModel,
    root: Path,
    episode: int = 0,
    reader: DatasetReader | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Each frame's time and probability, for one recorded episode."""
    reader = reader or DatasetReader(root)
    reader.require(model.inputs)
    features, state, times = _encode(reader, episode, None, model.inputs, model.device)
    if len(times) == 0:
        return times, np.zeros(0, dtype=np.float32)
    return times, model._from_features(features, model._standardise(state))


_MODEL_CACHE: dict[Path, tuple[int, LearnedCheckModel]] = {}


def load_model(directory: Path) -> LearnedCheckModel:
    """A saved check, loaded once per process until its files change."""
    directory = Path(directory).expanduser().resolve()
    path = directory / CHECK_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no learned check at {directory} (no {CHECK_FILE})")
    stamp = path.stat().st_mtime_ns + (directory / HEAD_FILE).stat().st_mtime_ns
    cached = _MODEL_CACHE.get(directory)
    if cached is None or cached[0] != stamp:
        cached = (stamp, LearnedCheckModel.load(directory))
        _MODEL_CACHE[directory] = cached
    return cached[1]


def judge(
    times: Sequence[float],
    probabilities: Sequence[float],
    threshold: float,
    hold_s: float,
    when: str = "any",
) -> dict[str, Any]:
    """Whether the probability stayed at or above `threshold` for `hold_s`."""
    runs = held_runs(list(times), [bool(p >= threshold) for p in probabilities])
    qualifying = [r for r in runs if r[1] - r[0] >= hold_s]
    if when == "end":
        qualifying = [r for r in qualifying if r[2]]
    return {
        "passed": bool(qualifying),
        "held_s": round(max((r[1] - r[0] for r in runs), default=0.0), 3),
        "first_at_s": round(qualifying[0][0], 3) if qualifying else None,
        "max_p": round(float(max(probabilities, default=0.0)), 4),
    }


# --- training ----------------------------------------------------------------


@dataclass(frozen=True)
class LabelledEpisode:
    """One labelled episode: where it is, and a target per frame."""

    root: Path
    episode: int
    targets: np.ndarray
    source: str  # "label" or "verdict"

    @property
    def reached(self) -> bool:
        return bool((self.targets == 1).any())


@dataclass(frozen=True)
class TrainingOptions:
    holdout: float = 0.2
    #: Left out just before each labelled first frame.
    margin_s: float = 0.5
    #: Use recorded rollouts' verdicts where an episode has no label.
    use_verdicts: bool = False
    #: A "success" verdict's positives: this much at the end of the rollout.
    success_tail_s: float = 0.5
    #: Train on every n-th labelled frame; held-out episodes are scored whole.
    stride: int = 1
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    hold_s: float = 0.2
    seed: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.holdout < 1.0:
            raise ValueError("holdout must be at least 0 and below 1")
        if self.stride < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("stride, epochs and batch_size must be at least 1")
        if self.margin_s < 0 or self.success_tail_s <= 0 or self.hold_s < 0:
            raise ValueError(
                "margin_s and hold_s must be 0 or more; success_tail_s above 0"
            )


def _dataset_roots(source: Path) -> list[Path]:
    """A dataset, or every dataset directly inside a folder (a session)."""
    source = Path(source).expanduser()
    if (source / "meta" / "info.json").is_file():
        return [source]
    roots = sorted(p.parent.parent for p in source.glob("*/meta/info.json"))
    if not roots:
        raise FileNotFoundError(f"{source} is neither a dataset nor a folder of them")
    return roots


def collect_episodes(
    sources: Sequence[Path], check: str, options: TrainingOptions
) -> list[LabelledEpisode]:
    """Every labelled episode of `check` in the sources, with per-frame targets."""
    found: list[LabelledEpisode] = []
    for source in sources:
        for root in _dataset_roots(source):
            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            fps = float(info.get("fps") or 30)
            lengths = episode_lengths(root)
            labels = labels_for(root, check)
            for episode, first in sorted(labels.items()):
                if episode not in lengths:
                    raise ValueError(f"{root}: label for missing episode {episode}")
                targets = frame_targets(
                    lengths[episode], first, round(options.margin_s * fps)
                )
                found.append(LabelledEpisode(root, episode, targets, "label"))
            if options.use_verdicts and len(lengths) == 1 and 0 not in labels:
                targets = verdict_targets(
                    lengths[0], read_outcome(root), round(options.success_tail_s * fps)
                )
                if targets is not None:
                    found.append(LabelledEpisode(root, 0, targets, "verdict"))
    return found


def _split(
    episodes: list[LabelledEpisode], holdout: float, seed: int
) -> tuple[list[LabelledEpisode], list[LabelledEpisode]]:
    """Hold out whole episodes, in the same proportion from those that reach
    the state and those that never do, keeping at least one of each to train."""
    rng = random.Random(seed)
    train: list[LabelledEpisode] = []
    held: list[LabelledEpisode] = []
    for group in (
        [e for e in episodes if e.reached],
        [e for e in episodes if not e.reached],
    ):
        group = list(group)
        rng.shuffle(group)
        count = min(int(round(len(group) * holdout)), max(len(group) - 1, 0))
        held += group[:count]
        train += group[count:]
    return train, held


def _first_pass(
    times: np.ndarray, probabilities: np.ndarray, threshold: float, hold_s: float
) -> int | None:
    """The frame where the first stretch long enough to pass begins."""
    runs = held_runs(list(times), [bool(p >= threshold) for p in probabilities])
    starts = [r[0] for r in runs if r[1] - r[0] >= hold_s]
    if not starts:
        return None
    return int(np.argmin(np.abs(np.asarray(times) - starts[0])))


def _outcome(episode: LabelledEpisode, first: int | None) -> str:
    """How the check did on one labelled episode.

    "false pass": it first passed on a frame labelled as not holding -- on an
    episode that never gets there, or before the labelled first frame of one
    that does. Whatever runs after the check would start from a failure.
    "missed": the state was reached and the check never passed.
    """
    if first is None:
        return "missed" if episode.reached else "no pass"
    return "false pass" if episode.targets[first] == 0 else "passed"


def _report(
    held: list[LabelledEpisode],
    scored: list[tuple[np.ndarray, np.ndarray]],
    threshold: float,
    hold_s: float,
) -> dict[str, Any]:
    """Per held-out episode: would the check have passed it, and was that
    right? Then the totals, at the model's threshold and across a sweep."""
    episodes = []
    for episode, (times, probs) in zip(held, scored, strict=True):
        positives = np.flatnonzero(episode.targets == 1)
        episodes.append(
            {
                "dataset": str(episode.root),
                "episode": episode.episode,
                "source": episode.source,
                "reached": episode.reached,
                "labelled_at_s": round(float(times[positives[0]]), 3)
                if episode.source == "label" and len(positives)
                else None,
                "result": _outcome(
                    episode, _first_pass(times, probs, threshold, hold_s)
                ),
                **judge(times, probs, threshold, hold_s),
            }
        )

    def totals(t: float) -> dict[str, Any]:
        results = [
            _outcome(e, _first_pass(times, probs, t, hold_s))
            for e, (times, probs) in zip(held, scored, strict=True)
        ]
        return {
            "threshold": t,
            "false_passes": results.count("false pass"),
            "misses": results.count("missed"),
        }

    known = [
        (episode.targets[episode.targets != IGNORED], probs[episode.targets != IGNORED])
        for episode, (_, probs) in zip(held, scored, strict=True)
    ]
    targets = np.concatenate([k[0] for k in known]) if known else np.zeros(0)
    predicted = (
        np.concatenate([k[1] for k in known]) >= threshold if known else np.zeros(0)
    )
    return {
        "episodes_reached": sum(1 for e in held if e.reached),
        "episodes_never": sum(1 for e in held if not e.reached),
        **totals(threshold),
        "frames": {
            "true_pos": int(((predicted == 1) & (targets == 1)).sum()),
            "false_pos": int(((predicted == 1) & (targets == 0)).sum()),
            "false_neg": int(((predicted == 0) & (targets == 1)).sum()),
            "true_neg": int(((predicted == 0) & (targets == 0)).sum()),
        },
        "sweep": [totals(t) for t in _SWEEP],
        "episodes": episodes,
    }


def train_check(
    sources: Sequence[Path],
    check: str,
    inputs: CheckInputs,
    out: Path,
    options: TrainingOptions | None = None,
    *,
    log: Any = None,
) -> dict[str, Any]:
    """Train `check` on the labelled episodes in `sources`, save it to `out`,
    and return the held-out report."""
    options = options or TrainingOptions()
    say = log or (lambda message: None)
    episodes = collect_episodes(sources, check, options)
    positives = sum(int((e.targets == 1).sum()) for e in episodes)
    negatives = sum(int((e.targets == 0).sum()) for e in episodes)
    if not positives or not negatives:
        raise ValueError(
            f"check {check!r} needs frames where its state holds and frames where it "
            f"does not; the labels give {positives} and {negatives}"
        )
    train, held = _split(episodes, options.holdout, options.seed)
    if not any(e.reached for e in train) or all((e.targets != 0).all() for e in train):
        raise ValueError(
            "after holding out episodes, training has no positive or no negative "
            "frames; label more episodes or lower holdout"
        )
    device = _device(options.device)
    torch.manual_seed(options.seed)
    readers: dict[Path, DatasetReader] = {}

    def reader(root: Path) -> DatasetReader:
        if root not in readers:
            readers[root] = DatasetReader(root)
            readers[root].require(inputs)
        return readers[root]

    started = time.monotonic()
    images: list[list[torch.Tensor]] = [[] for _ in inputs.cameras]
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for number, episode in enumerate(train, 1):
        indices = np.flatnonzero(episode.targets != IGNORED)[:: options.stride]
        features, state, _ = _encode(
            reader(episode.root), episode.episode, indices, inputs, device
        )
        for store, f in zip(images, features, strict=True):
            store.append(f)
        states.append(state)
        targets.append(episode.targets[indices])
        say(f"encoded {number}/{len(train)} training episodes")
    x_images = [torch.cat(store) for store in images]
    x_state = np.concatenate(states).astype(np.float32)
    y = torch.as_tensor(np.concatenate(targets).astype(np.float32), device=device)

    mean = x_state.mean(axis=0) if inputs.state else np.zeros(0, np.float32)
    std = x_state.std(axis=0) if inputs.state else np.zeros(0, np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    model = LearnedCheckModel(
        name=check,
        inputs=inputs,
        head=_Head(len(inputs.cameras), inputs.feature_dim, len(inputs.state)).to(
            device
        ),
        state_mean=mean.astype(np.float32),
        state_std=std,
        device=device,
    )
    s_all = model._standardise(x_state)

    # Balanced: a check that is mostly shown negatives would learn to say no.
    n_pos = float(y.sum())
    pos_weight = torch.tensor((len(y) - n_pos) / max(n_pos, 1.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(
        model.head.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(options.seed)
    losses = []
    for epoch in range(options.epochs):
        model.head.train()
        order = torch.randperm(len(y), generator=generator).to(device)
        total = 0.0
        for i in range(0, len(y), options.batch_size):
            batch = order[i : i + options.batch_size]
            logits = model.head(
                [x[batch].float() for x in x_images],
                s_all[batch] if s_all is not None else None,
            )
            loss = loss_fn(logits, y[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        losses.append(round(total / len(y), 5))
        say(f"epoch {epoch + 1}/{options.epochs}: loss {losses[-1]}")
    model.head.eval()

    scored = []
    for number, episode in enumerate(held, 1):
        scored.append(
            score_episode(model, episode.root, episode.episode, reader(episode.root))
        )
        say(f"scored {number}/{len(held)} held-out episodes")
    report = _report(held, scored, model.threshold, options.hold_s) if held else None
    model.trained = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": [str(Path(s)) for s in sources],
        "episodes": {"train": len(train), "held_out": len(held)},
        "frames": {"train": int(len(y)), "positive": int(n_pos)},
        "options": {k: v for k, v in asdict(options).items() if k != "device"},
        "device": str(device),
        "seconds": round(time.monotonic() - started, 1),
        "loss": losses,
        "held_out": None
        if report is None
        else {k: report[k] for k in ("false_passes", "misses", "frames")},
    }
    model.save(out)
    return {
        "check": check,
        "saved": str(Path(out)),
        "trained": model.trained,
        "held_out": report,
    }


# --- command line ------------------------------------------------------------


def _parse_crop(text: str) -> tuple[str, tuple[float, float, float, float]]:
    name, _, box = text.partition("=")
    values = tuple(float(v) for v in box.split(","))
    if not name or len(values) != 4:
        raise argparse.ArgumentTypeError("a crop is CAMERA=x0,y0,x1,y1 (fractions)")
    return name, values


def _parse_size(text: str) -> tuple[int, int]:
    height, _, width = text.lower().partition("x")
    try:
        return int(height), int(width)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "a size is HEIGHTxWIDTH, e.g. 224x224"
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m flexivtrainer.evaluation.learned_check",
        description="Label episodes, train a learned check, and score recordings.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    label = commands.add_parser("label", help="label one episode of a dataset")
    label.add_argument("dataset", type=Path)
    label.add_argument("--check", required=True)
    label.add_argument("--episode", type=int, required=True)
    which = label.add_mutually_exclusive_group(required=True)
    which.add_argument("--first-frame", type=int, help="first frame the state holds")
    which.add_argument("--never", action="store_true", help="it never holds")
    which.add_argument("--remove", action="store_true", help="remove the label")

    labels = commands.add_parser("labels", help="list a dataset's labels")
    labels.add_argument("dataset", type=Path)

    train = commands.add_parser("train", help="train a check on labelled episodes")
    train.add_argument("--check", required=True)
    train.add_argument(
        "--data",
        type=Path,
        action="append",
        required=True,
        help="a dataset, or a folder of datasets; repeatable",
    )
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--camera", action="append", default=[])
    train.add_argument(
        "--state",
        action="append",
        default=[],
        help="an observation.state column name; repeatable",
    )
    train.add_argument("--crop", type=_parse_crop, action="append", default=[])
    train.add_argument("--encoder", choices=ENCODERS, default="resnet18")
    train.add_argument("--grid", type=int, default=2)
    train.add_argument("--image-size", type=_parse_size, default=(224, 224))
    train.add_argument(
        "--verdicts",
        action="store_true",
        help="use rollout verdicts where an episode has no label",
    )
    defaults = TrainingOptions()
    for flag, kind in (
        ("holdout", float),
        ("margin_s", float),
        ("success_tail_s", float),
        ("stride", int),
        ("epochs", int),
        ("hold_s", float),
        ("seed", int),
    ):
        train.add_argument(
            "--" + flag.replace("_", "-"), type=kind, default=getattr(defaults, flag)
        )
    train.add_argument("--device", default=None)

    score = commands.add_parser("score", help="score a dataset's episodes")
    score.add_argument("dataset", type=Path)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--episode", type=int, action="append")
    score.add_argument("--threshold", type=float, default=None)
    score.add_argument("--hold-s", type=float, default=0.2)
    score.add_argument(
        "--frames", action="store_true", help="also print every frame's probability"
    )
    score.add_argument("--device", default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "label":
            if args.remove:
                done = remove_label(args.dataset, args.check, args.episode)
                result: Any = {"removed": done}
            else:
                first = None if args.never else args.first_frame
                write_label(args.dataset, args.check, args.episode, first)
                result = {"labelled": {"episode": args.episode, "first_frame": first}}
        elif args.command == "labels":
            result = {
                name: {"summary": describe(episodes), "episodes": episodes}
                for name, episodes in read_labels(args.dataset).items()
            }
        elif args.command == "train":
            inputs = CheckInputs(
                cameras=tuple(args.camera),
                state=tuple(args.state),
                image_size=args.image_size,
                crops=dict(args.crop),
                encoder=args.encoder,
                grid=args.grid,
            )
            options = TrainingOptions(
                holdout=args.holdout,
                margin_s=args.margin_s,
                use_verdicts=args.verdicts,
                success_tail_s=args.success_tail_s,
                stride=args.stride,
                epochs=args.epochs,
                hold_s=args.hold_s,
                seed=args.seed,
                device=args.device,
            )
            result = train_check(
                args.data,
                args.check,
                inputs,
                args.out,
                options,
                log=lambda message: print(message, file=sys.stderr),
            )
        else:
            model = LearnedCheckModel.load(args.model, args.device)
            threshold = model.threshold if args.threshold is None else args.threshold
            reader = DatasetReader(args.dataset)
            episodes = args.episode or sorted(episode_lengths(args.dataset))
            result = {"check": model.name, "threshold": threshold, "episodes": []}
            for episode in episodes:
                times, probs = score_episode(model, args.dataset, episode, reader)
                entry = {
                    "episode": episode,
                    **judge(times, probs, threshold, args.hold_s),
                }
                if args.frames:
                    entry["p"] = [round(float(p), 4) for p in probs]
                result["episodes"].append(entry)
    except (FileNotFoundError, IndexError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
