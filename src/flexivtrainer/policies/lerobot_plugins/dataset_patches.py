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

"""Runtime patches for LeRobot's dataset read path, installed before make_dataset.

Robust seek: merged datasets pack episodes into one mp4, and a segment boundary
landing beside an existing keyframe leaves two in a row both claiming
picture-order-count zero. LeRobot seeks one tick before its target, which lands
on the previous segment and forces a read across that boundary -- where the HEVC
decoder drops the second IDR instead of emitting it. The frame is in the file; only
random access to it fails. Seeking onto the target's own keyframe avoids the cross.

Depth filter: the reader decodes every video feature a dataset declares, whatever
the policy asked for. Opt-in via TRAIN_LOAD_DEPTH_ENV, which only the training
subprocess sets -- the app imports this package too and still needs depth.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from flexivtrainer.config import TRAIN_LOAD_DEPTH_ENV

logger = logging.getLogger(__name__)

LOAD_DEPTH_ENV = TRAIN_LOAD_DEPTH_ENV

_TRUTHY = {"1", "true", "yes", "on"}


def depth_filter_requested() -> bool:
    """Whether to drop depth features from metadata loaded in this process."""

    raw = os.environ.get(LOAD_DEPTH_ENV)
    if raw is None:
        return False
    return raw.strip().lower() not in _TRUTHY


def is_depth_feature(feature: dict) -> bool:
    """Mirror of LeRobot's ``LeRobotDatasetMetadata.depth_keys`` predicate."""

    info = feature.get("info") or {}
    video_info = feature.get("video_info") or {}
    return bool(
        info.get("is_depth_map", False)
        or info.get("video.is_depth_map", False)
        or video_info.get("video.is_depth_map", False)
    )


def _decode_from(
    container,
    stream,
    timestamps: list[float],
    seek_pts: int,
    *,
    any_frame: bool,
    is_depth: bool,
    log_loaded_timestamps: bool,
) -> tuple[list[torch.Tensor], list[float]]:
    """Frame handling of ``decode_video_frames_pyav``, with the seek target given."""

    last_ts = max(timestamps)
    frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []
    container.seek(seek_pts, backward=True, any_frame=any_frame, stream=stream)
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        current_ts = float(frame.pts * stream.time_base)
        if log_loaded_timestamps:
            logger.info(f"frame loaded at timestamp={current_ts:.4f}")
        if is_depth:
            array = frame.to_ndarray(format="gray12le")
            frames.append(torch.from_numpy(array).unsqueeze(0).contiguous())
        else:
            array = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break
    return frames, loaded_ts


def _covers(timestamps: list[float], loaded_ts: list[float], tolerance_s: float):
    """Return ``(ok, argmin)`` for matching each query timestamp to a loaded one."""

    if not loaded_ts:
        return False, None
    distance = torch.cdist(
        torch.tensor(timestamps)[:, None], torch.tensor(loaded_ts)[:, None], p=1
    )
    minimum, argmin = distance.min(1)
    return bool((minimum < tolerance_s).all()), argmin


def robust_decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    return_uint8: bool = False,
    is_depth: bool = False,
) -> torch.Tensor:
    """``decode_video_frames_pyav`` that retries the seek before giving up."""

    import av
    from lerobot.datasets.video_utils import FrameTimestampError

    video_path = str(video_path)
    first_ts = min(timestamps)

    frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []
    argmin = None
    used = None
    labels: list[str] = []
    # Retries reuse this container; reopening per attempt cost ~50% on the common
    # path, and seek() flushes the decoder anyway.
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        target_pts = round(first_ts / stream.time_base)
        attempts = [
            ("upstream", target_pts - 1, False),
            ("target-keyframe", target_pts, False),
            ("any-frame", target_pts, True),
            ("linear", 0, False),
        ]
        labels = [label for label, *_ in attempts]
        for label, seek_pts, any_frame in attempts:
            try:
                frames, loaded_ts = _decode_from(
                    container,
                    stream,
                    timestamps,
                    seek_pts,
                    any_frame=any_frame,
                    is_depth=is_depth,
                    log_loaded_timestamps=log_loaded_timestamps,
                )
            except av.error.FFmpegError:
                continue
            ok, argmin = _covers(timestamps, loaded_ts, tolerance_s)
            if ok:
                used = label
                break

    if used is None:
        raise FrameTimestampError(
            "One or several query timestamps unexpectedly violate the tolerance "
            f"({tolerance_s=}) after retrying every seek strategy "
            f"({', '.join(labels)})."
            f"\nqueried timestamps: {torch.tensor(timestamps)}"
            f"\nloaded timestamps: {torch.tensor(loaded_ts)}"
            f"\nvideo: {video_path}"
            "\nbackend: pyav"
        )

    if used != "upstream":
        logger.warning(
            "Recovered %s via the %r seek after the default seek missed it "
            "(likely an episode boundary with adjacent keyframes).",
            f"{video_path}@{first_ts:.4f}s",
            used,
        )

    closest = torch.stack([frames[index] for index in argmin])
    if len(timestamps) != len(closest):
        raise FrameTimestampError(
            f"Number of retrieved frames ({len(closest)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )
    if return_uint8 or is_depth:
        return closest
    return closest.type(torch.float32) / 255


_original_decode_pyav = None


def original_decode_video_frames_pyav():
    """LeRobot's unpatched pyav decode, so tests can compare against it."""

    return _original_decode_pyav


def _patch_robust_seek() -> None:
    global _original_decode_pyav

    from lerobot.datasets import video_utils

    if not hasattr(video_utils, "decode_video_frames_pyav"):
        raise RuntimeError(
            "LeRobot changed shape: video_utils.decode_video_frames_pyav is gone, "
            "so the robust-seek patch would silently stop applying."
        )
    if getattr(video_utils.decode_video_frames_pyav, "_flexiv_patched", False):
        return
    _original_decode_pyav = video_utils.decode_video_frames_pyav
    robust_decode_video_frames_pyav._flexiv_patched = True
    # Patching the module global also covers dataset_reader's from-import of the
    # decode_video_frames dispatcher, which resolves this name at call time.
    video_utils.decode_video_frames_pyav = robust_decode_video_frames_pyav


def _patch_depth_filter() -> None:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    if not hasattr(LeRobotDatasetMetadata, "depth_keys"):
        raise RuntimeError(
            "LeRobot changed shape: LeRobotDatasetMetadata.depth_keys is gone, so "
            "the depth filter would silently stop applying."
        )
    original_init = LeRobotDatasetMetadata.__init__
    if getattr(original_init, "_flexiv_patched", False):
        return

    def __init__(self, *args, **kwargs):  # noqa: N807 - patching a dunder
        original_init(self, *args, **kwargs)
        if not depth_filter_requested():
            return
        features = self.info.features
        dropped = [key for key, ft in features.items() if is_depth_feature(ft)]
        if not dropped:
            return
        # Only this load path, never create(): video_keys, delta_timestamps and
        # the policy's inferred inputs then agree on being depth-free.
        self.info.features = {
            key: ft for key, ft in features.items() if key not in dropped
        }
        logger.info(
            "Excluded %d depth feature(s) from training: %s (set %s=1 to keep them).",
            len(dropped),
            ", ".join(dropped),
            LOAD_DEPTH_ENV,
        )

    __init__._flexiv_patched = True
    LeRobotDatasetMetadata.__init__ = __init__


def apply_patches() -> None:
    """Install both dataset patches. Safe to call more than once."""

    _patch_robust_seek()
    _patch_depth_filter()
