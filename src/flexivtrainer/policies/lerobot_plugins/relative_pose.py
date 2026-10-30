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

"""Chunk-relative SE(3) pose actions, as in UMI.

A policy trained on absolute poses learns the frame they were recorded in: the
SLAM map for a handheld gripper, the robot base for teleoperation. Expressing
each action of a chunk in the frame of the gripper's pose when the chunk was
predicted, ``T_rel = T_cur^-1 * T_action``, removes that frame, so a policy
trained in one rolls out in the other as long as the tool tip is the same.
LeRobot's own relative actions subtract per dimension, which is right for
joints but not for a rotation, and leaves a translation in the world frame.

Poses are rewritten in place (position, then rotation-6D in the dataset's own
convention), so no feature changes shape; every other dimension (gripper,
twist, wrench) passes through untouched.

The pose in ``observation.state`` is replaced by the identity rather than made
relative to the newest frame. A policy with an observation history queues each
frame after preprocessing, so at rollout an older frame can never be
re-expressed relative to a pose that did not exist when it was queued; the
identity is the one value training and rollout agree on. Motion is still in
the images and in any twist the state carries.

Training: ``TRAIN_RELATIVE_POSE_ENV`` carries the layout (``pose_layout``) into
the lerobot-train subprocess. There the dataset's action statistics are
recomputed in the relative frame, and new ACT / Diffusion / DiT processors get
the step pair around their normalizers. Both are saved with the checkpoint, so
rollout needs no setting. The pair shares the anchor pose through ``_ANCHORS``
by ``pair_id``, which survives the round trip through the processor JSON.

The rotation columns of UMI's extra pose inputs (recent motion, rotation since
start; ``rollout/pose_history.py``) get statistics that normalize them to
themselves, as UMI leaves rotations unnormalized: their diagonal entries stay
near 1, and dividing by that tiny spread blows small turns up into huge inputs.
The gripper width, in state and action, gets statistics under which mean/std
maps its recorded range onto [-1, 1], as UMI's range normalizer does; its
values stay in metres.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    NormalizerProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    UnnormalizerProcessorStep,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from flexivtrainer.config import TRAIN_RELATIVE_POSE_ENV
from flexivtrainer.data.lerobot_io import ROTATION_6D_AXES
from flexivtrainer.rollout.pose_history import history_columns

logger = logging.getLogger(__name__)

RELATIVE_POSE_STEP = "flexiv_relative_pose"
ABSOLUTE_POSE_STEP = "flexiv_absolute_pose"
# Families whose processors are built by a factory LeRobot imports lazily, so
# patching that module attribute reaches lerobot-train. B-spline actions are
# spline coefficients over a converted dataset and are not covered.
RELATIVE_POSE_POLICIES = ("act", "diffusion", "multi_task_dit")
_FAMILY_FACTORIES = (
    ("lerobot.policies.act.processor_act", "make_act_pre_post_processors"),
    (
        "lerobot.policies.diffusion.processor_diffusion",
        "make_diffusion_pre_post_processors",
    ),
    (
        "lerobot.policies.multi_task_dit.processor_multi_task_dit",
        "make_multi_task_dit_pre_post_processors",
    ),
)

# Recorded by this repository: "<side>.tcp_pose.x" + rows of R as r1_*, r2_*.
_TCP_POSITION = ("tcp_pose.x", "tcp_pose.y", "tcp_pose.z")
_TCP_ROTATION = tuple(f"tcp_rotation_6d.{axis}" for axis in ROTATION_6D_AXES)
# Exported by flexiv_gdk's UMI pipeline: bare x, y, z + columns of R as r<row><col>.
_UMI_POSITION = ("x", "y", "z")
_UMI_ROTATION = ("r00", "r10", "r20", "r01", "r11", "r21")
# Position zero and R = I, in either convention.
_IDENTITY_POSE = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

# Statistics under which a column normalizes to itself in every LeRobot mode:
# MEAN_STD, MIN_MAX, QUANTILES (q01, q99) and QUANTILE10 (q10, q90).
_UNNORMALIZED_STATS = {
    "mean": 0.0,
    "std": 1.0,
    "min": -1.0,
    "max": 1.0,
    "q01": -1.0,
    "q99": 1.0,
    "q10": -1.0,
    "q90": 1.0,
}
# "<side>.gripper.width" in metres, or the UMI layout's normalized opening.
_GRIPPER_WIDTH = re.compile(r"^(?:[^.]+\.gripper\.width|gripper)$")
# A narrower recorded range is left as is, as UMI's range normalizer does.
_MIN_WIDTH_RANGE = 1e-7

# pair_id -> one (position, rotation) anchor per state pose group.
_ANCHORS: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}


def _feature_names(features: dict[str, Any], key: str) -> list[str]:
    names = (features.get(key) or {}).get("names")
    if isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{key} has no list of per-dimension names")
    return names


def _pose_groups(names: Sequence[str], key: str) -> list[dict[str, Any]]:
    index = {name: i for i, name in enumerate(names)}
    groups = []
    for name in names:
        if name.endswith(_TCP_POSITION[0]):
            prefix = name[: -len(_TCP_POSITION[0])]
            position, rotation, convention = _TCP_POSITION, _TCP_ROTATION, "rows"
        elif name == _UMI_POSITION[0]:
            prefix = ""
            position, rotation, convention = _UMI_POSITION, _UMI_ROTATION, "columns"
        else:
            continue
        wanted = [prefix + n for n in (*position, *rotation)]
        missing = [n for n in wanted if n not in index]
        if missing:
            raise ValueError(
                f"{key} has a pose at {name!r} but no {', '.join(missing)}; "
                "relative poses need position and rotation-6D (a quaternion "
                "layout is not supported)"
            )
        groups.append(
            {
                "prefix": prefix,
                "position": [index[n] for n in wanted[:3]],
                "rotation": [index[n] for n in wanted[3:]],
                "convention": convention,
            }
        )
    return groups


def pose_layout(features: dict[str, Any]) -> dict[str, Any]:
    """Where the poses sit in a dataset's state and action, by feature name.

    Each action pose is anchored to the state pose with the same prefix (the
    same arm). Raises ``ValueError`` when the dataset has no pose to convert or
    an action pose has no state pose to anchor to.
    """
    state = _pose_groups(_feature_names(features, OBS_STATE), OBS_STATE)
    action = _pose_groups(_feature_names(features, ACTION), ACTION)
    if not action:
        raise ValueError("action has no Cartesian pose to make relative")
    prefixes = [group["prefix"] for group in state]
    for group in action:
        if group["prefix"] not in prefixes:
            raise ValueError(
                f"action pose {group['prefix'] or '(unprefixed)'!r} has no "
                "matching pose in observation.state to anchor it"
            )
        group["anchor"] = prefixes.index(group["prefix"])
    return {"state": state, "action": action}


def requested_layout() -> dict[str, Any] | None:
    """The layout the training job asked for, or None when it did not."""
    raw = os.environ.get(TRAIN_RELATIVE_POSE_ENV, "").strip()
    return json.loads(raw) if raw else None


def checkpoint_pose_layout(model_dir: Path) -> dict[str, Any] | None:
    """The layout a checkpoint's preprocessor converts with, if it has one."""
    path = Path(model_dir) / "policy_preprocessor.json"
    try:
        steps = json.loads(path.read_text(encoding="utf-8")).get("steps") or []
    except (OSError, json.JSONDecodeError):
        return None
    for step in steps:
        if step.get("registry_name") == RELATIVE_POSE_STEP:
            return (step.get("config") or {}).get("layout")
    return None


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor, convention: str) -> torch.Tensor:
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    basis_1 = torch.nn.functional.normalize(first, dim=-1)
    basis_2 = torch.nn.functional.normalize(
        second - (basis_1 * second).sum(-1, keepdim=True) * basis_1, dim=-1
    )
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    rows = torch.stack([basis_1, basis_2, basis_3], dim=-2)
    return rows if convention == "rows" else rows.transpose(-1, -2)


def _matrix_to_rotation_6d(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    if convention != "rows":
        matrix = matrix.transpose(-1, -2)
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def _read_pose(
    values: torch.Tensor, group: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    position = values[..., group["position"]]
    rotation = _rotation_6d_to_matrix(
        values[..., group["rotation"]], group["convention"]
    )
    return position, rotation


def _write_pose(
    values: torch.Tensor,
    group: dict[str, Any],
    position: torch.Tensor,
    rotation: torch.Tensor,
) -> None:
    values[..., group["position"]] = position.to(values.dtype)
    values[..., group["rotation"]] = _matrix_to_rotation_6d(
        rotation, group["convention"]
    ).to(values.dtype)


def _broadcast_anchor(
    anchor: tuple[torch.Tensor, torch.Tensor], position: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give a (B, ...) anchor the time axes of a (B, T, ...) chunk."""
    anchor_position, anchor_rotation = anchor
    anchor_position = anchor_position.to(position)
    anchor_rotation = anchor_rotation.to(position)
    while anchor_position.ndim < position.ndim:
        anchor_position = anchor_position.unsqueeze(1)
        anchor_rotation = anchor_rotation.unsqueeze(1)
    return anchor_position, anchor_rotation


def to_relative(
    actions: torch.Tensor,
    anchors: list[tuple[torch.Tensor, torch.Tensor]],
    layout: dict[str, Any],
) -> torch.Tensor:
    """``T_cur^-1 * T`` for each action pose; other dimensions unchanged."""
    actions = actions.clone()
    for group in layout["action"]:
        position, rotation = _read_pose(actions, group)
        anchor_position, anchor_rotation = _broadcast_anchor(
            anchors[group["anchor"]], position
        )
        inverse = anchor_rotation.transpose(-1, -2)
        _write_pose(
            actions,
            group,
            (inverse @ (position - anchor_position).unsqueeze(-1)).squeeze(-1),
            inverse @ rotation,
        )
    return actions


def to_absolute(
    actions: torch.Tensor,
    anchors: list[tuple[torch.Tensor, torch.Tensor]],
    layout: dict[str, Any],
) -> torch.Tensor:
    """``T_cur * T_rel`` for each action pose; inverse of ``to_relative``."""
    actions = actions.clone()
    for group in layout["action"]:
        position, rotation = _read_pose(actions, group)
        anchor_position, anchor_rotation = _broadcast_anchor(
            anchors[group["anchor"]], position
        )
        _write_pose(
            actions,
            group,
            (anchor_rotation @ position.unsqueeze(-1)).squeeze(-1) + anchor_position,
            anchor_rotation @ rotation,
        )
    return actions


def state_anchors(
    state: torch.Tensor, layout: dict[str, Any]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The newest frame's pose of each state group, (B, 3) and (B, 3, 3)."""
    newest = state[:, -1] if state.ndim == 3 else state
    return [_read_pose(newest, group) for group in layout["state"]]


def _identity_state(state: torch.Tensor, layout: dict[str, Any]) -> torch.Tensor:
    state = state.clone()
    identity = torch.tensor(_IDENTITY_POSE, dtype=state.dtype, device=state.device)
    for group in layout["state"]:
        state[..., group["position"] + group["rotation"]] = identity
    return state


@ProcessorStepRegistry.register(RELATIVE_POSE_STEP)
@dataclass
class RelativePoseProcessorStep(ProcessorStep):
    """Anchor on the newest state pose; actions relative to it, state pose to I.

    ``hold_anchor`` keeps the anchor of the chunk being executed while the
    rollout pops cached actions from it (set per tick by the runner).
    """

    layout: dict[str, Any] = field(default_factory=dict)
    pair_id: str = ""
    hold_anchor: bool = field(default=False, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None:
            return transition
        anchors = state_anchors(state, self.layout)
        if not (self.hold_anchor and self.pair_id in _ANCHORS):
            _ANCHORS[self.pair_id] = [
                (position.detach(), rotation.detach())
                for position, rotation in anchors
            ]
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = {
            **observation,
            OBS_STATE: _identity_state(state, self.layout),
        }
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            new_transition[TransitionKey.ACTION] = to_relative(
                action, anchors, self.layout
            )
        return new_transition

    def reset(self) -> None:
        _ANCHORS.pop(self.pair_id, None)
        self.hold_anchor = False

    def get_config(self) -> dict[str, Any]:
        return {"layout": self.layout, "pair_id": self.pair_id}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(ABSOLUTE_POSE_STEP)
@dataclass
class AbsolutePoseProcessorStep(ProcessorStep):
    """Return predicted relative poses to the frame of their anchor."""

    layout: dict[str, Any] = field(default_factory=dict)
    pair_id: str = ""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        anchors = _ANCHORS.get(self.pair_id)
        if anchors is None:
            raise RuntimeError(
                "No anchor pose for this relative-pose policy: the preprocessor "
                "must run on an observation before its actions are postprocessed"
            )
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = to_absolute(action, anchors, self.layout)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"layout": self.layout, "pair_id": self.pair_id}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def add_relative_pose_steps(
    preprocessor: Any, postprocessor: Any, layout: dict[str, Any]
) -> None:
    """Wrap the normalizers: relative before normalizing, absolute after."""
    pair_id = uuid.uuid4().hex
    pre_steps = list(preprocessor.steps)
    at = next(
        i for i, s in enumerate(pre_steps) if isinstance(s, NormalizerProcessorStep)
    )
    pre_steps.insert(at, RelativePoseProcessorStep(layout=layout, pair_id=pair_id))
    preprocessor.steps = pre_steps
    post_steps = list(postprocessor.steps)
    at = next(
        i for i, s in enumerate(post_steps) if isinstance(s, UnnormalizerProcessorStep)
    )
    post_steps.insert(at + 1, AbsolutePoseProcessorStep(layout=layout, pair_id=pair_id))
    postprocessor.steps = post_steps


def uses_relative_pose(preprocessor: Any) -> bool:
    return any(
        isinstance(step, RelativePoseProcessorStep)
        for step in getattr(preprocessor, "steps", ())
    )


def hold_relative_pose_anchor(preprocessor: Any, hold: bool) -> None:
    """Keep the executing chunk's anchor while its cached actions are popped."""
    for step in getattr(preprocessor, "steps", ()):
        if isinstance(step, RelativePoseProcessorStep):
            step.hold_anchor = hold


def relative_action_stats(
    actions: np.ndarray,
    states: np.ndarray,
    episode_index: np.ndarray,
    offsets: Sequence[int],
    layout: dict[str, Any],
    batch_frames: int = 4096,
) -> dict[str, np.ndarray]:
    """Action statistics over every chunk the policy trains on, made relative.

    Each frame anchors a chunk at ``offsets`` (LeRobot's action delta indices),
    clamped to its episode as LeRobot pads it.
    """
    from lerobot.datasets.compute_stats import (  # noqa: PLC0415
        RunningQuantileStats,
    )

    episode_index = np.asarray(episode_index).reshape(-1)
    frames = np.arange(len(episode_index))
    starts = np.r_[0, np.flatnonzero(np.diff(episode_index)) + 1]
    ends = np.r_[starts[1:], len(episode_index)] - 1
    run = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(frames)]))
    first, last = starts[run], ends[run]
    offsets = np.asarray(offsets)

    actions_t = torch.as_tensor(actions, dtype=torch.float64)
    states_t = torch.as_tensor(states, dtype=torch.float64)
    running = RunningQuantileStats()
    for begin in range(0, len(frames), batch_frames):
        rows = frames[begin : begin + batch_frames]
        chunk = np.clip(
            rows[:, None] + offsets[None, :], first[rows, None], last[rows, None]
        )
        relative = to_relative(
            actions_t[chunk], state_anchors(states_t[rows], layout), layout
        )
        running.update(relative.reshape(-1, actions.shape[-1]).numpy())
    return running.get_statistics()


def unnormalized_rotation_columns(names: Sequence[str]) -> list[int]:
    """Indices of the rotation entries of UMI's extra pose inputs in the state."""
    spec = history_columns(list(names))
    rotations = {
        name
        for name in (spec.names if spec is not None else ())
        if name.rsplit(".", 1)[-1] in ROTATION_6D_AXES
    }
    return [i for i, name in enumerate(names) if name in rotations]


def unnormalized_rotation_stats(
    stats: dict[str, np.ndarray], names: Sequence[str]
) -> dict[str, np.ndarray]:
    """A copy of the state statistics that leaves those rotation entries as is."""
    columns = unnormalized_rotation_columns(names)
    out = {key: np.array(value, copy=True) for key, value in stats.items()}
    for key, value in _UNNORMALIZED_STATS.items():
        if key in out and columns:
            out[key][..., columns] = value
    return out


def _leave_rotations_unnormalized(meta: Any) -> None:
    """Rewrite the dataset's state statistics, which the normalizer reads."""
    stats = (meta.stats or {}).get(OBS_STATE)
    if stats is None:
        return
    names = _feature_names(meta.features, OBS_STATE)
    columns = unnormalized_rotation_columns(names)
    if not columns:
        return
    meta.stats = {**meta.stats, OBS_STATE: unnormalized_rotation_stats(stats, names)}
    logger.info(
        "State rotations left unnormalized: %s", ", ".join(names[i] for i in columns)
    )


def gripper_width_columns(names: Sequence[str]) -> list[int]:
    """Indices of the gripper widths (not their change or force)."""
    return [i for i, name in enumerate(names) if _GRIPPER_WIDTH.match(name)]


def range_normalized_width_stats(
    stats: dict[str, np.ndarray], names: Sequence[str]
) -> dict[str, np.ndarray]:
    """A copy whose mean/std map each gripper width's [min, max] onto [-1, 1]."""
    out = {key: np.array(value, copy=True) for key, value in stats.items()}
    if not {"mean", "std", "min", "max"} <= out.keys():
        return out
    for i in gripper_width_columns(names):
        low, high = out["min"][..., i], out["max"][..., i]
        if np.all(high - low >= _MIN_WIDTH_RANGE):
            out["mean"][..., i] = (low + high) / 2
            out["std"][..., i] = (high - low) / 2
    return out


def _range_normalize_widths(meta: Any, key: str) -> None:
    """Rewrite the dataset's statistics of ``key``, which the normalizer reads."""
    stats = (meta.stats or {}).get(key)
    if stats is None:
        return
    names = _feature_names(meta.features, key)
    columns = gripper_width_columns(names)
    if not columns:
        return
    meta.stats = {**meta.stats, key: range_normalized_width_stats(stats, names)}
    logger.info(
        "%s gripper width range-normalized: %s",
        key,
        ", ".join(names[i] for i in columns),
    )


def _patch_dataset_stats() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    original_init = LeRobotDataset.__init__
    if getattr(original_init, "_flexiv_patched", False):
        return

    def __init__(self, *args, **kwargs):  # noqa: N807 - patching a dunder
        original_init(self, *args, **kwargs)
        layout = requested_layout()
        if layout is None:
            return
        _leave_rotations_unnormalized(self.meta)
        _range_normalize_widths(self.meta, OBS_STATE)
        offsets = (getattr(self.reader, "delta_indices", None) or {}).get(ACTION)
        if offsets is None:
            _range_normalize_widths(self.meta, ACTION)
            return
        hf_dataset = self.hf_dataset
        stats = relative_action_stats(
            np.array(hf_dataset[ACTION], dtype=np.float32),
            np.array(hf_dataset[OBS_STATE], dtype=np.float32),
            np.array(hf_dataset["episode_index"]),
            offsets,
            layout,
        )
        # The normalizer reads dataset.meta.stats for a new policy and as the
        # override when fine-tuning, so both see relative statistics.
        self.meta.stats = {**self.meta.stats, ACTION: stats}
        logger.info(
            "Relative-pose action stats over %d frames x %d offsets",
            len(hf_dataset),
            len(offsets),
        )
        _range_normalize_widths(self.meta, ACTION)

    __init__._flexiv_patched = True
    LeRobotDataset.__init__ = __init__


def _patch_processor_factories() -> None:
    for module_name, name in _FAMILY_FACTORIES:
        module = importlib.import_module(module_name)
        original = getattr(module, name, None)
        if original is None:
            raise RuntimeError(
                f"LeRobot changed shape: {module_name}.{name} is gone, so "
                "relative-pose steps would silently stop being added."
            )
        if getattr(original, "_flexiv_patched", False):
            continue

        def make(*args, _original=original, **kwargs):
            preprocessor, postprocessor = _original(*args, **kwargs)
            layout = requested_layout()
            if layout is not None:
                add_relative_pose_steps(preprocessor, postprocessor, layout)
            return preprocessor, postprocessor

        make._flexiv_patched = True
        setattr(module, name, make)


def apply_patches() -> None:
    """Install the stats and processor patches. Safe to call more than once."""

    _patch_dataset_stats()
    _patch_processor_factories()
