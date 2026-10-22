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

"""A practice space: how a skill steps, what it sees, what its action means.

A step skill makes one small move per step, `rate_hz` steps a second. Its
action is a vector in [-1, 1], and one unit on an axis is that axis's bound:

    step_m    tool axis -> the largest move per step, in metres
    step_deg  tool axis -> the largest rotation per step, in degrees
    force_n   tool axis -> the largest force target, in newtons

The action's columns are the moves, then the rotations, then the forces, each
in x, y, z order of the axes listed. A move is relative to where the tool
measurably IS when the step starts, in the tool's own frame. An axis under
force control is held where it is and pressed with its force target. An axis
in none of the three is held where it is.

What the skill sees at a step:

    observation.images.<view>    each listed camera, as captured (RGB, HWC)
    observation.state            the tool's position (m), its rotation (the first
                                 two rows of its matrix) and its velocity, as the
                                 recorder stores them
    observation.wrench           the last `wrench_samples` force readings, oldest
                                 first, sampled at `sense_hz`: the force and
                                 moment the robot applies, in its tool frame
    observation.last_actions     its last `last_actions` commands, newest first,
                                 each re-expressed relative to where the tool is
                                 NOW, in units of the bounds

The last commands are there because a skill that cannot see what it last asked
for reads the contact force as "how deep was my last target" instead, and
because the arm follows a command a step or two late.

ONE CLASS BUILDS BOTH THE DEMONSTRATIONS AND THE LIVE OBSERVATION. A
`StepTracker` is fed robot samples one at a time -- from a recording by the demo
converter, from the arm by the practice environment -- and turns them into
observations and commands. A skill trained on converted demos therefore sees,
on the robot, the numbers it was trained on.

The space is stored with the skill: `meta/practice_space.json` in a converted
recording, `practice_space.json` beside a checkpoint.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from flexivtrainer.data.lerobot_io import rotation_6d_to_matrix

__all__ = [
    "AXES",
    "SPACE_FILENAME",
    "SPACE_RELATIVE_PATH",
    "Command",
    "PracticeSpace",
    "Sample",
    "StepTracker",
    "read_space",
    "write_space",
]

SPACE_FILENAME = "practice_space.json"
SPACE_RELATIVE_PATH = Path("meta") / SPACE_FILENAME
FORMAT_VERSION = 1
AXES = ("x", "y", "z")

IMAGE_PREFIX = "observation.images."
STATE = "observation.state"
WRENCH = "observation.wrench"
LAST_ACTIONS = "observation.last_actions"
STATE_NAMES = (
    *(f"tcp_pose.{a}" for a in AXES),
    *(f"tcp_rotation_6d.r{row}_{a}" for row in (1, 2) for a in AXES),
    "tcp_twist.vx",
    "tcp_twist.vy",
    "tcp_twist.vz",
    "tcp_twist.wx",
    "tcp_twist.wy",
    "tcp_twist.wz",
)
WRENCH_AXES = ("fx", "fy", "fz", "mx", "my", "mz")
#: An action this far outside [-1, 1] is a fault, not rounding.
_ACTION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Sample:
    """One reading of the arm: where the tool is, how it moves, what it feels.

    `wrench` is the driver's `ext_wrench_in_world`: what the world exerts on
    the tool, in the base frame.
    """

    position: np.ndarray  # (3,) m, base frame
    rotation: np.ndarray  # (3, 3) tool -> base
    twist: np.ndarray  # (6,) as the driver reports it
    wrench: np.ndarray  # (6,)

    @classmethod
    def from_driver(
        cls,
        tcp_pose: Sequence[float],
        tcp_vel: Sequence[float],
        ext_wrench_in_world: Sequence[float],
    ) -> Sample:
        """From the driver's states: a pose is [x, y, z, qw, qx, qy, qz]."""
        pose = np.asarray(tcp_pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"a driver pose has 7 numbers, got {pose.shape}")
        w, x, y, z = pose[3:]
        return cls(
            position=pose[:3].copy(),
            rotation=Rotation.from_quat([x, y, z, w]).as_matrix(),
            twist=np.asarray(tcp_vel, dtype=np.float64).reshape(6),
            wrench=np.asarray(ext_wrench_in_world, dtype=np.float64).reshape(6),
        )

    @classmethod
    def from_recorded(
        cls, pose: Sequence[float], twist: Sequence[float], wrench: Sequence[float]
    ) -> Sample:
        """From a recording: a pose is [x, y, z] then its rotation-6D rows."""
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (9,):
            raise ValueError(f"a recorded pose has 9 numbers, got {pose.shape}")
        return cls(
            position=pose[:3].copy(),
            rotation=rotation_6d_to_matrix(pose[3:]),
            twist=np.asarray(twist, dtype=np.float64).reshape(6),
            wrench=np.asarray(wrench, dtype=np.float64).reshape(6),
        )

    def tool_wrench(self) -> np.ndarray:
        """(6,): the force and moment the robot applies, in its tool frame.

        `R^T v` turns a base-frame vector into the tool frame; the sign turns
        what the world exerts on the tool into what the tool exerts on it.
        """
        to_tool = self.rotation.T
        return -np.concatenate([to_tool @ self.wrench[:3], to_tool @ self.wrench[3:]])


@dataclass(frozen=True)
class Command:
    """Where one step sends the tool, and how hard it presses.

    `force` is what the robot applies, in the tool frame; only the space's
    force axes of it are commanded.
    """

    position: np.ndarray  # (3,) m, base frame
    rotation: np.ndarray  # (3, 3) tool -> base
    force: np.ndarray  # (3,) N, tool frame

    def pose(self) -> list[float]:
        """The target as the driver takes it: [x, y, z, qw, qx, qy, qz]."""
        x, y, z, w = Rotation.from_matrix(self.rotation).as_quat()
        if w < 0:  # one of the two equal quaternions, always the same one
            x, y, z, w = -x, -y, -z, -w
        return [float(v) for v in self.position] + [
            float(w),
            float(x),
            float(y),
            float(z),
        ]


def _bounds(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must map tool axes (x, y, z) to numbers")
    out: dict[str, float] = {}
    for axis in AXES:  # x, y, z order, whatever order they were given in
        if axis not in value:
            continue
        bound = value[axis]
        if isinstance(bound, bool) or not isinstance(bound, int | float):
            raise ValueError(f"{name}.{axis} must be a number, got {bound!r}")
        if not (math.isfinite(bound) and bound > 0):
            raise ValueError(f"{name}.{axis} must be positive, got {bound}")
        out[axis] = float(bound)
    unknown = sorted(set(value) - set(AXES))
    if unknown:
        raise ValueError(f"{name} names unknown axes {unknown}; use x, y, z")
    return out


@dataclass(frozen=True)
class PracticeSpace:
    """How a skill steps, what it sees, and what one unit of its action means."""

    rate_hz: float
    step_m: Mapping[str, float] = field(default_factory=dict)
    step_deg: Mapping[str, float] = field(default_factory=dict)
    force_n: Mapping[str, float] = field(default_factory=dict)
    views: tuple[str, ...] = ()
    last_actions: int = 3
    wrench_samples: int = 10
    sense_hz: float = 30.0

    def __post_init__(self) -> None:
        for name in ("rate_hz", "sense_hz"):
            value = getattr(self, name)
            if not (isinstance(value, int | float) and math.isfinite(value)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        stride = self.sense_hz / self.rate_hz
        if abs(stride - round(stride)) > 1e-6 or round(stride) < 1:
            raise ValueError(
                f"sense_hz {self.sense_hz} must be a whole multiple of rate_hz "
                f"{self.rate_hz}: a step starts on a sample"
            )
        object.__setattr__(self, "step_m", _bounds(self.step_m, "step_m"))
        object.__setattr__(self, "step_deg", _bounds(self.step_deg, "step_deg"))
        object.__setattr__(self, "force_n", _bounds(self.force_n, "force_n"))
        both = sorted(set(self.step_m) & set(self.force_n))
        if both:
            raise ValueError(
                f"tool axis {both} is both moved and force controlled; "
                "an axis is one or the other"
            )
        if not (self.step_m or self.step_deg or self.force_n):
            raise ValueError("a practice space needs at least one action axis")
        views = tuple(str(v) for v in self.views)
        if len(set(views)) != len(views) or any(not v for v in views):
            raise ValueError(f"views must be distinct camera names, got {views}")
        object.__setattr__(self, "views", views)
        for name in ("last_actions", "wrench_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a whole number >= 0, got {value!r}")

    # --- what it means ---------------------------------------------------

    @property
    def stride(self) -> int:
        """Samples per step."""
        return round(self.sense_hz / self.rate_hz)

    @property
    def action_names(self) -> list[str]:
        return (
            [f"move.{a}" for a in self.step_m]
            + [f"turn.{a}" for a in self.step_deg]
            + [f"force.{a}" for a in self.force_n]
        )

    @property
    def action_bounds(self) -> list[tuple[float, str]]:
        """What one unit of each action column is: (bound, "m" | "deg" | "N")."""
        return (
            [(b, "m") for b in self.step_m.values()]
            + [(b, "deg") for b in self.step_deg.values()]
            + [(b, "N") for b in self.force_n.values()]
        )

    @property
    def action_dim(self) -> int:
        return len(self.step_m) + len(self.step_deg) + len(self.force_n)

    @property
    def force_axes(self) -> list[bool]:
        """Which of the six Cartesian axes are under force control."""
        return [axis in self.force_n for axis in AXES] + [False] * 3

    def state_dims(self) -> dict[str, list[str]]:
        """Every vector the skill sees, with the name of each number in it."""
        dims: dict[str, list[str]] = {STATE: list(STATE_NAMES)}
        if self.wrench_samples:
            dims[WRENCH] = [
                f"{axis}[t-{age}]"
                for age in range(self.wrench_samples - 1, -1, -1)
                for axis in WRENCH_AXES
            ]
        if self.last_actions:
            dims[LAST_ACTIONS] = [
                f"{name}[-{back}]"
                for back in range(1, self.last_actions + 1)
                for name in self.action_names
            ]
        return dims

    # --- stored with the skill ---------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "rate_hz": self.rate_hz,
            "step_m": dict(self.step_m),
            "step_deg": dict(self.step_deg),
            "force_n": dict(self.force_n),
            "views": list(self.views),
            "last_actions": self.last_actions,
            "wrench_samples": self.wrench_samples,
            "sense_hz": self.sense_hz,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PracticeSpace:
        if not isinstance(data, Mapping):
            raise ValueError("a practice space is a JSON object")
        values = dict(data)
        version = values.pop("format_version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise ValueError(f"practice space format {version} is not supported")
        known = {
            "rate_hz",
            "step_m",
            "step_deg",
            "force_n",
            "views",
            "last_actions",
            "wrench_samples",
            "sense_hz",
        }
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown practice space settings {unknown}")
        if "rate_hz" not in values:
            raise ValueError("a practice space needs rate_hz")
        if isinstance(values.get("views"), str):
            raise ValueError("views must be a list of camera names")
        values["views"] = tuple(values.get("views") or ())
        return cls(**values)


def write_space(root: Path, space: PracticeSpace, *, checkpoint: bool = False) -> Path:
    """Write `meta/practice_space.json` of a dataset, or that of a checkpoint."""
    path = Path(root) / (SPACE_FILENAME if checkpoint else SPACE_RELATIVE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(space.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def read_space(path: Path) -> PracticeSpace:
    """A practice space from its JSON file, naming the file when it is wrong."""
    path = Path(path)
    try:
        return PracticeSpace.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid practice space in {path}: {exc}") from exc


class StepTracker:
    """What one episode in a practice space has sensed and commanded so far.

    Feed it every sample at `sense_hz`: `reset` with the first, `sense` with
    each one after. At a step, `observation` is what the skill sees, and `take`
    turns its action into the command to send. Demonstrations and takeovers go
    through `action_toward` first, which finds the action that heads for a
    target they commanded, and then through `take` like any other action.
    """

    def __init__(self, space: PracticeSpace) -> None:
        self.space = space
        self._sample: Sample | None = None
        self._wrenches: deque[np.ndarray] = deque(maxlen=max(space.wrench_samples, 1))
        self._commands: deque[Command] = deque(maxlen=max(space.last_actions, 1))
        self._move = [AXES.index(a) for a in space.step_m]
        self._move_bound = np.array(list(space.step_m.values()))
        self._turn = [AXES.index(a) for a in space.step_deg]
        self._turn_bound = np.array(list(space.step_deg.values()))
        self._force = [AXES.index(a) for a in space.force_n]
        self._force_bound = np.array(list(space.force_n.values()))

    @property
    def sample(self) -> Sample:
        if self._sample is None:
            raise RuntimeError("reset() the tracker with the episode's first sample")
        return self._sample

    def reset(self, sample: Sample) -> None:
        """Start an episode. Nothing has been commanded yet, so every slot of
        the last commands holds "stay here, pressing as now"; the force window
        repeats the first reading."""
        self._sample = sample
        wrench = sample.tool_wrench()
        self._wrenches.clear()
        self._wrenches.extend([wrench] * self._wrenches.maxlen)
        hold = Command(sample.position, sample.rotation, wrench[:3])
        self._commands.clear()
        self._commands.extend([hold] * self._commands.maxlen)

    def sense(self, sample: Sample) -> None:
        """One more sample, at `sense_hz`."""
        if self._sample is None:
            raise RuntimeError("reset() the tracker with the episode's first sample")
        self._sample = sample
        self._wrenches.append(sample.tool_wrench())

    def observation(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """What the skill sees now. `images` maps camera names to RGB frames."""
        sample = self.sample
        missing = [v for v in self.space.views if v not in images]
        if missing:
            raise KeyError(f"no frame from camera {missing}")
        out: dict[str, np.ndarray] = {
            IMAGE_PREFIX + view: np.asarray(images[view]) for view in self.space.views
        }
        out[STATE] = np.concatenate(
            [sample.position, sample.rotation[:2].reshape(6), sample.twist]
        ).astype(np.float32)
        if self.space.wrench_samples:
            out[WRENCH] = np.concatenate(list(self._wrenches)).astype(np.float32)
        if self.space.last_actions:
            newest_first = reversed(self._commands)
            out[LAST_ACTIONS] = np.concatenate(
                [self._units(c) for c in newest_first]
            ).astype(np.float32)
        return out

    def action_toward(self, target: Command) -> tuple[np.ndarray, np.ndarray]:
        """(action, unclipped): the step that heads for `target`, from here.

        Each axis is in units of its bound and the action is clipped to
        [-1, 1]; the unclipped value says by how much a demonstration
        outran the bounds.
        """
        units = self._units(target)
        return np.clip(units, -1.0, 1.0).astype(np.float32), units

    def take(self, action: Sequence[float]) -> Command:
        """The command for this step's action; it becomes the newest of the
        last commands. Refuses an action outside [-1, 1]: a skill that
        produces one is broken, and a clamped one is a different command."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.space.action_dim,):
            raise ValueError(
                f"an action has {self.space.action_dim} numbers "
                f"{self.space.action_names}, got {action.shape[0]}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"the action is not finite: {action.tolist()}")
        if np.any(np.abs(action) > 1.0 + _ACTION_TOLERANCE):
            raise ValueError(f"the action leaves [-1, 1]: {action.tolist()}")
        action = np.clip(action, -1.0, 1.0)
        sample = self.sample
        n_move, n_turn = len(self._move), len(self._turn)
        move = np.zeros(3)
        move[self._move] = action[:n_move] * self._move_bound
        turn = np.zeros(3)
        turn[self._turn] = action[n_move : n_move + n_turn] * self._turn_bound
        force = np.zeros(3)
        force[self._force] = action[n_move + n_turn :] * self._force_bound
        command = Command(
            position=sample.position + sample.rotation @ move,
            rotation=sample.rotation
            @ Rotation.from_rotvec(turn, degrees=True).as_matrix(),
            force=force,
        )
        if self.space.last_actions:
            self._commands.append(command)
        return command

    def _units(self, command: Command) -> np.ndarray:
        """`command` relative to where the tool is now, in units of the bounds."""
        sample = self.sample
        to_tool = sample.rotation.T
        move = to_tool @ (command.position - sample.position)
        turn = Rotation.from_matrix(to_tool @ command.rotation).as_rotvec(degrees=True)
        return np.concatenate(
            [
                move[self._move] / self._move_bound,
                turn[self._turn] / self._turn_bound,
                np.asarray(command.force)[self._force] / self._force_bound,
            ]
        )
