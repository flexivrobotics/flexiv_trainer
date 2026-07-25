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

"""Fit recorded Cartesian TCP actions into fixed-size B-spline targets.

The representation follows the B-spline Policy data layout:

``[parameter_row, knot + control_channels]``

The knot vector occupies column zero. Remaining columns contain Cartesian
control points represented as XYZ plus rotation-6D and optional gripper width
for every selected arm. The fixed-size matrix can be flattened into a normal
one-dimensional LeRobot ``action`` feature for training.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import (
    BSpline,
    generate_knots,
    make_interp_spline,
    make_lsq_spline,
)
from scipy.spatial.transform import Rotation

_POSE_AXES = ("x", "y", "z", "q_w", "q_x", "q_y", "q_z")
_ROTATION_6D_AXES = ("r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z")


@dataclass(frozen=True, slots=True)
class TCPActionLayout:
    """Indices and output control names for one arm's recorded action."""

    side: str
    pose_indices: tuple[int, ...]
    gripper_width_index: int | None = None

    @property
    def control_names(self) -> tuple[str, ...]:
        position = tuple(f"{self.side}.tcp_pose.{axis}" for axis in _POSE_AXES[:3])
        rotation = tuple(
            f"{self.side}.tcp_rotation_6d.{axis}" for axis in _ROTATION_6D_AXES
        )
        gripper = (
            (f"{self.side}.gripper.width",)
            if self.gripper_width_index is not None
            else ()
        )
        return position + rotation + gripper


@dataclass(frozen=True, slots=True)
class AdaptiveSplineFit:
    """Result of adaptive FITPACK-style knot insertion."""

    spline: BSpline
    max_abs_error: float
    tolerance_reached: bool

    @property
    def knots(self) -> np.ndarray:
        return np.asarray(self.spline.t)

    @property
    def control_points(self) -> np.ndarray:
        return np.asarray(self.spline.c)


@dataclass(frozen=True, slots=True)
class EpisodeSplineTargets:
    """One fixed-size spline parameter matrix for every episode frame."""

    parameters: np.ndarray
    controls: np.ndarray
    fit: AdaptiveSplineFit


def detect_tcp_action_layouts(
    action_names: Sequence[str],
    sides: Sequence[str] | None = None,
) -> list[TCPActionLayout]:
    """Find named ``<side>.tcp_pose.*`` runs in a LeRobot action vector."""

    names = [str(name) for name in action_names]
    name_to_index = {name: index for index, name in enumerate(names)}
    if len(name_to_index) != len(names):
        raise ValueError("Action feature names must be unique")

    suffix = ".tcp_pose.x"
    detected = [name[: -len(suffix)] for name in names if name.endswith(suffix)]
    selected = list(sides) if sides else detected
    if not selected:
        raise ValueError(
            "No named TCP pose action found. Expected axes such as "
            "'left_arm.tcp_pose.x' in meta/info.json."
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate arm side requested: {selected}")

    layouts: list[TCPActionLayout] = []
    for side in selected:
        expected = [f"{side}.tcp_pose.{axis}" for axis in _POSE_AXES]
        missing = [name for name in expected if name not in name_to_index]
        if missing:
            raise ValueError(
                f"Incomplete TCP pose action for side '{side}'; missing {missing}"
            )
        layouts.append(
            TCPActionLayout(
                side=side,
                pose_indices=tuple(name_to_index[name] for name in expected),
                gripper_width_index=name_to_index.get(f"{side}.gripper.width"),
            )
        )
    return layouts


def quaternion_wxyz_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    """Convert unit quaternions to the first two rows of rotation matrices."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {quaternion.shape}")
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("TCP pose contains a non-finite or near-zero quaternion")
    normalized = quaternion / norms
    quaternion_xyzw = np.concatenate(
        [normalized[..., 1:], normalized[..., :1]], axis=-1
    )
    matrices = Rotation.from_quat(quaternion_xyzw.reshape(-1, 4)).as_matrix()
    rotation_6d = matrices[:, :2, :].reshape(-1, 6)
    return rotation_6d.reshape(quaternion.shape[:-1] + (6,))


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Project rotation-6D values onto SO(3) using Gram-Schmidt."""

    rotation_6d = np.asarray(rotation_6d, dtype=np.float64)
    if rotation_6d.shape[-1] != 6:
        raise ValueError(
            f"Expected rotation-6D shape (..., 6), got {rotation_6d.shape}"
        )
    flat = rotation_6d.reshape(-1, 6)
    first = flat[:, :3]
    second = flat[:, 3:]

    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(~np.isfinite(first_norm)) or np.any(first_norm < 1e-8):
        raise ValueError("Rotation-6D first basis vector is degenerate")
    basis_1 = first / first_norm

    orthogonal = second - np.sum(basis_1 * second, axis=1, keepdims=True) * basis_1
    second_norm = np.linalg.norm(orthogonal, axis=1, keepdims=True)
    if np.any(~np.isfinite(second_norm)) or np.any(second_norm < 1e-8):
        raise ValueError("Rotation-6D second basis vector is degenerate")
    basis_2 = orthogonal / second_norm
    basis_3 = np.cross(basis_1, basis_2)

    matrices = np.stack([basis_1, basis_2, basis_3], axis=1)
    return matrices.reshape(rotation_6d.shape[:-1] + (3, 3))


def rotation_6d_to_quaternion_wxyz(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert rotation-6D values to normalized ``[qw, qx, qy, qz]``."""

    matrices = rotation_6d_to_matrix(rotation_6d)
    quaternion_xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
    quaternion_wxyz = np.concatenate(
        [quaternion_xyzw[:, 3:4], quaternion_xyzw[:, :3]], axis=1
    )
    # Choose one deterministic representative of q == -q.
    flip = quaternion_wxyz[:, :1] < 0
    quaternion_wxyz = np.where(flip, -quaternion_wxyz, quaternion_wxyz)
    return quaternion_wxyz.reshape(matrices.shape[:-2] + (4,))


def extract_cartesian_controls(
    actions: np.ndarray,
    layouts: Sequence[TCPActionLayout],
) -> np.ndarray:
    """Extract spline controls from flat recorded action vectors."""

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"Expected action matrix [frames, dims], got {actions.shape}")
    if not layouts:
        raise ValueError("At least one TCP action layout is required")

    controls: list[np.ndarray] = []
    for layout in layouts:
        pose = actions[:, layout.pose_indices]
        position = pose[:, :3]
        rotation_6d = quaternion_wxyz_to_rotation_6d(pose[:, 3:7])
        arm_controls = [position, rotation_6d]
        if layout.gripper_width_index is not None:
            arm_controls.append(actions[:, layout.gripper_width_index, None])
        controls.append(np.concatenate(arm_controls, axis=1))
    result = np.concatenate(controls, axis=1)
    if np.any(~np.isfinite(result)):
        raise ValueError("Cartesian control trajectory contains non-finite values")
    return result


def fit_adaptive_bspline(
    controls: np.ndarray,
    *,
    degree: int = 3,
    max_error: float = 0.002,
    smoothing: float = 1e-12,
    min_knots: int | None = None,
    max_knots: int | None = None,
) -> AdaptiveSplineFit:
    """Fit a multidimensional trajectory with adaptive knot insertion."""

    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] == 0:
        raise ValueError(
            f"Expected non-empty controls [frames, dims], got {controls.shape}"
        )
    if degree < 1:
        raise ValueError("B-spline degree must be at least 1")
    if len(controls) <= degree:
        raise ValueError(
            f"Degree-{degree} fitting requires at least {degree + 1} frames; "
            f"received {len(controls)}"
        )
    if max_error <= 0:
        raise ValueError("max_error must be positive")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    minimum_valid_knots = 2 * (degree + 1)
    if min_knots is not None and min_knots < minimum_valid_knots:
        raise ValueError(
            f"min_knots must be at least {minimum_valid_knots} for degree {degree}"
        )
    if max_knots is not None and max_knots < 2 * (degree + 1):
        raise ValueError(
            f"max_knots must be at least {minimum_valid_knots} for degree {degree}"
        )
    if (
        min_knots is not None
        and max_knots is not None
        and max_knots < min_knots
    ):
        raise ValueError("max_knots cannot be smaller than min_knots")
    maximum_available_knots = len(controls) + degree + 1
    if min_knots is not None and min_knots > maximum_available_knots:
        raise ValueError(
            f"{len(controls)} frames can provide at most "
            f"{maximum_available_knots} knots, fewer than the required "
            f"{min_knots}. Record a longer episode or reduce chunk_size."
        )
    if np.any(~np.isfinite(controls)):
        raise ValueError("Cannot fit non-finite controls")

    sample_times = np.arange(len(controls), dtype=np.float64)
    last_spline: BSpline | None = None
    last_error = float("inf")
    for knots in generate_knots(
        sample_times,
        controls,
        k=degree,
        s=smoothing,
        nest=max_knots,
    ):
        spline = make_lsq_spline(
            sample_times,
            controls,
            knots,
            k=degree,
        )
        error = float(np.max(np.abs(spline(sample_times) - controls)))
        last_spline = spline
        last_error = error
        enough_knots = min_knots is None or len(knots) >= min_knots
        if enough_knots and error <= max_error:
            return AdaptiveSplineFit(
                spline=spline,
                max_abs_error=error,
                tolerance_reached=True,
            )

    if last_spline is None:  # pragma: no cover - defensive SciPy contract guard
        raise RuntimeError("SciPy did not generate a candidate knot vector")
    if min_knots is not None and len(last_spline.t) < min_knots:
        # A trajectory that is already represented exactly by a low-order
        # polynomial can make generate_knots stop before the fixed policy target
        # size is reached. Add a valid, evenly distributed subset of the
        # interpolation knots instead of repeating an endpoint more than p+1
        # times (which would create an invalid spline basis).
        interpolation_knots = np.asarray(
            make_interp_spline(sample_times, controls, k=degree).t
        )
        internal = interpolation_knots[degree + 1 : -(degree + 1)]
        required_internal = min_knots - 2 * (degree + 1)
        if required_internal:
            indices = np.linspace(
                0,
                len(internal) - 1,
                required_internal,
                dtype=int,
            )
            selected_internal = internal[indices]
        else:
            selected_internal = np.empty(0, dtype=np.float64)
        expanded_knots = np.concatenate(
            [
                np.repeat(sample_times[0], degree + 1),
                selected_internal,
                np.repeat(sample_times[-1], degree + 1),
            ]
        )
        last_spline = make_lsq_spline(
            sample_times,
            controls,
            expanded_knots,
            k=degree,
        )
        last_error = float(
            np.max(np.abs(last_spline(sample_times) - controls))
        )
    return AdaptiveSplineFit(
        spline=last_spline,
        max_abs_error=last_error,
        tolerance_reached=last_error <= max_error,
    )


def _chunk_fitted_spline(
    spline: BSpline,
    *,
    degree: int,
    chunk_size: int,
    stride: int,
) -> list[np.ndarray]:
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    full_knots = np.asarray(spline.t, dtype=np.float64)
    full_controls = np.asarray(spline.c, dtype=np.float64)
    unique_span = full_knots[degree:-degree]
    expected_rows = chunk_size + 2 * degree
    active_control_rows = expected_rows - (degree + 1)
    if len(full_knots) < expected_rows:
        raise ValueError(
            f"Fitted spline has {len(full_knots)} knots but fixed targets require "
            f"{expected_rows}"
        )
    chunks: list[np.ndarray] = []

    for start_index in range(0, len(unique_span) - 1, stride):
        slice_start = min(start_index, len(full_knots) - expected_rows)
        knots = full_knots[slice_start : slice_start + expected_rows]
        active_controls = full_controls[
            slice_start : slice_start + active_control_rows
        ]
        if len(active_controls) != active_control_rows:
            raise AssertionError("Incomplete active B-spline control-point window")

        # The policy-facing format has as many control rows as knot rows. SciPy
        # uses only len(knots) - degree - 1 of them; pad the ignored tail without
        # increasing boundary-knot multiplicity.
        controls = np.concatenate(
            [
                active_controls,
                np.repeat(active_controls[-1:], degree + 1, axis=0),
            ],
            axis=0,
        )
        chunks.append(np.concatenate([knots[:, None], controls], axis=1))

    if not chunks:
        raise RuntimeError("Fitted spline did not produce any parameter chunks")
    return chunks


def build_episode_spline_targets(
    controls: np.ndarray,
    *,
    degree: int = 3,
    chunk_size: int = 10,
    stride: int = 1,
    max_error: float = 0.002,
    smoothing: float = 1e-12,
    max_knots: int | None = None,
) -> EpisodeSplineTargets:
    """Create one local-time spline parameter target for every frame."""

    fit = fit_adaptive_bspline(
        controls,
        degree=degree,
        max_error=max_error,
        smoothing=smoothing,
        min_knots=chunk_size + 2 * degree,
        max_knots=max_knots,
    )
    chunks = _chunk_fitted_spline(
        fit.spline,
        degree=degree,
        chunk_size=chunk_size,
        stride=stride,
    )

    frame_count = len(controls)
    parameters = np.empty(
        (frame_count, chunk_size + 2 * degree, 1 + controls.shape[1]),
        dtype=np.float32,
    )
    next_frame = 0
    last_chunk = chunks[-1]
    for chunk in chunks:
        boundary_time = float(chunk[degree, 0])
        while next_frame < frame_count and next_frame <= boundary_time:
            local = chunk.copy()
            local[:, 0] -= next_frame
            parameters[next_frame] = local
            next_frame += 1

    while next_frame < frame_count:
        local = last_chunk.copy()
        local[:, 0] -= next_frame
        parameters[next_frame] = local
        next_frame += 1

    return EpisodeSplineTargets(
        parameters=parameters,
        controls=np.asarray(controls, dtype=np.float64),
        fit=fit,
    )


def evaluate_parameter_matrix(
    parameter_matrix: np.ndarray,
    times: Iterable[float] | np.ndarray,
    *,
    degree: int = 3,
) -> np.ndarray:
    """Evaluate one fixed-size knot/control-point parameter matrix."""

    parameter_matrix = np.asarray(parameter_matrix, dtype=np.float64)
    if parameter_matrix.ndim != 2 or parameter_matrix.shape[1] < 2:
        raise ValueError(
            "Expected parameter matrix [rows, knot + controls], got "
            f"{parameter_matrix.shape}"
        )
    if len(parameter_matrix) <= degree + 1:
        raise ValueError("Parameter matrix is too short for the configured degree")

    knots = parameter_matrix[:, 0]
    if np.any(np.diff(knots) < 0):
        raise ValueError("B-spline knots must be nondecreasing")
    controls = parameter_matrix[: -(degree + 1), 1:]
    spline = BSpline(knots, controls, degree, extrapolate=False)
    return np.asarray(spline(np.asarray(list(times), dtype=np.float64)))


def parameter_feature_names(
    layouts: Sequence[TCPActionLayout],
    *,
    parameter_rows: int,
) -> list[str]:
    """Names for a row-major flattened spline parameter matrix."""

    if parameter_rows < 1:
        raise ValueError("parameter_rows must be positive")
    control_names = [name for layout in layouts for name in layout.control_names]
    if not control_names:
        raise ValueError("At least one TCP action layout is required")
    names: list[str] = []
    for row in range(parameter_rows):
        names.append(f"bspline.row_{row:02d}.knot")
        names.extend(f"bspline.row_{row:02d}.{name}" for name in control_names)
    return names


def parameter_matrix_shape(
    layouts: Sequence[TCPActionLayout],
    *,
    parameter_rows: int,
) -> tuple[int, int]:
    """Return the logical row-major spline action shape."""

    names = parameter_feature_names(layouts, parameter_rows=parameter_rows)
    if len(names) % parameter_rows:
        raise ValueError("B-spline feature names do not form complete rows")
    return parameter_rows, len(names) // parameter_rows


def validate_parameter_matrix_shape(
    parameters: np.ndarray,
    layouts: Sequence[TCPActionLayout],
    *,
    parameter_rows: int,
) -> None:
    """Validate the logical spline dimensions of one or more targets."""

    actual = np.asarray(parameters).shape
    expected = parameter_matrix_shape(layouts, parameter_rows=parameter_rows)
    if len(actual) < 2 or actual[-2:] != expected:
        raise ValueError(
            f"B-spline parameters end with shape {actual[-2:]}, expected {expected}"
        )
