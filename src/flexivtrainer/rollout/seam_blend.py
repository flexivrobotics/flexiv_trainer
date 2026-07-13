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

"""Pure-numpy seam-smoothing helpers for the rollout command stream.

``apply_twist_mode`` rewrites the commanded velocity (twist) channel of a real-
unit action chunk before dispatch. It is an experiment lever: ``raw`` sends the
diffusion-sampled twist verbatim (current behaviour), ``zero`` sends no target
velocity, and ``fd`` replaces it with the finite-difference velocity of the pose
sequence (mirroring ``replay_episode._finite_difference_velocity``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.spatial.transform as st

_TWIST_DIM = 6


def _quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> st.Rotation:
    quat_wxyz = np.asarray(quat_wxyz)
    quat_xyzw = np.concatenate([quat_wxyz[..., 1:], quat_wxyz[..., :1]], axis=-1)
    return st.Rotation.from_quat(quat_xyzw)


def _rotation_to_quat_wxyz(rot: st.Rotation) -> np.ndarray:
    quat_xyzw = rot.as_quat()
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)


def _fd_twist(
    poses: np.ndarray,
    index: int,
    dt: float,
    prev_pose: np.ndarray | None,
) -> np.ndarray:
    """Finite-difference TCP twist ``[vx,vy,vz, wx,wy,wz]`` at ``index``.

    Central difference in the interior; at index 0 the boundary ``prev_pose``
    (last dispatched pose) is used as the previous sample for a central diff over
    ``2*dt`` when available, else forward difference; the last index backward-
    differences. Mirrors ``replay_episode._finite_difference_velocity``.
    """
    n = len(poses)
    if index == 0 and prev_pose is not None:
        lo_pose = prev_pose
        hi_pose = poses[1] if n > 1 else poses[0]
        span = 2.0 * dt if n > 1 else dt
    else:
        lo = max(index - 1, 0)
        hi = min(index + 1, n - 1)
        span = (hi - lo) * dt
        lo_pose = poses[lo]
        hi_pose = poses[hi]
    if span <= 0:
        return np.zeros(_TWIST_DIM)
    lin = (hi_pose[:3] - lo_pose[:3]) / span
    rot_lo = _quat_wxyz_to_rotation(lo_pose[3:7])
    rot_hi = _quat_wxyz_to_rotation(hi_pose[3:7])
    ang = (rot_hi * rot_lo.inv()).as_rotvec() / span
    return np.concatenate([lin, ang])


def apply_twist_mode(
    actions: np.ndarray,
    layout: list[dict[str, Any]],
    mode: str,
    dt: float,
    boundary_prev_poses: list[list[float] | None] | None = None,
) -> np.ndarray:
    """Rewrite each arm's twist slice of ``actions`` per ``mode``.

    ``raw`` returns the input unchanged; ``zero`` zeroes every arm's twist slice;
    ``fd`` fills it from the finite-difference velocity of that arm's pose
    sequence. Arms without a twist slice are skipped. Pose and wrench dims are
    never touched.
    """
    if mode == "raw":
        return actions
    if mode not in ("zero", "fd"):
        raise ValueError(f"unknown twist mode {mode!r}")

    out = actions.copy()
    for arm, arm_plan in enumerate(layout):
        twist_slice = arm_plan.get("twist")
        if twist_slice is None:
            continue
        if mode == "zero":
            out[:, twist_slice] = 0.0
            continue
        pose_slice = arm_plan.get("pose")
        if pose_slice is None:
            continue
        poses = actions[:, pose_slice]
        prev = None
        if boundary_prev_poses is not None and arm < len(boundary_prev_poses):
            prev_pose = boundary_prev_poses[arm]
            if prev_pose is not None:
                prev = np.asarray(prev_pose, dtype=actions.dtype)
        for k in range(len(actions)):
            out[k, twist_slice] = _fd_twist(poses, k, dt, prev if k == 0 else None)
    return out


def seam_weights(n: int, freeze: int, fade_end: int, schedule: str) -> np.ndarray:
    """Blend weights over ``n`` executed steps: 1.0 (old) fading to 0.0 (new).

    ``1.0`` for ``k < freeze``; decays to exactly ``0.0`` at ``k == fade_end``
    (linear = evenly spaced; ``exp`` = a simple exponential decay); ``0.0`` for
    ``k >= fade_end``. If ``fade_end <= freeze`` the ramp collapses to a hard
    switch (1.0 up to ``freeze``, then 0.0).
    """
    weights = np.zeros(n, dtype=np.float64)
    freeze = max(int(freeze), 0)
    fade_end = min(int(fade_end), n)
    if freeze >= n:
        weights[:] = 1.0
        return weights
    weights[:freeze] = 1.0
    if fade_end <= freeze:
        return weights
    span = fade_end - freeze
    ramp_idx = np.arange(span, dtype=np.float64)
    frac = (ramp_idx + 1.0) / span  # 0 -> 1 across [freeze, fade_end)
    if schedule == "exp":
        # Exponential decay reaching ~0 at fade_end, renormalized to hit 0 exactly.
        decay = np.exp(-3.0 * frac)
        edge = np.exp(-3.0)
        ramp = (decay - edge) / (1.0 - edge)
    else:
        ramp = 1.0 - frac
    weights[freeze:fade_end] = ramp
    return weights


def _blend_xyz(
    prev_tail: np.ndarray,
    fresh: np.ndarray,
    anchor: np.ndarray | None,
    w: np.ndarray,
    p: int,
) -> np.ndarray:
    """Velocity-continuous xyz blend: blend per-step deltas, integrate from a.

    ``a`` is the virtual index -1 sample (anchor last-dispatched pose, else
    ``prev_tail[0]``). ``out[0] = a + d_blend[0]`` (never ``a`` -- pinning to the
    anchor would inject a one-step stall).
    """
    n = len(fresh)
    a = anchor if anchor is not None else prev_tail[0]
    d_old = np.empty((n, 3), dtype=np.float64)
    d_new = np.empty((n, 3), dtype=np.float64)
    d_old[0] = prev_tail[0] - a
    d_new[0] = fresh[0] - a
    d_new[1:] = np.diff(fresh, axis=0)
    # Where no old data exists (k >= p), fall back to the fresh delta.
    for k in range(1, n):
        d_old[k] = (prev_tail[k] - prev_tail[k - 1]) if k < p else d_new[k]
    d_blend = w[:, None] * d_old + (1.0 - w[:, None]) * d_new
    out = np.empty((n, 3), dtype=np.float64)
    out[0] = a + d_blend[0]
    for k in range(1, n):
        out[k] = out[k - 1] + d_blend[k]
    return out


def _blend_quat(
    prev_tail: np.ndarray,
    fresh: np.ndarray,
    anchor_quat: np.ndarray | None,
    w: np.ndarray,
    p: int,
) -> np.ndarray:
    """Velocity-continuous quaternion blend on SO(3): blend rotvec deltas.

    Same recurrence as ``_blend_xyz`` but composing rotations; index 0 uses the
    anchor quaternion (else ``prev_tail[0]``) as the virtual index -1 sample.
    Returns wxyz rows.
    """
    n = len(fresh)
    a_rot = (
        _quat_wxyz_to_rotation(anchor_quat)
        if anchor_quat is not None
        else _quat_wxyz_to_rotation(prev_tail[0])
    )
    old_rots = _quat_wxyz_to_rotation(prev_tail)
    new_rots = _quat_wxyz_to_rotation(fresh)
    rv_old = np.empty((n, 3), dtype=np.float64)
    rv_new = np.empty((n, 3), dtype=np.float64)
    rv_new[0] = (new_rots[0] * a_rot.inv()).as_rotvec()
    for k in range(1, n):
        rv_new[k] = (new_rots[k] * new_rots[k - 1].inv()).as_rotvec()
    rv_old[0] = (old_rots[0] * a_rot.inv()).as_rotvec()
    for k in range(1, n):
        rv_old[k] = (old_rots[k] * old_rots[k - 1].inv()).as_rotvec() if k < p else rv_new[k]
    rv_blend = w[:, None] * rv_old + (1.0 - w[:, None]) * rv_new
    out = np.empty((n, 4), dtype=np.float64)
    cur = st.Rotation.from_rotvec(rv_blend[0]) * a_rot
    out[0] = _rotation_to_quat_wxyz(cur)
    for k in range(1, n):
        cur = st.Rotation.from_rotvec(rv_blend[k]) * cur
        out[k] = _rotation_to_quat_wxyz(cur)
    return out


def blend_seam(
    prev_tail: np.ndarray | None,
    fresh: np.ndarray,
    layout: list[dict[str, Any]],
    anchor_poses: list[list[float] | None] | None,
    weights: np.ndarray,
) -> np.ndarray:
    """Velocity-continuous blend of ``prev_tail`` (old executed rows) into ``fresh``.

    ``prev_tail`` and ``fresh`` are real-unit executed slices aligned at fresh
    index 0. Returns ``fresh`` unchanged (same object) if there is nothing to
    blend (no old data or all-zero weights). xyz and quaternion channels blend
    per-step deltas and integrate from the anchor (see ``_blend_xyz`` /
    ``_blend_quat``); twist and wrench blend positionally.
    """
    if prev_tail is None or len(prev_tail) == 0 or float(np.max(weights)) == 0.0:
        return fresh
    n = len(fresh)
    p = len(prev_tail)
    w = np.asarray(weights, dtype=np.float64).copy()
    w[np.arange(n) >= p] = 0.0  # no old data past the tail
    out = fresh.copy()
    for arm, arm_plan in enumerate(layout):
        anchor_pose = None
        if anchor_poses is not None and arm < len(anchor_poses):
            raw = anchor_poses[arm]
            if raw is not None:
                anchor_pose = np.asarray(raw, dtype=np.float64)
        pose_slice = arm_plan.get("pose")
        if pose_slice is not None:
            xyz = slice(pose_slice.start, pose_slice.start + 3)
            quat = slice(pose_slice.start + 3, pose_slice.start + 7)
            anchor_xyz = anchor_pose[:3] if anchor_pose is not None else None
            anchor_quat = anchor_pose[3:7] if anchor_pose is not None else None
            out[:, xyz] = _blend_xyz(
                prev_tail[:, xyz], fresh[:, xyz], anchor_xyz, w, p
            )
            out[:, quat] = _blend_quat(
                prev_tail[:, quat], fresh[:, quat], anchor_quat, w, p
            )
        for channel in ("twist", "wrench"):
            chan_slice = arm_plan.get(channel)
            if chan_slice is None:
                continue
            m = min(p, n)
            wm = w[:m, None]
            out[:m, chan_slice] = (
                wm * prev_tail[:m, chan_slice] + (1.0 - wm) * fresh[:m, chan_slice]
            )
    return out
