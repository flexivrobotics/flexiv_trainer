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

import numpy as np
import pytest
import scipy.spatial.transform as st

from flexivtrainer.rollout.replay_episode import _finite_difference_velocity
from flexivtrainer.rollout.seam_blend import (
    apply_twist_mode,
    blend_seam,
    seam_weights,
)


def _single_arm_layout() -> list[dict]:
    return [
        {
            "side": "single_arm",
            "pose": slice(0, 7),
            "twist": slice(7, 13),
            "wrench": slice(13, 19),
        }
    ]


def _dual_arm_layout() -> list[dict]:
    return [
        {
            "side": "left",
            "pose": slice(0, 7),
            "twist": slice(7, 13),
            "wrench": slice(13, 19),
        },
        {
            "side": "right",
            "pose": slice(19, 26),
            "twist": None,
            "wrench": slice(26, 32),
        },
    ]


def _identity_pose_row(x: float) -> list[float]:
    return [x, 2 * x, 3 * x, 1.0, 0.0, 0.0, 0.0]


def _single_arm_chunk(n: int) -> np.ndarray:
    rows = []
    for k in range(n):
        pose = _identity_pose_row(float(k))
        twist = [float(k) + 0.5] * 6
        wrench = [float(k) + 0.25] * 6
        rows.append(pose + twist + wrench)
    return np.asarray(rows, dtype=np.float32)


def test_raw_returns_input_unchanged() -> None:
    layout = _single_arm_layout()
    actions = _single_arm_chunk(5)

    out = apply_twist_mode(actions, layout, "raw", 0.1)

    assert out is actions
    assert np.array_equal(out, actions)


def test_zero_zeroes_only_twist_slice() -> None:
    layout = _single_arm_layout()
    actions = _single_arm_chunk(5)

    out = apply_twist_mode(actions, layout, "zero", 0.1)

    assert np.array_equal(out[:, 0:7], actions[:, 0:7])
    assert np.array_equal(out[:, 13:19], actions[:, 13:19])
    assert np.all(out[:, 7:13] == 0.0)
    # Input is not mutated.
    assert not np.all(actions[:, 7:13] == 0.0)


def test_zero_dual_arm_skips_missing_twist_no_cross_bleed() -> None:
    layout = _dual_arm_layout()
    n = 4
    left = _single_arm_chunk(n)
    right_pose = np.stack([_identity_pose_row(float(k) + 10.0) for k in range(n)])
    right_wrench = np.full((n, 6), 7.0, dtype=np.float32)
    actions = np.concatenate([left, right_pose, right_wrench], axis=1).astype(
        np.float32
    )

    out = apply_twist_mode(actions, layout, "zero", 0.1)

    # Left arm twist zeroed.
    assert np.all(out[:, 7:13] == 0.0)
    # Right arm (no twist slice) fully untouched: pose + wrench identical.
    assert np.array_equal(out[:, 19:26], actions[:, 19:26])
    assert np.array_equal(out[:, 26:32], actions[:, 26:32])
    # Left pose/wrench untouched.
    assert np.array_equal(out[:, 0:7], actions[:, 0:7])
    assert np.array_equal(out[:, 13:19], actions[:, 13:19])


def test_fd_matches_replay_reference() -> None:
    layout = _single_arm_layout()
    actions = _single_arm_chunk(6)
    dt = 0.1

    out = apply_twist_mode(actions, layout, "fd", dt)

    poses = actions[:, 0:7]
    for k in range(len(actions)):
        expected = _finite_difference_velocity(poses, k, dt)
        assert np.allclose(out[k, 7:13], expected)
    # Pose / wrench untouched.
    assert np.array_equal(out[:, 0:7], actions[:, 0:7])
    assert np.array_equal(out[:, 13:19], actions[:, 13:19])


def test_fd_boundary_prev_pose_gives_index0_continuity() -> None:
    layout = _single_arm_layout()
    dt = 0.1
    # Constant velocity along x of 1 unit/step -> 1/dt unit/s.
    actions = _single_arm_chunk(5)
    # Boundary sample is the point one step before index 0 (x = -1).
    boundary = [_identity_pose_row(-1.0)]

    out = apply_twist_mode(actions, layout, "fd", dt, boundary_prev_poses=boundary)

    # xyz step is (1, 2, 3) per unit x -> velocity (1,2,3)/dt.
    expected_lin = np.array([1.0, 2.0, 3.0]) / dt
    assert np.allclose(out[0, 7:10], expected_lin)


def test_fd_angular_constant_z_rotation() -> None:
    layout = _single_arm_layout()
    dt = 0.1
    omega = 0.2  # rad/step
    n = 5
    rows = []
    for k in range(n):
        rot = st.Rotation.from_rotvec([0.0, 0.0, omega * k])
        quat_xyzw = rot.as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        pose = [0.0, 0.0, 0.0, *quat_wxyz]
        rows.append(pose + [0.0] * 6 + [0.0] * 6)
    actions = np.asarray(rows, dtype=np.float64)

    out = apply_twist_mode(actions, layout, "fd", dt)

    interior = out[2, 10:13]
    assert np.allclose(interior, [0.0, 0.0, omega / dt], atol=1e-6)


def test_invalid_mode_raises() -> None:
    layout = _single_arm_layout()
    actions = _single_arm_chunk(3)

    with pytest.raises(ValueError):
        apply_twist_mode(actions, layout, "bogus", 0.1)


# --- seam_weights ----------------------------------------------------------


def test_seam_weights_linear_ramp() -> None:
    w = seam_weights(10, freeze=2, fade_end=6, schedule="linear")

    assert np.all(w[:2] == 1.0)
    ramp = w[2:6]
    assert np.all(np.diff(ramp) < 0)  # strictly decreasing
    assert w[5] == pytest.approx(0.0, abs=1e-9)  # exactly 0 at fade_end - 1
    assert np.all(w[6:] == 0.0)


def test_seam_weights_exp_ramp_reaches_zero() -> None:
    w = seam_weights(10, freeze=1, fade_end=7, schedule="exp")

    assert w[0] == 1.0
    ramp = w[1:7]
    assert np.all(np.diff(ramp) < 0)
    assert ramp[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(w[7:] == 0.0)
    # Exp decays faster early than linear over the same span.
    linear = seam_weights(10, freeze=1, fade_end=7, schedule="linear")
    assert w[2] < linear[2]


def test_seam_weights_hard_switch_when_fade_end_le_freeze() -> None:
    w = seam_weights(8, freeze=3, fade_end=3, schedule="linear")

    assert np.all(w[:3] == 1.0)
    assert np.all(w[3:] == 0.0)


# --- blend_seam ------------------------------------------------------------


def _const_velocity_chunk(n: int, v: np.ndarray, start: np.ndarray) -> np.ndarray:
    """(n, 19) chunk: xyz moves at constant velocity ``v`` from ``start``."""
    rows = []
    for k in range(n):
        xyz = start + v * k
        pose = [*xyz.tolist(), 1.0, 0.0, 0.0, 0.0]
        rows.append(pose + [0.0] * 6 + [0.0] * 6)
    return np.asarray(rows, dtype=np.float64)


def test_blend_seam_velocity_continuity_and_no_stall() -> None:
    layout = _single_arm_layout()
    n = 10
    v1 = np.array([0.01, 0.0, 0.0])
    v2 = np.array([0.05, 0.0, 0.0])
    # prev_tail and fresh both start at the same first position (RTC pins it).
    start = np.array([1.0, 0.0, 0.0])
    prev_tail = _const_velocity_chunk(n, v1, start)
    fresh = _const_velocity_chunk(n, v2, start)
    # Anchor consistent with v1: one step before prev_tail[0].
    anchor_pose = [*(start - v1).tolist(), 1.0, 0.0, 0.0, 0.0]

    w = seam_weights(n, freeze=1, fade_end=7, schedule="linear")
    out = blend_seam(prev_tail, fresh, layout, [anchor_pose], w)

    out_xyz = out[:, 0:3]
    a = np.array(anchor_pose[:3])
    # Full blended-velocity sequence including index 0 (delta from the anchor).
    vel = np.diff(np.vstack([a, out_xyz]), axis=0)[:, 0]
    # Velocity moves monotonically from v1 (index 0) toward v2.
    assert vel[0] == pytest.approx(v1[0], abs=1e-9)  # out[0]=anchor+v1, no stall
    assert np.all(np.diff(vel) >= -1e-9)  # non-decreasing toward v2
    assert vel[-1] == pytest.approx(v2[0], abs=1e-9)  # settles at fresh velocity
    # No stall at index 0: out[0] == anchor + v1 (not the anchor itself).
    assert out_xyz[0, 0] == pytest.approx(start[0], abs=1e-9)
    # The handoff kink is spread, strictly smaller than the raw disagreement.
    second = np.diff(np.diff(out_xyz[:, 0]))
    assert np.max(np.abs(second)) < abs(v2[0] - v1[0])


def test_blend_seam_noop_prev_none_returns_same_object() -> None:
    layout = _single_arm_layout()
    fresh = _single_arm_chunk(5).astype(np.float64)
    w = seam_weights(5, freeze=1, fade_end=4, schedule="linear")

    out = blend_seam(None, fresh, layout, None, w)

    assert out is fresh


def test_blend_seam_noop_zero_weights_returns_same_object() -> None:
    layout = _single_arm_layout()
    fresh = _single_arm_chunk(5).astype(np.float64)
    prev_tail = _single_arm_chunk(5).astype(np.float64)
    w = np.zeros(5)

    out = blend_seam(prev_tail, fresh, layout, None, w)

    assert out is fresh


def test_blend_seam_quaternion_continuity_and_unit_norm() -> None:
    layout = _single_arm_layout()
    n = 8
    dt_ang_old = 0.05  # rad/step
    dt_ang_new = 0.20

    def _rot_chunk(omega: float) -> np.ndarray:
        rows = []
        for k in range(n):
            rot = st.Rotation.from_rotvec([0.0, 0.0, omega * k])
            q = rot.as_quat()  # xyzw
            pose = [0.0, 0.0, 0.0, q[3], q[0], q[1], q[2]]
            rows.append(pose + [0.0] * 6 + [0.0] * 6)
        return np.asarray(rows, dtype=np.float64)

    prev_tail = _rot_chunk(dt_ang_old)
    fresh = _rot_chunk(dt_ang_new)
    # Anchor one step before index 0 on the old rate.
    a_rot = st.Rotation.from_rotvec([0.0, 0.0, -dt_ang_old])
    aq = a_rot.as_quat()
    anchor_pose = [0.0, 0.0, 0.0, aq[3], aq[0], aq[1], aq[2]]

    w = seam_weights(n, freeze=1, fade_end=6, schedule="linear")
    out = blend_seam(prev_tail, fresh, layout, [anchor_pose], w)

    quats = out[:, 3:7]
    norms = np.linalg.norm(quats, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)

    # Angular velocity (z rotvec of successive deltas) transitions monotonically.
    ang = []
    rots = [
        st.Rotation.from_quat([q[1], q[2], q[3], q[0]]) for q in quats
    ]
    prev = a_rot
    for r in rots:
        ang.append((r * prev.inv()).as_rotvec()[2])
        prev = r
    ang = np.asarray(ang)
    assert ang[0] == pytest.approx(dt_ang_old, abs=1e-9)  # no stall at index 0
    assert np.all(np.diff(ang) >= -1e-9)
    assert ang[-1] == pytest.approx(dt_ang_new, abs=1e-6)


def test_blend_seam_prev_shorter_than_fresh_falls_back_to_fresh_velocity() -> None:
    layout = _single_arm_layout()
    n = 10
    p = 4
    v1 = np.array([0.01, 0.0, 0.0])
    v2 = np.array([0.05, 0.0, 0.0])
    start = np.array([1.0, 0.0, 0.0])
    prev_tail = _const_velocity_chunk(p, v1, start)
    fresh = _const_velocity_chunk(n, v2, start)
    anchor_pose = [*(start - v1).tolist(), 1.0, 0.0, 0.0, 0.0]

    # All-ones weights up to fade_end well past p, so only the P clamp matters.
    w = seam_weights(n, freeze=1, fade_end=9, schedule="linear")
    out = blend_seam(prev_tail, fresh, layout, [anchor_pose], w)

    dx = np.diff(out[:, 0], axis=0)
    # Indices >= P use the pure-fresh delta (v2).
    assert np.allclose(dx[p:], v2[0], atol=1e-9)


def test_blend_seam_twist_wrench_positional_blend() -> None:
    layout = _single_arm_layout()
    n = 6
    prev_tail = np.zeros((n, 19), dtype=np.float64)
    fresh = np.zeros((n, 19), dtype=np.float64)
    prev_tail[:, 3] = 1.0  # identity quaternion (wxyz)
    fresh[:, 3] = 1.0
    prev_tail[:, 7:13] = 2.0  # twist
    fresh[:, 7:13] = 6.0
    prev_tail[:, 13:19] = 1.0  # wrench
    fresh[:, 13:19] = 5.0

    w = seam_weights(n, freeze=1, fade_end=n, schedule="linear")
    out = blend_seam(prev_tail, fresh, layout, None, w)

    # At index 0 (w=1) the old value dominates; later fades toward fresh.
    assert out[0, 7] == pytest.approx(2.0)
    assert out[-1, 7] == pytest.approx(6.0)
    assert out[0, 13] == pytest.approx(1.0)
    assert out[-1, 13] == pytest.approx(5.0)
