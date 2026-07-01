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

"""Time-parameterized Cartesian pose spline so the sender always has a smooth
target between sparse policy waypoints.

Ported from diffusion_policy's ``PoseTrajectoryInterpolator`` (Chi et al.),
adapted from 6-DoF axis-angle to our 7-element quaternion pose
``[x, y, z, qw, qx, qy, qz]``. Position is linearly interpolated, orientation
Slerp'd.
"""

from __future__ import annotations

import numbers

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st

# scipy.Rotation quaternions are [qx, qy, qz, qw]; our poses carry [qw, qx, qy,
# qz] after the position triplet. These helpers convert between the two layouts.
_POSE_DIM = 7


def _quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> st.Rotation:
    quat_wxyz = np.asarray(quat_wxyz)
    quat_xyzw = np.concatenate([quat_wxyz[..., 1:], quat_wxyz[..., :1]], axis=-1)
    return st.Rotation.from_quat(quat_xyzw)


def _rotation_to_quat_wxyz(rot: st.Rotation) -> np.ndarray:
    quat_xyzw = np.atleast_2d(rot.as_quat())
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)
    return quat_wxyz


def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()


def pose_distance(start_pose, end_pose) -> tuple[float, float]:
    start_pose = np.asarray(start_pose)
    end_pose = np.asarray(end_pose)
    pos_dist = float(np.linalg.norm(end_pose[:3] - start_pose[:3]))
    rot_dist = float(
        rotation_distance(
            _quat_wxyz_to_rotation(start_pose[3:7]),
            _quat_wxyz_to_rotation(end_pose[3:7]),
        )
    )
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator:
    """Spline over (time, 7-vec quaternion pose) waypoints.

    Queries are clamped to the trajectory span, so past-the-end holds the last
    pose -- this is how the sender holds position when no new waypoint arrived.
    """

    def __init__(self, times, poses) -> None:
        times = np.asarray(times, dtype=np.float64)
        poses = np.asarray(poses, dtype=np.float64)
        assert len(times) >= 1
        assert len(poses) == len(times)

        if len(times) == 1:
            # A single waypoint can't be interpolated; hold it for all queries.
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self.pos_interp = si.interp1d(
                times, poses[:, :3], axis=0, assume_sorted=True
            )
            self.rot_interp = st.Slerp(times, _quat_wxyz_to_rotation(poses[:, 3:7]))

    @property
    def times(self) -> np.ndarray:
        return self._times if self.single_step else self.pos_interp.x

    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        n = len(self.times)
        poses = np.zeros((n, _POSE_DIM))
        poses[:, :3] = self.pos_interp.y
        poses[:, 3:7] = _rotation_to_quat_wxyz(self.rot_interp(self.times))
        return poses

    def trim(self, start_t: float, end_t: float) -> PoseTrajectoryInterpolator:
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # Slerp requires strictly increasing knots, so drop duplicates.
        all_times = np.unique(all_times)
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)

    def schedule_waypoint(
        self,
        pose,
        time,
        max_pos_speed: float = np.inf,
        max_rot_speed: float = np.inf,
        curr_time: float | None = None,
        last_waypoint_time: float | None = None,
    ) -> PoseTrajectoryInterpolator:
        """Blend a new target ``pose`` at absolute ``time`` into the trajectory,
        preserving it up to ``curr_time`` and re-planning the far end. Mirrors
        the reference logic."""
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # Target is in the past relative to now -- ignore it.
                return self
            start_time = max(curr_time, start_time)
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        assert start_time <= end_time
        assert end_time <= time
        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)

        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)
        return PoseTrajectoryInterpolator(times, poses)

    def __call__(self, t: numbers.Number | np.ndarray) -> np.ndarray:
        is_single = isinstance(t, numbers.Number)
        t = np.array([t]) if is_single else np.asarray(t, dtype=np.float64)

        pose = np.zeros((len(t), _POSE_DIM))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            t = np.clip(t, self.times[0], self.times[-1])
            pose[:, :3] = self.pos_interp(t)
            pose[:, 3:7] = _rotation_to_quat_wxyz(self.rot_interp(t))

        return pose[0] if is_single else pose

    def velocity(self, t: float, dt: float = 1e-3) -> np.ndarray:
        """Finite-difference TCP velocity ``[vx,vy,vz, wx,wy,wz]`` at ``t`` -- a
        cleaner velocity target for ``SendCartesianMotionForce`` than the
        policy's raw twist."""
        if self.single_step:
            return np.zeros(6)
        lo = float(np.clip(t - dt, self.times[0], self.times[-1]))
        hi = float(np.clip(t + dt, self.times[0], self.times[-1]))
        span = hi - lo
        if span <= 0:
            return np.zeros(6)
        p_lo = self(lo)
        p_hi = self(hi)
        lin = (p_hi[:3] - p_lo[:3]) / span
        rot_lo = _quat_wxyz_to_rotation(p_lo[3:7])
        rot_hi = _quat_wxyz_to_rotation(p_hi[3:7])
        ang = (rot_hi * rot_lo.inv()).as_rotvec() / span
        return np.concatenate([lin, ang])
