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

from flexivtrainer.rollout.pose_interpolator import PoseTrajectoryInterpolator

# Identity orientation quaternion [qw, qx, qy, qz].
_IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]


def _pose(x, y, z, quat=_IDENTITY_QUAT):
    return [x, y, z, *quat]


def _unit(quat):
    return pytest.approx(1.0, abs=1e-6) == float(np.linalg.norm(quat))


def test_interpolates_position_between_waypoints() -> None:
    interp = PoseTrajectoryInterpolator(
        times=[0.0, 1.0],
        poses=[_pose(0.0, 0.0, 0.0), _pose(1.0, 2.0, 3.0)],
    )
    mid = interp(0.5)
    assert mid[:3] == pytest.approx([0.5, 1.0, 1.5])
    assert _unit(mid[3:7])


def test_query_past_end_holds_last_pose() -> None:
    interp = PoseTrajectoryInterpolator(
        times=[0.0, 1.0],
        poses=[_pose(0.0, 0.0, 0.0), _pose(1.0, 1.0, 1.0)],
    )
    # Times are clamped to the span, so before-start and past-end hold the ends.
    assert interp(-5.0)[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert interp(10.0)[:3] == pytest.approx([1.0, 1.0, 1.0])


def test_single_waypoint_holds_for_all_queries() -> None:
    interp = PoseTrajectoryInterpolator(times=[0.0], poses=[_pose(0.3, 0.4, 0.5)])
    assert interp(0.0)[:3] == pytest.approx([0.3, 0.4, 0.5])
    assert interp(99.0)[:3] == pytest.approx([0.3, 0.4, 0.5])
    assert interp.velocity(50.0) == pytest.approx(np.zeros(6))


def test_orientation_is_slerped_to_unit_quaternion() -> None:
    # 0 deg -> 90 deg about z; halfway should be ~45 deg and unit-norm.
    q0 = _IDENTITY_QUAT
    half = np.sqrt(0.5)
    q90 = [half, 0.0, 0.0, half]  # 90 deg about z, [qw,qx,qy,qz]
    interp = PoseTrajectoryInterpolator(
        times=[0.0, 1.0], poses=[_pose(0, 0, 0, q0), _pose(0, 0, 0, q90)]
    )
    mid = interp(0.5)
    assert _unit(mid[3:7])
    # 45 deg about z -> qw = cos(22.5deg).
    assert mid[3] == pytest.approx(np.cos(np.deg2rad(22.5)), abs=1e-6)


def test_schedule_waypoint_blends_far_end_without_disturbing_near_term() -> None:
    interp = PoseTrajectoryInterpolator(
        times=[0.0, 1.0],
        poses=[_pose(0.0, 0.0, 0.0), _pose(1.0, 0.0, 0.0)],
    )
    # At curr_time=0.5, schedule a new target at t=2.0. The near-term pose at
    # curr_time must be preserved; the far end now heads to the new target.
    near_before = interp(0.5)
    blended = interp.schedule_waypoint(
        pose=_pose(5.0, 0.0, 0.0), time=2.0, curr_time=0.5
    )
    assert blended(0.5)[:3] == pytest.approx(near_before[:3], abs=1e-6)
    assert blended(2.0)[:3] == pytest.approx([5.0, 0.0, 0.0], abs=1e-6)


def test_schedule_waypoint_in_the_past_is_ignored() -> None:
    interp = PoseTrajectoryInterpolator(
        times=[0.0, 1.0],
        poses=[_pose(0.0, 0.0, 0.0), _pose(1.0, 0.0, 0.0)],
    )
    same = interp.schedule_waypoint(pose=_pose(9, 9, 9), time=0.2, curr_time=0.5)
    assert same is interp


def test_velocity_matches_constant_linear_motion() -> None:
    # 2 m over 1 s along x -> vx = 2 m/s, no rotation.
    interp = PoseTrajectoryInterpolator(
        times=[0.0, 1.0], poses=[_pose(0.0, 0.0, 0.0), _pose(2.0, 0.0, 0.0)]
    )
    vel = interp.velocity(0.5)
    assert vel[:3] == pytest.approx([2.0, 0.0, 0.0], abs=1e-3)
    assert vel[3:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
