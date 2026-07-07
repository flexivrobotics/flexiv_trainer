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

"""On-robot A/B replay of a recorded episode's action sequence, OPEN-LOOP.

This standalone script replays the Cartesian waypoints of one LeRobot episode
on the real Flexiv robot without any policy or cameras, executing the same
waypoints two ways so the hand-rolled spline interpolation layer can be judged
against the robot's own online trajectory generator (OTG):

  * ``--interpolator true`` (mode A): build a single
    :class:`PoseTrajectoryInterpolator` over the whole episode up front, then a
    high-rate loop streams ``interp(now)`` pose + ``interp.velocity(now)`` via
    ``SendCartesianMotionForce`` -- the same behaviour as the rollout sender in
    ``service.py``.
  * ``--interpolator false`` (mode B): send each raw waypoint exactly once at
    its target time and let the robot's internal OTG blend the sparse poses.
    The commanded velocity is chosen by ``--velocity {twist,fd,zero}``.

A background thread samples the measured TCP pose for the whole replay and the
result is saved to an ``.npz`` with summary motion statistics, so the two modes
can be compared with numbers rather than eyeballs.

Run (dry-run, never touches the robot)::

    .venv/bin/python -m flexivtrainer.rollout.replay_episode \
        --dataset .local/episodes/DP_push_t_10hz/20260702_130029 \
        --interpolator false --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.spatial.transform as st

from flexivtrainer.rollout.pose_interpolator import PoseTrajectoryInterpolator

# Scalars per metric: tcp_pose is [x,y,z,qw,qx,qy,qz]; tcp_twist (velocity) is
# 6-axis [vx,vy,vz, wx,wy,wz]. Matches the layout in service.py.
_POSE_DIM = 7
_TWIST_DIM = 6

# Default motion limits, mirroring RolloutConfig so the two modes drive the arm
# under identical constraints.
_DEFAULT_MAX_LINEAR_VEL = 0.25  # m/s
_DEFAULT_MAX_ANGULAR_VEL = 0.6  # rad/s
_DEFAULT_MAX_LINEAR_ACC = 1.0  # m/s^2
_DEFAULT_MAX_ANGULAR_ACC = 2.5  # rad/s^2

_ROBOT_SERIALS_PATH = Path(".local") / "robot_serials.json"
_DEFAULT_LOG_HZ = 100.0


def _str2bool(value: str) -> bool:
    """Parse a case-insensitive truthy/falsy string for argparse."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> st.Rotation:
    """Convert our [qw,qx,qy,qz] quaternion(s) to a scipy Rotation ([qx,qy,qz,qw])."""
    quat_wxyz = np.asarray(quat_wxyz)
    quat_xyzw = np.concatenate([quat_wxyz[..., 1:], quat_wxyz[..., :1]], axis=-1)
    return st.Rotation.from_quat(quat_xyzw)


def _normalize_pose_quaternion(pose: list[float]) -> list[float]:
    """Renormalize the orientation quaternion of a ``[x,y,z,qw,qx,qy,qz]`` pose.

    Recorded poses may carry a slightly non-unit quaternion (the policy regresses
    each component independently); ``SendCartesianMotionForce`` expects a unit
    quaternion. A near-zero norm is left untouched to avoid dividing by ~0. Local
    copy so this script stays independent of ``service.py``.
    """
    pose = list(pose)
    if len(pose) < _POSE_DIM:
        return pose
    quat = pose[3:7]
    norm = sum(component * component for component in quat) ** 0.5
    if norm > 1e-6:
        pose[3:7] = [component / norm for component in quat]
    return pose


def _find_run(names: list[str], prefix: str) -> int | None:
    """Index of the first axis name starting with ``prefix``, or ``None``."""
    for index, name in enumerate(names):
        if name.startswith(prefix):
            return index
    return None


def _detect_side(action_names: list[str]) -> str:
    """First arm side prefix present in the action names (e.g. ``single_arm``)."""
    for name in action_names:
        if ".tcp_pose." in name:
            return name.split(".tcp_pose.", 1)[0]
    raise ValueError("no '<side>.tcp_pose.*' axes found in action feature names")


def _load_default_serial() -> str | None:
    """First non-empty follower serial from ``.local/robot_serials.json``."""
    if not _ROBOT_SERIALS_PATH.exists():
        return None
    data = json.loads(_ROBOT_SERIALS_PATH.read_text(encoding="utf-8"))
    for serial in data.get("follower_robot_serials", []):
        serial = str(serial).strip()
        if serial:
            return serial
    return None


def _load_episode(dataset: Path, episode: int) -> tuple[np.ndarray, list[str], int]:
    """Return ``(action_matrix, action_names, fps)`` for one episode.

    ``action_matrix`` has shape ``(n_frames, action_dim)`` ordered by frame.
    """
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing dataset info: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = int(info["fps"])
    action_names = list(info["features"]["action"]["names"])

    parquets = sorted(
        glob.glob(str(dataset / "data" / "**" / "*.parquet"), recursive=True)
    )
    if not parquets:
        raise FileNotFoundError(f"no parquet files under {dataset / 'data'}")

    frames: list[pd.DataFrame] = []
    for path in parquets:
        chunk = pd.read_parquet(
            path, columns=["action", "frame_index", "episode_index"]
        )
        frames.append(chunk[chunk["episode_index"] == episode])
    table = pd.concat(frames, ignore_index=True)
    if table.empty:
        raise ValueError(f"episode {episode} not found in {dataset}")
    table = table.sort_values("frame_index")

    action = np.stack([np.asarray(row, dtype=np.float64) for row in table["action"]])
    return action, action_names, fps


def _finite_difference_velocity(poses: np.ndarray, index: int, dt: float) -> np.ndarray:
    """Central-difference TCP velocity ``[vx,vy,vz, wx,wy,wz]`` at ``index``.

    Linear part is the position slope; angular part is the rotvec of the
    neighbouring-quaternion delta over the span (same pattern as
    ``PoseTrajectoryInterpolator.velocity``). Falls back to forward/backward
    differencing at the trajectory ends.
    """
    n = len(poses)
    lo = max(index - 1, 0)
    hi = min(index + 1, n - 1)
    span = (hi - lo) * dt
    if span <= 0:
        return np.zeros(_TWIST_DIM)
    lin = (poses[hi, :3] - poses[lo, :3]) / span
    rot_lo = _quat_wxyz_to_rotation(poses[lo, 3:7])
    rot_hi = _quat_wxyz_to_rotation(poses[hi, 3:7])
    ang = (rot_hi * rot_lo.inv()).as_rotvec() / span
    return np.concatenate([lin, ang])


def _build_timeline(
    action: np.ndarray,
    action_names: list[str],
    fps: int,
    velocity_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice poses/twists out of the action matrix and build the command timeline.

    Returns ``(times, poses, velocities)`` where ``times`` are per-waypoint
    offsets in seconds from the first waypoint (t0 = 0), ``poses`` are unit-
    quaternion 7-vectors, and ``velocities`` are the 6-vec commanded velocities
    selected by ``velocity_mode`` (used by mode B; mode A derives its own from
    the spline).
    """
    side = _detect_side(action_names)
    pose_start = _find_run(action_names, f"{side}.tcp_pose.")
    twist_start = _find_run(action_names, f"{side}.tcp_twist.")
    if pose_start is None:
        raise ValueError(f"no '{side}.tcp_pose.*' run in action names")
    pose_slice = slice(pose_start, pose_start + _POSE_DIM)
    twist_slice = (
        None if twist_start is None else slice(twist_start, twist_start + _TWIST_DIM)
    )

    dt = 1.0 / float(fps)
    n = len(action)
    times = np.arange(n, dtype=np.float64) * dt
    poses = np.stack(
        [
            np.asarray(_normalize_pose_quaternion(list(action[k, pose_slice])))
            for k in range(n)
        ]
    )

    if velocity_mode == "twist":
        if twist_slice is None:
            raise ValueError(
                f"--velocity twist requires a '{side}.tcp_twist.*' run in the action "
                "names, which this dataset does not have"
            )
        velocities = np.stack(
            [np.asarray(action[k, twist_slice], dtype=np.float64) for k in range(n)]
        )
    elif velocity_mode == "fd":
        velocities = np.stack(
            [_finite_difference_velocity(poses, k, dt) for k in range(n)]
        )
    elif velocity_mode == "zero":
        velocities = np.zeros((n, _TWIST_DIM))
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown velocity mode {velocity_mode!r}")

    return times, poses, velocities


def _motion_stats(t: np.ndarray, measured_pose: np.ndarray) -> dict[str, float]:
    """Linear speed / acceleration / jerk statistics from the measured trace.

    Uses simple finite differences over the (uneven) sample times; the measured
    trace is dense (~100 Hz) so smoothing-free diffs are adequate for A/B numbers.
    """
    if len(t) < 4:
        return {}
    pos = measured_pose[:, :3]
    dt = np.diff(t)
    dt = np.where(dt > 1e-9, dt, 1e-9)
    vel = np.diff(pos, axis=0) / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    # Acceleration/jerk on the velocity midpoints; use the average spacing.
    acc = np.diff(vel, axis=0) / dt[:-1, None]
    jerk = np.diff(acc, axis=0) / dt[:-2, None]
    return {
        "mean_linear_speed": float(np.mean(speed)),
        "max_linear_speed": float(np.max(speed)),
        "rms_linear_acc": float(np.sqrt(np.mean(np.sum(acc**2, axis=1)))),
        "rms_linear_jerk": float(np.sqrt(np.mean(np.sum(jerk**2, axis=1)))),
    }


class _PoseLogger:
    """Background sampler of the measured TCP pose during a replay.

    Records ``(t, measured_pose, commanded_pose)`` at ~``hz`` for the whole run.
    The commanded pose is read from a shared slot the main thread updates on each
    send, so every measured sample carries the command target that was active.
    """

    def __init__(self, robot: Any, hz: float = _DEFAULT_LOG_HZ) -> None:
        self._robot = robot
        self._period = 1.0 / float(hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._command = [0.0] * _POSE_DIM
        self.t: list[float] = []
        self.measured_pose: list[list[float]] = []
        self.commanded_pose: list[list[float]] = []

    def set_command(self, pose: list[float]) -> None:
        with self._lock:
            self._command = [float(v) for v in pose]

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="replay-logger"
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                measured = [float(v) for v in self._robot.states().tcp_pose]
            except Exception:  # pragma: no cover - hardware specific
                measured = [float("nan")] * _POSE_DIM
            with self._lock:
                command = list(self._command)
            self.t.append(start - self._t0)
            self.measured_pose.append(measured)
            self.commanded_pose.append(command)
            rest = self._period - (time.monotonic() - start)
            if rest > 0:
                self._stop.wait(rest)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "t": np.asarray(self.t, dtype=np.float64),
            "measured_pose": np.asarray(self.measured_pose, dtype=np.float64),
            "commanded_pose": np.asarray(self.commanded_pose, dtype=np.float64),
        }


def _connect_robot(serial: str, stop_event: threading.Event) -> Any:
    """Connect/enable a Flexiv robot and switch to NRT Cartesian motion-force.

    Mirrors ``RolloutService._connect_robot``: clear fault, enable, wait until
    operational, then ``SwitchMode(NRT_CARTESIAN_MOTION_FORCE)``.
    """
    import flexivrdk  # noqa: PLC0415

    robot = flexivrdk.Robot(serial)
    if robot.fault():
        robot.ClearFault()
    robot.Enable()
    while not robot.operational():
        if stop_event.wait(0.1):
            break
    robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
    return robot


def _send(robot: Any, pose: list[float], velocity: list[float], limits: tuple) -> None:
    """SendCartesianMotionForce with a zero wrench and the shared motion limits.

    Signature matches the call used throughout this repo:
    ``(pose7, wrench6, velocity6, max_lin_vel, max_ang_vel, max_lin_acc, max_ang_acc)``.
    """
    max_lin_vel, max_ang_vel, max_lin_acc, max_ang_acc = limits
    robot.SendCartesianMotionForce(
        [float(v) for v in pose],
        [0.0] * _TWIST_DIM,
        [float(v) for v in velocity],
        max_lin_vel,
        max_ang_vel,
        max_lin_acc,
        max_ang_acc,
    )


def _replay_interpolated(
    robot: Any,
    logger: _PoseLogger,
    times: np.ndarray,
    poses: np.ndarray,
    limits: tuple,
    sender_hz: float,
    stop_event: threading.Event,
) -> None:
    """Mode A: one spline over all waypoints, streamed at ``sender_hz``.

    Open-loop replay has a fixed timeline (no replanning), so the spline is built
    in a single shot -- ``times = [t_start, t0..tN]``, ``poses = [measured, p0..pN]``
    -- rather than spliced with ``schedule_waypoint``. The first knot seeds from
    the measured pose so the arm eases from where it actually is.
    """
    measured = _normalize_pose_quaternion([float(v) for v in robot.states().tcp_pose])
    now = time.monotonic()
    # A couple of dt of lead so the first commanded pose is reachable.
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.1
    t_start = now
    knot_times = np.concatenate([[t_start], now + 2.0 * dt + times])
    knot_poses = np.concatenate([[np.asarray(measured)], poses])
    interp = PoseTrajectoryInterpolator(times=knot_times, poses=knot_poses)

    period = 1.0 / sender_hz
    end_time = knot_times[-1]
    while not stop_event.is_set():
        tick = time.monotonic()
        if robot.fault():
            raise RuntimeError("robot reported a fault during replay")
        pose = interp(tick)
        velocity = interp.velocity(tick)
        logger.set_command(list(pose))
        _send(robot, list(pose), list(velocity), limits)
        if tick >= end_time:
            break
        rest = period - (time.monotonic() - tick)
        if rest > 0:
            stop_event.wait(rest)


def _replay_sparse(
    robot: Any,
    logger: _PoseLogger,
    times: np.ndarray,
    poses: np.ndarray,
    velocities: np.ndarray,
    limits: tuple,
    stop_event: threading.Event,
) -> None:
    """Mode B: send each waypoint once at its target time; robot OTG smooths.

    The first send is scheduled a couple of ``dt`` in the future so the OTG plans
    the initial ease-in from the current robot state.
    """
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.1
    t0 = time.monotonic() + 2.0 * dt
    for k in range(len(times)):
        if stop_event.is_set():
            break
        target = t0 + float(times[k])
        # Wait until this waypoint's target time.
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            if stop_event.wait(min(remaining, 0.01)):
                break
        if robot.fault():
            raise RuntimeError("robot reported a fault during replay")
        pose = list(poses[k])
        logger.set_command(pose)
        _send(robot, pose, list(velocities[k]), limits)


def _print_dry_run(
    times: np.ndarray,
    poses: np.ndarray,
    velocities: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Print the planned command timeline and waypoint-spacing stats; no robot."""
    n = len(times)
    spacing = np.diff(times) if n > 1 else np.array([])
    print("\n=== DRY RUN (no robot, no flexivrdk import) ===")
    print(f"pose slice dims : {poses.shape[1]}")
    print(f"twist slice dims: {velocities.shape[1]}")
    print(f"waypoints       : {n}")
    if spacing.size:
        print(
            "waypoint spacing: "
            f"mean={spacing.mean():.4f}s min={spacing.min():.4f}s "
            f"max={spacing.max():.4f}s (expected {1.0 / args.fps:.4f}s)"
        )
    step_pos = (
        np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) if n > 1 else np.array([])
    )
    if step_pos.size:
        print(
            "per-step xyz move: "
            f"mean={step_pos.mean() * 1000:.2f}mm max={step_pos.max() * 1000:.2f}mm"
        )
    print(f"\nfirst {min(10, n)} planned commands (t_offset s | pose xyz | vel lin):")
    for k in range(min(10, n)):
        xyz = poses[k, :3]
        vlin = velocities[k, :3]
        print(
            f"  t={times[k]:6.3f}  "
            f"xyz=[{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}]  "
            f"vlin=[{vlin[0]:+.4f}, {vlin[1]:+.4f}, {vlin[2]:+.4f}]"
        )


def _default_out_path(dataset: Path, args: argparse.Namespace) -> Path:
    mode = "interp" if args.interpolator else "sparse"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{dataset.name}_ep{args.episode}_{mode}_{args.velocity}_{stamp}.npz"
    return Path(".local") / "replay_logs" / name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-loop A/B replay of a recorded episode on the Flexiv robot.",
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="LeRobot dataset dir"
    )
    parser.add_argument(
        "--episode", type=int, default=0, help="episode index to replay"
    )
    parser.add_argument(
        "--interpolator",
        type=_str2bool,
        default=True,
        help="true: spline sender (mode A); false: raw waypoints, robot OTG (mode B)",
    )
    parser.add_argument(
        "--velocity",
        choices=("twist", "fd", "zero"),
        default="twist",
        help="commanded velocity source for mode B (and dry-run preview)",
    )
    parser.add_argument(
        "--serial", default=None, help="robot serial (overrides config)"
    )
    parser.add_argument(
        "--sender-hz", type=float, default=200.0, help="mode A streaming rate (Hz)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="cap on frames replayed (0 = all)"
    )
    parser.add_argument("--max-linear-vel", type=float, default=_DEFAULT_MAX_LINEAR_VEL)
    parser.add_argument(
        "--max-angular-vel", type=float, default=_DEFAULT_MAX_ANGULAR_VEL
    )
    parser.add_argument("--max-linear-acc", type=float, default=_DEFAULT_MAX_LINEAR_ACC)
    parser.add_argument(
        "--max-angular-acc", type=float, default=_DEFAULT_MAX_ANGULAR_ACC
    )
    parser.add_argument("--out", type=Path, default=None, help="output .npz path")
    parser.add_argument(
        "--yes", action="store_true", help="skip the pre-motion confirmation prompt"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse + build timeline + print plan; never import flexivrdk",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset

    action, action_names, fps = _load_episode(dataset, args.episode)
    if args.max_frames > 0:
        action = action[: args.max_frames]
    # Expose the resolved fps on args so downstream helpers can reference it.
    args.fps = fps

    times, poses, velocities = _build_timeline(action, action_names, fps, args.velocity)
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    mode = "A (spline sender)" if args.interpolator else "B (raw waypoints / robot OTG)"
    limits = (
        args.max_linear_vel,
        args.max_angular_vel,
        args.max_linear_acc,
        args.max_angular_acc,
    )

    print("=== Episode replay ===")
    print(f"dataset      : {dataset}")
    print(f"episode      : {args.episode}")
    print(f"frames       : {len(times)} (fps={fps})")
    print(f"duration     : {duration:.2f}s")
    print(f"mode         : {mode}")
    print(f"velocity src : {args.velocity}")
    print(
        "limits       : "
        f"lin_vel={limits[0]} ang_vel={limits[1]} "
        f"lin_acc={limits[2]} ang_acc={limits[3]}"
    )

    if args.dry_run:
        _print_dry_run(times, poses, velocities, args)
        return 0

    serial = args.serial or _load_default_serial()
    if not serial:
        raise SystemExit(
            "no robot serial: pass --serial or populate follower_robot_serials in "
            f"{_ROBOT_SERIALS_PATH}"
        )
    print(f"serial       : {serial}")
    if args.interpolator:
        print(f"sender rate  : {args.sender_hz} Hz")

    if not args.yes:
        input("\nPress Enter to start motion (Ctrl-C to abort)... ")

    stop_event = threading.Event()
    robot = _connect_robot(serial, stop_event)
    logger = _PoseLogger(robot)
    logger.start()
    try:
        if args.interpolator:
            _replay_interpolated(
                robot, logger, times, poses, limits, args.sender_hz, stop_event
            )
        else:
            _replay_sparse(robot, logger, times, poses, velocities, limits, stop_event)
    except KeyboardInterrupt:
        print("\nInterrupted -- stopping robot.")
        stop_event.set()
        robot.Stop()
    except Exception as exc:
        stop_event.set()
        robot.Stop()
        raise SystemExit(f"replay aborted: {exc}") from exc
    finally:
        stop_event.set()
        logger.stop()
        robot.Stop()

    data = logger.arrays()
    stats = _motion_stats(data["t"], data["measured_pose"])

    out_path = args.out or _default_out_path(dataset, args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        t=data["t"],
        measured_pose=data["measured_pose"],
        commanded_pose=data["commanded_pose"],
        # attrs for later comparison of the two modes.
        dataset=str(dataset),
        episode=args.episode,
        fps=fps,
        mode="interp" if args.interpolator else "sparse",
        velocity_mode=args.velocity,
        sender_hz=args.sender_hz,
        limits=np.asarray(limits, dtype=np.float64),
    )

    print(f"\nSaved log -> {out_path}")
    if stats:
        print("=== Measured motion stats ===")
        print(f"mean linear speed : {stats['mean_linear_speed'] * 1000:.2f} mm/s")
        print(f"max linear speed  : {stats['max_linear_speed'] * 1000:.2f} mm/s")
        print(f"RMS linear acc    : {stats['rms_linear_acc']:.4f} m/s^2")
        print(f"RMS linear jerk   : {stats['rms_linear_jerk']:.2f} m/s^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
