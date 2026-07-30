#!/usr/bin/env python
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

"""Offline probe: compare demonstrated vs predicted vs commanded B-spline motion.

No robot and no hardware are involved. The script answers two questions:

  Part A (per-frame plan comparison)
      For probe frames, run the policy on the *dataset's own* observations and
      compare the predicted spline parameters against the ground-truth
      parameters stored in the dataset. Reports geometry error separately from
      timing error, which is what distinguishes "went the wrong way" from
      "went the right way too slowly".

  Part B (virtual-clock closed-loop replay)
      Drive the real ``BSplineExecutor`` over a virtual clock, replanning
      exactly when production would, and log every commanded pose next to the
      demonstrated pose at the same wall-clock instant. This exposes lag: if
      the commanded trajectory trails the demonstration in time, the policy is
      reacting late rather than aiming wrong.

Usage:

    .venv/bin/python scripts/probe_bspline_rollout.py \
        --checkpoint .local/training/<run>/checkpoints/last \
        --dataset .local/datasets/<bspline-dataset> \
        --episode 0 --probes 12 --replay-seconds 20 \
        --out /tmp/bspline_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.interpolate import BSpline

from flexivtrainer.rollout.checkpoint import _default_policy_loader
from flexivtrainer.rollout.executors.bspline import _parse_layout


class _CaptureRobot:
    """Stand-in for a Flexiv follower that records commanded Cartesian poses."""

    def __init__(self) -> None:
        self.commands: list[list[float]] = []

    def SendCartesianMotionForce(  # noqa: N802 - mirrors the RDK method name
        self, pose, *_args, **_kwargs
    ) -> None:
        self.commands.append([float(v) for v in pose])

    def fault(self) -> bool:
        return False


def _spline_from_matrix(matrix: np.ndarray, degree: int) -> BSpline:
    """Decode a parameter matrix exactly as BSplineExecutor._decode does."""
    knots = np.asarray(matrix[:, 0], dtype=np.float64).copy()
    for index in range(1, len(knots)):
        if knots[index] < knots[index - 1]:
            knots[index] = knots[index - 1] + 1e-6
    controls = np.asarray(matrix[: -(degree + 1), 1:], dtype=np.float64)
    return BSpline(knots, controls, degree, extrapolate=False)


def _domain(spline: BSpline, degree: int) -> tuple[float, float]:
    return float(spline.t[degree]), float(spline.t[-degree - 1])


def _positions(sample: np.ndarray, layouts) -> np.ndarray:
    """Stack per-arm XYZ from one evaluated spline sample."""
    return np.concatenate([sample[list(a.position_indices)] for a in layouts])


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _load_dataset(root: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset("local/probe", root=root)


def _episode_bounds(dataset, episode: int) -> tuple[int, int]:
    """Global frame range [from, to) of one episode (LeRobot v3 layout)."""
    episodes = dataset.meta.episodes
    if episode >= len(episodes):
        raise SystemExit(f"Episode {episode} not in dataset (has {len(episodes)})")
    row = episodes[episode]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


class _Inferencer:
    """Runs the policy on dataset observations the way the runner does."""

    def __init__(self, policy, preprocessor, postprocessor, dataset, device: str):
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._ds = dataset
        self._device = device
        self._keys = list(policy.config.image_features) + ["observation.state"]
        self._n_obs = int(policy.config.n_obs_steps)

    def predict(self, frame: int, lo: int) -> tuple[np.ndarray, float]:
        """Return (flat 304-vector, measured inference latency in seconds)."""
        self._policy.reset()
        for offset in range(self._n_obs - 1, -1, -1):
            item = self._ds[max(lo, frame - offset)]
            batch = {k: item[k].unsqueeze(0) for k in self._keys}
            batch["task"] = [item["task"]]
            self._policy.enqueue_observation(self._pre(batch))

        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        actions = self._policy.predict_action_chunk()
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        actions = self._post(actions)
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        return actions.detach().cpu().numpy().reshape(-1).astype(np.float64), latency


def part_a(
    args, inf: _Inferencer, dataset, layouts, degree: int, rows: int, channels: int
):
    """Per-frame comparison of demonstrated vs predicted spline parameters."""
    lo, hi = _episode_bounds(dataset, args.episode)
    frames = np.linspace(lo + 1, hi - 2, args.probes).round().astype(int)
    print(
        f"\n{'=' * 78}\nPART A - demonstrated vs predicted plan (episode {args.episode})\n{'=' * 78}"
    )
    print(
        f"{'frame':>7} {'gt_dom_s':>9} {'pred_dom_s':>10} {'dom_ratio':>9} "
        f"{'geom_err_m':>10} {'gt_speed':>9} {'pred_speed':>10} {'spd_ratio':>9} {'infer_ms':>9}"
    )

    records = []
    for frame in frames:
        pred_vec, latency = inf.predict(int(frame), lo)
        gt_vec = dataset[int(frame)]["action"].numpy().astype(np.float64)

        gt_spline = _spline_from_matrix(gt_vec.reshape(rows, channels), degree)
        pred_spline = _spline_from_matrix(pred_vec.reshape(rows, channels), degree)
        gt_lo, gt_hi = _domain(gt_spline, degree)
        pred_lo, pred_hi = _domain(pred_spline, degree)

        # Sample both curves over their own full domain at matched normalized
        # phase. This compares SHAPE independently of how long each curve says
        # it should take, which is the whole point of the split.
        phase = np.linspace(0.0, 1.0, args.samples)
        gt_pts = np.array(
            [
                _positions(np.asarray(gt_spline(gt_lo + p * (gt_hi - gt_lo))), layouts)
                for p in phase
            ]
        )
        pred_pts = np.array(
            [
                _positions(
                    np.asarray(pred_spline(pred_lo + p * (pred_hi - pred_lo))), layouts
                )
                for p in phase
            ]
        )

        gt_dom_s = (gt_hi - gt_lo) / args.knot_rate
        pred_dom_s = (pred_hi - pred_lo) / args.knot_rate
        geom_err = float(np.max(np.abs(pred_pts - gt_pts)))
        gt_speed = _path_length(gt_pts) / gt_dom_s if gt_dom_s > 0 else math.nan
        pred_speed = _path_length(pred_pts) / pred_dom_s if pred_dom_s > 0 else math.nan

        rec = {
            "frame": int(frame),
            "gt_domain_s": gt_dom_s,
            "pred_domain_s": pred_dom_s,
            "domain_ratio": pred_dom_s / gt_dom_s if gt_dom_s > 0 else math.nan,
            "geometry_max_err_m": geom_err,
            "gt_speed_m_per_s": gt_speed,
            "pred_speed_m_per_s": pred_speed,
            "speed_ratio": pred_speed / gt_speed if gt_speed > 0 else math.nan,
            "infer_ms": latency * 1000.0,
            "gt_knots": gt_vec.reshape(rows, channels)[:, 0].tolist(),
            "pred_knots": pred_vec.reshape(rows, channels)[:, 0].tolist(),
        }
        records.append(rec)
        print(
            f"{rec['frame']:>7} {gt_dom_s:>9.3f} {pred_dom_s:>10.3f} "
            f"{rec['domain_ratio']:>9.3f} {geom_err:>10.4f} {gt_speed:>9.4f} "
            f"{pred_speed:>10.4f} {rec['speed_ratio']:>9.3f} {latency * 1000:>9.1f}"
        )

    dr = np.array([r["domain_ratio"] for r in records])
    sr = np.array([r["speed_ratio"] for r in records])
    ge = np.array([r["geometry_max_err_m"] for r in records])
    print(
        f"\n  geometry max err   : median {np.median(ge):.4f} m   p95 {np.percentile(ge, 95):.4f} m"
    )
    print(
        f"  domain ratio       : median {np.median(dr):.3f}  (>1 = plan claims MORE time than demo)"
    )
    print(
        f"  speed ratio        : median {np.median(sr):.3f}  (<1 = commanded motion SLOWER than demo)"
    )
    print("\n  Interpretation:")
    print(
        "    geometry small + speed ratio < 1  -> right path, too slow (timing/knot problem)"
    )
    print(
        "    geometry large                    -> wrong path (policy quality problem)"
    )
    return records


def _smoothness_report(
    rows_out: list[dict[str, Any]], n_arms: int, period: float
) -> None:
    """Separate commanded-motion roughness at replans from roughness within a plan.

    The handoff is only C0 (pose-continuous): `_align` matches position, never
    velocity. Every install can therefore inject a velocity step. If boundary
    acceleration dominates within-plan acceleration, replanning is the jitter
    source and replanning *more often* makes jitter worse, not better.
    """
    if len(rows_out) < 4:
        return
    pos = np.array([[r[f"cmd_{i}"] for i in range(3 * n_arms)] for r in rows_out])
    plan = np.array([r["plan"] for r in rows_out])
    stamp = np.array([r["t"] for r in rows_out])

    # Use measured dt, not the nominal period: a mismatch would fake a spike.
    dt = np.diff(stamp)[:, None]
    step = np.max(
        [
            np.linalg.norm(np.diff(pos, axis=0)[:, 3 * a : 3 * a + 3], axis=1)
            for a in range(n_arms)
        ],
        axis=0,
    )
    edge_step = plan[1:] != plan[:-1]
    print("\n  commanded position step per control tick (worst arm)")
    print(
        f"    within a plan     : median {np.median(step[~edge_step]) * 1000:8.4f} mm"
    )
    print(
        f"    across a replan   : median {np.median(step[edge_step]) * 1000:8.4f} mm  "
        f"max {step[edge_step].max() * 1000:.4f} mm"
    )
    print(
        f"    step-up factor    : {np.median(step[edge_step]) / max(np.median(step[~edge_step]), 1e-12):.0f}x"
    )

    vel = np.diff(pos, axis=0) / dt

    # Heading change is what reads as "zigzag": position and speed can both be
    # continuous while the direction of travel snaps at every plan handoff.
    for arm in range(n_arms):
        v = vel[:, 3 * arm : 3 * arm + 3]
        speed = np.linalg.norm(v, axis=1)
        moving = speed > 1e-4
        unit = np.zeros_like(v)
        unit[moving] = v[moving] / speed[moving, None]
        cos = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
        turn = np.degrees(np.arccos(cos))
        ok = moving[1:] & moving[:-1]
        edge = (plan[1:-1] != plan[:-2]) & ok
        core = (plan[1:-1] == plan[:-2]) & ok
        if not edge.any() or not core.any():
            continue
        print(f"    arm {arm} heading change per tick:")
        print(f"      within a plan   : median {np.median(turn[core]):6.2f} deg  p99 {np.percentile(turn[core], 99):6.2f} deg")
        print(f"      across a replan : median {np.median(turn[edge]):6.2f} deg  max {turn[edge].max():6.2f} deg")

    acc = np.diff(vel, axis=0) / dt[:-1]
    # Per-arm magnitudes, then worst arm per tick.
    acc_mag = np.max(
        [np.linalg.norm(acc[:, 3 * a : 3 * a + 3], axis=1) for a in range(n_arms)],
        axis=0,
    )
    # acc[i] spans ticks i..i+2; a plan change inside that window is a boundary.
    boundary = (plan[1:-1] != plan[:-2]) | (plan[2:] != plan[1:-1])
    inside, edge = acc_mag[~boundary], acc_mag[boundary]
    if not len(edge) or not len(inside):
        return

    print("\n  commanded-motion smoothness (finite-difference on the sent poses)")
    print(
        f"    within-plan accel : median {np.median(inside):8.3f}  p99 {np.percentile(inside, 99):8.3f} m/s^2"
    )
    print(
        f"    at-replan  accel  : median {np.median(edge):8.3f}  max {edge.max():8.3f} m/s^2"
    )
    ratio = np.median(edge) / max(np.median(inside), 1e-9)
    print(f"    boundary/interior : {ratio:.1f}x")
    if ratio > 3.0:
        print(
            "    -> replan handoffs are the dominant roughness; more replanning = more jitter"
        )
    else:
        print(
            "    -> handoffs are not the dominant roughness; look at send_hz / robot-side tracking"
        )


def part_b(
    args,
    inf: _Inferencer,
    dataset,
    layouts,
    degree: int,
    rows: int,
    channels: int,
    feature_names,
):
    """Virtual-clock replay through the real BSplineExecutor."""
    from flexivtrainer.rollout.executors.bspline import BSplineExecutor

    lo, hi = _episode_bounds(dataset, args.episode)
    fps = float(args.knot_rate)
    duration = min(args.replay_seconds, (hi - lo - 2) / fps)
    period = 1.0 / args.control_hz

    now = 0.0
    clock = lambda: now  # noqa: E731 - executor takes an injectable clock
    robots = [_CaptureRobot() for _ in layouts]
    executor = BSplineExecutor(
        robots,
        feature_names,
        threading.Event(),
        (0.5, 1.0, 2.0, 4.0),
        checkpoint_fps=fps,
        degree=degree,
        control_hz=args.control_hz,
        speed_scale=args.speed_scale,
        predict_before_end_s=args.predict_before_end_s,
        handoff_blend_s=args.handoff_blend_s,
        handoff_max_accel=args.handoff_max_accel,
        clock=clock,
    )

    print(
        f"\n{'=' * 78}\nPART B - virtual-clock replay ({duration:.1f}s, control_hz={args.control_hz})\n{'=' * 78}"
    )

    rows_out: list[dict[str, Any]] = []
    plan_index = 0
    installed = False
    plan_events: list[dict[str, Any]] = []

    def tick() -> None:
        """One executor control step; records commanded vs demonstrated pose."""
        before = len(robots[0].commands)
        executor.execute_once(now)
        if len(robots[0].commands) == before:
            return
        commanded = np.concatenate([np.asarray(r.commands[-1][:3]) for r in robots])
        frame = int(np.clip(lo + round(now * fps), lo, hi - 1))
        gt_vec = dataset[frame]["action"].numpy().astype(np.float64)
        gt_spline = _spline_from_matrix(gt_vec.reshape(rows, channels), degree)
        g_lo, g_hi = _domain(gt_spline, degree)
        demo = _positions(np.asarray(gt_spline(np.clip(0.0, g_lo, g_hi))), layouts)
        remaining = executor.remaining_s(now)
        rows_out.append(
            {
                "t": round(now, 4),
                "frame": frame,
                "plan": plan_index,
                "remaining_s": None if remaining is None else round(remaining, 4),
                "lag_m": float(np.linalg.norm(commanded - demo)),
                **{f"cmd_{i}": float(v) for i, v in enumerate(commanded)},
                **{f"demo_{i}": float(v) for i, v in enumerate(demo)},
            }
        )

    while now < duration:
        if not installed or executor.replan_needed(now):
            frame = int(np.clip(lo + round(now * fps), lo, hi - 1))
            pred_vec, latency = inf.predict(frame, lo)
            # Faithful to production: the old plan keeps being commanded while
            # inference runs, so advance the virtual clock through that window.
            deadline = now + latency
            while installed and now < deadline and now < duration:
                tick()
                now += period
            now = max(now, deadline)
            result = executor.install(pred_vec, observation_age_s=latency, now=now)
            plan_index += 1
            installed = True
            plan_events.append(
                {
                    "t": round(now, 4),
                    "plan": plan_index,
                    "infer_ms": round(latency * 1000, 1),
                    "start_time": round(result.start_time, 4),
                    "alignment_error": round(result.alignment_error, 6),
                    "align_searched": result.align_searched,
                    "align_capped": result.align_capped,
                    "align_endpoint_error": round(result.align_endpoint_error, 6),
                    "warning": result.warning,
                }
            )
        tick()
        now += period

    lag = np.array([r["lag_m"] for r in rows_out]) if rows_out else np.zeros(1)
    rem = [r["remaining_s"] for r in rows_out if r["remaining_s"] is not None]
    print(f"  control ticks      : {len(rows_out)}")
    print(
        f"  plans installed    : {plan_index}  ({plan_index / duration:.2f} replans/s)"
    )
    print(
        f"  mean seconds/plan  : {duration / max(plan_index, 1):.2f} s of open-loop motion"
    )
    print(
        f"  infer_ms           : median {np.median([e['infer_ms'] for e in plan_events]):.1f}"
    )
    print(
        f"  |commanded - demo| : median {np.median(lag):.4f} m  p95 {np.percentile(lag, 95):.4f} m  max {lag.max():.4f} m"
    )
    if rem:
        print(
            f"  remaining_s        : min {min(rem):.3f}  max {max(rem):.3f}  (sawtooth depth = plan length)"
        )
    warned = [e for e in plan_events if e["warning"]]
    print(f"  handoff warnings   : {len(warned)} / {plan_index}")
    searched = sum(1 for e in plan_events if e["align_searched"])
    capped = sum(1 for e in plan_events if e["align_capped"])
    print(f"  align searched     : {searched} / {plan_index}")
    print(f"  align capped @20%  : {capped} / {plan_index}")
    print(
        f"  alignment_error    : median {np.median([e['alignment_error'] for e in plan_events]):.6f}"
    )
    # ~0 means the residual is curve disagreement between samples, not phase error.
    hits = [e for e in plan_events if e["align_searched"]]
    if hits:
        gain = [e["align_endpoint_error"] - e["alignment_error"] for e in hits]
        print(
            f"  search gain        : median {np.median(gain):.6f}  "
            f"max {max(gain):.6f}  improved {sum(g > 1e-9 for g in gain)}/{len(hits)}"
        )
    _smoothness_report(rows_out, len(layouts), period)
    return rows_out, plan_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="checkpoint dir (…/checkpoints/last)"
    )
    parser.add_argument(
        "--dataset", required=True, help="B-spline LeRobot dataset root"
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--probes", type=int, default=12, help="Part A probe frames")
    parser.add_argument(
        "--samples", type=int, default=64, help="samples per curve in Part A"
    )
    parser.add_argument(
        "--replay-seconds", type=float, default=20.0, help="Part B length; 0 to skip"
    )
    parser.add_argument("--control-hz", type=float, default=200.0)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--predict-before-end-s", type=float, default=0.06)
    parser.add_argument(
        "--handoff-blend-s",
        type=float,
        default=0.15,
        help="handoff offset-decay window; 0 reproduces the old stepping handoff",
    )
    parser.add_argument("--handoff-max-accel", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", default="", help="prefix for CSV/JSON output (optional)"
    )
    args = parser.parse_args()

    policy, pre, post = _default_policy_loader(args.checkpoint, args.device)
    config = policy.config
    feature_names = list(config.action_feature_names)
    degree = int(config.spline_degree)
    _, layouts = _parse_layout(feature_names)
    rows = int(config.horizon)
    channels = len(feature_names) // rows
    args.knot_rate = float(config.knot_rate_hz)

    print(f"checkpoint     : {args.checkpoint}")
    print(f"action layout  : {rows} rows x {channels} channels = {len(feature_names)}")
    print(f"degree         : {degree}   knot_rate_hz: {args.knot_rate}")
    print(f"arms           : {[a.side for a in layouts]}")

    dataset = _load_dataset(args.dataset)
    inf = _Inferencer(policy, pre, post, dataset, args.device)

    with torch.inference_mode():
        records = part_a(args, inf, dataset, layouts, degree, rows, channels)
        replay, events = ([], [])
        if args.replay_seconds > 0:
            replay, events = part_b(
                args, inf, dataset, layouts, degree, rows, channels, feature_names
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{out}_part_a.json").write_text(json.dumps(records, indent=1))
        print(f"\nwrote {out}_part_a.json")
        if replay:
            with open(f"{out}_replay.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(replay[0]))
                writer.writeheader()
                writer.writerows(replay)
            Path(f"{out}_plans.json").write_text(json.dumps(events, indent=1))
            print(f"wrote {out}_replay.csv ({len(replay)} rows)")
            print(f"wrote {out}_plans.json ({len(events)} plans)")


if __name__ == "__main__":
    main()
