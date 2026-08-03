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

"""Offline probe: compile and run several ACT checkpoints back to back.

No robot and no HTTP server are involved. Each checkpoint runs on its own fresh
thread, exactly as ``WaypointRunner`` does, calling the same production helpers
in the same order. That reproduces the sequential-rollout compile failure (or
proves it fixed) in seconds instead of needing two live rollouts.

Pass ``--checkpoint`` more than once, and repeat one of them to show that N
sequential rollouts work rather than only two. Action widths may differ; they
are irrelevant to the failure, which is about thread identity.

Usage:

    .venv/bin/python scripts/probe_act_cudagraph_reuse.py \
        --checkpoint .local/training/<run-38d>/checkpoints/last \
        --checkpoint .local/training/<run-26d>/checkpoints/last \
        --checkpoint .local/training/<run-38d>/checkpoints/last \
        --device cuda:0 --ticks 200
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flexivtrainer.config import AppSettings
from flexivtrainer.policies import act as act_policy
from flexivtrainer.rollout import _cudagraph_state, observations
from flexivtrainer.rollout.checkpoint import (
    _default_policy_loader,
    checkpoint_action_output_dim,
    resolve_checkpoint_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="checkpoint dir; repeat to run several in sequence",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=("reduce-overhead", "default"),
    )
    return parser.parse_args()


def _synthetic_observation(policy: Any) -> dict[str, Any]:
    """Zero-filled inputs matching the checkpoint's own feature shapes.

    numpy, not tensors: LeRobot's prepare_observation_for_inference calls
    torch.from_numpy on every entry.
    """
    observation: dict[str, Any] = {}
    for key, feature in policy.config.input_features.items():
        shape = tuple(feature.shape)
        if len(shape) == 3:
            # Config stores CHW; prepare_observation_for_inference permutes as
            # if the caller handed it HWC, which is what the runner does.
            channels, height, width = shape
            shape = (height, width, channels)
        observation[key] = np.zeros(shape, dtype=np.float32)
    return observation


def _rollout(args: argparse.Namespace, checkpoint: str, index: int) -> bool:
    label = f"rollout #{index} {checkpoint}"
    print(f"\n=== {label}")
    print(f"    thread={threading.current_thread().name}")
    seeded = False
    policy = None
    try:
        if args.compile_mode == "reduce-overhead" and args.device.startswith("cuda"):
            _cudagraph_state.seed_thread_local_state()
            seeded = True
        resolved = str(
            resolve_checkpoint_path(
                str(Path(checkpoint).expanduser().resolve()),
                AppSettings().storage.root,
            )
        )
        action_dim = checkpoint_action_output_dim(resolved)
        policy, preprocessor, postprocessor = _default_policy_loader(
            resolved, args.device
        )
        act_policy.compile_model(policy, mode=args.compile_mode)
        policy.reset()
        print(f"    compiled: action_dim={action_dim} mode={args.compile_mode}")

        observation = _synthetic_observation(policy)
        latencies_ms: list[float] = []
        for _ in range(args.ticks):
            if seeded:
                torch.compiler.cudagraph_mark_step_begin()
            started = time.perf_counter()
            observations._predict_action_chunk(
                observation,
                policy,
                args.device,
                preprocessor,
                postprocessor,
                force_refresh=True,
            )
            observations._cuda_sync(args.device)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

        warm = latencies_ms[1:] or latencies_ms
        ordered = sorted(warm)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        print(
            f"    OK  ticks={args.ticks} "
            f"mean={statistics.mean(warm):.2f}ms p95={p95:.2f}ms "
            f"first={latencies_ms[0]:.1f}ms (compile+capture)"
        )
        print(f"    budget at 60Hz = 16.67ms -> {'PASS' if p95 < 16.67 else 'OVER'}")
        return True
    except Exception:
        print(f"    FAILED {label}")
        traceback.print_exc()
        return False
    finally:
        policy = None
        _cudagraph_state.teardown_rollout_gpu_state(
            args.device, cudagraphs_seeded=seeded
        )
        if args.device.startswith("cuda") and torch.cuda.is_available():
            print(
                f"    after teardown: allocated="
                f"{torch.cuda.memory_allocated() // 1024**2}MiB "
                f"reserved={torch.cuda.memory_reserved() // 1024**2}MiB"
            )


def main() -> None:
    args = _parse_args()
    print(f"torch={torch.__version__} device={args.device}")
    failures: list[int] = []
    for index, checkpoint in enumerate(args.checkpoint):
        outcome: list[bool] = []
        thread = threading.Thread(
            target=lambda: outcome.append(_rollout(args, checkpoint, index)),
            name=f"rollout-policy-planner-{index}",
        )
        thread.start()
        thread.join()
        if not outcome or not outcome[0]:
            failures.append(index)

    total = len(args.checkpoint)
    print(f"\n{total - len(failures)}/{total} rollouts ok")
    if failures:
        raise SystemExit(f"failed rollouts: {failures}")


if __name__ == "__main__":
    main()
