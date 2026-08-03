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

"""Per-thread isolation for torch's CUDA-graph-tree state.

``torch._inductor.cudagraph_trees`` keeps its per-device tree managers in a
module-level ``threading.local()`` that is populated only on the thread which
executes the module body. Its C++ TLS fallback is propagated to threads the
autograd engine spawns, never to a plain ``threading.Thread``, so ``get_obj()``
trips a bare, message-less ``assert`` on any other thread.

Every rollout runs on a fresh planner thread, so the first rollout in a process
works (its planner thread imports the module) and every later one fails. Two
call paths reach that assert:

* the first cudagraph-captured forward, via ``get_container()``
* ``torch.compiler.reset()``, which walks the process-global ``cached_backends``
  registry and calls ``reset_cudagraph_trees()`` on the *calling* thread

Seeding fresh containers per planner thread fixes both and gives each rollout a
private ``CUDAGraphTreeManager`` and memory pool, so checkpoints cannot share
compiled state.
"""

from __future__ import annotations

import gc
import threading
from collections import defaultdict
from typing import Any

from flexivtrainer.observability import describe_traceback, warn

_REQUIRED_ATTRS = ("local", "reset_cudagraph_trees")


class CudagraphTreesUnavailable(RuntimeError):
    """torch's cudagraph-tree internals no longer match what this module seeds."""


def _import_cudagraph_trees() -> Any:
    from torch._inductor import cudagraph_trees  # noqa: PLC0415

    missing = [name for name in _REQUIRED_ATTRS if not hasattr(cudagraph_trees, name)]
    if missing:
        raise CudagraphTreesUnavailable(
            f"torch._inductor.cudagraph_trees is missing {missing}; "
            "flexivtrainer's per-thread CUDA-graph seeding needs updating for "
            "this torch version"
        )
    if not isinstance(cudagraph_trees.local, threading.local):
        raise CudagraphTreesUnavailable(
            "torch._inductor.cudagraph_trees.local is no longer a threading.local; "
            "per-thread CUDA-graph seeding would silently stop being per-thread"
        )
    return cudagraph_trees


def seed_thread_local_state() -> None:
    """Give the calling thread private cudagraph-tree bookkeeping.

    Call once at the top of a planner thread, before anything compiles or calls
    ``torch.compiler.reset()``. Raises rather than degrading: without it, a
    compiled rollout fails with an unactionable bare AssertionError.
    """
    cudagraph_trees = _import_cudagraph_trees()
    cudagraph_trees.local.tree_manager_containers = {}
    cudagraph_trees.local.tree_manager_locks = defaultdict(threading.Lock)


def teardown_rollout_gpu_state(device: str, *, cudagraphs_seeded: bool) -> None:
    """Release this thread's cudagraph trees and cached GPU memory.

    Must run on the thread that seeded, since torch resolves the containers
    per-thread. Best-effort: never raises, so cleanup cannot mask the error that
    ended the rollout or skip the robot release that follows it.
    """
    try:
        import torch  # noqa: PLC0415
    except Exception:  # pragma: no cover - torch optional
        return

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    try:
        gc.collect()
        if is_cuda:
            torch.cuda.synchronize()
        if cudagraphs_seeded:
            _import_cudagraph_trees().reset_cudagraph_trees()
        if is_cuda:
            torch.cuda.empty_cache()
    except Exception as exc:
        warn("Failed to release rollout GPU state", describe_traceback(exc))
