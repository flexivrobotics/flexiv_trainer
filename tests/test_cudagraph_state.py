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

"""Per-thread CUDA-graph-tree seeding.

These exercise torch's real thread-local bookkeeping, not a mock, and need no
GPU: get_container/reset_cudagraph_trees only touch Python-level state.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest

torch = pytest.importorskip("torch")
cudagraph_trees = pytest.importorskip("torch._inductor.cudagraph_trees")

from flexivtrainer.rollout._cudagraph_state import (  # noqa: E402
    seed_thread_local_state,
    teardown_rollout_gpu_state,
)


def _in_fresh_thread(fn: Callable[[], Any]) -> BaseException | None:
    """Return whatever fn raised on a brand-new thread, as a rollout would."""
    captured: list[BaseException] = []

    def target() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - the assertion under test
            captured.append(exc)

    thread = threading.Thread(target=target, name="planner-probe")
    thread.start()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    return captured[0] if captured else None


def test_unseeded_thread_trips_a_message_less_assert() -> None:
    # The failure this module exists to prevent: an empty message is why it
    # surfaced as a bare "AssertionError" with no actionable detail.
    exc = _in_fresh_thread(lambda: cudagraph_trees.get_container(0))
    assert isinstance(exc, AssertionError)
    assert str(exc) == ""


def test_seeding_survives_sequential_planner_threads() -> None:
    def rollout() -> None:
        seed_thread_local_state()
        cudagraph_trees.get_container(0)
        teardown_rollout_gpu_state("cpu", cudagraphs_seeded=True)

    for index in range(3):
        assert _in_fresh_thread(rollout) is None, f"rollout {index} failed"


def test_seeding_stays_local_to_the_thread_that_seeded() -> None:
    assert _in_fresh_thread(
        lambda: (seed_thread_local_state(), cudagraph_trees.get_container(0))
    ) is None
    # A shared-dict fix would make this sibling pass too, hiding the isolation.
    assert isinstance(
        _in_fresh_thread(lambda: cudagraph_trees.get_container(0)), AssertionError
    )


def test_teardown_never_raises_on_an_unseeded_thread() -> None:
    assert _in_fresh_thread(
        lambda: teardown_rollout_gpu_state("cpu", cudagraphs_seeded=False)
    ) is None
