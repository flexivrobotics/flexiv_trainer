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

from collections import deque
from types import SimpleNamespace

import torch

from flexivtrainer.rollout.observations import _predict_action_chunk


def test_fresh_inference_returns_and_postprocesses_the_complete_chunk(
    monkeypatch,
) -> None:
    queue = deque()
    policy = SimpleNamespace(_action_queue=queue)
    postprocessed: list[torch.Tensor] = []

    def postprocessor(actions):
        postprocessed.append(actions)
        return actions + 10

    def predict_action(*args, **kwargs):
        queue.extend([
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 4.0]]),
        ])
        return torch.tensor([[10.0, 11.0]])

    monkeypatch.setattr(
        "lerobot.utils.control_utils.predict_action", predict_action
    )

    chunk, fresh = _predict_action_chunk(
        {}, policy, "cpu", lambda value: value, postprocessor
    )

    assert fresh is True
    assert torch.equal(
        chunk,
        torch.tensor([
            [10.0, 11.0],
            [11.0, 12.0],
            [13.0, 14.0],
        ]),
    )
    assert len(postprocessed) == 1


def test_cached_action_skips_reprocessing_the_pending_tail(monkeypatch) -> None:
    queue = deque([
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
    ])
    policy = SimpleNamespace(_action_queue=queue)
    postprocessed: list[torch.Tensor] = []

    def postprocessor(actions):
        postprocessed.append(actions)
        return actions + 10

    def predict_action(*args, **kwargs):
        return postprocessor(queue.popleft())

    monkeypatch.setattr(
        "lerobot.utils.control_utils.predict_action", predict_action
    )

    chunk, fresh = _predict_action_chunk(
        {}, policy, "cpu", lambda value: value, postprocessor
    )

    assert fresh is False
    assert torch.equal(chunk, torch.tensor([[11.0, 12.0]]))
    assert len(queue) == 1
    assert len(postprocessed) == 1
