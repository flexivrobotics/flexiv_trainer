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

import torch

from flexivtrainer.jobs.train_policy import resolve_training_device


def test_resolve_device_passes_explicit_through() -> None:
    # An explicit device is honoured as-is, regardless of what's available.
    assert resolve_training_device("cpu") == "cpu"
    assert resolve_training_device("cuda") == "cuda"
    assert resolve_training_device("mps") == "mps"


def test_resolve_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_training_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_training_device("auto") == "cpu"
