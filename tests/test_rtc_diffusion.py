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

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flexivtrainer.policies import rtc_diffusion  # noqa: E402


def test_rtc_generate_actions_stashes_full_horizon(monkeypatch) -> None:
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8
    action_dim = 19

    model = SimpleNamespace(
        config=SimpleNamespace(
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        ),
        _prepare_global_conditioning=lambda batch: None,
    )
    full = torch.arange(
        horizon * action_dim, dtype=torch.float32
    ).reshape(1, horizon, action_dim)
    monkeypatch.setattr(
        rtc_diffusion, "rtc_conditional_sample", lambda *a, **k: full
    )

    batch = {"observation.state": torch.zeros(1, n_obs_steps, action_dim)}
    executed = rtc_diffusion.rtc_generate_actions(model, batch)

    # The stash carries the full normalized horizon (index 0 == first step).
    assert model._rtc_last_full_horizon.shape == (1, horizon, action_dim)
    assert torch.equal(model._rtc_last_full_horizon, full)
    # The executed slice is [n_obs_steps-1 : +n_action_steps] as before.
    start = n_obs_steps - 1
    assert executed.shape == (1, n_action_steps, action_dim)
    assert torch.equal(executed, full[:, start : start + n_action_steps])
