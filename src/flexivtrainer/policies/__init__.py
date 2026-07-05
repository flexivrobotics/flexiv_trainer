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

"""Per-policy configuration, one module per policy family (diffusion, act, ...).

Each policy family owns its own ``TrainingConfig`` + ``RolloutConfig`` in its
module rather than scattering them across the training and rollout configs.
Callers reach a family's config through ``policies.<family>`` and share the
family-agnostic rollout knobs via ``SharedRolloutConfig``. ``PolicyConfig`` is
the tree the app settings mount under ``policies``; the main config only
selects a policy, all family knobs live here.
"""

from pydantic import BaseModel, Field

from flexivtrainer.policies import diffusion
from flexivtrainer.policies._shared import SharedRolloutConfig

__all__ = ["PolicyConfig", "SharedRolloutConfig", "diffusion"]


class DiffusionConfig(BaseModel):
    """Training knobs (baked into checkpoint) + rollout knobs (applied at load)."""

    training: diffusion.TrainingConfig = Field(
        default_factory=diffusion.TrainingConfig
    )
    rollout: diffusion.RolloutConfig = Field(default_factory=diffusion.RolloutConfig)


class PolicyConfig(BaseModel):
    """Per-policy-family knobs; one sub-model per family."""

    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)

    def rollout_for(self, policy_type: str) -> SharedRolloutConfig:
        family = getattr(self, policy_type, None)
        rollout = getattr(family, "rollout", None)
        return rollout if rollout is not None else SharedRolloutConfig()
