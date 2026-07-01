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

Each policy family owns its own knobs here rather than scattering them across the
training and rollout configs. Re-export each family's config so callers import the
``policies`` namespace once instead of reaching into submodules.
"""

from flexivtrainer.policies.diffusion import DiffusionPolicyConfig

__all__ = ["DiffusionPolicyConfig"]
