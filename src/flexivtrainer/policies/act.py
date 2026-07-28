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

"""ACT policy configuration: training knobs + rollout overrides.

``TrainingConfig`` holds the train-time knobs, emitted as ``lerobot-train``
flags and rendered as the Web UI form (defaults track LeRobot's ACTConfig).

``RolloutConfig`` holds the inference-time overrides applied when a checkpoint
is loaded for rollout (``apply_rollout_overrides``), letting temporal ensembling
be turned off without retraining.
"""

from __future__ import annotations

from typing import Any

from numpy import true_divide
from pydantic import Field

from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.policies._shared import SharedRolloutConfig, SharedTrainingConfig


class TrainingConfig(SharedTrainingConfig):
    chunk_size: int = Field(100, ge=1, le=1000, description="action-chunk length")
    n_action_steps: int = Field(
        100, ge=1, le=1000,
        description="actions executed per chunk; must be <= chunk_size",
    )
    n_encoder_layers: int = Field(
        4, ge=1, le=12, description="transformer encoder layers (ACT paper: 4)"
    )
    n_decoder_layers: int = Field(
        7, ge=1, le=12, description="transformer decoder layers (ACT paper: 7)"
    )
    dim_model: int = Field(512, ge=64, le=2048, description="transformer width")
    optimizer_lr: float = Field(1e-5, gt=0, le=1.0, description="learning rate")
    # Checkbox-gated in the Web UI; plain float (not float | None) so the schema
    # builder types it "float". Last field so its checkbox has no cell to overlap.
    temporal_ensemble_coeff: float = Field(
        0.1,
        gt=0,
        le=1.0,
        description="temporal-ensemble weight; forces n_action_steps=1 when enabled",
    )


class RolloutConfig(SharedRolloutConfig):
    """ACT rollout overrides applied at load."""

    # Ensembling forces a forward pass per step and ignores n_action_steps, so no
    # chunk is ever committed. Disabling trades smoothing for chunked execution.
    disable_temporal_ensemble: bool = False
    # ACT's forward is kernel-launch-bound (~1700 dispatches), so CUDA-graph
    # replay roughly halves it. Measured 7.99ms -> 4.00ms on an RTX 5090.
    compile_model: bool = True


def _disable_temporal_ensemble(policy: Any) -> bool:
    config = getattr(policy, "config", None)
    if config is None or getattr(config, "temporal_ensemble_coeff", None) is None:
        return False
    try:
        # Must clear before n_action_steps is raised: LeRobot rejects the two
        # together, and select_action only reads its queue when this is None.
        config.temporal_ensemble_coeff = None
        policy.temporal_ensembler = None
    except Exception as exc:
        warn("Failed to disable ACT temporal ensembling", describe_exception(exc))
        return False
    return True


def _compile_model(policy: Any) -> bool:
    """Compile the tensor core only; select_action mutates a Python deque."""
    import torch  # noqa: PLC0415

    model = getattr(policy, "model", None)
    if model is None:
        return False
    try:
        policy.model = torch.compile(model, mode="reduce-overhead")
    except Exception as exc:
        warn("Failed to compile ACT model; using eager", describe_exception(exc))
        return False
    return True


def apply_rollout_overrides(policy: Any, rollout_cfg: RolloutConfig) -> bool:
    """Apply ACT-specific rollout overrides to a freshly loaded policy."""
    applied = False
    if getattr(rollout_cfg, "disable_temporal_ensemble", False):
        applied |= _disable_temporal_ensemble(policy)
    if getattr(rollout_cfg, "compile_model", False):
        applied |= _compile_model(policy)
    return applied
