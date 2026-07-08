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

"""Multi-Task DiT policy configuration: training knobs + rollout overrides.

Language-conditioned Diffusion Transformer (CLIP vision + frozen CLIP text
encoder conditioning a DiT). ``TrainingConfig`` defaults track this repo's
10 Hz data (~1 s horizon) rather than LeRobot's 30 Hz defaults. The denoising
sampler is baked into the checkpoint at train time; ``RolloutConfig`` retunes
it at load without retraining (``apply_rollout_overrides``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from flexivtrainer.observability import describe_exception, warn
from flexivtrainer.policies._shared import SharedRolloutConfig, SharedTrainingConfig


class TrainingConfig(SharedTrainingConfig):
    horizon: int = Field(10, ge=2, le=64, description="action-chunk length")
    n_obs_steps: int = Field(2, ge=1, le=8, description="observation frames fed in")
    n_action_steps: int = Field(
        8, ge=1, le=48, description="<= horizon - n_obs_steps + 1"
    )
    objective: Literal["diffusion", "flow_matching"] = Field(
        "diffusion", description="diffusion or flow_matching"
    )
    noise_scheduler_type: Literal["DDPM", "DDIM"] = Field(
        "DDIM", description="DDIM or DDPM (diffusion objective)"
    )
    num_inference_steps: int = Field(
        10, ge=1, le=100, description="baked default; 5-16"
    )
    num_layers: int = Field(6, ge=1, le=24, description="transformer layers")
    hidden_dim: int = Field(
        512, ge=64, le=2048, description="hidden dim; divisible by num_heads"
    )
    num_heads: int = Field(8, ge=1, le=32, description="attention heads")
    dropout: float = Field(0.1, ge=0, le=1, description="dropout rate")
    vision_encoder_name: str = Field(
        "openai/clip-vit-base-patch16",
        description="CLIP model (must contain 'clip')",
    )
    text_encoder_name: str = Field(
        "openai/clip-vit-base-patch16",
        description="CLIP model (must contain 'clip')",
    )
    # Bare tuples: tuple|None would degrade to "str" in training_field_schema.
    image_resize_shape: tuple[int, int] = Field(
        (240, 320), description="H,W; resize before crop"
    )
    image_crop_shape: tuple[int, int] = Field(
        (216, 288), description="H,W; <= resize"
    )
    optimizer_lr: float = Field(2e-5, gt=0, le=1.0, description="learning rate")
    vision_encoder_lr_multiplier: float = Field(
        0.1, gt=0, le=1.0, description="LR multiplier for the CLIP vision encoder"
    )


class RolloutConfig(SharedRolloutConfig):
    """Multi-Task DiT rollout overrides applied at load."""

    # "" = leave the checkpoint's own sampler/steps untouched.
    noise_scheduler_type: Literal["", "DDPM", "DDIM"] = "DDIM"
    num_denoise_steps: int = Field(default=10, ge=1, le=1000)


def apply_rollout_overrides(policy: Any, rollout_cfg: RolloutConfig) -> bool:
    """Apply DiT-specific rollout overrides to a freshly loaded policy."""
    scheduler = getattr(rollout_cfg, "noise_scheduler_type", "")
    if not scheduler:
        return False
    # Flow-matching has no noise scheduler to swap.
    if getattr(getattr(policy, "config", None), "objective", "") != "diffusion":
        return False
    objective = getattr(policy, "objective", None)
    existing = getattr(objective, "noise_scheduler", None)
    if objective is None or existing is None:
        return False
    steps = getattr(rollout_cfg, "num_denoise_steps", 0)
    try:
        from diffusers import DDIMScheduler, DDPMScheduler  # noqa: PLC0415

        # Reuse the trained schedule so only the sampler family changes; pass
        # the same kwargs DiT's DiffusionObjective uses when building one.
        cfg = existing.config
        kwargs = dict(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range,
            prediction_type=cfg.prediction_type,
        )
        builder = DDIMScheduler if scheduler == "DDIM" else DDPMScheduler
        objective.noise_scheduler = builder(**kwargs)
        objective.num_inference_steps = steps
    except Exception as exc:
        warn("Failed to override DiT scheduler", describe_exception(exc))
        return False
    return True
