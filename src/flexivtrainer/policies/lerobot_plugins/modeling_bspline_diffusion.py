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

"""LeRobot B-spline diffusion policy."""

from __future__ import annotations

import threading
from collections import deque
from copy import deepcopy

import torch
from lerobot.configs.types import PolicyFeature
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor

from .configuration_bspline_diffusion import BSplineDiffusionConfig


class BSplineDiffusionPolicy(PreTrainedPolicy):
    config_class = BSplineDiffusionConfig
    name = "bspline_diffusion"

    def __init__(self, config: BSplineDiffusionConfig, **kwargs) -> None:
        del kwargs
        super().__init__(config)
        config.validate_features()
        self.parameter_rows, self.parameter_channels = config.logical_action_shape()

        model_config = deepcopy(config)
        action_feature = config.action_feature
        assert action_feature is not None
        model_config.output_features = {
            ACTION: PolicyFeature(
                type=action_feature.type,
                shape=(self.parameter_channels,),
            )
        }
        self.diffusion = DiffusionModel(model_config)
        self._queue_lock = threading.Lock()
        self.reset()

    def get_optim_params(self):
        return self.diffusion.parameters()

    def reset(self) -> None:
        with self._queue_lock:
            self._queues = {
                OBS_STATE: deque(maxlen=self.config.n_obs_steps),
                ACTION: deque(maxlen=1),
            }
            if self.config.image_features:
                self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
            if self.config.env_state_feature:
                self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    def _prepare_observations(
        self,
        batch: dict[str, Tensor],
        *,
        ensure_time_dimension: bool,
    ) -> dict[str, Tensor]:
        batch = dict(batch)
        if self.config.image_features:
            for key in self.config.image_features:
                if (
                    ensure_time_dimension
                    and self.config.n_obs_steps == 1
                    and batch[key].ndim == 4
                ):
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features],
                dim=-4,
            )
        return batch

    def _reshape_training_action(
        self, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        batch = dict(batch)
        action = batch[ACTION]
        expected = self.parameter_rows * self.parameter_channels
        if action.ndim != 3 or action.shape[1:] != (1, expected):
            raise ValueError(
                "Expected B-spline action shape "
                f"[batch, 1, {expected}], got {tuple(action.shape)}"
            )
        batch[ACTION] = action.reshape(
            action.shape[0],
            self.parameter_rows,
            self.parameter_channels,
        )
        if "action_is_pad" in batch:
            padding = batch["action_is_pad"]
            if padding.ndim != 2 or padding.shape[1] != 1:
                raise ValueError(
                    "Expected B-spline action padding shape [batch, 1], got "
                    f"{tuple(padding.shape)}"
                )
            batch["action_is_pad"] = padding.expand(-1, self.parameter_rows)
        return batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self._prepare_observations(
            batch,
            ensure_time_dimension=True,
        )
        batch = self._reshape_training_action(batch)
        return self.diffusion.compute_loss(batch), None

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor] | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        del batch
        with self._queue_lock:
            observations = {
                key: torch.stack(list(queue), dim=1)
                for key, queue in self._queues.items()
                if key != ACTION and queue
            }

        if OBS_STATE not in observations:
            raise RuntimeError("No observations are queued for B-spline inference")
        global_condition = self.diffusion._prepare_global_conditioning(observations)
        parameters = self.diffusion.conditional_sample(
            observations[OBS_STATE].shape[0],
            global_cond=global_condition,
            noise=noise,
        )
        return parameters.reshape(parameters.shape[0], 1, -1)

    def enqueue_observation(self, batch: dict[str, Tensor]) -> None:
        batch = dict(batch)
        batch.pop(ACTION, None)
        batch = self._prepare_observations(
            batch,
            ensure_time_dimension=False,
        )
        with self._queue_lock:
            self._queues = populate_queues(self._queues, batch)

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> Tensor:
        self.enqueue_observation(batch)
        with self._queue_lock:
            needs_inference = not self._queues[ACTION]
        if needs_inference:
            actions = self.predict_action_chunk(noise=noise)
            with self._queue_lock:
                self._queues[ACTION].extend(actions.transpose(0, 1))
        with self._queue_lock:
            return self._queues[ACTION].popleft()
