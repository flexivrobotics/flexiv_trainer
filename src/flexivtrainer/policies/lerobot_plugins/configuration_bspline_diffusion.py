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

"""LeRobot configuration for B-spline diffusion."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

_FEATURE_PATTERN = re.compile(r"^bspline\.row_(\d+)\.(.+)$")


@PreTrainedConfig.register_subclass("bspline_diffusion")
@dataclass
class BSplineDiffusionConfig(DiffusionConfig):
    horizon: int = 16
    n_action_steps: int = 1
    drop_n_last_frames: int = 0
    do_mask_loss_for_padding: bool = False
    action_feature_names: list[str] | None = None
    spline_degree: int = 3
    knot_rate_hz: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps != 1:
            raise ValueError("B-spline diffusion requires n_action_steps=1")
        if self.drop_n_last_frames != 0:
            raise ValueError("B-spline diffusion requires drop_n_last_frames=0")
        if self.do_mask_loss_for_padding:
            raise ValueError("B-spline diffusion does not support padded-loss masking")
        if (
            isinstance(self.spline_degree, bool)
            or not isinstance(self.spline_degree, int)
            or self.spline_degree < 1
        ):
            raise ValueError("B-spline spline_degree must be a positive integer")
        if self.knot_rate_hz is not None and (
            isinstance(self.knot_rate_hz, bool)
            or not math.isfinite(self.knot_rate_hz)
            or self.knot_rate_hz <= 0
        ):
            raise ValueError("B-spline knot_rate_hz must be positive when set")

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]

    def logical_action_shape(self) -> tuple[int, int]:
        feature = self.action_feature
        if feature is None or len(feature.shape) != 1:
            raise ValueError("B-spline action must be a flat one-dimensional feature")
        if not self.action_feature_names:
            raise ValueError("B-spline action feature names are required")
        if len(self.action_feature_names) != feature.shape[0]:
            raise ValueError(
                "B-spline action feature names do not match the flat action width"
            )

        rows: list[list[str]] = []
        for name in self.action_feature_names:
            match = _FEATURE_PATTERN.fullmatch(name)
            if match is None:
                raise ValueError(f"Malformed B-spline action feature name: {name!r}")
            row = int(match.group(1))
            if row == len(rows):
                rows.append([])
            if row != len(rows) - 1:
                raise ValueError(
                    "B-spline action rows must be contiguous and row-major"
                )
            rows[row].append(match.group(2))

        if len(rows) != self.horizon:
            raise ValueError(
                f"B-spline action has {len(rows)} rows, expected horizon={self.horizon}"
            )
        channels = rows[0]
        if len(channels) < 2 or channels[0] != "knot":
            raise ValueError("Each B-spline row must start with a knot channel")
        if len(set(channels)) != len(channels):
            raise ValueError("B-spline channels must be unique within each row")
        if any(row != channels for row in rows[1:]):
            raise ValueError("B-spline action rows must have identical channel layouts")
        if feature.shape[0] % self.horizon:
            raise ValueError(
                "Flat B-spline action width is not divisible by the horizon"
            )
        return self.horizon, len(channels)

    def validate_features(self) -> None:
        super().validate_features()
        self.logical_action_shape()
