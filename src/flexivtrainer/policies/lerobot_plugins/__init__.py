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

"""Internal LeRobot policy plugins."""

from .configuration_bspline_diffusion import BSplineDiffusionConfig
from .dataset_patches import LOAD_DEPTH_ENV, apply_patches
from .modeling_bspline_diffusion import BSplineDiffusionPolicy
from .processor_bspline_diffusion import (
    make_bspline_diffusion_pre_post_processors,
)

# LeRobot imports this package before it builds the dataset, the only window in
# which these patches can take effect on the training subprocess.
apply_patches()

__all__ = [
    "LOAD_DEPTH_ENV",
    "BSplineDiffusionConfig",
    "BSplineDiffusionPolicy",
    "apply_patches",
    "make_bspline_diffusion_pre_post_processors",
]
