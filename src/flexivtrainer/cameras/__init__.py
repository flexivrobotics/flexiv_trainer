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

"""Depth-camera support: a vendor-agnostic service over per-SDK backends."""

from flexivtrainer.cameras.backends import (
    CameraBackend,
    CameraStream,
    DeviceInfo,
    FramePair,
    OrbbecBackend,
    RealSenseBackend,
)
from flexivtrainer.cameras.service import CameraRuntime, CameraService

__all__ = [
    "CameraBackend",
    "CameraRuntime",
    "CameraService",
    "CameraStream",
    "DeviceInfo",
    "FramePair",
    "OrbbecBackend",
    "RealSenseBackend",
]
