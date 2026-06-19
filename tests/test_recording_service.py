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

import numpy as np

from flexivtrainer.data.recording_service import RecordingService


def test_grab_images_converts_bgr_capture_to_rgb() -> None:
    # Cameras capture a red pixel as BGR [0, 0, 255]; recorded frames must be
    # RGB [255, 0, 0] so LeRobot playback shows red (not blue/purple).
    bgr_red = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr_red[0, 0] = [0, 0, 255]

    service = RecordingService.__new__(RecordingService)
    service._cameras = SimpleNamespace(
        capture_frame=lambda name, **kwargs: {"image": bgr_red}
    )

    images = service._grab_images(["ego"], require_all=True, attempts=1)

    assert images["ego"][0, 0].tolist() == [255, 0, 0]
    assert images["ego"].flags["C_CONTIGUOUS"]
