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

import pytest

from flexivtrainer.runtime.gripper_session import (
    GripperIdentity,
    GripperInitializationBusyError,
    GripperInitializationRegistry,
    GripperInitializationState,
)


def test_claim_complete_reuse_and_force_reinitialize() -> None:
    registry = GripperInitializationRegistry()
    left = GripperIdentity("LEFT", "Flexiv-GN01")
    right = GripperIdentity("RIGHT", "Flexiv-GN01")

    first = registry.claim([left, right])
    assert first.initialize == (left, right)
    assert first.reused == ()
    assert registry.state(left) is GripperInitializationState.INITIALIZING

    registry.complete(first.initialize)
    second = registry.claim([left, right])
    assert second.initialize == ()
    assert second.reused == (left, right)

    forced = registry.claim([left, right], force=True)
    assert forced.initialize == (left, right)
    assert forced.reused == ()
    registry.complete(forced.initialize)
    assert registry.state(right) is GripperInitializationState.READY


def test_busy_claim_is_atomic_and_failed_claim_can_retry() -> None:
    registry = GripperInitializationRegistry()
    left = GripperIdentity("LEFT", "Flexiv-GN01")
    right = GripperIdentity("RIGHT", "Flexiv-GN01")

    registry.claim([left])
    with pytest.raises(GripperInitializationBusyError):
        registry.claim([left, right])
    assert registry.state(right) is GripperInitializationState.UNINITIALIZED

    registry.fail([left])
    retry = registry.claim([left, right])
    assert retry.initialize == (left, right)
