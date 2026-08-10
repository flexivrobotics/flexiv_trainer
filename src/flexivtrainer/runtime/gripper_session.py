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

"""Process-lifetime coordination for physical gripper initialization."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class GripperInitializationState(StrEnum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class GripperIdentity:
    """Stable identity for one physical follower gripper in this process."""

    follower_serial: str
    gripper_model: str

    def __post_init__(self) -> None:
        if not self.follower_serial.strip():
            raise ValueError("Follower serial is required for gripper identity")
        if not self.gripper_model.strip():
            raise ValueError("Gripper model is required for gripper identity")

    def describe(self) -> str:
        return f"{self.follower_serial}/{self.gripper_model}"


@dataclass(frozen=True, slots=True)
class GripperInitializationClaim:
    """Atomic decision about which grippers need mechanical initialization."""

    initialize: tuple[GripperIdentity, ...]
    reused: tuple[GripperIdentity, ...]


class GripperInitializationBusyError(RuntimeError):
    pass


class GripperInitializationRegistry:
    """Thread-safe, process-local state for non-queryable RDK initialization.

    The RDK does not expose a reliable initialized bit. This registry therefore
    records only initialization performed during the current backend process;
    it deliberately owns no RDK objects or robot connections.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[GripperIdentity, GripperInitializationState] = {}

    @staticmethod
    def _unique(
        identities: Iterable[GripperIdentity],
    ) -> tuple[GripperIdentity, ...]:
        return tuple(dict.fromkeys(identities))

    def claim(
        self,
        identities: Iterable[GripperIdentity],
        *,
        force: bool = False,
    ) -> GripperInitializationClaim:
        """Reserve every required initialization as one atomic batch."""

        requested = self._unique(identities)
        with self._lock:
            busy = [
                identity
                for identity in requested
                if self._states.get(identity)
                is GripperInitializationState.INITIALIZING
            ]
            if busy:
                names = ", ".join(identity.describe() for identity in busy)
                raise GripperInitializationBusyError(
                    f"Gripper initialization is already running: {names}"
                )

            initialize = tuple(
                identity
                for identity in requested
                if force
                or self._states.get(identity)
                is not GripperInitializationState.READY
            )
            initialize_set = set(initialize)
            reused = tuple(
                identity for identity in requested if identity not in initialize_set
            )
            for identity in initialize:
                self._states[identity] = GripperInitializationState.INITIALIZING
            return GripperInitializationClaim(initialize=initialize, reused=reused)

    def complete(self, identities: Iterable[GripperIdentity]) -> None:
        with self._lock:
            for identity in self._unique(identities):
                if (
                    self._states.get(identity)
                    is GripperInitializationState.INITIALIZING
                ):
                    self._states[identity] = GripperInitializationState.READY

    def fail(self, identities: Iterable[GripperIdentity]) -> None:
        """Release failed claims so a later explicit attempt can retry."""

        with self._lock:
            for identity in self._unique(identities):
                if (
                    self._states.get(identity)
                    is GripperInitializationState.INITIALIZING
                ):
                    self._states.pop(identity, None)

    def state(self, identity: GripperIdentity) -> GripperInitializationState:
        with self._lock:
            return self._states.get(
                identity, GripperInitializationState.UNINITIALIZED
            )
