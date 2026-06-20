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

"""End effector control during teleoperation.

Each arm pair can configure a leader *digital input* device (a trigger such as a
button wired to one of the leader robot's DI ports) and a follower end effector
(a digital output device or a gripper). While the pairs are engaged, this
controller polls every pair's leader DI and mirrors its state onto the follower:
the follower output/gripper is driven to its configured "activated" state while
the leader trigger is active, and to the opposite state otherwise.

The configuration is the per-side :class:`EndEffectorSideConfig` cached in
``robot_serials.json`` and keyed by arm side ("left_arm"/"right_arm"/
"single_arm"); the side order matches the teleop pair index.
"""

from __future__ import annotations

import threading
from typing import Any

from flexivtrainer.config import EndEffectorSideConfig
from flexivtrainer.observability import describe_exception, warn

try:
    import flexivrdk
except ImportError:  # pragma: no cover - environment-specific
    flexivrdk = None

Gripper = getattr(flexivrdk, "Gripper", None) if flexivrdk is not None else None


def _side_config(value: Any) -> EndEffectorSideConfig | None:
    """Coerce a cached entry (model or plain dict) to EndEffectorSideConfig."""
    if value is None:
        return None
    if isinstance(value, EndEffectorSideConfig):
        return value
    if isinstance(value, dict):
        try:
            return EndEffectorSideConfig(**value)
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return max(low, min(high, value))


class EndEffectorController:
    """Background thread that mirrors leader DIs onto follower end effectors."""

    # The poll rate only needs to track button presses, so a modest rate keeps
    # the (blocking) digital-output / gripper writes from saturating the bus.
    POLL_HZ = 30.0
    # Defaults used when commanding a gripper, clamped into each gripper's own
    # reported parameter range before use.
    GRIPPER_VELOCITY = 0.1  # [m/s]
    GRIPPER_FORCE = 20.0  # [N]

    def __init__(
        self,
        controller: Any,
        sides: list[str],
        configs: dict[str, Any],
    ) -> None:
        self._controller = controller
        # Per-pair config in pair-index order (None for sides without a usable
        # leader-trigger -> follower-effector mapping).
        self._configs: list[EndEffectorSideConfig | None] = []
        for side in sides:
            cfg = _side_config(configs.get(side))
            usable = cfg if cfg is not None and self._has_work(cfg) else None
            self._configs.append(usable)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Lazily-enabled grippers and the last commands issued, keyed by pair
        # index, so we only issue a (blocking) command on an actual state change.
        self._grippers: dict[int, Any] = {}
        self._gripper_params: dict[int, Any] = {}
        self._last_do: dict[int, bool] = {}
        self._last_gripper_target: dict[int, str] = {}
        self._errors: dict[int, str] = {}

    @staticmethod
    def _has_work(cfg: EndEffectorSideConfig) -> bool:
        # Mirroring needs a leader trigger to read and a follower effector to drive.
        return cfg.leader == "digital_input" and cfg.follower in {
            "digital_output",
            "gripper",
        }

    def has_work(self) -> bool:
        return any(cfg is not None for cfg in self._configs)

    def start(self) -> None:
        if not self.has_work() or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="end-effector-control", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        self._grippers.clear()
        self._gripper_params.clear()
        self._last_do.clear()
        self._last_gripper_target.clear()

    def _run(self) -> None:  # pragma: no cover - hardware specific
        period = 1.0 / self.POLL_HZ
        while not self._stop_event.is_set():
            for index, cfg in enumerate(self._configs):
                if cfg is None:
                    continue
                try:
                    self._tick(index, cfg)
                    self._errors.pop(index, None)
                except Exception as exc:
                    message = describe_exception(exc)
                    # Log only on the first failure (or when it changes) to avoid
                    # flooding the console at the poll rate.
                    if self._errors.get(index) != message:
                        warn(
                            f"End effector control failed for pair {index}", message
                        )
                    self._errors[index] = message
            self._stop_event.wait(period)

    def _tick(self, index: int, cfg: EndEffectorSideConfig) -> None:
        triggered = self._leader_triggered(index, cfg)
        if cfg.follower == "digital_output":
            self._drive_digital_output(index, cfg, triggered)
        elif cfg.follower == "gripper":
            self._drive_gripper(index, cfg, triggered)

    def _leader_triggered(self, index: int, cfg: EndEffectorSideConfig) -> bool:
        # digital_inputs(idx) returns (leader_ports, follower_ports); the leader
        # is the trigger source.
        leader_ports, _ = self._controller.digital_inputs(index)
        port_high = bool(leader_ports[cfg.leader_channel])
        # The leader trigger is "active" when its port matches the activating state.
        return port_high if cfg.leader_activating_state == "high" else not port_high

    def _drive_digital_output(
        self, index: int, cfg: EndEffectorSideConfig, triggered: bool
    ) -> None:
        activated_high = cfg.follower_activated_state == "high"
        # Drive the port to its activated state while triggered, else the opposite.
        value = activated_high if triggered else not activated_high
        if self._last_do.get(index) == value:
            return
        _, follower = self._controller.instances(index)
        follower.SetDigitalOutputs({cfg.follower_channel: value})
        self._last_do[index] = value

    def _drive_gripper(
        self, index: int, cfg: EndEffectorSideConfig, triggered: bool
    ) -> None:
        # Activated state ("close"/"open") is applied while triggered; the
        # opposite is applied otherwise.
        if triggered:
            target = cfg.gripper_activated_state
        else:
            target = "open" if cfg.gripper_activated_state == "close" else "close"
        if self._last_gripper_target.get(index) == target:
            return

        gripper = self._ensure_gripper(index, cfg)
        params = self._gripper_params[index]
        width = params.min_width if target == "close" else params.max_width
        velocity = _clamp(self.GRIPPER_VELOCITY, params.min_vel, params.max_vel)
        force = _clamp(self.GRIPPER_FORCE, params.min_force, params.max_force)
        gripper.Move(width, velocity, force)
        self._last_gripper_target[index] = target

    def _ensure_gripper(self, index: int, cfg: EndEffectorSideConfig) -> Any:
        gripper = self._grippers.get(index)
        if gripper is not None:
            return gripper
        if Gripper is None:
            raise RuntimeError("flexivrdk is not available; cannot control gripper")
        _, follower = self._controller.instances(index)
        gripper = Gripper(follower)
        gripper.Enable(cfg.gripper_model)
        self._grippers[index] = gripper
        self._gripper_params[index] = gripper.params()
        return gripper
