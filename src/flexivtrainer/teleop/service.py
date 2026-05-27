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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
import time

from flexivtrainer.config import AppSettings, TeleopRobotPair
from flexivtrainer.observability import describe_exception

try:
    import flexivtdk
except (
    ImportError
):  # pragma: no cover - dependency availability is environment-specific
    flexivtdk = None

TransparentCartesianTeleopLAN = (
    getattr(flexivtdk, "TransparentCartesianTeleopLAN", None)
    if flexivtdk is not None
    else None
)


@dataclass
class TeleopSnapshot:
    configured: bool
    available: bool
    initialized: bool
    started: bool
    stopped: bool
    fault: str | None
    error: str | None


class TeleopService:
    def __init__(
        self,
        settings: AppSettings,
        get_robot_pairs: Callable[[], list[TeleopRobotPair]] | None = None,
    ) -> None:
        self._settings = settings
        self._get_robot_pairs = get_robot_pairs or (lambda: settings.teleop_robot_pairs)
        self._controller: Any | None = None
        self._error: str | None = None
        self._initialized = False

    def _instance_provider_results(self, provider: Any) -> list[Any]:
        if not callable(provider):
            return []

        try:
            result = provider()
        except TypeError:
            result = None
        except Exception:
            result = None
        else:
            return [result]

        results: list[Any] = []
        for index in range(8):
            try:
                result = provider(index)
            except TypeError:
                break
            except Exception:
                break

            results.append(result)
        return results

    def _extract_robot_handles(self, value: Any) -> list[Any]:
        handles: list[Any] = []
        stack = [value]
        seen: set[int] = set()

        while stack:
            current = stack.pop()
            if current is None:
                continue

            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)

            if callable(getattr(current, "ExecutePrimitive", None)):
                handles.append(current)
                continue

            if isinstance(current, dict):
                stack.extend(current.values())
                continue

            if isinstance(current, (list, tuple, set)):
                stack.extend(current)
                continue

            for attr_name in (
                "robot",
                "robots",
                "leader",
                "follower",
                "leader_robot",
                "follower_robot",
                "rdk_robot",
                "rdk_robots",
            ):
                if hasattr(current, attr_name):
                    stack.append(getattr(current, attr_name))

        return handles

    def _home_robot_handles(self) -> list[Any]:
        if self._controller is None:
            return []

        provider_results: list[Any] = [self._controller]
        provider_results.extend(
            self._instance_provider_results(
                getattr(self._controller, "instances", None)
            )
        )

        class_provider = getattr(TransparentCartesianTeleopLAN, "instances", None)
        if class_provider is not None:
            provider_results.extend(self._instance_provider_results(class_provider))

        robots = self._extract_robot_handles(provider_results)
        deduped: list[Any] = []
        seen: set[int] = set()
        for robot in robots:
            marker = id(robot)
            if marker in seen:
                continue
            deduped.append(robot)
            seen.add(marker)
        return deduped

    def _wait_for_home_completion(self, robot: Any, timeout_s: float = 60.0) -> None:
        states_reader = getattr(robot, "primitive_states", None) or getattr(
            robot, "primitiveStates", None
        )
        if not callable(states_reader):
            return

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            states = states_reader()
            reached_target = (
                states.get("reachedTarget") if isinstance(states, dict) else None
            )
            if isinstance(reached_target, list):
                reached_target = reached_target[0] if reached_target else False
            if reached_target:
                return
            time.sleep(0.2)

        raise TimeoutError("Timed out while waiting for Home primitive to finish")

    def initialize(self) -> TeleopSnapshot:
        robot_pairs_config = self._get_robot_pairs()
        if TransparentCartesianTeleopLAN is None:
            self._error = "flexivtdk is not importable in the selected environment"
            return self.snapshot()
        if not any(
            pair.leader_serial and pair.follower_serial for pair in robot_pairs_config
        ):
            self._error = "No teleoperation robot pairs are configured"
            return self.snapshot()
        if self._controller is None:
            robot_pairs = [
                (pair.leader_serial, pair.follower_serial)
                for pair in robot_pairs_config
                if pair.leader_serial and pair.follower_serial
            ]
            try:
                self._controller = TransparentCartesianTeleopLAN(
                    robot_pairs_sn=robot_pairs,
                    network_interface_whitelist=self._settings.network_interface_whitelist,
                )
                init_method = getattr(self._controller, "Init", None)
                if callable(init_method):
                    init_method()
                self._initialized = True
                self._error = None
            except Exception as exc:  # pragma: no cover - hardware specific
                self._error = describe_exception(exc)
                self._controller = None
        return self.snapshot()

    def start(self) -> TeleopSnapshot:
        self.initialize()
        if self._controller is None:
            return self.snapshot()
        try:
            self._controller.Start()
            self._error = None
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
        return self.snapshot()

    def stop(self) -> TeleopSnapshot:
        if self._controller is None:
            return self.snapshot()
        try:
            self._controller.Stop()
            self._error = None
        except Exception as exc:  # pragma: no cover - hardware specific
            self._error = describe_exception(exc)
        return self.snapshot()

    def shutdown(self) -> None:
        if self._controller is None:
            return

        try:
            self.stop()
        except Exception:
            pass

        for method_name in (
            "Close",
            "close",
            "Shutdown",
            "shutdown",
            "Disconnect",
            "disconnect",
        ):
            method = getattr(self._controller, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:  # pragma: no cover - hardware specific
                    self._error = describe_exception(exc)
                break

        self._controller = None
        self._initialized = False
        self._error = None

    def reset_home(self) -> dict[str, Any]:
        if self._controller is None:
            return {"ok": False, "error": "Teleoperation controller is not initialized"}

        robots = self._home_robot_handles()
        if not robots:
            return {
                "ok": False,
                "error": "Unable to access robot instances from TransparentCartesianTeleopLAN.instances()",
            }

        warnings: list[str] = []
        for index, robot in enumerate(robots):
            primitive_sent = False
            last_error: Exception | None = None
            for primitive_name in ("Home", "Home()"):
                try:
                    robot.ExecutePrimitive(primitive_name, dict())
                    primitive_sent = True
                    break
                except TypeError:
                    try:
                        robot.ExecutePrimitive(primitive_name)
                        primitive_sent = True
                        break
                    except Exception as exc:  # pragma: no cover - hardware specific
                        last_error = exc
                except Exception as exc:  # pragma: no cover - hardware specific
                    last_error = exc
            if not primitive_sent:
                warnings.append(
                    f"Home primitive failed for robot {index}: {describe_exception(last_error or RuntimeError('Home primitive invocation failed'))}"
                )
                continue

            try:
                self._wait_for_home_completion(robot)
            except Exception as exc:  # pragma: no cover - hardware specific
                warnings.append(
                    f"Home completion failed for robot {index}: {describe_exception(exc)}"
                )

        return {"ok": not warnings, "warnings": warnings}

    def snapshot(self) -> TeleopSnapshot:
        robot_pairs = self._get_robot_pairs()
        configured = any(
            pair.leader_serial and pair.follower_serial for pair in robot_pairs
        )
        started = False
        stopped = True
        fault: str | None = None
        if self._controller is not None:
            started = not bool(getattr(self._controller, "stopped", True))
            stopped = bool(getattr(self._controller, "stopped", True))
            raw_fault = getattr(self._controller, "fault", None)
            fault = str(raw_fault) if raw_fault else None
        return TeleopSnapshot(
            configured=configured,
            available=TransparentCartesianTeleopLAN is not None,
            initialized=self._initialized and self._controller is not None,
            started=started,
            stopped=stopped,
            fault=fault,
            error=self._error,
        )
