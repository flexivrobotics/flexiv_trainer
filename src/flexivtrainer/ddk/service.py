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

from typing import Callable
from typing import Any

from flexivtrainer.config import AppSettings
from flexivtrainer.observability import describe_exception

try:
    from flexivddk import Client
except (
    ImportError
):  # pragma: no cover - dependency availability is environment-specific
    Client = None


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return {
            key: _serialize_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


class DDKService:
    def __init__(
        self,
        settings: AppSettings,
        get_remote_robot_serials: Callable[[], list[str]] | None = None,
    ) -> None:
        self._settings = settings
        self._get_remote_robot_serials = get_remote_robot_serials or (
            lambda: settings.remote_robot_serials
        )
        self._clients: dict[str, Any] = {}
        self._errors: dict[str, str] = {}

    def initialize(self) -> dict[str, Any]:
        if Client is None:
            return {
                "available": False,
                "configured": bool(self._get_remote_robot_serials()),
                "errors": {
                    "import": "flexivddk is not importable in the selected environment"
                },
                "robots": {},
            }

        for serial in self._get_remote_robot_serials():
            if serial in self._clients:
                continue
            try:
                self._clients[serial] = Client(
                    robot_sn=serial,
                    network_interface_whitelist=self._settings.network_interface_whitelist,
                    verbose=False,
                )
                self._errors.pop(serial, None)
            except Exception as exc:  # pragma: no cover - hardware specific
                self._errors[serial] = describe_exception(exc)

        return self.status()

    def shutdown(self) -> None:
        for serial, client in list(self._clients.items()):
            for method_name in (
                "Close",
                "close",
                "Disconnect",
                "disconnect",
                "Stop",
                "stop",
            ):
                method = getattr(client, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:  # pragma: no cover - hardware specific
                        self._errors[serial] = describe_exception(exc)
                    break
        self._clients.clear()
        self._errors.clear()

    def status(self) -> dict[str, Any]:
        robots = {}
        for serial, client in self._clients.items():
            robots[serial] = {
                "connected": bool(getattr(client, "connected", False)),
                "error": self._errors.get(serial),
            }
        for serial, error in self._errors.items():
            robots.setdefault(serial, {"connected": False, "error": error})

        return {
            "available": Client is not None,
            "configured": bool(self._get_remote_robot_serials()),
            "errors": dict(self._errors),
            "robots": robots,
        }

    def snapshot(self, initialize: bool = True) -> dict[str, Any]:
        if initialize:
            self.initialize()
        robots: dict[str, Any] = {}
        for serial, client in self._clients.items():
            try:
                robots[serial] = {
                    "connected": bool(getattr(client, "connected", False)),
                    "server_time": _serialize_value(
                        getattr(client, "server_time", None)
                    ),
                    "cartesian_state": _serialize_value(
                        getattr(client, "cartesian_states", None)
                    ),
                    "cartesian_command": _serialize_value(
                        getattr(client, "cartesian_commands", None)
                    ),
                    "digital_inputs": _serialize_value(
                        getattr(client, "digital_inputs", None)
                    ),
                }
                self._errors.pop(serial, None)
            except Exception as exc:  # pragma: no cover - hardware specific
                self._errors[serial] = describe_exception(exc)
                robots[serial] = {
                    "connected": False,
                    "error": describe_exception(exc),
                }

        return {
            "robots": robots,
            "errors": dict(self._errors),
        }
