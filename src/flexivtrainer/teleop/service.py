from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Any

from flexivtrainer.config import AppSettings, TeleopRobotPair
from flexivtrainer.observability import describe_exception

try:
    from flexivtdk import TransparentCartesianTeleopLAN
except (
    ImportError
):  # pragma: no cover - dependency availability is environment-specific
    TransparentCartesianTeleopLAN = None


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
                if hasattr(self._controller, "Init"):
                    self._controller.Init()
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

        warnings: list[str] = []
        for index, pair in enumerate(self._get_robot_pairs()):
            if pair.leader_home_posture:
                try:
                    self._controller.SetLeaderNullSpacePosture(pair.leader_home_posture)
                except TypeError:
                    try:
                        self._controller.SetLeaderNullSpacePosture(
                            index, pair.leader_home_posture
                        )
                    except Exception as exc:  # pragma: no cover - hardware specific
                        warnings.append(
                            f"Leader home posture failed for pair {index}: {describe_exception(exc)}"
                        )
                except Exception as exc:  # pragma: no cover - hardware specific
                    warnings.append(
                        f"Leader home posture failed for pair {index}: {describe_exception(exc)}"
                    )

            if pair.follower_home_posture:
                try:
                    self._controller.SetFollowerNullSpacePosture(
                        pair.follower_home_posture
                    )
                except TypeError:
                    try:
                        self._controller.SetFollowerNullSpacePosture(
                            index, pair.follower_home_posture
                        )
                    except Exception as exc:  # pragma: no cover - hardware specific
                        warnings.append(
                            f"Follower home posture failed for pair {index}: {describe_exception(exc)}"
                        )
                except Exception as exc:  # pragma: no cover - hardware specific
                    warnings.append(
                        f"Follower home posture failed for pair {index}: {describe_exception(exc)}"
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
