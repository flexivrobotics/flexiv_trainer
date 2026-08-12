# Copyright 2026 Flexiv Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared gripper command metadata validation and persistence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRIPPER_COMMAND_FILENAME = "gripper_command.json"
GRIPPER_COMMAND_RELATIVE_PATH = Path("meta") / GRIPPER_COMMAND_FILENAME
TARGET_WIDTH_AXIS_SUFFIX = ".gripper.target_width"
LEGACY_WIDTH_AXIS_SUFFIX = ".gripper.width"
REMOVED_CLOSE_AXIS_SUFFIX = ".gripper.close"
COMMAND_PARAMETER_DECIMALS = 6


@dataclass(frozen=True)
class GripperCommandMetadata:
    velocity_m_s: float
    force_limit_n: float
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("Unsupported gripper command metadata format_version")
        for field_name in ("velocity_m_s", "force_limit_n"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{field_name} must be a finite positive number")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{field_name} must be a finite positive number")
            normalized = round(numeric, COMMAND_PARAMETER_DECIMALS)
            if normalized <= 0:
                raise ValueError(
                    f"{field_name} is below the supported metadata precision"
                )
            object.__setattr__(self, field_name, normalized)

    @classmethod
    def from_dict(cls, payload: Any) -> GripperCommandMetadata:
        if not isinstance(payload, dict):
            raise ValueError("Gripper command metadata must be a JSON object")
        velocity = payload.get("velocity_m_s")
        force = payload.get("force_limit_n")
        if isinstance(velocity, bool) or not isinstance(velocity, int | float):
            raise ValueError("velocity_m_s must be a finite positive number")
        if isinstance(force, bool) or not isinstance(force, int | float):
            raise ValueError("force_limit_n must be a finite positive number")
        velocity = float(velocity)
        force = float(force)
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError("velocity_m_s must be a finite positive number")
        if not math.isfinite(force) or force <= 0:
            raise ValueError("force_limit_n must be a finite positive number")
        return cls(
            velocity_m_s=velocity,
            force_limit_n=force,
            format_version=payload.get("format_version"),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "format_version": self.format_version,
            "velocity_m_s": self.velocity_m_s,
            "force_limit_n": self.force_limit_n,
        }


def read_gripper_command_metadata(path: Path) -> GripperCommandMetadata:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid gripper command metadata: {path}") from exc
    try:
        return GripperCommandMetadata.from_dict(payload)
    except ValueError as exc:
        raise ValueError(f"Invalid gripper command metadata at {path}: {exc}") from exc


def write_gripper_command_metadata(
    root: Path,
    metadata: GripperCommandMetadata,
    *,
    checkpoint: bool = False,
) -> Path:
    path = (
        root / GRIPPER_COMMAND_FILENAME
        if checkpoint
        else root / GRIPPER_COMMAND_RELATIVE_PATH
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def action_gripper_mode(names: list[str]) -> str | None:
    target_width = any(name.endswith(TARGET_WIDTH_AXIS_SUFFIX) for name in names)
    legacy_width = any(name.endswith(LEGACY_WIDTH_AXIS_SUFFIX) for name in names)
    removed_close = any(name.endswith(REMOVED_CLOSE_AXIS_SUFFIX) for name in names)
    if removed_close:
        raise ValueError(
            "Boolean gripper.close actions are unsupported; convert the dataset "
            "to gripper.target_width"
        )
    if target_width and legacy_width:
        raise ValueError(
            "Action schema cannot mix gripper.target_width and legacy gripper.width"
        )
    if target_width:
        return "target_width"
    if legacy_width:
        return "legacy_width"
    return None
