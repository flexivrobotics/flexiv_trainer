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

"""Per-policy configuration, one module per policy family (diffusion, act, ...).

Each family owns a ``TrainingConfig`` (train-time knobs, emitted as ``lerobot-train``
flags) and a ``RolloutConfig`` (inference-time knobs). Field metadata (default, bounds,
description) doubles as the Web UI form schema via ``training_field_schema``.
"""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel

from flexivtrainer.policies import act, diffusion, pi0, smolvla

TRAINING_CONFIGS: dict[str, type[BaseModel]] = {
    "diffusion": diffusion.TrainingConfig,
    "act": act.TrainingConfig,
    "smolvla": smolvla.TrainingConfig,
    "pi0": pi0.TrainingConfig,
}


def _numeric_bound(metadata: list[Any], *attrs: str) -> float | int | None:
    for meta in metadata:
        for attr in attrs:
            value = getattr(meta, attr, None)
            if value is not None:
                return value
    return None


def training_field_schema(policy_type: str) -> list[dict[str, Any]]:
    """Derive the Web UI form schema from a policy's ``TrainingConfig`` model.

    Each field yields ``{name, flag, type, arity, default, min, max, choices, hint}``.
    ``flag`` is the ``lerobot-train`` argument; ``json_schema_extra={"flag": ...}`` on
    the field overrides the default ``--policy.<name>`` mapping (shared knobs use
    ``--<name>``). Tuple fields carry ``arity`` for multi-box rendering.
    """
    model = TRAINING_CONFIGS[policy_type]
    schema: list[dict[str, Any]] = []
    for name, field in model.model_fields.items():
        extra = field.json_schema_extra or {}
        annotation = field.annotation
        origin = get_origin(annotation)
        if origin is tuple:
            field_type, arity = "tuple", len(get_args(annotation))
        elif origin is Literal:
            field_type, arity = "enum", 0
        elif annotation is bool:  # bool before int: bool subclasses int
            field_type, arity = "bool", 0
        elif annotation is int:
            field_type, arity = "int", 0
        elif annotation is float:
            field_type, arity = "float", 0
        else:
            field_type, arity = "str", 0
        default = field.default
        schema.append(
            {
                "name": name,
                "flag": extra.get("flag", f"--policy.{name}"),
                "type": field_type,
                "arity": arity,
                "default": list(default) if isinstance(default, tuple) else default,
                "min": _numeric_bound(field.metadata, "ge", "gt"),
                "max": _numeric_bound(field.metadata, "le", "lt"),
                "choices": list(get_args(annotation)) if field_type == "enum" else None,
                "hint": field.description or "",
            }
        )
    return schema
