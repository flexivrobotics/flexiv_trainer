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

from flexivtrainer.policies import training_field_schema


def test_act_field_schema_order_and_flags() -> None:
    schema = training_field_schema("act")
    by_name = {f["name"]: f for f in schema}

    # Base (shared) fields come first, then ACT declaration order.
    # temporal_ensemble_coeff is declared last so its Web UI "enable" checkbox has no
    # cell to its right to overlap.
    act_order = [
        "chunk_size",
        "n_action_steps",
        "n_encoder_layers",
        "n_decoder_layers",
        "dim_model",
        "optimizer_lr",
        "temporal_ensemble_coeff",
    ]
    names = [f["name"] for f in schema]
    assert names[-len(act_order):] == act_order

    assert by_name["n_action_steps"] == {
        "name": "n_action_steps",
        "flag": "--policy.n_action_steps",
        "type": "int",
        "arity": 0,
        "default": 100,
        "min": 1,
        "max": 1000,
        "choices": None,
        "hint": by_name["n_action_steps"]["hint"],
    }
    assert by_name["n_encoder_layers"]["default"] == 4
    assert by_name["n_encoder_layers"]["type"] == "int"
    assert by_name["n_encoder_layers"]["max"] == 12
    assert by_name["n_decoder_layers"]["default"] == 7
    assert by_name["n_decoder_layers"]["type"] == "int"
    assert by_name["n_decoder_layers"]["max"] == 12


def test_temporal_ensemble_coeff_is_float_not_optional_string() -> None:
    # Regression guard: modeling the coeff as float | None would make the schema
    # generator misclassify it as type="str" with a null default (a UnionType
    # matches none of the tuple/Literal/bool/int/float branches).
    schema = training_field_schema("act")
    coeff = next(f for f in schema if f["name"] == "temporal_ensemble_coeff")
    assert coeff["type"] == "float"
    assert coeff["default"] == 0.1
    assert coeff["flag"] == "--policy.temporal_ensemble_coeff"
    assert coeff["min"] == 0  # from gt=0
    assert coeff["max"] == 1.0  # from le=1.0
