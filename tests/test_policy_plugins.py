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

"""Policy types registered by packages outside this repository."""

import sys

import pytest
from test_rollout_evaluation import _service
from test_rollout_service import (
    _checkpoint,
    _FakePolicy,
    _FakeRobot,
)

from flexivtrainer.config import AppSettings
from flexivtrainer.policies import PolicyConfig, import_policy_plugins
from flexivtrainer.policies._shared import SharedRolloutConfig


def test_a_family_keeps_its_own_rollout_knobs_and_others_get_the_plugin_ones() -> None:
    config = PolicyConfig(plugin_rollout=SharedRolloutConfig(n_action_steps=1))
    assert config.rollout_for("act") is config.act.rollout
    assert config.rollout_for("my_family").n_action_steps == 1
    # A field that is not a family is not mistaken for one.
    assert config.rollout_for("plugin_packages") is config.plugin_rollout


def test_the_plugin_knobs_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("FLEXIV_TRAINER_POLICIES__PLUGIN_PACKAGES", '["pkg_a", "pkg_b"]')
    monkeypatch.setenv("FLEXIV_TRAINER_POLICIES__PLUGIN_ROLLOUT__N_ACTION_STEPS", "1")
    monkeypatch.setenv("FLEXIV_TRAINER_POLICIES__PLUGIN_ROLLOUT__REPLAN_STEPS", "0")
    settings = AppSettings()
    assert settings.policies.plugin_packages == ["pkg_a", "pkg_b"]
    rollout = settings.policies.rollout_for("my_family")
    assert (rollout.n_action_steps, rollout.replan_steps) == (1, 0)


def test_plugin_packages_are_imported(tmp_path, monkeypatch) -> None:
    package = tmp_path / "registers_a_policy"
    package.mkdir()
    (package / "__init__.py").write_text("REGISTERED = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "registers_a_policy", raising=False)
    assert import_policy_plugins([" registers_a_policy ", ""]) == ["registers_a_policy"]
    assert sys.modules["registers_a_policy"].REGISTERED


def test_a_plugin_that_does_not_import_stops_the_rollout_by_name(tmp_path) -> None:
    robot = _FakeRobot("F1")

    def configure(settings):
        settings.policies.plugin_packages = ["no_such_policy_package"]

    service = _service(tmp_path, _FakePolicy([0.0] * 19), robot, configure=configure)
    with pytest.raises(RuntimeError, match="no_such_policy_package"):
        service.start(_checkpoint(tmp_path))
    assert not robot.enabled
