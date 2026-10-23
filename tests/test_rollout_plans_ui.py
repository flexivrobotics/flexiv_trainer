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

"""The rollout page's Chains & Practice panel: its node tests, and its wiring
pinned as text for machines without node."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "flexivtrainer" / "web"


def _function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    following = re.search(r"\n(?:async )?function ", source[start + 1 :])
    return (
        source[start : start + 1 + following.start()] if following else source[start:]
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_rollout_plans_javascript() -> None:
    script = Path(__file__).with_name("rollout_plans_ui.test.cjs")
    result = subprocess.run(
        [shutil.which("node"), "--test", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_rollout_page_hosts_the_plans_panel_and_uses_the_plans_api() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    render = _function(source, "renderRollout")
    assert 'id="rollout-plans"' in render
    assert render.count("renderRolloutPlans();") == 2, "both render paths refresh it"
    assert "refreshRolloutPlans()" in _function(source, "bootstrapRollout")
    assert "/rollout/plans/${encodeURIComponent(name)}/start" in _function(
        source, "startRolloutPlan"
    )
    panel = _function(source, "renderRolloutPlans")
    assert '"/rollout/chain/continue", { action: "retry" }' in panel
    assert '"/rollout/chain/continue", { action: "skip" }' in panel
    assert '"/rollout/practice/continue"' in panel
    assert "refreshRolloutChainRuns()" in _function(source, "bootstrapRollout")
    assert '"/rollout/chain/verdict"' in _function(source, "saveChainVerdict")
    assert '"/rollout/chains"' in _function(source, "refreshRolloutChainRuns")
    assert ".rollout-chain-step--current" in (WEB_ROOT / "styles.css").read_text(
        encoding="utf-8"
    )
