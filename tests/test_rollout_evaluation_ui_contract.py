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

"""Front-end contract for recording, verdicts and evaluation sessions.

CI runs only pytest, so the wiring the operator depends on is pinned here as
text; tests/rollout_evaluation_ui.test.cjs exercises the same code in node.
"""

import re
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "flexivtrainer" / "web"


def _app_js() -> str:
    return (WEB_ROOT / "app.js").read_text(encoding="utf-8")


def _function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    following = re.search(r"\n(?:async )?function ", source[start + 1 :])
    return (
        source[start : start + 1 + following.start()] if following else source[start:]
    )


def test_the_rollout_page_hosts_the_evaluation_panel() -> None:
    source = _app_js()
    render = _function(source, "renderRollout")
    assert 'id="rollout-evaluation"' in render
    assert render.count("renderRolloutEvaluation();") == 2, (
        "both render paths refresh it"
    )


def test_a_single_rollout_sends_the_recording_and_feedforward_choices() -> None:
    start = _function(_app_js(), "startRolloutRun")
    assert "record: !!state.rolloutRecord" in start
    assert "send_twist" in start


def test_verdicts_and_trials_use_the_rollout_api() -> None:
    source = _app_js()
    assert '"/rollout/verdict"' in _function(source, "saveRolloutVerdict")
    trial = _function(source, "startRolloutTrial")
    assert "/rollout/evaluations/${encodeURIComponent(name)}/next" in trial


def test_a_trial_is_shown_by_its_label() -> None:
    panel = _function(_app_js(), "renderRolloutEvaluation")
    assert "status.checkpoint_label" in panel
    assert "checkpoint_path" not in panel


def test_the_page_loads_the_new_script_and_styles() -> None:
    index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    assert "/static/app.js?v=20261015-rollout-eval" in index
    assert "/static/styles.css?v=20261015-rollout-eval" in index
    assert ".rollout-eval-summary" in (WEB_ROOT / "styles.css").read_text(
        encoding="utf-8"
    )
