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

"""Front-end contract for the one-click rollout Start."""

import re
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "flexivtrainer" / "web"


def _app_js() -> str:
    return (WEB_ROOT / "app.js").read_text(encoding="utf-8")


def _render_rollout_body(source: str) -> str:
    start = source.index("function renderRollout()")
    return source[start : source.index("function renderRolloutCameras", start)]


def test_rollout_tab_has_no_service_bubbles() -> None:
    source = _app_js()
    styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

    for token in (
        "buildRolloutServiceCard",
        "data-rollout-service",
        "controlRolloutService",
        "isRolloutServiceConnected",
        "rolloutServiceBusy",
        "Disconnect before rollout",
    ):
        assert token not in source, f"{token} survived the bubble removal"
    assert "rollout-service-card" not in styles


def test_rollout_start_does_not_gate_on_service_state() -> None:
    body = _render_rollout_body(_app_js())
    match = re.search(r"const canStart = (.*?);", body, re.DOTALL)
    assert match, "renderRollout no longer computes canStart"
    can_start = match.group(1)

    assert "hasCheckpointSelection" in can_start
    for token in ("teleop", "cameras", "Connected", "serviceConnectBusy"):
        assert token not in can_start


def test_rollout_render_key_drops_service_state() -> None:
    body = _render_rollout_body(_app_js())
    match = re.search(r"const renderKey = \[(.*?)\]\.join", body, re.DOTALL)
    assert match
    render_key = match.group(1)

    assert "rolloutServiceKey" not in render_key
    assert "rolloutServiceBusyKey" not in render_key
    assert "state.rolloutRequiresTask" in render_key


def test_start_resyncs_service_state_after_the_server_preflight() -> None:
    source = _app_js()
    start = source.index("async function startRolloutRun()")
    body = source[start : source.index("function renderRollout()", start)]

    assert "} finally {" in body
    assert "refreshTeleopStatus()" in body
    assert "refreshSummary()" in body


def test_home_page_keeps_its_own_service_tiles() -> None:
    source = _app_js()

    assert "function createServiceStatusCard(" in source
    assert "controlHomeService(definition.serviceName, definition.control)" in source
