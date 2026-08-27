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

from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "flexivtrainer" / "web"


def test_gripper_panel_uses_backend_session_lifecycle() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "GRIPPER_INIT_WAIT_MS" not in source
    assert 'gripperSession[side] === "ready"' in source
    assert '"/teleop/gripper/reinitialize"' in source
    assert "window.confirm" in source
    assert 'markup = sessionReady ? "Prepare" : "Initialize"' in source
    assert "showReinitialize = done || anySessionReady" in source
    assert "params?.takeover_pending" in source
    assert "Grasp preserved — request Close once to enable opening." in source


def test_gripper_panel_asset_revision_is_current() -> None:
    index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    assert "/static/app.js?v=20260812-07" in index
    assert "/static/styles.css?v=20260812-03" in index


def test_recording_uses_accepted_gripper_target_width_command() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "label: `${side}.gripper.target_width`" in source
    assert 'payload: "gripper_command"' in source
    assert 'verifyField: "target_width"' in source
    assert "flexivtrainer.gripperParams" not in source
    assert 'api("/teleop/gripper/params"' in source
