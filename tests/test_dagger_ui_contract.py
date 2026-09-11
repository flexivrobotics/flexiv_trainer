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

"""Front-end contract for the DAgger leader-match toggle."""

from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "flexivtrainer" / "web"


def test_dagger_toggle_is_wired() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="teleop-dagger-mode"' in html
    assert 'byId("teleop-dagger-mode")' in source
    assert "state.ui.daggerMode" in source


def test_dagger_mode_targets_the_match_endpoint() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert '"/teleop/match-leader"' in source
    assert '"/teleop/home"' in source


def test_home_button_swaps_label_through_render_keys() -> None:
    # Both keys must exist: setMarkupIfChanged only repaints on a key change, so
    # a single key would leave the button stuck on one label.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert '"teleop-home:match"' in source
    assert '"teleop-home:home"' in source
    assert "Match Leader" in source
