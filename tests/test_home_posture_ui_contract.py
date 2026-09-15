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

"""Front-end contract for the home-posture Record button."""

from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "flexivtrainer" / "web"


def test_record_button_lives_in_the_home_posture_panel() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'id="home-posture-record"' in html
    panel = html.split('id="home-posture-panel"', 1)[1]
    assert 'id="home-posture-record"' in panel.split("</section>", 1)[0]


def test_record_button_reads_the_follower_and_persists() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'byId("home-posture-record")' in source
    assert '"/teleop/follower-posture"' in source
    assert "saveRobotConfigNow" in source
