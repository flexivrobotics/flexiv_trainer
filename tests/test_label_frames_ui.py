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

"""The Labelling Frames page: its node tests, and its wiring pinned as text
for machines without node."""

import shutil
import subprocess
from pathlib import Path

import pytest

WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "flexivtrainer" / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_label_frames_javascript() -> None:
    script = Path(__file__).with_name("label_frames_ui.test.cjs")
    result = subprocess.run(
        [shutil.which("node"), "--test", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_processing_page_offers_labelling_and_uses_the_label_api() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    assert 'cardTitle: "Labelling Frames"' in source
    assert "openOtherBrowser: () => openProcessingDatasetBrowser()" in source
    assert 'if (state.processingMode === "labels") {' in source
    assert "/datasets/check-labels?path=" in source
    assert 'method: "PUT"' in source and 'method: "DELETE"' in source
    assert ".label-badge--never" in (WEB_ROOT / "styles.css").read_text(
        encoding="utf-8"
    )
