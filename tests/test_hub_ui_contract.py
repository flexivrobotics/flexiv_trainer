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

"""Front-end contract for the HuggingFace source toggles."""

import re
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "flexivtrainer" / "web"


def _rollout_render_key(source: str) -> str:
    """The renderKey literal inside renderRollout()."""
    start = source.index("function renderRollout()")
    body = source[start : source.index("container.innerHTML", start)]
    match = re.search(r"const renderKey = \[(.*?)\]\.join", body, re.DOTALL)
    assert match, "renderRollout no longer builds a renderKey array"
    return match.group(1)


def test_rollout_render_key_tracks_hub_source() -> None:
    # renderRollout() returns early when the renderKey is unchanged. Hub state
    # left out of that key means clicking the toggle updates state but never
    # redraws, so the button appears dead.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    render_key = _rollout_render_key(source)

    assert "state.rolloutCheckpointSource" in render_key
    assert "state.rolloutActionNamesWarning" in render_key


def test_rollout_render_key_excludes_free_text_inputs() -> None:
    # Text fields must stay out of the key: the 1s status poll would otherwise
    # rebuild the panel mid-typing and steal focus.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    render_key = _rollout_render_key(source)

    assert "rolloutCheckpointRepoId" not in render_key
    assert "rolloutCheckpointRevision" not in render_key
    assert "rolloutTaskText" not in render_key


def test_repo_id_input_updates_start_button_directly() -> None:
    # Because the repo id is not in the renderKey, the Start button has to be
    # enabled imperatively as the user types.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'byId("rollout-primary-action")' in source
    assert "primary.disabled = !state.rolloutCheckpointRepoId" in source


def test_rollout_hub_controls_exist() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="rollout-source-toggle"' in source
    assert 'id="rollout-hub-repo"' in source
    assert 'id="rollout-hub-load"' in source
    assert "loadHubCheckpointInfo" in source


def test_rollout_start_sends_source_discriminator() -> None:
    # A repo id must never be sent on checkpoint_path, or it would reach the
    # local path resolver server-side.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert '{ source: "hub", repo_id: repoId, task,' in source
    assert '{ source: "local", checkpoint_path: checkpoint, task }' in source


def test_training_dataset_hub_controls_exist() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="training-dataset-source-toggle"' in source
    assert 'id="training-hub-repo"' in source
    assert 'dataset_source: "hub"' in source
    assert 'dataset_source: "local", dataset_path: state.mergedDatasetPath' in source
