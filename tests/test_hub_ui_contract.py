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
    # Without this the button never repaints from Load to Downloading.
    assert "state.rolloutHubLoadState" in render_key


def test_hub_load_button_reports_download_progress() -> None:
    # The Hub fetch is synchronous and can run for minutes, so the button must
    # show it is working rather than looking unresponsive.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert '"Downloading…"' in source
    assert '"Loaded"' in source
    assert 'state.rolloutHubLoadState = "loading"' in source
    assert 'state.rolloutHubLoadState = "loaded"' in source
    assert 'state.rolloutHubLoadState = "error"' in source


def test_hub_load_button_disabled_while_downloading() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'const hubLoading = state.rolloutHubLoadState === "loading"' in source
    assert "isRunning || hubLoading ? \"disabled\" : \"\"" in source
    # Starting a rollout mid-download would race the same cache directory.
    assert "canStart = hasCheckpointSelection && !isRunning && !hubLoading" in source


def test_loading_state_renders_before_the_request() -> None:
    # renderRollout() must run between setting "loading" and awaiting the fetch,
    # or the spinner state is never painted.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    start = source.index("async function loadHubCheckpointInfo()")
    body = source[start : source.index("async function startRolloutRun()", start)]

    loading = body.index('state.rolloutHubLoadState = "loading"')
    render = body.index("renderRollout();", loading)
    request = body.index("await api(")
    assert loading < render < request


def test_ui_uses_server_layout_verdict() -> None:
    # The server knows the action width and the arm count, so the UI must not
    # re-derive a vague "if it has a gripper" caveat of its own.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "info.layout_warning" in source
    assert "If its layout" not in source
    assert "includes a gripper, the rollout will refuse" not in source


def test_editing_repo_id_clears_loaded_label() -> None:
    # "Loaded" refers to one specific repo; typing a different one must not keep
    # claiming the new id is already downloaded.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "state.rolloutHubLoadedRepoId" in source
    assert 'loadBtn.textContent = "Load"' in source


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


def test_output_dir_is_named_from_the_hub_repo() -> None:
    # A Hub dataset has no local path, so mergedDatasetPath is empty. Without a
    # name the output falls back to the bare training root, which the backend
    # rejects with "output must be within training root".
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    start = source.index("function getTrainingOutputDir()")
    body = source[start : source.index("\n}", start)]

    assert 'state.trainingDatasetSource === "hub"' in body
    assert "state.trainingDatasetRepoId" in body
    # The repo id contains "/", which must not become a path separator.
    assert "replace(/[^A-Za-z0-9._-]+/g" in body


def test_hub_dataset_skips_the_preview_step() -> None:
    # Only meta/ is fetched for a Hub dataset, so there are no frames to show;
    # the preview page would sit empty forever.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "const usingHubDataset = state.trainingDatasetSource === \"hub\"" in source
    assert (
        "stepAfterDataset = usingHubDataset ? previewStep + 1 : previewStep" in source
    )
    assert "state.trainingStep = stepAfterDataset" in source
    # And Back from policy selection must not land on the skipped page.
    assert "usingHubDataset ? datasetStep : previewStep" in source


def test_training_dataset_hub_controls_exist() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="training-dataset-source-toggle"' in source
    assert 'id="training-hub-repo"' in source
    assert 'dataset_source: "hub"' in source
    assert 'dataset_source: "local", dataset_path: state.mergedDatasetPath' in source
