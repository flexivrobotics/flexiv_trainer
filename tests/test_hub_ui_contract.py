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
    # Wrapped for the token-retry flow, so match the await, not a bare api(.
    request = body.index("await withHubToken(")
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


def test_hub_token_prompt_is_wired() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    for element_id in (
        "hub-token-modal",
        "hub-token-input",
        "hub-token-reason",
        "hub-token-submit",
        "hub-token-cancel",
    ):
        assert f'id="{element_id}"' in html
        assert f'byId("{element_id}")' in source
    # A credential must never be a plain text field.
    assert 'id="hub-token-input" type="password"' in html

    assert "error.status = response.status" in source
    assert "error.detail = detail" in source

    assert source.count("withHubToken(") >= 4
    assert 'api("/system/hub-token"' in source


def test_hub_token_prompt_only_triggers_on_auth_failure() -> None:
    # Pinned to the Hub's auth wording: prompting on any error would ask for a
    # token when the repo simply does not exist.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    start = source.index("function isHubTokenError(")
    body = source[start : source.index("function promptForHubToken(", start)]

    assert "no token is configured" in body
    assert "is gated" in body
    assert "authentication failed" in body
    for status in ("401", "403", "404"):
        assert status in body


def test_ui_renders_layout_ok_verdict() -> None:
    # Reading only layout_warning left a successful load showing nothing.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "info.layout_ok" in source
    # Both loaders must clear it on failure, or a stale verdict outlives it.
    assert source.count("state.rolloutActionNamesOk = (info && info.layout_ok)") == 2
    assert source.count('state.rolloutActionNamesOk = ""') >= 3
    # Must be in the render key or the panel never repaints.
    assert 'state.rolloutActionNamesOk || ""' in source
    # A warning outranks the confirmation; they must never show together.
    assert (
        "!state.rolloutActionNamesWarning && state.rolloutActionNamesOk" in source
    )


def test_layout_ok_renders_green_not_as_an_error() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

    assert "rollout-ok" in source
    assert ".rollout-ok {" in styles
    assert "color: var(--success);" in styles.split(".rollout-ok {", 1)[1][:120]


def test_training_hub_dataset_load_controls_exist() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="training-hub-load"' in source
    assert "loadHubDatasetInfo" in source
    assert "/training/hub-dataset-info?" in source
    # meta/ only, so "Downloading" would overstate what the button does.
    assert '"Loading…"' in source


def test_training_loading_state_renders_before_the_request() -> None:
    # As with the rollout loader: painted before the await, or never seen.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    start = source.index("async function loadHubDatasetInfo()")
    body = source[start : source.index("async function startTrainingRun(", start)]

    loading = body.index('state.trainingHubLoadState = "loading"')
    render = body.index("renderTraining();", loading)
    request = body.index("await withHubToken(")
    assert loading < render < request


def test_training_next_requires_a_clean_dataset_load() -> None:
    # Without this the feature is optional and problems still reach Start.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'const hubLoaded = state.trainingHubLoadState === "loaded"' in source
    assert "&& !state.trainingDatasetVerdictWarning" in source
    assert '${hubLoaded ? "" : "disabled"}>Next' in source
    assert "nextBtn.disabled = !state.trainingDatasetRepoId" not in source


def test_training_dataset_verdict_uses_the_server_strings() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "info.dataset_ok" in source
    assert "info.dataset_warning" in source
    assert (
        "!state.trainingDatasetVerdictWarning && state.trainingDatasetVerdict" in source
    )
    assert "rollout-ok" in source and "rollout-error" in source


def test_editing_training_repo_id_invalidates_the_load() -> None:
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "const invalidateHubLoad = () =>" in source
    assert 'loadBtn.textContent = "Load"' in source
    # The revision is part of the cache identity, so it must invalidate too.
    assert "state.trainingHubLoadedRevision" in source
    assert source.count("invalidateHubLoad();") >= 2


def test_hub_dataset_step_estimate_uses_the_hub_frame_count() -> None:
    # buildTrainingExtraArgs drops --steps when steps is 0.
    source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    start = source.index("function computeTrainingSteps(")
    body = source[start : start + 400]

    assert "const frames = trainingDatasetFrames();" in body
    hub_frames = "state.trainingHubDatasetInfo.num_frames"
    assert hub_frames in source
