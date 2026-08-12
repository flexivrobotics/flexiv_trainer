# HuggingFace Hub loading — session change summary

Adds the ability to train from a Hub **dataset**, roll out from a Hub
**checkpoint**, and fine-tune from a Hub checkpoint. Previously both training and
rollout were local-filesystem-only.

**Status:** training from a Hub dataset is verified running end to end. **Rollout
from a Hub checkpoint is not fully verified yet** — the download, path
validation, and metadata reads work, but a full rollout against hardware has not
been exercised.

---

## Design constraint that shaped everything

The repo has careful path security (`tests/test_path_security.py`). The guiding
rule for this patch was: **do not weaken any existing validator.** Hub
identifiers travel on their own explicitly-typed fields (`source` + `repo_id`),
never on the fields that feed a path resolver.

Consequence: `tests/test_path_security.py` is **unmodified** and still passes.

---

## Files changed

| File | Kind | What |
|---|---|---|
| `src/flexivtrainer/data/hub.py` | **new** | All Hub identity, validation, caching, error mapping |
| `src/flexivtrainer/config.py` | modified | `HubConfig`, `StorageConfig.hub_cache_root` |
| `src/flexivtrainer/rollout/checkpoint.py` | modified | Hub checkpoint resolution + action-name recovery chain |
| `src/flexivtrainer/rollout/service.py` | modified | `source`/`repo_id`/`revision`/`action_names` on `start()` |
| `src/flexivtrainer/jobs/train_policy.py` | modified | Hub dataset/checkpoint resolution, `action_names.json` sidecar |
| `src/flexivtrainer/api/routes/rollout.py` | modified | Source discriminator, Hub error → HTTP codes, layout warning |
| `src/flexivtrainer/api/routes/training.py` | modified | Source discriminator, Hub error → HTTP codes |
| `src/flexivtrainer/web/app.js` | modified | "Use HuggingFace" toggles, load-progress button, flow fixes |
| `src/flexivtrainer/web/styles.css` | modified | Hub input + load button styles |
| `src/flexivtrainer/web/index.html` | modified | Asset cache-bust version |
| `tests/test_hub.py` | **new** | Unit tests for `hub.py` |
| `tests/test_hub_integration.py` | **new** | End-to-end wiring, no network |
| `tests/test_hub_ui_contract.py` | **new** | Front-end contract guards |

`src/flexivtrainer/web/digital-io-gripper-pipeline.html` is an unrelated
standalone design page that was already untracked; it is not part of this work.

---

## New helper functions

### `src/flexivtrainer/data/hub.py` (new module)

**Identity and validation**

| Function | Why it exists |
|---|---|
| `HubRef` (dataclass) | Carries `repo_id` + optional `revision` as one validated value, so the pair can't drift apart between call sites. |
| `validate_repo_id()` | **The security boundary.** Accepts only `owner/name`. Runs before any part of the string touches a path, so a traversal like `../etc` can never reach the filesystem. |
| `validate_revision()` | Same, for branch/tag/sha. |
| `parse_hub_ref()` | Convenience wrapper returning a validated `HubRef`. |
| `sanitize_repo_id()` | Turns `lerobot/pusht` into `lerobot__pusht-3f2a9c11`. Needed because a dataset directory **name** is reused to build the B-spline conversion output directory — an unsanitized `/` would become a path separator. The sha256 suffix keeps truncated slugs distinct and gives each revision its own directory. |
| `is_hub_repo_id()` | Distinguishes a real Hub id from the synthetic `local/<name>` ids this app generates for its own datasets, so `local/*` is never sent to the Hub. |

**Cache and auth**

| Function | Why it exists |
|---|---|
| `hub_cache_root()` / `hub_cache_dir()` | Resolves `.local/cache/hub/{datasets,checkpoints}/<slug>`, asserting containment. Reuses the previously-unused `StorageConfig.cache_root`. |
| `hub_token()` | Resolves a token from config → `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` → `None` (falls through to the `hf auth login` cache). |
| `_fetch_lock()` | Per-repo lock so two concurrent loads of the same repo serialize instead of racing into one `local_dir`. |
| `_read_marker()` / `_write_marker()` | The completion marker `.flexivtrainer_hub.json`, written **only on success** — so an interrupted download is never mistaken for a usable cache entry. |
| `_require_hub_enabled()` | Honors `HubConfig.enabled`. |

**Fetching**

| Function | Why it exists |
|---|---|
| `fetch_dataset_metadata()` | Downloads **only** `meta/` via `LeRobotDatasetMetadata(root=...)`, which pulls with `allow_patterns="meta/"`. This is the key primitive: it gives the training pre-flight a real directory to read while leaving gigabytes of video to the training subprocess. |
| `fetch_checkpoint_snapshot()` | Downloads a checkpoint in full. Checkpoints are small relative to datasets and every metadata helper reads them **by path**, so a full snapshot keeps all of them working unchanged. |
| `_has_model_config()` | Post-download check mirroring `_checkpoint_model_dir`'s root-vs-`pretrained_model/` layout handling, so a malformed repo fails fast. |

**Error reporting** (all added while debugging real failures)

| Function | Why it exists |
|---|---|
| `HubError` + `HubNotFoundError` / `HubAuthError` / `HubUnavailableError` / `HubDatasetNotTaggedError` | Typed errors so routes map them to distinct HTTP codes instead of a blanket 409/502. |
| `describe_hub_error()` | Turns a Hub exception into an actionable message. |
| `_classify_hub_error()` | Picks the typed error class. |
| `_safe_str()` | **Bug fix.** `HfHubHTTPError.__str__` dereferences `response.headers` and can itself raise. Without this the error path crashed and reported its own secondary failure, burying the real cause. |
| `_status_code()` | Same defensiveness for a `response` attribute that may raise. |
| `_describe_untagged_dataset()` | Recognizes LeRobot's "dataset has no codebase-version tag" failure. LeRobot raises `RevisionNotFoundError` without the keyword-only `response` that `huggingface_hub>=1.0` requires, so the real cause arrives as an unrelated `TypeError`. This translates it into "tag the repo with `create_tag(...)`". |

### `src/flexivtrainer/rollout/checkpoint.py`

| Function | Why it exists |
|---|---|
| `resolve_hub_checkpoint()` | Materializes a Hub checkpoint, then hands it to the **existing, unmodified** `resolve_checkpoint_path`. |
| `checkpoint_sidecar_action_names()` | Reads `action_names.json` from the checkpoint. Makes a checkpoint self-describing. |
| `checkpoint_config_action_names()` | Reads `output_features.action.names` from `config.json` when present. |
| `_hub_dataset_action_names()` | Last-resort recovery: fetches the training dataset's `meta/` from the Hub. |
| `_dataset_action_names()` | Extracted from the old loop body inside `checkpoint_action_names` so the same validation serves local and Hub datasets. |
| `_validated_action_names()` | Shared validation (non-empty, all strings, unique) for every tier above. |

### `src/flexivtrainer/jobs/train_policy.py`

| Function | Why it exists |
|---|---|
| `_resolve_hub_dataset()` | Fetches `meta/` and returns a local root, so every existing pre-flight read works unchanged. |
| `_resolve_hub_checkpoint()` | Snapshots a Hub checkpoint then applies the **same** `config.json` + `model.safetensors` validation as the local resolver. |
| `inspect_hub_checkpoint()` | Hub entry point for fine-tune inspection. |
| `_inspect_resolved_checkpoint()` | Extracted body of the old `inspect_checkpoint`, so local and Hub share identical inspection logic. |
| `_dataset_action_names()` | Reads axis names from a dataset for the sidecar. |
| `_checkpoint_model_dirs()` | Extracted from `_sync_gripper_command_metadata`; finds every saved checkpoint dir. |
| `_sync_action_names()` | Writes `action_names.json` beside each saved checkpoint at training time. |

### `src/flexivtrainer/api/routes/rollout.py`

| Function | Why it exists |
|---|---|
| `_layout_warning()` | Decides server-side whether a checkpoint will actually fail to start, by comparing its action width against the configured arm count. Replaces a vague UI-side caveat. |

---

## What is reused vs. newly added

### Reused unchanged — verified untouched in the diff

These are the load-bearing pieces of the old code, deliberately not modified:

- **`resolve_checkpoint_path()`** (`rollout/checkpoint.py`) — still does string
  prefix matching, per-segment `iterdir` walk, and symlink-escape rejection.
  Because the Hub cache lives at `.local/cache/hub/`, **inside** the storage
  root, a downloaded checkpoint passes this validator honestly. Downloaded repo
  content therefore gets the *same* symlink-escape protection as local content —
  a security gain over adding a parallel resolver that bypassed it.
- **`TrainingService._resolve_dataset()`** — unchanged; still requires
  containment in `merged_root`. The Hub path is a *sibling* method.
- **`TrainingService._resolve_checkpoint()`** — unchanged; the Hub variant
  applies the same two file checks to the snapshot.
- **`_resolve_output_dir()`** — unchanged.
- **`canonical_action_names()`** (`rollout/executors/waypoint.py`) — unchanged.
  Still the fallback layout guesser recognizing only widths 13 and 19 per arm.
- **`_default_policy_loader()`** — unchanged. LeRobot's `from_pretrained`
  already resolves Hub ids, but it receives a local snapshot path here.
- **All checkpoint metadata helpers** (`checkpoint_action_output_dim`,
  `checkpoint_gripper_command_metadata`, `checkpoint_image_resolutions`,
  `_checkpoint_policy_type`, `_checkpoint_target_hz`, `_checkpoint_task`) —
  unchanged, because they receive a local path after materialization.
- **The whole training pre-flight** (`_dataset_gripper_command_metadata`,
  `_bspline_dataset_contract`, `_rgb_only_policy_input_features`,
  `_validate_checkpoint_dataset`) — unchanged, because `_resolve_hub_dataset`
  hands them a real directory containing `meta/`.

### Modified in place

- **`checkpoint_action_names()`** — kept its signature and existing local-dataset
  loop; gained keyword-only `settings` / `override` and the tier ordering.
  Existing callers still work.
- **`inspect_checkpoint()`** — body extracted to `_inspect_resolved_checkpoint`;
  behavior identical.
- **`_build_training_command()`** — gained an optional `hub_ref`. With it, passes
  the real `repo_id` and **omits `--dataset.root`** so LeRobot fetches bulk data
  itself; without it, the old behavior is byte-for-byte unchanged.
- **`_training_env()`** — injects `HF_TOKEN` when configured, since the
  subprocess is what downloads a private dataset.
- **`RolloutService.start()` / `TrainingService.start()`** — gained keyword-only
  source arguments, defaulting to `"local"`. Existing calls are unaffected.
- **`_sync_gripper_command_metadata()`** — its directory-scan loop was extracted
  to `_checkpoint_model_dirs()` and is now shared with `_sync_action_names()`.

---

## Why the action-name recovery chain exists

This was the one genuinely hard problem, and it is a **safety** issue rather than
a convenience one.

`checkpoint_action_names()` recovered axis names from the training dataset's
local `meta/info.json`. A Hub checkpoint has no local sibling dataset, so it
returned `None`, and `_preflight_waypoint` fell back to `canonical_action_names`,
which **raises for any gripper-bearing width**. Hub checkpoints with grippers
could not start at all.

Recovery is now tried in descending order of authority:

1. Explicit `action_names` on the start request
2. `action_names.json` sidecar in the checkpoint
3. `output_features.action.names` in `config.json`
4. A local copy of the training dataset (the original path)
5. The training dataset's `meta/` from the Hub

If every tier misses, the rollout **raises rather than infers**. A guessed
gripper axis would send a position command to a force channel on live hardware,
so failing loudly is the correct behavior.

`action_names.json` (written by `_sync_action_names` at training time) is the
durable fix: it makes checkpoints self-describing and also repairs the
pre-existing *local* case where the training dataset is deleted or renamed.

**Note:** existing checkpoints predate this and have no sidecar. It applies to
future training runs only.

---

## Bugs found and fixed during the session

1. **Toggle appeared dead.** `renderRollout()` returns early when its render key
   is unchanged (a guard that stops the 1 s poll from flashing the camera feed).
   Hub state was missing from that key, so state changed but nothing repainted.
2. **`str(exc)` could crash the error path**, masking real causes — see
   `_safe_str`.
3. **Untagged dataset reported as a nonsense 502.** Now reported as a 400 with
   the `create_tag` fix.
4. **Empty preview page** for Hub datasets (only `meta/` is local, so there are
   no frames). The flow now skips that step in both directions.
5. **`403 output must be within training root`.** `getTrainingOutputDir()` took
   the run name from `mergedDatasetPath`, empty for Hub, so it sent the bare
   training root. Now derived from a sanitized repo id.
6. **Misleading gripper warning.** Replaced with a server-computed verdict that
   compares action width against the configured arm count.

---

## Known limitations

- **Rollout from a Hub checkpoint is not fully verified.**
- **B-spline + Hub dataset is refused** with a clear error. Conversion is a
  separate subprocess reading parquet directly, which a meta-only fetch never
  materializes.
- **Dataset download happens inside the training subprocess**, so there is a
  silent period before training starts and progress appears only in the training
  log. A network drop mid-training fails the run with a LeRobot traceback.
- **No cache eviction.** `.local/cache/hub/` grows per repo and per revision.
- **Checkpoint download is synchronous** — `POST /rollout/start` blocks.
- **Hub dataset repos must carry a codebase-version git tag** (a LeRobot
  requirement).

---

## Verification

- `pytest tests/` — **597 passed, 4 skipped** (baseline before this work: 453).
- `tests/test_path_security.py` **unmodified** and passing — the check that
  security was not weakened.
- `ruff check src tests` — **17 issues, all pre-existing**; none added.
- Live Hub calls verified: checkpoint download + validation
  (`lerobot/diffusion_pusht_keypoints`), metadata-only dataset fetch
  (`lerobot/pusht`, 200 KB on disk), and `flexivrobotics/push_t_dual` metadata
  (217 episodes, 26 action axes, dual-arm).
- All other tests monkeypatch the network.
