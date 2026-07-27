# Rollout Performance & Smoothness — Remaining Fixes

Follow-up work from the investigation into an ACT rollout that ran at ~20 Hz instead of
30 Hz with visibly jerky motion. Fix 1 (resize rollout frames to the checkpoint's trained
resolution) has landed; this document covers fixes 2 through 4.

Reference checkpoint:
`.local/training/merged_20260724_174104_rgb-act_20260724_185117/checkpoints/156000/pretrained_model`
(ACT, 3 cameras, bimanual, 240x320, `chunk_size=60`, `temporal_ensemble_coeff=0.01`).

---

## Measured baseline

All numbers from the reference checkpoint on an RTX 5090 (torch 2.10+cu128, native
`sm_120`, so no PTX-JIT penalty). These drive every decision below.

| Configuration | Forward pass |
|---|---|
| 240x320 fp32 | **8.0 ms** |
| 480x640 fp32 | **9.6 ms** |
| 64x64 fp32 | 7.7 ms |
| batch=4 @ 240x320 | 9.2 ms |
| `select_action` + `.cpu()` | 8.3 ms |
| **CPU-side enqueue only, no GPU sync** | **7.94 ms** |
| autocast bf16 | 10.1 ms (slower) |
| fp32 + TF32 | 8.05 ms (unchanged) |

Two conclusions that shape everything else:

1. **The forward pass is 100% CPU kernel-launch-bound.** Enqueue time (7.94 ms) equals
   total wall time (8.0 ms) — the GPU is idle throughout. There are **1721 aten ops per
   forward**, each ~5 us of Python/dispatch/launch overhead.
2. **Image resolution is nearly irrelevant to speed.** Quartering the pixels buys 1.6 ms;
   a 64x64 image is barely faster than 640x480. This is why the earlier 640→320 training
   change produced no speedup. Fix 1 was a *correctness* fix, not a performance one.

Under GIL contention, same checkpoint, one competing CPU-bound Python thread:

| Condition | Forward pass |
|---|---|
| No contention | **10.3 ms** |
| 1 busy thread, `switchinterval=0.005` (default) | **19,551 ms** |
| 1 busy thread, `switchinterval=0.001` | 2,433 ms |
| 1 busy thread, `switchinterval=0.0001` | 315 ms |

A ~1900x inflation from a single saturated thread, scaling linearly with the switch
interval — the signature of 1721 GIL handoffs per forward. Real competing threads are
bursty rather than saturated, which is exactly why the observed cost is a diluted
8 ms → 35-50 ms with 210 ms spikes.

---

## Fix 2 — Serve without temporal ensembling (highest leverage)

**Do this one first.** It addresses the frequency *and* the smoothness without touching
threading, and it is a load-time config change rather than a redesign.

### Problem

`temporal_ensemble_coeff = 0.01` is baked into the checkpoint. In
`.venv/.../lerobot/policies/act/modeling_act.py:108-113`, `select_action` checks
`temporal_ensemble_coeff is not None` **first** and, when set, calls
`predict_action_chunk` unconditionally on every step — a full forward, every control
tick, returning exactly one action. 59 of the 60 predicted actions are discarded. Chunk
caching is impossible by construction.

Two consequences compound it:

- **The existing chunk-length knob is silently dead.**
  `SharedRolloutConfig.n_action_steps` already defaults to **16**
  ([src/flexivtrainer/policies/_shared.py:56](src/flexivtrainer/policies/_shared.py#L56)),
  and `_apply_n_action_steps`
  ([src/flexivtrainer/rollout/service.py:461-490](src/flexivtrainer/rollout/service.py#L461-L490),
  called at [service.py:188](src/flexivtrainer/rollout/service.py#L188)) writes
  `config.n_action_steps = 16`. But it never clears `temporal_ensemble_coeff`, so lerobot
  ignores the value entirely. It even logs `"Action chunk length overridden"`, which reads
  as confirmation that something took effect when nothing did.
- **The chunk reader looks for the wrong attribute.**
  [observations.py:52-53](src/flexivtrainer/rollout/observations.py#L52-L53) reads
  `getattr(policy, "_queues", None)`, but `ACTPolicy` stores its cache as
  `_action_queue` (singular) — see `modeling_act.py:98`. So `action_queue` is always
  `None`, `fresh` is always `True`, `tail` is always empty, and
  `_predict_action_chunk` always returns a **length-1** chunk regardless of policy state.

Net effect: one waypoint per inference, no buffer, a full 40 ms forward every tick.

### Change

In `_apply_n_action_steps` ([service.py:461](src/flexivtrainer/rollout/service.py#L461)),
when the policy config carries a non-`None` `temporal_ensemble_coeff` and the operator has
requested `n_action_steps > 1`, clear the coefficient (and drop the stale ensembler
object) before setting `n_action_steps`, so lerobot takes the action-queue path. Log the
trade explicitly — ensembling is being disabled in exchange for chunked execution, and
that is a behaviour change the operator should see in the rollout log.

Also fix the `_queues` → `_action_queue` lookup in `observations.py` so ACT's real cached
chunk is read. Prefer a small helper that resolves whichever attribute the policy family
exposes, since the other families (`diffusion`, `pi0`, `smolvla`, `multi_task_dit`, …) do
use `_queues`.

Note `policy.reset()` is called inside each runner
([waypoint.py:134](src/flexivtrainer/rollout/runners/waypoint.py#L134),
[bspline.py:236](src/flexivtrainer/rollout/runners/bspline.py#L236)) — after the override.
`ACTPolicy.reset` builds `_action_queue` with `maxlen=config.n_action_steps`, so the
override must land before `reset()`, which it currently does. Keep that ordering.

### Expected result

At `n_action_steps=16` and `replan_steps=16` (both already the defaults), inference runs
roughly every 8th-16th tick instead of every tick. The 40 ms forward is amortised and
hidden behind a committed 16-waypoint path, so the loop holds 30 Hz even with inference
cost unchanged.

### Trade-off to validate on hardware

Temporal ensembling is a smoothing mechanism; removing it removes that smoothing. With
`coeff=0.01` over a 60-step chunk the weights are near-uniform and skewed toward *older*
actions, which adds substantial command lag — so removing it should reduce lag, but it may
expose per-chunk discontinuities at replan boundaries. **Bench this before running on
hardware.** If discontinuities appear, the fix is blending at chunk boundaries in the
executor, not reinstating a full-forward-every-step ensembler.

---

## Fix 3 — Stop dropping every waypoint as stale

### Problem

This is the direct cause of the jerky motion, and it is independent of the frequency.
**`sched=0` on every single logged line** in the rollout output is the proof.

The arithmetic:

- `action_anchor_offset_steps = 1`
  ([_shared.py:65](src/flexivtrainer/policies/_shared.py#L65)) and `dt = 1/30`, so
  waypoint 0 targets `loop_start + 33.3 ms`
  ([waypoint.py:243-246](src/flexivtrainer/rollout/runners/waypoint.py#L243-L246)).
- `replace_waypoints` drops any waypoint whose `target_time <= now`
  ([executors/waypoint.py:119-120](src/flexivtrainer/rollout/executors/waypoint.py#L119-L120)).
- Dispatch happens ~42-47 ms after `loop_start` (inference alone is 35-50 ms).

33.3 ms < 42 ms, so **the only waypoint produced is always already stale and always
dropped.** `self._waypoints` becomes `[]`, `scheduled_count` becomes 0, and
`SendCartesianMotionForce` is never called for it. The arm receives commands only
sporadically.

The config comment at [_shared.py:64-65](src/flexivtrainer/policies/_shared.py#L64-L65)
states the intent — "offset >= 1 keeps waypoint 0 ahead of the past-filter (inference
latency would drop it)" — but `offset = 1` only buys 33.3 ms of lead against 42+ ms of
actual latency. The default is simply too small for this policy.

### Change

1. **Raise the lead time.** Either raise `action_anchor_offset_steps` to 2-3 (66-100 ms of
   lead), or derive the anchor from measured inference latency so it adapts instead of
   being hand-tuned per checkpoint. The field is already bounded `ge=0, le=8`, so 2-3 needs
   no schema change.
2. **Make the drop visible.** `replace_waypoints` currently discards stale waypoints
   silently. When it drops *all* of them, warn (rate-limited — this is a hot path). A
   silent `sched=0` is what let this go unnoticed; it should be loud.
3. Fix 2 largely subsumes this: with a 16-waypoint chunk, later waypoints are far enough
   ahead to survive the filter even if waypoint 0 does not. Do both anyway — the lead-time
   bug is real on its own and would resurface for any single-step policy.

---

## Fix 4 — Reduce GIL contention (only if still short after 2 and 3)

Everything runs in **one process sharing one GIL**: uvicorn, three `camera-acquire-*`
threads, the `rollout-policy-planner` thread, the waypoint executor thread, and the WebUI
poller. Because the forward pass is launch-bound with 1721 GIL handoffs, it is unusually
sensitive to any competing Python work.

Options, cheapest first:

1. **Throttle the status endpoint.** The WebUI polls `GET /rollout/status` every 330 ms
   ([web/app.js:6338-6344](src/flexivtrainer/web/app.js#L6338-L6344)) and
   `RolloutService.status()` ([service.py:96-114](src/flexivtrainer/rollout/service.py#L96-L114))
   copies and JSON-serialises up to 300 metric dicts and 2000 log strings **per call**, on
   a sync route that Starlette runs in a threadpool. This is the largest avoidable GIL hog.
   Send deltas or cap the payload.
2. **Disable depth during RGB-only rollout.** `use_depth` defaults to `True`
   ([config.py:42](src/flexivtrainer/config.py#L42)). For an RGB-only policy this is pure
   waste in the camera threads — dropping it removes per-frame work and bandwidth.
3. **CUDA graphs / `torch.compile(mode="reduce-overhead")`.** The real structural fix:
   collapses 1721 launches into a single graph replay, making the forward both faster and
   largely GIL-immune. Highest effort, highest ceiling. Input shapes are fixed every step,
   which is the ideal case. Verify numerics against eager before trusting it.
4. **`sys.setswitchinterval(0.0005)`** at rollout start — a cheap partial mitigation with a
   measured ~10x improvement under contention, but it raises context-switch overhead for
   every other thread. Treat as a stopgap, not a fix.
5. **Move inference to a separate process.** Fully sidesteps the GIL. Largest change; only
   worth it if 1-3 prove insufficient.

Not worth pursuing, with measurements: AMP/bf16 autocast (**slower** — 10.1 ms vs 8.0 ms),
TF32 (no change), lower resolution (1.6 ms), larger batches (GPU is already idle).
`cudnn.benchmark = True` is untested and plausibly helps the 3x ResNet18 convs slightly,
but cannot address launch-bound overhead.

---

## Suggested order

1. **Fix 2** — biggest win, config-level, fixes frequency and smoothness together.
2. **Fix 3** — small, independent, and prevents the staleness bug recurring.
3. Re-measure. If 30 Hz holds, stop.
4. **Fix 4** items 1-2 if short; item 3 only if a structural fix is warranted.

## Verification

- Bench inference cadence before/after Fix 2 with the reference checkpoint — confirm
  `inference=` no longer appears on every logged step.
- Confirm `sched=` is non-zero in the rollout log after Fix 3. This is the single clearest
  smoothness signal; `sched=0` means the arm is receiving nothing.
- Watch `freq=` approach 30.0/30.0 Hz and check `infer_max` spikes shrink.
- Compare commanded vs measured pose lag (`cmd_xyz` vs `meas_xyz`) before and after — the
  reference log shows measured trailing commanded by roughly 15 steps (~0.5 s).
- Run `.venv/bin/python -m pytest tests/ -q` (295 tests pass at the time of writing) and
  `.venv/bin/ruff check src tests` on changed files. Note the repo is **not**
  `ruff format` clean overall; check only the files you touch.
