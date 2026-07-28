# Rollout Inference Speedup — ACT now, diffusion next

Companion to `ROLLOUT_PERFORMANCE_FIXES.md` (depth-alignment work, already landed) and
`POLICY_INFERENCE_BACKEND_PLAN.md` (the ONNX/backend design). This document records what was
**measured** on 2026-07-28 and what follows from it.

Reference checkpoint throughout:
`.local/training/merged_20260724_174104_rgb-act_20260724_185117/checkpoints/156000/pretrained_model`
— ACT, 240x320, 3 cameras, bimanual, `chunk_size=60`, temporal ensembling on.
Hardware: RTX 5090, torch 2.10.0+cu128, CUDA 12.8.

---

## Motivation

Not ACT for its own sake. If ACT only just holds 30 Hz, a diffusion action head — or any
larger policy — cannot. The goal is headroom.

---

## Two independent problems (only the second is about speed)

### 1. At 60 Hz the arm is barely being commanded

From the 12:01 log: **`sched=0` on 32 of 33 logged steps**, and measured pose lagging
commanded by up to 40 mm (step 690: `cmd_x=0.472` vs `meas_x=0.512`).

This is arithmetic, not performance:

| Target | `dt` | waypoint-0 lead at `anchor=1` | vs ~22 ms dispatch |
|---|---|---|---|
| 30 Hz | 33.3 ms | 33.3 ms | survives |
| **60 Hz** | **16.7 ms** | **16.7 ms** | **dropped every tick** |

`target_times = loop_start + (k + anchor) * dt`
([runners/waypoint.py:248-251](src/flexivtrainer/rollout/runners/waypoint.py#L248-L251)) is
anchored to the tick's *start*, so the entire inference latency eats waypoint 0's lead.
`replace_waypoints` then discards anything with `target_time <= now`
([executors/waypoint.py:119-120](src/flexivtrainer/rollout/executors/waypoint.py#L119-L120)).

Because temporal ensembling is kept (`disable_temporal_ensemble: False`), ACT returns a
**length-1** chunk — waypoint 0 is the only one, so the whole chunk dies.

Two aggravating factors worth knowing:

- `_log_step` logs `action_lists[0]` as "commanded" regardless of whether it was ever
  dispatched, so the commanded trace in those logs is **aspirational**, not what the robot
  received.
- `WaypointExecutor._execute_loop` holds **indefinitely** on an empty list
  ([:167-184](src/flexivtrainer/rollout/executors/waypoint.py#L167-L184)) with no starvation
  signal — the robot silently freezes at its last pose. `SendCartesianMotionForce` is a
  non-real-time discrete API that tolerates gaps at the firmware level, so nothing complains.
  **This is why the same bug went unnoticed twice.**

**Faster inference is worthless while commands are being discarded. Fix this first.**

### 2. Inference is genuinely too slow for 60 Hz

~20 ms per tick against a 16.67 ms budget. See below.

---

## Measured baseline

| Lever | model core | full tick (numpy→action) | verdict |
|---|---|---|---|
| eager | 7.99–9.76 ms | 12.77 ms | baseline |
| `torch.set_num_threads(1)` | 9.66 ms | — | **+1.0% — skip** |
| `+ cudnn.benchmark = True` | 9.66 ms | — | **0% — skip** |
| `torch.compile(mode="default")` | 5.46 ms (1.46x) | — | hazard-free fallback |
| **`torch.compile("reduce-overhead")`** | **4.00 ms (2.0x)** | **7.31 ms (−43%)** | **recommended** |
| backbone 3×batch-1 → 1×batch-3 | 3.67 → 1.31 ms | — | deferred |

**Compile cost: 89 s cold, 3.0 s warm.** The 54 MB inductor cache lives at
`/tmp/torchinductor_$USER`, which is cleared on reboot — so the first rollout after a boot
looks like a 90 s hang unless `TORCHINDUCTOR_CACHE_DIR` is pointed somewhere persistent.

**Unexplained gap:** live is ~22 ms where the isolated full tick benches at 12.77 ms — **1.7x
that I could not reproduce**, even with simulated camera/status-endpoint contention. If the
compiled ratio holds live, 22 → ~12.6 ms, which fits 60 Hz. That is an extrapolation.

### Corrections to earlier claims

Both of these were mine, and both were wrong:

- I described the 8-thread torch CPU pool as "~1 core wasted" and implied a latency win.
  The wasted core is real (7 OpenMP workers spinning at ~14.5% each); the **latency benefit is
  +1.0%**. Do not re-try this expecting a speedup.
- `POLICY_INFERENCE_BACKEND_PLAN.md` cites ACT at "~35–50 ms during normal thread contention."
  Those numbers were measured with the depth-alignment bug active. Realistic contention
  without it costs **4%** (10.33 → 10.73 ms). The budget available to a larger policy is
  therefore much better than that document assumes.

### The parity gate in `POLICY_INFERENCE_BACKEND_PLAN.md` is the wrong unit

That document sets a `1e-4` **mean absolute error on normalized actions**. Compiled-vs-eager
measures **1.241e-4 mean / 5.767e-4 max** over 20 distinct observations — so it *fails* the
stated gate.

Against this dataset's action std (max 0.1486 m across translation dims), that is:

- mean error → **0.017 mm**
- max error → **0.077 mm**

Negligible against the 1–3 mm commanded-vs-measured tracking error already visible in the
logs. **The gate should be expressed in millimetres, not normalized units**, or it rejects
perfectly good optimizations.

TF32 is not the cause — disabling it made parity slightly *worse* (1.939e-4). This is fp32
reduction-order nondeterminism from inductor's fusion, which is expected and benign here.

### CUDA-graph output aliasing — real, but fails loudly

`reduce-overhead` uses CUDA graphs, so output buffers are reused. Holding an output across
the next forward raises:

```
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a
subsequent run. ... clone the tensor outside of torch.compile() or call
torch.compiler.cudagraph_mark_step_begin() before each model invocation.
```

This is the good outcome — loud, not silent corruption. Consequence:

- **Ensembling on (today):** safe. The ensembler consumes the output immediately and the
  postprocessor copies to CPU, so nothing is held.
- **Chunked execution (`disable_temporal_ensemble=True`):** ACT's `_action_queue` holds GPU
  tensors across forwards, so `compile_model=True` + chunking **will crash** without a
  `.clone()` of the chunk.

---

## Plan

### Step 1 — Restore commanding: **faster inference is the only fix**

My first plan here was wrong, and a pre-existing test caught it. Both proposed remedies —
anchoring `target_times` on inference completion, and raising `action_anchor_offset_steps` —
**break the 30 Hz case that currently works.**

Because `replace_waypoints` wipes the pending list on every replan, and `fresh=True` on every
tick under ensembling, waypoint 0 fires only if its due time lands in the window
`(now, next_replan)`:

| Regime | window | `a=1` on `loop_start` | `a=2` on `loop_start` | `a=1` on completion |
|---|---|---|---|---|
| 30 Hz, dt=33.3 ms, I=12 ms | (12.0, 45.3) ms | due 33.3 → **fires** | due 66.7 → never | due 45.3 → never |
| 60 Hz, dt=16.7 ms, I=22 ms | (22.0, 44.0) ms | due 16.7 → never | due 33.3 → fires | due 38.7 → fires |

Anything that pushes the due time *later* survives the behind-regime but overshoots the next
replan in the keeping-up regime. The one configuration that works while keeping up is the
existing `anchor=1` on `loop_start` — which is why 30 Hz has always worked and 60 Hz has not.

**The condition for waypoint 0 to fire is `I < dt`.** So `sched=0` at 60 Hz is not an anchor
bug to be tuned around; it is the frequency problem itself. Make inference faster than
16.7 ms and `anchor=1` starts firing again. That is Step 2.

A structural fix (don't wipe still-valid waypoints, or don't replan every tick) would remove
the coupling entirely — see the b-spline section — but it is not needed if `I < dt`.

Shipped instead: a rate-limited warning when a replan schedules nothing, so this failure can
never again be silent.

### Step 2 — Opt-in `torch.compile`

`compile_model: bool = False` on ACT's `RolloutConfig`, applied in `_default_policy_loader`
([rollout/checkpoint.py:76-101](src/flexivtrainer/rollout/checkpoint.py#L76-L101)) after
`.to(device)` / `.eval()`.

Compile **`policy.model`** (the `ACT` nn.Module), not `ACTPolicy` — `select_action` mutates a
Python deque and is not a pure graph. `ACT.forward` is compile-clean: no `.item()`, no
data-dependent control flow, every branch on config or `self.training` (static at inference).

lerobot already gates compile behind a `compile_model` flag for `diffusion`, `pi0`, and
`smolvla`. **ACT is the only family missing it**, so this follows the upstream pattern — and
it is the same seam the diffusion work will use.

Must get right: the `.clone()` for the chunked path, a persistent inductor cache dir, and a
warn-and-continue-eager fallback if compile raises.

### Step 3 — ONNX benchmark for ACT (parallel track)

Install `onnx`, `onnxscript`, `onnxruntime-gpu` as experiment-only dependencies. Export a
fixed-shape ACT wrapper (explicit positional tensors, batch 1, whole chunk out, pre/post
outside the graph) and benchmark three levels against the 4.00 ms compiled number: plain
`InferenceSession.run`, I/O-binding with persistent CUDA buffers, and I/O-binding + ONNX
CUDA graphs.

Gate: `CUDAExecutionProvider` active with no material CPU fallback, or the benchmark is
invalid. ONNX must beat 4.00 ms to justify a second inference implementation plus a
per-family export adapter.

Keep this under `scripts/` — do not add `src/flexivtrainer/rollout/inference/` until a
backend wins.

### Step 4 — Deferred: batch the per-camera backbone calls

`ACT.forward` loops `self.backbone(img)` once per camera, i.e. 3 batch-1 ResNet18 passes
instead of 1 batch-3 (3.67 → 1.31 ms). Numerically safe — `FrozenBatchNorm2d` uses fixed
running stats and convs are per-sample independent — but it requires all cameras at one
resolution, which `checkpoint_image_resolutions` does not guarantee, and it means patching
vendored lerobot.

`torch.compile` may already recover most of it. **Measure the two combined; never sum the
savings.**

---

## If inference stays spiky, reuse the b-spline runner

`bspline.py` already solves this, and the waypoint path is the only one still doing
synchronous in-loop inference:

- `ThreadPoolExecutor(max_workers=1, ...)` + a `Future` harvested only when `.done()`
  ([runners/bspline.py:250-252](src/flexivtrainer/rollout/runners/bspline.py#L250-L252)) — the
  loop never awaits inference. `max_workers=1` also gives single-owner access to
  `policy` / `_action_queue` / `temporal_ensembler` for free, which matters because
  **`_predict_action_chunk` is not re-entrant** (it mutates the policy's deque from outside
  the policy, unsynchronized).
- A separate observe cadence with catch-up-without-drift, and a genuinely independent
  executor thread at its own `control_hz`.
- Its executor **holds gracefully** by continuing to sample the existing spline, rather than
  the waypoint executor's silent freeze.

Adapting that handoff also requires splitting `target_hz`, which currently drives *both* the
planner tick period and the waypoint spacing
([runners/waypoint.py:138-140](src/flexivtrainer/rollout/runners/waypoint.py#L138-L140)).

---

## Landmine if chunked execution is re-enabled

`_predict_action_chunk`
([rollout/observations.py:89-95](src/flexivtrainer/rollout/observations.py#L89-L95)) re-runs
the **full postprocessor over the entire pending tail on every tick** — including a GPU→CPU
transfer — even on ticks that skip the network. Harmless today (ensembling means `tail` is
always empty and it early-returns at line 90), but it would eat much of chunking's benefit.
Cache the postprocessed chunk once per fresh inference and slice it.

---

## Verification

1. `.venv/bin/python -m pytest tests/ -q` — 317 passing at time of writing.
2. `.venv/bin/ruff check` on changed files only; the repo is not `ruff format` clean overall.
3. Parity asserted in **millimetres** over ≥20 recorded observations, not normalized units.
4. Test: `compile_model=True` + chunked execution does not raise the CUDA-graph error.
5. Test: a compile failure falls back to eager with a warning rather than aborting.
6. Live, in this order:
   1. Commanding fix alone at 60 Hz → `sched` non-zero, `cmd`/`meas` gap collapses. **Confirm
      the arm is actually being commanded before judging any speed work.**
   2. `compile_model=True` → read `inference=`. Expect ~20 → ~12 ms. **If it does not move,
      the 1.7x bench-vs-live gap is the real bottleneck** and the next step is py-spy stacks
      (needs sudo; `ptrace_scope=1` and no passwordless sudo on this machine), not a faster
      backend.
   3. Second rollout start costs ~3 s of compile, not 89 s.
7. ONNX track: p50/p95/p99 and peak GPU memory for all three levels, plus the active
   execution providers, against the 4.00 ms compiled baseline.

---

## Then

Once ACT holds 60 Hz with headroom, the same `compile_model` seam applies to diffusion —
the actual goal. Re-measure the diffusion checkpoint against its own p99 budget before
building any diffusion ONNX export path; a 267M-parameter model at 16 DDIM steps may already
clear a 267 ms replan budget in eager.
