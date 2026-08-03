# ACT Rollout Compilation: the sequential-checkpoint failure

Why running two ACT checkpoints back to back in one backend process failed with a
bare `AssertionError`, why four reasonable fixes did not work, and what the fix
actually is.

Investigated and fixed 2026-08-03, against `torch 2.10.0+cu128` on an RTX 5090.

---

## 0. The one-paragraph version

ACT rollouts compile with `torch.compile(model, mode="reduce-overhead")`, which
routes through torch's CUDA-graph-trees runtime. That runtime keeps its per-device
bookkeeping in a module-level `threading.local()` which is populated **only on the
thread that first executes the module body**. Every rollout runs on a brand-new
planner thread, so the first rollout in a process works and every later one dies on
a message-less `assert` deep inside torch. It has nothing to do with the action
dimension — the same checkpoint run twice fails identically. The fix gives each
planner thread its own CUDA-graph bookkeeping before anything compiles, and tears
it down on that same thread when the rollout ends.

## 1. Symptom

Start an ACT rollout, stop it, start a second ACT checkpoint in the same backend
process:

```
RuntimeError: Failed to compile ACT model: AssertionError
```

- The first rollout compiled and ran at 60 Hz.
- The second failed, symmetrically in either order (38D-first or 26D-first).
- Failed runs recorded **zero** rollout metrics, so the failure happened before the
  planner loop ever appended a sample.
- Eager execution worked for either checkpoint, but at ~19 ms per inference against
  a 16.67 ms budget at 60 Hz, which produced stale single-action waypoints and
  visible robot jitter. Silently degrading to eager was not acceptable.

The error text is useless, and that is itself a bug: a bare `assert cond` with no
message produces an exception whose `str()` is empty, and
`describe_exception` ([`console.py:121`](src/flexivtrainer/observability/console.py))
returns just the type name when the message is blank. So a rich failure deep in a
dependency arrived as the seven-character string `AssertionError`, with no traceback
recorded anywhere.

## 2. "Different action dimensions" was the wrong theory

The two checkpoints differed in action width — 38D (pose + twist + wrench per arm)
versus 26D (pose + twist per arm) — so the natural hypothesis was that torch had
cached a graph specialized to the first output width. It had not.

The falsifying experiment is one line of operator work: **run the same checkpoint
twice.** It fails the second time too.

A four-rollout reproduction on the real GPU, each rollout on a fresh thread, with
alternating widths, makes it unambiguous:

| Rollout | Action dim | Result |
|---|---|---|
| #0 | 38 | OK |
| #1 | 26 | `FAIL AssertionError: AssertionError()` |
| #2 | 38 | `FAIL AssertionError: AssertionError()` |
| #3 | 26 | `FAIL AssertionError: AssertionError()` |

Rollout #2 has the **same** action width as rollout #0, which had just succeeded,
and it still fails. The variable is not the model. It is which thread is running.

## 3. The actual mechanism

In `torch/_inductor/cudagraph_trees.py`, the module body runs this once, at import:

```python
local = threading.local()                                            # :279
local.tree_manager_containers = {}                                   # :282
local.tree_manager_locks = defaultdict(threading.Lock)               # :283
...
torch._C._stash_obj_in_tls("tree_manager_containers", local.tree_manager_containers)  # :293
torch._C._stash_obj_in_tls("tree_manager_locks", local.tree_manager_locks)            # :294
```

Both attributes land on the **importing thread's** thread-local, and the C++ TLS
stash exists so that, per the comment above it, the objects "will be copied over as
TLS when new threads are created" — by the autograd engine. It does not propagate
to a plain `threading.Thread`.

Every lookup goes through:

```python
def get_obj(local: Any, attr_name: str) -> Any:            # :323
    if hasattr(local, attr_name):
        return getattr(local, attr_name)
    else:
        assert torch._C._is_key_in_tls(attr_name)          # :327  bare, no message
        return torch._C._get_obj_in_tls(attr_name)
```

On any thread other than the importer, `hasattr` is `False` and
`_is_key_in_tls` is `False`, so line 327 raises `AssertionError("")`. Verified
directly, without a GPU or a model:

```
main thread      hasattr(local)=True   _is_key_in_tls=True   get_obj -> OK
fresh thread #1  hasattr(local)=False  _is_key_in_tls=False  get_obj -> AssertionError  str(exc)=''
fresh thread #2  hasattr(local)=False  _is_key_in_tls=False  get_obj -> AssertionError  str(exc)=''
```

Now map that onto this repo. `WaypointRunner.start()` creates a **new**
`threading.Thread` for every rollout, and `RolloutService.start()` builds a new
runner each time. So:

- **Rollout #1** — nothing has imported `cudagraph_trees` yet. The planner thread
  imports it lazily (via `torch.compiler.cudagraph_mark_step_begin()`, whose body is
  `from torch._inductor import cudagraph_trees`), so the module body executes *on
  that planner thread* and `local` is populated for it. Everything works.
- **Rollout #2** — a different planner thread. The module is already in
  `sys.modules`, so the body does not re-run, `local` is empty for this thread, and
  the first lookup asserts.

## 4. Two call paths reach the same assert

This is the least obvious part of the investigation, and it is why the symptom moved
around as fixes were attempted.

**Path A — the first cudagraph-captured forward.** Inductor calls `get_container()`
(`cudagraph_trees.py:331`), which calls `get_obj`. This is the original baseline
failure: it fired at the first prediction.

**Path B — `torch.compiler.reset()` itself.** `torch._dynamo.reset()` does not
touch cudagraph trees in its own body, which is why reading it suggests the reset is
harmless. But it calls `_reset_guarded_backend_cache()`
(`torch/_dynamo/eval_frame.py:356`), which iterates a **process-global** dict
`cached_backends` (`:144`, populated at `:731` by *any* thread on every
`torch.compile()` call) and calls `.reset()` on each entry. For a
`reduce-overhead` backend that is `_TorchCompileInductorWrapper.reset()`
(`torch/__init__.py:2447`), which calls `reset_cudagraph_trees()` — which calls
`get_obj` on **whatever thread is currently resetting**.

Reproduced verbatim on the GPU:

```
File "torch/compiler/__init__.py", line 66, in reset -> torch._dynamo.reset()
File "torch/_dynamo/__init__.py", line 154, in reset -> _reset_guarded_backend_cache()
File "torch/_dynamo/eval_frame.py", line 360, in _reset_guarded_backend_cache -> backend.reset()
File "torch/__init__.py", line 2454, in reset -> reset_cudagraph_trees()
File "torch/_inductor/cudagraph_trees.py", line 307, in reset_cudagraph_trees
    container_dict = get_obj(local, "tree_manager_containers")
File "torch/_inductor/cudagraph_trees.py", line 327, in get_obj
    assert torch._C._is_key_in_tls(attr_name)
AssertionError
```

This explains the asymmetry between runs. On rollout #1 `cached_backends` is empty,
so the reset is a no-op and cannot fail. On rollout #2 it holds rollout #1's
backend, so the reset trips the assert **before `torch.compile()` is even reached**.
The `torch.compiler.reset()` call that had been added as a fix was itself the crash
site.

## 5. Why each earlier attempt could not have worked

| Attempt | Why it could not work |
|---|---|
| Call `torch.compiler.reset()` before compiling | It reaches `reset_cudagraph_trees()` on the unseeded planner thread — this *is* the assert (path B above), not a cure for it. |
| Call `cudagraph_mark_step_begin()` before each inference | Its entire body is `MarkStepBox.mark_step_counter -= 1`. It touches no thread-local and invalidates nothing. |
| Move `torch.compile()` into the planner thread | The planner thread is a *new* thread every rollout, so moving work onto it moves work onto the thing that is broken. |
| Hard-stop on compile failure | Correct and safe, but it addresses the blast radius, not the cause: the second checkpoint stays unusable. |
| Fall back to eager | Sidesteps cudagraphs entirely, hence no crash — and 19 ms inference against a 16.67 ms deadline, hence jitter. |

## 6. The fix

[`src/flexivtrainer/rollout/_cudagraph_state.py`](src/flexivtrainer/rollout/_cudagraph_state.py)
gives the calling thread its own cudagraph bookkeeping:

- `seed_thread_local_state()` (`:70`) assigns fresh `tree_manager_containers` and
  `tree_manager_locks` onto `cudagraph_trees.local` for the current thread.
- `teardown_rollout_gpu_state(device, *, cudagraphs_seeded)` (`:82`) drops the trees
  and cached allocator memory: `gc.collect()`, `torch.cuda.synchronize()`,
  `reset_cudagraph_trees()`, `torch.cuda.empty_cache()`. Best-effort — it never
  raises, so cleanup cannot mask the error that ended the rollout.

Two details are load-bearing:

**Seeding must precede `compile_model`.** Because of path B, seeding after
`torch.compiler.reset()` still fails. In
[`runners/waypoint.py`](src/flexivtrainer/rollout/runners/waypoint.py) the seed is at
`:186`, immediately before `self._prepare_policy(policy)` at `:190`. Measured both
ways: seeding after the reset gives rollout #0 OK / #1 FAIL; seeding before gives
both OK.

**Containers are fresh per run, not shared.** `get_container()` creates a new
`TreeManagerContainer` per `(thread-local dict, device)`, with its own
`torch.cuda.graph_pool_handle()`. Fresh-per-thread means rollout N can never read or
shut down rollout N±1's graphs or memory pool — which is the isolation the earlier
global-reset attempts were reaching for and never achieved.

Teardown runs in the planner thread's `finally` (`:342`), after dropping every
reference that could alias a CUDA-graph output buffer, since the pool cannot be
reclaimed while anything still points into it.

Also added: a `compile_mode` knob on `act.RolloutConfig`
([`act.py:75`](src/flexivtrainer/policies/act.py)). Setting
`FLEXIV_TRAINER_POLICIES__ACT__ROLLOUT__COMPILE_MODE=default` keeps Inductor but
drops CUDA graphs, sidestepping this whole class of problem at some speed cost.

## 7. Diagnosability

The bug was hard to find mainly because the error carried no information.

- `describe_traceback()` ([`console.py:129`](src/flexivtrainer/observability/console.py))
  formats a full stack; the one-line `describe_exception` still feeds the UI and the
  status endpoint, while both runners now send the full traceback to the server
  console (`waypoint.py:330`).
- `compile_model` ([`act.py:93`](src/flexivtrainer/policies/act.py)) splits its
  single `try` into two — reset at `:103`, compile at `:114` — so a failure names
  which call raised. Under the old single block, path A and path B were
  indistinguishable.

## 8. Lifecycle bugs fixed alongside

These are independent of the compile bug but could each produce jitter on a second
rollout, so they were fixed in the same pass.

**Shared stop event.** `RolloutService` created one `threading.Event` for the whole
process and `.clear()`ed it inside `start()`. A planner thread's loop condition is
`while not self._stop_event.is_set()`, so clearing it *un-stopped* any planner from
the previous run that had not yet exited. Now each run gets a fresh event
([`service.py:261`](src/flexivtrainer/rollout/service.py)); a zombie keeps the old
one, which stays set.

**`start()` overlapping a live planner.** `stop()` joins with a 2 s timeout and
previously marked the service idle even when the join timed out, so the next
`start()` sailed past the guard while the old planner still owned the GPU and the
robots. A Python thread cannot be killed and `torch.compile` is not interruptible,
so `start()` now refuses (`:167`) with an actionable message.

**`stop()` reporting idle when it was not.** It now reports a distinct `"stopping"`
status (`:130`, `:445`) while a planner is still alive, and leaves robot release to
that planner's own `finally`.

**Cross-run robot release.** `_release_robots` read-and-cleared the *service's*
robot list, so a late planner reaching its `finally` could stop the **next** run's
robots. Release is now scoped to each run's own list (`_make_release_robots`,
`:705`).

**Per-tick whole-device sync.** `observations._cuda_sync`
([`observations.py:128`](src/flexivtrainer/rollout/observations.py)) called
`torch.cuda.synchronize()` on every tick, inside the 16.67 ms budget, purely so the
`infer_ms` metric attributed GPU time — and it swallowed every exception, which is
exactly where an async kernel fault would first surface. It is now behind
`debug_timing` (default off, `waypoint.py:164`/`:249`) and warns instead of
discarding. With it off, `infer_ms` measures launch time; the GPU wait shows up in
`to_list`, which already syncs the tensors it copies.

## 9. Verification

Three real checkpoints, run sequentially in one process, each on its own fresh
thread, 120 ticks each:

| Rollout | Action dim | Result | mean | p95 | reserved after teardown |
|---|---|---|---|---|---|
| #0 | 38 | OK | 5.47 ms | 5.43 ms | 6 MiB |
| #1 | 26 | OK | 4.97 ms | 5.28 ms | 8 MiB |
| #2 | 38 | OK | 5.44 ms | 5.83 ms | 6 MiB |

All comfortably inside the 16.67 ms budget, against ~19 ms for eager. First tick per
rollout is 1.3-2.1 s, which is compile plus graph capture. GPU memory is flat across
runs; the unfixed reproduction left 122 MiB reserved after its one successful
rollout.

Test suite: 388 passing, up from 380.

[`tests/test_cudagraph_state.py`](tests/test_cudagraph_state.py) pins the mechanism
with no GPU: an unseeded thread raises `AssertionError` **with an empty message**,
three sequential seeded threads all succeed, and seeding one thread does *not* make
an unseeded sibling work — that last one is what distinguishes real per-thread
isolation from a shared-dict fix that would merely silence the assert.

This matters because every pre-existing compile test monkeypatches `torch.compile`,
`torch.compiler.reset`, and `cudagraph_mark_step_begin` to recorder lambdas. They
assert call *ordering* and can never execute torch's real thread-local code, so CI —
which has no GPU and installs a narrower lerobot extras set than `pyproject.toml` —
could not have caught this and still cannot catch it without the new file.

## 10. How to re-verify

```
.venv/bin/python scripts/probe_act_cudagraph_reuse.py \
    --checkpoint .local/training/<run-38d>/checkpoints/<step> \
    --checkpoint .local/training/<run-26d>/checkpoints/<step> \
    --checkpoint .local/training/<run-38d>/checkpoints/<step> \
    --device cuda:0 --ticks 120
```

No robot and no HTTP server. It calls the same production helpers in the same order
`WaypointRunner._run` does, one fresh thread per checkpoint, and prints per-run
latency, the full traceback on failure, and GPU memory after each teardown. Repeat a
checkpoint to prove N sequential rollouts work rather than only two.

**A running server holds the old code in memory.** The package resolves to `src/`,
so no reinstall is needed, but the backend must be restarted to pick up changes —
otherwise the fix appears not to work.

## 11. If you upgrade torch

The fix assigns to a private torch attribute, so it is guarded rather than trusted.
`_import_cudagraph_trees()` (`_cudagraph_state.py:52`) raises
`CudagraphTreesUnavailable` (`:48`) if `cudagraph_trees` loses `local` or
`reset_cudagraph_trees`, or if `local` stops being a `threading.local` — that last
check matters because an attribute of a different type would pass `hasattr` while
silently ceasing to be per-thread. Failure is loud and degrades to today's behavior
(compilation fails visibly), never to a silent regression.

Worth re-reading on upgrade: whether `_reset_guarded_backend_cache` still walks a
process-global registry, and whether `get_obj`'s bare assert has gained a message.

Known rough edge, unfixed: the web UI gates on `status === "running"`
(`web/app.js:3186`, `:5048`), so during the new `"stopping"` status the Start button
looks enabled. Pressing it is harmless — the backend refuses with "The previous
rollout planner has not exited yet" — but treating `"stopping"` like `"running"` in
those two checks would make it coherent.

## Source map

| Concern | Location |
|---|---|
| Per-thread seed and teardown | `src/flexivtrainer/rollout/_cudagraph_state.py:48,70,82` |
| Seed site, before compile | `src/flexivtrainer/rollout/runners/waypoint.py:186` |
| Compile invocation | `src/flexivtrainer/rollout/runners/waypoint.py:190` |
| Teardown in planner `finally` | `src/flexivtrainer/rollout/runners/waypoint.py:342` |
| Traceback on planner crash | `src/flexivtrainer/rollout/runners/waypoint.py:330` |
| `debug_timing` gate | `src/flexivtrainer/rollout/runners/waypoint.py:164,249` |
| `compile_model`, split try | `src/flexivtrainer/policies/act.py:93,103,114` |
| `compile_mode` knob | `src/flexivtrainer/policies/act.py:75` |
| `start()` refusal | `src/flexivtrainer/rollout/service.py:167` |
| `"stopping"` status | `src/flexivtrainer/rollout/service.py:130,445` |
| Per-run stop event | `src/flexivtrainer/rollout/service.py:261` |
| Per-run robot release | `src/flexivtrainer/rollout/service.py:705` |
| `uses_cuda_graphs` gate | `src/flexivtrainer/rollout/service.py:228` |
| `_cuda_sync` | `src/flexivtrainer/rollout/observations.py:128` |
| `describe_exception` / `describe_traceback` | `src/flexivtrainer/observability/console.py:121,129` |
| Regression tests | `tests/test_cudagraph_state.py` |
| Manual probe | `scripts/probe_act_cudagraph_reuse.py` |

Torch internals, `torch 2.10.0+cu128`:

| Concern | Location |
|---|---|
| Thread-local + TLS stash | `torch/_inductor/cudagraph_trees.py:279-294` |
| `reset_cudagraph_trees` | `torch/_inductor/cudagraph_trees.py:304` |
| `get_obj`, bare assert | `torch/_inductor/cudagraph_trees.py:323,327` |
| `get_container` | `torch/_inductor/cudagraph_trees.py:331` |
| Process-global backend registry | `torch/_dynamo/eval_frame.py:144,731` |
| `_reset_guarded_backend_cache` | `torch/_dynamo/eval_frame.py:356` |
| `_TorchCompileInductorWrapper.reset` | `torch/__init__.py:2447` |
