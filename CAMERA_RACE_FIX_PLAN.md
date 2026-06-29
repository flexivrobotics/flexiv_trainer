# Camera Duplicate-Feed / 0-FPS Race — Fix Plan

Status: **plan only, no code changes made.**
Scope: `src/flexivtrainer/cameras/realsense.py` (`RealSenseService`).

## Symptom

When connecting two cameras (ego + wrist) with **two different serials** configured,
sometimes both UI panels show the **same image** and one shows **0 FPS**. It takes a
few connect/reconnect iterations to land on two correct, live 30-fps feeds.

A live backend status snapshot during the bug showed contradictory runtime state:

```
ego:   started=False, actual_serial=750612070842, fps=30.0,
       error="Camera serial 750612070842 is not detected"
wrist: started=True,  actual_serial=323622272310, fps=0.0,
       error="wait_for_frames cannot be called before start()"
```

`started=False` with `fps=30.0`, and `started=True` with a "before start()" error, are
both internally impossible for a single consistent state — proving the state is being
read/written mid-mutation.

## Root cause (confirmed by read-only investigation)

This is a **concurrency / state-corruption race**, NOT an assignment-logic bug.
(Two earlier attempts to fix assignment logic — `ensure_default_assignment` reservation
and a post-start bind check — did not help and were reverted.)

Three coupled defects:

1. **`_stop_runtime` releases `self._lock` mid-operation.**
   At `realsense.py:204-208` it does `self._lock.release()` → `thread.join(timeout=2.0)`
   → `self._lock.acquire()`. During that open window, another camera operation can run
   and mutate shared runtime state. Concurrent callers that reach the lock:
   - `GET /teleop/cameras/{name}/frame` — frontend polls ~30 Hz (very frequent).
   - `POST /teleop/status` — frontend polls ~2 Hz.
   - `set_device_serials` / `start_streams` / `stop_streams` / `set_active_locations`
     — user clicks (connect / swap serial / arm-mode change).
   All HTTP routes are sync `def`, so uvicorn runs them on a threadpool → genuinely
   concurrent OS threads contending for the one `self._lock`.

2. **The acquire loop uses a pipeline captured by value and keeps polling a dead one.**
   `_acquire_loop` (`realsense.py:401`) binds `pipeline = runtime.pipeline` once. After
   `_stop_runtime` stops/replaces the pipeline, the orphaned thread keeps calling
   `pipeline.wait_for_frames()` → `"wait_for_frames cannot be called before start()"`,
   and the UI keeps showing that camera's **last retained frame** (the visual
   "duplicate"), while its fps reads 0.

3. **`discover()` and `status()` touch the RealSense context / shared state without the
   lock.** `discover()` (`realsense.py:82`) enumerates `rs.context().devices` and is
   reachable while pipelines are starting/stopping → intermittent
   "serial not detected" flapping. `status()` (`realsense.py:370`) reads many runtime
   fields without the lock → a single status read can capture a half-updated runtime.

### Exact race interleaving (abridged)

1. ego started: `started=True`, `actual_serial=750...`, fps flowing.
2. User changes serial → `set_device_serials` (holds lock) → `_restart_started_cameras`
   → `_stop_runtime(ego)` → `stop_event.set()` → **`self._lock.release()`** (line 204).
3. In the open window, a frame request / the ego acquire thread runs and updates
   `runtime.fps` / caches a frame.
4. `_stop_runtime` re-acquires, calls `pipeline.stop()`, sets `started=False` — but
   `actual_serial` and the just-updated `fps=30` remain stale.
5. `_start_runtime(wrist)` sets `started=True`; meanwhile the orphaned ego acquire
   thread calls `wait_for_frames()` on the stopped pipeline → exception recorded.
6. A lock-free `status()` read sees the contradictory mix above.

> Note: uvicorn here runs a **single** process (default `workers=1`) with a sync-route
> threadpool — not multiprocess. That does not change the diagnosis: multiple threads in
> one process still race the same lock.

## Files / lines implicated

- `_stop_runtime` lock release: `realsense.py:195-218` (critical: 204, 208)
- `_acquire_loop` stale pipeline + spin-on-error: `realsense.py:401-405`, `450-452`
- `discover()` unlocked: `realsense.py:82-100` (called from `_available_serials` 185-193)
- `status()` unlocked read: `realsense.py:370-393`

## Planned changes (all in `realsense.py`)

**Change 1 — Remove the lock-release window in `_stop_runtime` (core fix).**
Hold `self._lock` for the whole stop: set `stop_event`, stop the pipeline, clear
`pipeline`/`started`/`fps`/`actual_serial` — all under the lock. Do the thread `join()`
without releasing the service lock. Break the original deadlock reason (the acquire
loop grabbing the lock to cache a final frame) by having the acquire loop check
`stop_event` *before* taking the lock and skip the final cache write when stopping, so a
stopping thread never needs the lock.

**Change 2 — Make the acquire loop exit cleanly; never use a dead pipeline.**
Loop guard exits when `stop_event` is set OR when `runtime.pipeline` is no longer the
pipeline this thread started (it was replaced/stopped). On the `wait_for_frames` error
path, `break` if `stop_event` is set instead of spinning.

**Change 3 — Make lifecycle/state reads consistent.**
Ensure `_available_serials()` does not call `discover()` while a stop is mid-flight
(largely fixed once Change 1 keeps the lock held throughout). Take the lock for the
`status()` snapshot so one read can't capture a half-updated runtime.

**Change 4 — (small, defensive) Clear stale fields up front in `_start_runtime`.**
So a failed/not-started runtime cannot retain `fps`/`actual_serial` from a prior run.

## Explicitly NOT touched

- Assignment logic (`set_device_serials`, `ensure_default_assignment`) — confirmed not
  the cause; left byte-for-byte as committed.
- The frame endpoint, recording path, and frontend.

## Risks / caveats

- Concurrency code: main risk is **deadlock** if the lock is held across a blocking
  call. Only non-blocking work runs under-lock (set event, quick `pipeline.stop()`,
  field updates); the acquire thread must never block on the lock while stopping.
  `join()` keeps the existing 2s timeout and threads are daemon, so a stuck SDK call
  can't hang shutdown.
- If the RealSense SDK ever ignores `enable_device`, the *assignment* is correct but the
  bind could still be wrong — out of scope here; this plan targets the state-corruption
  race that matches the observed symptom.

## Verification

1. Import check + a scripted simulation of stop/start interleaving with fake
   pipeline/thread objects (no hardware), asserting no contradictory state across rapid
   start/stop/restart.
2. **Restart the backend**, then in the UI: connect cameras / swap serials a few times.
   Confirm two distinct 30-fps feeds on the **first** try, `/frame` returns **distinct**
   bytes per camera, and `status()` is clean (each: `started=True`, `fps>0`,
   `actual_serial==configured_serial`, `error=None`).

## Success criteria

Two distinct serials → two distinct, live (~30 fps) feeds on first connect; no
contradictory runtime state; no `started=True` with a `wait_for_frames ... before
start()` error.

## Possible phased option

If preferred, implement **Change 1 + 2 first** (the minimal core that closes the
corruption window), validate, then add Changes 3–4.
