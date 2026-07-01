# Rollout Motion Smoothing — Session Changes & TODO

Investigation and changes to make diffusion-policy rollout motion smooth on the
Flexiv follower arm. Goal: eliminate the jerky / "on-and-off" motion seen when
streaming policy actions to `SendCartesianMotionForce`.

## Summary of the investigation (what we learned)

1. **Apparent ~2 Hz was partly a logging artifact.** The step log printed every
   `loop_hz // 2` steps; the control loop ran faster than it appeared.
2. **Diffusion inference is the dominant per-step cost.** Cameras, RDK reads, and
   CPU↔GPU copies are ~0 ms; a chunk refill ran the U-Net once per
   `n_action_steps` (=8) steps.
3. **DDPM/100 refill ≈ 460 ms; DDIM/10 ≈ 50 ms; DDIM/5 ≈ 26 ms.** DDIM reuses the
   same trained weights and samples in far fewer steps. The denoise is
   **launch-bound** (many tiny sequential CUDA kernels), so a 5090 doesn't help
   and TF32/AMP/`torch.compile` barely move it (~20%). The fast reference repo
   (`Humanoid-Policy` / iDP3) avoids this by exporting the whole sampling loop to
   an unrolled ONNX graph — a heavier change, not taken.
4. **Chunk reuse already works** (LeRobot's `select_action` re-samples only when
   its action queue empties). No bug there.
5. **Root cause of jerk: no interpolation layer.** We sent raw, discrete policy
   waypoints straight to `SendCartesianMotionForce` once per policy step. The
   reference repo (`diffusion_policy`, Chi et al.) streams a *continuous
   time-parameterized spline* at a high rate instead. The RDK velocity cap
   (0.5 m/s) was never the issue (measured motion ~0.04 m/s).
6. **The RDK controller:** installed `flexivrdk` 1.9.2 exposes only NRT
   `SendCartesianMotionForce` (no RT `StreamCartesianMotionForce`). Per the robot
   owner, NRT handles up to 1000 Hz; 100–200 Hz is ideal — so high-rate NRT
   streaming is in spec, no upgrade needed.

## Code changes (all uncommitted)

### New: `src/flexivtrainer/rollout/pose_interpolator.py`
- Port of `diffusion_policy`'s `PoseTrajectoryInterpolator`, adapted from 6-DoF
  axis-angle to our **7-element quaternion pose** `[x,y,z,qw,qx,qy,qz]`.
- Position via scipy `interp1d` (linear), orientation via `Slerp`; both **clamped**
  to the trajectory span (past-the-end holds the last pose — no overshoot).
- Added `velocity(t)`: finite-difference of the spline (cleaner velocity target
  than the policy's raw twist).
- Unit tests: `tests/test_pose_interpolator.py` (interp between waypoints,
  schedule_waypoint blend, hold-past-end, unit quaternion, velocity).

### `src/flexivtrainer/rollout/service.py`
- **DDIM/DDPM scheduler override at load** (`_apply_diffusion_scheduler_override`)
  — swaps the diffusion sampler + `num_inference_steps` at rollout start using the
  checkpoint's own schedule kwargs (weights unchanged). Gated to diffusion
  policies; best-effort.
- **Per-stage timing diagnostics** in `_run` (fault_check/grab_images/read_states/
  build_obs/inference/to_list/dispatch + `total`), plus un-smoothed `infer_steps`/
  `infer_max`, and a `freq=actual/expected Hz` line. `_cuda_sync` added so async
  CUDA inference is attributed to the inference stage. **(diagnostic — to trim)**
- **`_StreamingController`** (new private class): the high-rate sender. Holds one
  `PoseTrajectoryInterpolator` per arm; `schedule()` blends the policy's commanded
  pose as a timestamped waypoint; `_sender_loop` ticks at `interp_hz` and calls
  `_send_interpolated_action` → the **single** `SendCartesianMotionForce` call
  site. Seeds each arm's interpolator from the robot's **measured** pose so the
  first tick eases (no jump).
- **Single integrated send path**: removed the old direct-dispatch path and the
  `interpolate` flag; deleted `_dispatch_action`. Streaming is the only path.
- **Safety — stop ordering**: `stop()` now joins the sender (`self._controller`)
  **before** `_release_robots()`, independent of the producer thread, so the arm
  cannot be commanded after stop even if the producer wedges.
- **Safety — hardware speed caps**: passes explicit
  `max_linear_vel/max_angular_vel/max_linear_acc/max_angular_acc` to
  `SendCartesianMotionForce` (a hard ceiling the interpolated velocity can't
  exceed).

### `src/flexivtrainer/config.py` (`RolloutConfig`)
- `diffusion_scheduler` (`""`/`DDPM`/`DDIM`), `diffusion_inference_steps`.
- `interp_hz` (sender rate, default 200).
- `max_linear_vel=0.25`, `max_angular_vel=0.6`, `max_linear_acc=1.0`,
  `max_angular_acc=2.5` (all `gt=0`; conservative for running near people).
- Current live config: `loop_hz=10`, `diffusion_scheduler="DDPM"`,
  `diffusion_inference_steps=100` (reference-style).

### `pyproject.toml`
- Added `scipy>=1.11` (interpolator dependency; installed in the venv).

### `tests/test_rollout_service.py`
- Dropped the direct-dispatch test; folded loop-lifecycle intent into
  `test_rollout_loop_streams_commands_and_stops` (verifies streaming, measured-
  pose seed range, unit quaternion, hardware caps reach the robot, clean sender
  shutdown). Scheduler-override test sets its scheduler explicitly.
- All **132 tests pass**; ruff clean on changed regions (pre-existing findings
  left untouched).

## Verification status

- ✅ Unit/integration tests green (fake robots).
- ⚠️ **Live behavior: still jittery** — see the open bug below. The interpolation
  did not smooth motion; it made it worse in the last run.

---

## TODO

### 🔴 Blocking bug — interpolation reach-and-freeze (fix before next tuning)
The waypoint horizon is wrong. In `_run`, `schedule(..., target_time=now + period)`
uses `period = 1/loop_hz = 0.1 s`, but the **real** gap between waypoints is one
full inference (~0.5 s with DDPM/100). The 200 Hz sender races to the target in
100 ms, then **clamps at the last pose** (interpolator clamps past-the-end) for
the remaining ~400 ms, then jumps when the next waypoint arrives → reach-and-freeze
jitter, worse than the old direct send.

Candidate fixes (decide, then implement):
- **A. Horizon = measured step interval.** Set `target_time` to the actual
  inter-step time (from `work_times`, with a floor) so the sender keeps
  interpolating across the real gap. Minimal change.
- **B. Schedule the whole action chunk.** Call `predict_action_chunk` to get all
  `n_action_steps` poses and schedule them with staggered timestamps
  (`obs_time + k*dt`), like the reference repo. Most correct/smooth; bigger change
  (bypasses LeRobot's `select_action` queue).
- **C. Add logging first** — actual step interval vs. the 0.1 s horizon — to
  confirm reach-and-freeze on hardware before changing scheduling.

### 🟠 Remaining safety items (from the pre-run audit)
- **Workspace / pose-jump clip.** No bound on where a diverged policy can command;
  only the speed cap limits it, not position. Reference repo clips target pose to a
  workspace box. Consider adding.
- **Clamp commanded velocity in `velocity()`** — belt-and-suspenders on top of the
  hardware cap.

### 🟢 Follow-ups / cleanup (non-blocking)
- **Cubic Hermite interpolation** using the policy's predicted twist as velocity
  boundary conditions (keeps the `twist` slice / `_TWIST_DIM` plumbing, which was
  deliberately retained for this). Orientation-velocity handling to be decided at
  implementation time.
- **Wind down diagnostics** once smooth: remove `_cuda_sync`, `stage_times`,
  `infer_raw`, `_log_timing`, and the verbose `infer_steps` logging; keep a
  lightweight `freq`/sender-Hz line.
- **Tune for target rate**: if staying at `loop_hz=10` with DDPM/100 (reference
  config), confirm task success; otherwise DDIM + fewer steps at higher `loop_hz`.
- **ONNX export** (optional): unroll the diffusion sampling loop to erase the
  launch-bound cost, if inference speed ever needs to drop further.

### First live-run checklist (safety)
- Clear the workspace — no people in reach during first motion checks.
- E-stop physically in hand.
- Start the arm near the policy's expected start pose.
- Consider `max_linear_vel=0.1` for the very first run, then raise.
- Watch the seed→first-chunk transition (riskiest moment).
