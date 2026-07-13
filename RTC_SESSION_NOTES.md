# RTC Rollout — Session Notes

Date: 2026-07-13

Goal: eliminate the jerk that occurs when the diffusion policy's consecutive
action chunks disagree at the replan seam during rollout, by porting **Real-Time
Chunking (RTC)** into the LeRobot diffusion rollout path. Inference-only (no
retraining). Reference: RTC / "Real-Time Execution of Action Chunking Flow
Policies", and the pattern already present in the Psi0 codebase
(`predict_action_with_rtc_flow_naive_inpaint`).

## What RTC does here (one line)

When a fresh action chunk is sampled, freeze its head onto the still-executing
tail of the previous chunk (RePaint-style inpainting inside the denoiser), so the
new chunk *continues* the old one instead of contradicting it.

---

## Files changed

| File | Role |
|---|---|
| `src/flexivtrainer/policies/rtc_diffusion.py` | **New.** The RTC sampler (RePaint inpaint loop) that replaces `DiffusionModel.conditional_sample`/`generate_actions`. |
| `src/flexivtrainer/policies/diffusion.py` | RTC config knobs on `RolloutConfig` + `_attach_rtc` that binds the RTC sampler onto a loaded policy. |
| `src/flexivtrainer/rollout/service.py` | Planner-loop plumbing: capture previous chunk's tail, normalize it, pass as prefix; RTC metrics + logging. |
| `src/flexivtrainer/policies/_shared.py` | `replan_steps` default tuned 8 → 5 (more chunk overlap). |
| `rtc_diagram.html` | Standalone visual explainer of the RTC cycle (repo root; not referenced by code). |

Everything except the final `replan_steps=5` tweak was committed incrementally
during the session (see commits `2de0b8a add rtc`, `03a585c bug fix`,
`bb81ef4 rtc rollout test`, and the `test` commits).

---

## How it works (data flow)

1. Policy loads → `apply_rollout_overrides` → `_attach_rtc` binds
   `rtc_generate_actions` onto `policy.diffusion` and stashes the RTC config.
   (No weights reloaded; site-packages `lerobot` never edited.)
2. Each planner tick, on a **replan** (`step % replan_steps == 0`), the service:
   - slices the previous chunk's still-unexecuted **tail**,
   - **normalizes** it into the model's action space (via the preprocessor's
     `NormalizerProcessorStep` — the exact inverse of the postprocessor's
     unnormalize),
   - stashes it on `policy.diffusion` as `_rtc_prev_actions` + freeze length.
3. `rtc_conditional_sample` runs the DDIM denoise loop; at every step it
   re-noises the clean prefix to the current noise level (`scheduler.add_noise`,
   which works on DDIM/DDPM), blends it into the sample under fade weights
   (hard-freeze first `d`, linear fade to `s`), then denoises one step.
4. Fade-weight curve reused from LeRobot's own
   `RTCProcessor.get_prefix_weights` (velocity-free schedule math).

`prev_actions is None` → the sampler is bit-for-bit the stock loop (first chunk
and RTC-off are safe).

---

## Bugs found and fixed (chronological)

### 1. `TypeError: int() argument ... not 'NoneType'`
`rtc_conditional_sample` did `int(inference_delay)` unconditionally, but the
config default (`0`) becomes `None` in `_attach_rtc`. **Fix:** `0`/`None` →
auto (`d=1`, `s=horizon//2`), all clamped to valid range.

### 2. Valley collapse (robot parks at the mean pose)
The service passed `elapsed` (≈ replan cadence, ~8) as BOTH the tail-slice index
(correct) AND the sampler's freeze length `d` (wrong — should be ~2). With `d=8`
and a real prefix of only 2 steps, `_pad_prev_actions`'s **zero padding**
occupied frozen positions 2–7. Normalized zero = mid-range = dataset mean pose,
so the robot was commanded to the mean pose and each replan re-anchored there.
**Fixes:**
- Sampler clamps `d`/`s` to the real (pre-padding) prefix length — padding can
  never be frozen. (`rtc_diffusion.py`)
- Service separates the two quantities: `elapsed` slices the tail; freeze length
  = `(n_obs_steps−1) + rtc_inference_delay`, which also fixes an off-by-one vs
  where `generate_actions` slices the executed window. (`service.py`)
- `_normalize_actions` now returns `None` (RTC skipped) instead of silently
  passing real-unit actions when the normalizer step isn't found.

### 3. RTC never engaged on first replan (`rtc-skip elapsed=11 >= chunk_len=10`)
`elapsed` was computed as wall-clock ÷ `dt`. The first chunk's one-time ~365 ms
startup inference inflated wall-clock so `elapsed` overshot the chunk length and
RTC bailed. **Fix:** `elapsed = step − prev_chunk_step` (integer planner ticks,
immune to inference-time jitter).

### 4. Logs always showed `d=None len=None`; then `seam_gap=0.0000`
- The `rtc` log fired on the `log_every` cadence (steps 0,5,10,…) which almost
  never coincides with replan steps (0,8,16,…), so it sampled the reset `None`
  values. **Fix:** log RTC on `fresh` (replan) steps.
- `seam_gap` measured position 0 — which is exactly the frozen head, always ~0.
  **Fix:** `seam_gap` now measures the **velocity discontinuity at the fade
  edge** (2nd difference at index `d`) — the actual felt-jerk proxy. Verified
  ~0 for smooth motion, >0 for a kink.

---

## Current config (defaults, in `RolloutConfig` / `SharedRolloutConfig`)

```
noise_scheduler_type  = DDIM
num_denoise_steps     = 16
n_action_steps        = 10
replan_steps          = 5          # was 8; more chunk overlap
rtc_enabled           = True
rtc_inference_delay   = 0          # -> auto: freeze (n_obs_steps-1)+1 = 2 head steps
rtc_execution_horizon = 8          # fade old->new over steps d..8
rtc_prefix_schedule   = linear     # gentler handoff than exp
```

With `n_action_steps=10`, `replan_steps=5`: expect `d=2`, `len=6` per replan.

---

## Verification done

- **Offline sampler tests** (`scratchpad/test_rtc.py`, NOT yet in the repo):
  RTC-off is bit-exact to stock; freeze+fade correct; `None`/`0` args auto;
  valley regression (short prefix + oversized `d` must not pin to zero);
  slice arithmetic; real→normalized round-trip. All pass; ruff clean.
- **Hardware runs:** valley fixed (normal trajectory); first-replan skip fixed;
  RTC engages every replan.
- **Latest run metrics:** inference 8–17 ms (vs 100 ms budget — no staleness);
  `seam_gap` mean ≈ 0.009, max 0.020 — comparable to the trajectory's own
  per-step motion, i.e. the RTC seam is NOT producing an abnormal kink.

---

## Open status / next steps

**Key finding:** `seam_gap` is small but the operator still reports clear
jerkiness. Since the commanded seam is smooth, the residual jerk is most likely
**not** the RTC fade edge. Two remaining suspects:

1. **Policy chunk-to-chunk disagreement** past the frozen head — RTC (inference-
   only) can soften but not erase a genuine plan change. The `replan_steps=5`
   change (wider overlap, `len` 3→6) is the current experiment to test this.
2. **Controller / tracking lag** — logs show `cmd_xyz` consistently leads
   `meas_xyz` by ~5–8 mm (lag-then-catch), which reads as jerk regardless of how
   smooth the *commanded* path is. If `replan_steps=5` doesn't improve the
   *feel*, investigate the motion limits / gains in
   `WaypointExecutor.SendCartesianMotionForce` and the anchor timing.

**Decisive next experiment:** run with `replan_steps=5`, confirm `len=6` in the
logs, and judge by **feel** (seam_gap won't move much more):
- smoother → it was planning disagreement; optionally push `replan_steps=4`.
- no change → pivot to the controller/tracking layer (stop tuning RTC).

**Housekeeping deferred by user:**
- Promote `scratchpad/test_rtc.py` into the repo test suite (`tests/`) — it's the
  regression guard for bugs 1 & 2 and currently only lives in scratch.
- Decide whether to keep/move/delete `rtc_diagram.html`.
- Temporary `rtc-skip` diagnostic log was already removed; the `step=N rtc`
  per-replan log line remains (useful for tuning).

## Tuning knobs quick reference

| Symptom | Knob |
|---|---|
| Seam jerk, inference slow (`infer_ms > 2·dt`) | raise `rtc_inference_delay` |
| Abrupt handoff at fade edge | `rtc_prefix_schedule=linear`, raise `rtc_execution_horizon` |
| Thin anchor (`len` small) | lower `replan_steps` (more overlap) |
| Small seam_gap but still jerky | not the seam — check policy / controller |
