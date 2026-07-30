# How B-Spline Rollout Works

A detailed, intuitive walkthrough of what happens when you press **Start** on a
`bspline_diffusion` checkpoint — using the run you just trained as the concrete
example:

```
.local/training/merged_20260728_163313_nodepth_bspline-bspline_diffusion_20260728_171259/checkpoints/last
```

---

## 0. The one-paragraph version

A normal diffusion policy predicts **a list of poses**, and the robot walks that
list one waypoint at a time. A B-spline policy instead predicts **the parameters
of a smooth curve** — a knot vector plus control points. The rollout loop never
"steps through waypoints." It hands that curve to a 200 Hz thread that *samples*
the curve as a function of wall-clock time and streams Cartesian pose commands
to the robot. When the curve is nearly used up, the policy predicts a *new*
curve, and the executor splices it in at the point on the new curve that best
matches where the outgoing plan is right then — then smooths out whatever
mismatch alignment could not remove.

Three things make this different from the waypoint (ACT / plain diffusion) path:

1. **Time is continuous.** The commanded pose is `spline(t)` where `t` advances
   with real elapsed time, not with loop iterations. Jitter in the Python loop
   changes *when* you sample, not *what* the trajectory is.
2. **Replanning is phase-aligned, not position-faded.** The old smoothing hack
   (blend from current pose toward the new plan) injected velocity
   discontinuities at every replan. Here the executor searches for the spline
   parameter `t*` on the *new* curve whose pose matches the outgoing plan, and
   starts there.
3. **The splice is C¹.** Alignment never lands exactly, and it never considers
   velocity at all, so the raw splice stepped both the commanded pose *and* its
   direction — the arm was told to reverse heading by ~90° at every replan. Both
   residuals are now carried as decaying corrections (§5, §12.5). This was the
   dominant source of rollout jitter and zigzag; the history is worth reading
   before touching `_align` or the blend.

---

## 1. What the policy actually outputs

### Shape of the action

Your checkpoint's `config.json`:

| field | value | meaning |
|---|---|---|
| `output_features.action.shape` | `[304]` | one flat vector per inference |
| `horizon` | `16` | 16 **parameter rows** |
| `spline_degree` | `3` | cubic B-spline |
| `knot_rate_hz` | `30.0` | knot units are 1/30 s (dataset FPS) |
| `n_action_steps` | `1` | one *plan*, not one waypoint |
| `n_obs_steps` | `2` | conditions on 2 stacked observations |

304 = **16 rows × 19 channels**. Reshaped row-major, one inference produces this
matrix (see [data/bspline.py:475-491](src/flexivtrainer/data/bspline.py#L475-L491)
for the naming scheme `bspline.row_NN.<channel>`):

```
       col 0     cols 1..9                    cols 10..18
       ─────     ──────────────────────       ──────────────────────
row 00 knot[0]   left_arm  x y z r1 r2        right_arm  x y z r1 r2
row 01 knot[1]   left_arm  x y z r1 r2        right_arm  x y z r1 r2
 ...
row 15 knot[15]  left_arm  x y z r1 r2        right_arm  x y z r1 r2
       └─ knot   └─ control point for left    └─ control point for right
          vector    (3 pos + 6 rotation-6D)      (3 pos + 6 rotation-6D)
```

So this bimanual checkpoint predicts **no gripper channel** — 19 = 1 knot + 2 ×
(3 position + 6 rotation-6D). If a `<side>.gripper.width` channel were present
it would be a 20th/21st column and the gripper executor would engage; here it
does not (see §6).

### Why rotation-6D and not quaternions

Quaternions are a bad thing to interpolate linearly and a bad thing for a
network to regress (double cover: `q ≡ -q`). Rotation-6D is the first two rows
of the rotation matrix — an unconstrained ℝ⁶ that the network can regress
freely, projected back onto SO(3) with Gram–Schmidt at execution time
([data/bspline.py:151-190](src/flexivtrainer/data/bspline.py#L151-L190)). Because
6D is closed under interpolation, **the spline can be evaluated directly in 6D**
and only the final sample gets projected to a quaternion. That is the key trick
that lets one spline carry both translation and rotation.

### Reading the matrix as a spline

Only `rows - (degree + 1)` = 16 − 4 = **12 control rows are active** — the decoder
below ignores rows 12–15. They are not filler, though: they hold the *next four
real control points* of the episode fit, so the policy gets genuine lookahead
supervision in them ([data/bspline.py:369-392](src/flexivtrainer/data/bspline.py#L369-L392)).
Only the last few windows of an episode, which run past the end of the
coefficient array, fall back to repeating the last available row. They used to
repeat unconditionally, which held 23.7% of the training target constant — see
§12.7.

```python
knots    = matrix[:, 0]                    # 16 values, nondecreasing
controls = matrix[:-(degree+1), 1:]        # 12 × 18
spline   = BSpline(knots, controls, k=3, extrapolate=False)
```

The valid domain of a degree-3 B-spline with 16 knots is
`[knots[3], knots[-4]]` = `[knots[3], knots[12]]`
([executors/bspline.py:256-266](src/flexivtrainer/rollout/executors/bspline.py#L256-L266)).

### Local time — the important convention

Knots are stored **relative to the current frame**. During dataset conversion
every frame's chunk gets `local[:, 0] -= frame_index`
([data/bspline.py:429-441](src/flexivtrainer/data/bspline.py#L429-L441)). So at
inference the knot vector says *"this curve starts roughly now and extends
forward"*, in units of 1/`knot_rate_hz` seconds. A domain of, say, `[0, 9]`
means **9 / 30 Hz = 0.3 s of motion**. The executor never needs to know what
absolute timestep it is on.

---

## 2. Preflight — what is checked before a robot moves

`RolloutService.start()` treats `bspline_diffusion` as a special branch
([rollout/service.py:192-214](src/flexivtrainer/rollout/service.py#L192-L214)).
`_preflight_bspline` ([service.py:441-497](src/flexivtrainer/rollout/service.py#L441-L497))
refuses to proceed unless:

- `action_feature_names` exists in the checkpoint and parses into contiguous,
  identically-shaped rows each starting with `knot`
- rows == `horizon` (16 == 16)
- the checkpoint's arm sides **exactly match the app's active sides**, in order —
  your checkpoint demands `("left_arm", "right_arm")`, so a single-arm session
  is rejected rather than silently half-executed
- every active arm has a follower serial
- `spline_degree` is a positive int and `rows > degree + 1`
- the policy exposes `enqueue_observation()` and `predict_action_chunk()`

Two more B-spline-only behaviors:

- **`knot_rate_hz` is mandatory.** `_resolve_target_hz(..., require_metadata=True)`
  ([service.py:415-439](src/flexivtrainer/rollout/service.py#L415-L439)) raises
  rather than falling back to an app default, because the whole time base is
  derived from it. Yours is 30 Hz.
- **`prepare_motion` is deferred.** For waypoint policies the robot is put into
  Cartesian motion mode at connect time; for B-spline, `connect_robot` is called
  with `prepare_motion=None` ([service.py:225-227](src/flexivtrainer/rollout/service.py#L225-L227))
  so the robot stays **IDLE**. That's required because grippers can only be
  `Init()`-ed from IDLE. The runner switches modes afterward, once executors
  exist ([runners/bspline.py:177-178](src/flexivtrainer/rollout/runners/bspline.py#L177-L178)).

---

## 3. The three threads

```
┌──────────────────────────────────────────────────────────────────────┐
│ PLANNER THREAD  "rollout-policy-planner"     ~30 Hz (target_hz)      │
│                                                                      │
│  grab images → read robot state → preprocess → policy.enqueue_obs()  │
│  if executor.replan_needed() and no inference in flight:             │
│        submit inference to the 1-worker pool ──────────┐             │
│  publish metrics; sleep until next obs or replan point │             │
└────────────────────────────────────────────────────────┼─────────────┘
                                                         │
┌────────────────────────────────────────────────────────▼─────────────┐
│ INFERENCE POOL  "rollout-bspline-inference"   1 worker               │
│   predict_action_chunk() → postprocess (unnormalize)                 │
│   → executor.install(flat_304_vector, observation_age_s)             │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ EXECUTOR THREAD "rollout-bspline-executor"   control_hz = 200 Hz     │
│   t = start_time + (now - installed_at) * source_rate                │
│   raw = spline(t) + handoff correction  (18 values, in 6D)           │
│   per arm: pose = [xyz] + rot6d_to_quat(6d)                          │
│   robot.SendCartesianMotionForce(pose, 0, 0, limits...)              │
└──────────────────────────────────────────────────────────────────────┘

(+ "rollout-gripper-executor" at ≤30 Hz — only if the checkpoint predicts
   gripper width. Not active for this checkpoint.)
```

Why split them: diffusion inference takes tens of milliseconds and is
GIL-hostile. If it ran inline, the 200 Hz command stream would stall every
replan. Running inference on its own worker keeps the executor's send cadence
flat, and the executor holds *the previous plan* until the new one is installed —
so there is never a gap with nothing to command.

The planner's sleep is adaptive
([runners/bspline.py:354-366](src/flexivtrainer/rollout/runners/bspline.py#L354-L366)):
normally it wakes for the next observation, but if inference is in flight it
polls every 10 ms so the fresh plan gets installed promptly, and if the spline is
about to run out it wakes exactly `predict_before_end_s` before the end.

---

## 4. Sampling the curve: from wall-clock to spline parameter

The entire time base is these three lines
([`_spline_time`](src/flexivtrainer/rollout/executors/bspline.py#L579-L587)):

```python
source_rate = checkpoint_fps * speed_scale     # 30.0 * 1.0 = 30.0
t = clip(plan.start_time + (now - plan.installed_at) * source_rate,
         plan.min_time, plan.max_time)
```

Read it as: *"since installing this plan, `now - installed_at` seconds have
elapsed; at 30 knot-units per second that is this far along the curve."*

Consequences worth internalizing:

- **`speed_scale` is a true time warp.** `speed_scale = 2.0` walks the same
  geometric path twice as fast — doubling commanded velocity — without changing
  the path. Motion limits are *not* rescaled to compensate, so this is the knob
  that will hit the robot's velocity ceiling first.
- **The clip is a hold, not a stretch.** If replanning is late, `t` saturates at
  `max_time` and the robot simply **holds the final pose** of the old curve. No
  extrapolation (`extrapolate=False`), no drift. Slow inference degrades into
  stutter, never into divergence.
- **Missed executor deadlines are harmless to the path.** The loop tracks
  `missed_deadlines` for telemetry and advances its deadline, but `t` is
  recomputed from the clock, so a hiccup just skips a sample — it does not shift
  the trajectory in time.

Each sample becomes, per arm
([`execute_once`](src/flexivtrainer/rollout/executors/bspline.py#L588-L629)):

```python
raw        = self._sample(plan, now)     # spline(t) + any active handoff blend
position   = raw[position_indices]                       # 3
quaternion = rotation_6d_to_quaternion_wxyz(raw[rot_i])  # Gram-Schmidt → wxyz
robot.SendCartesianMotionForce(position + quaternion,
                               [0]*6,        # zero wrench
                               [0]*6,        # zero stiffness override
                               max_lin_vel, max_ang_vel,
                               max_lin_acc, max_ang_acc)
```

Everything commanded flows through [`_sample`](src/flexivtrainer/rollout/executors/bspline.py#L383-L399),
which adds the handoff correction described in §5. The blend is keyed on **real**
seconds, not spline time, so `speed_scale` cannot change how fast a correction is
retired.

Note the commanded quantity is a **pose**, with zero feed-forward wrench. There
is no velocity command — a deliberate change from the waypoint path, where
commanding raw `tcp_twist` was one of the two causes of jerky rollout.

---

## 5. The handoff: how a new plan splices into the old one

This is the heart of the design. Naively, installing a new spline at `t =
min_time` would jump the commanded pose from wherever the old curve had reached
to wherever the new curve begins. Two reasons they differ: the policy conditioned
on an observation from ~one inference-latency ago, and diffusion sampling is
stochastic.

`install()` ([executors/bspline.py:491-577](src/flexivtrainer/rollout/executors/bspline.py#L491-L577))
handles it in two cases:

**First plan** — nothing has been commanded yet, so start at
`clip(0.0, min_time, max_time)`; local-time convention says 0 is "now".
Alignment error is 0 by definition, and no blend is needed.

**Every later plan** — call `_align()`
([executors/bspline.py:431-489](src/flexivtrainer/rollout/executors/bspline.py#L431-L489)),
which answers: *at what parameter `t*` on the new curve does the pose most
closely match where the outgoing plan is at this instant?*

```
old curve  ────────────●━━━━━━ (t saturating / still running)
                       │ _sample(previous, install_time)
                       │  ↓ find matching phase
new curve  ─────────────●──────────────────────────────────►
           min_time     t*                          max_time
                        └─ start_time; plan runs forward from here
```

Mechanics:

1. Seed the search window from the **age of the observation** the plan was
   conditioned on: `initial_max = min_time + observation_age_s * source_rate`.
   Intuition — the policy planned from a stale observation, so the matching phase
   is roughly *age × rate* knot-units in. The age is measured from when the
   planner grabbed that observation, so it covers the ≤ 33 ms queueing wait at
   30 Hz *plus* the ~25 ms of inference compute. It used to be the compute time
   alone, which made the window roughly 2× too small.
2. Hard-cap the window at `min_time + (max_time - min_time) * time_align_max_fraction`
   (default 0.2). **This is a safety rail**: it forbids skipping more than 20% of
   the new curve, so a bad match can never fast-forward the robot deep into the
   plan. Hitting it sets `align_capped`.
3. **Always** run at least one `scipy.optimize.minimize_scalar(..., method="bounded")`
   pass on the L1 pose difference over the aligned channels (positions +
   rotation-6D; gripper is excluded from alignment). `min_time` stays a competing
   candidate and the optimizer's result is taken only if it is strictly better —
   a bounded search can land worse than an endpoint on a non-unimodal objective.
   `align_searched` records that the search ran; it is false only when the window
   collapses to a point (`observation_age_s == 0`).
4. If the best max-abs error still exceeds `time_align_error_threshold`
   (default 0.1), widen the window ×1.5 and search again — up to 20× or until the
   cap is hit. The threshold governs *widening*, never whether to search at all;
   §12.6 is what happens when it does gate the search.
5. If it *still* exceeds threshold, install anyway but emit a warning, bump
   `handoff_warnings`, and surface it in the UI log
   ([runners/bspline.py:322-329](src/flexivtrainer/rollout/runners/bspline.py#L322-L329)).

### Why alignment alone is not enough

Alignment gets you close, never exact — `minimize_scalar` returns the *best
available* match — and it compares **pose only, never velocity**. So on its own
the splice steps both the commanded pose (median 6 mm, vs 0.45 mm of normal
per-tick motion) and its direction of travel (median ~90°, up to a near-U-turn).
That was the jitter and the zigzag; §12.4 has the measurements.

So `install()` captures *both* residuals and `_sample` retires them smoothly:

```
commanded(t) = new(t) + p·W0(u) + v·T·W1(u),      u = (t − T0)/T
p = old(T0) − new(T0)                 position gap  → W0: 1→0, flat both ends
v = old′(T0) − new′(T0)               velocity gap  → W1: 0→0, unit slope at u=0
```

At `u = 0` this reproduces the outgoing plan's pose *and* its velocity exactly; by
`u = 1` both corrections and their slopes are zero, so it lands cleanly on the new
plan. That makes the handoff C¹. It is **not** the old position-fade, which
blended toward the plan linearly and therefore injected a velocity step at each
end of the window.

`T` comes from [`_blend_duration`](src/flexivtrainer/rollout/executors/bspline.py#L349-L381):
at least `handoff_blend_s`, stretched to bound the acceleration each term adds
(`√(5.77·|p| / a_max)` and `6·|v| / a_max`), and never more than half the
outgoing plan's remaining time so a correction cannot outlive the plan it fixes.

Two details that are easy to get wrong, both learned the hard way:

- **The target is the outgoing plan evaluated at the install instant, not
  `_last_raw_command`.** The last-sent value is up to one control period stale, and
  replaying it commands a literal dead stop for one tick — measured 0.055 → 0.000
  → 0.033 m/s. That is how the first attempt at this fix swapped a position step
  for a velocity notch.
- **The target is a *commanded* pose, not the measured robot pose.** Matching the
  command stream keeps it continuous even when the robot lags under load; matching
  measurement would bake tracking error into the plan. And `_last_raw_command`
  stores the blended pose actually sent, because the next handoff aligns to it.

---

## 6. Grippers

The gripper is *not* on the spline command path, even when predicted. Where a
`<side>.gripper.width` channel exists, `GripperExecutor` runs on its own thread
and **pulls** the latest sampled width via
`target_source=lambda: bspline_executor.last_gripper_widths`
([runners/bspline.py:165-176](src/flexivtrainer/rollout/runners/bspline.py#L165-L176)).

Why separate: gripper I/O is blocking, rate-limited (`MAX_COMMAND_HZ = 30`), and
would destroy a 200 Hz loop. Latest-only pull semantics means the gripper always
chases the newest width and never queues up stale commands.

**This checkpoint has no gripper channel**, so `layout.gripper_sides` is empty,
no `GripperExecutor` is built, and `_preflight_bspline_grippers` is a no-op. If
you retrain with gripper width included, preflight will then *require* a
configured follower gripper (with `gripper_model`) for each predicting side
([service.py:498-521](src/flexivtrainer/rollout/service.py#L498-L521)).

---

## 7. Knobs (`RolloutConfig` for `bspline_diffusion`)

From [policies/bspline_diffusion.py:39-51](src/flexivtrainer/policies/bspline_diffusion.py#L39-L51),
plus the inherited diffusion knobs:

| knob | default | what it does | when to change it |
|---|---|---|---|
| `control_hz` | 200 | executor send rate | lower if the robot's RDK link can't take 200 Hz; it does not change the path, only its sampling density |
| `speed_scale` | 1.0 | time warp on the curve | <1 to slow a too-aggressive policy; >1 hits velocity limits |
| `predict_before_end_s` | 0.06 | replan trigger — replan when this much curve remains | **raise for reactivity.** Plans are seconds long (§12.2), so 0.06 means ~1.2 replans/s and up to ~1.5 s of open-loop commitment. 0.3 roughly doubles the replan rate. Safe to raise now that the handoff is C¹ — before the fix it multiplied jitter |
| `handoff_blend_s` | 0.15 | window over which the handoff position/velocity mismatch decays to zero | raise (0.25–0.3) if motion still feels rough at the replan cadence; lower toward 0.05 if it feels mushy or laggy into contact; `0` restores the old stepping handoff (useful as an A/B) |
| `handoff_max_accel` | 2.0 | m/s² ceiling used to stretch `handoff_blend_s` when the gap is large | lower for a gentler correction on big mismatches; it only ever lengthens the window, never shortens it |
| `time_align_error_threshold` | 0.1 | mismatch above which a handoff **warning** is logged | mostly a reporting knob. Note it *also* drives how hard `_align` searches, so tightening it makes alignment widen its window chasing a better match and skip further into the plan — with blending a small residual is harmless, so leaving it alone is deliberate |
| `time_align_max_fraction` | 0.2 | max fraction of the new curve alignment may skip | raise cautiously if latency is high and alignment keeps saturating the cap |
| `playback_speed` | 1.0 | scales `target_hz` **before** it becomes `checkpoint_fps` | logs a warning; overlaps with `speed_scale`, prefer one |
| `num_denoise_steps` | 16 | DDIM steps actually used at rollout; `apply_rollout_overrides` assigns it onto the model, overriding the checkpoint's stored 8 (§12.3) | the main lever on `infer_ms` |

`playback_speed` and `speed_scale` multiply into the same `source_rate` from
different directions (`_apply_playback_speed` scales `target_hz`, which becomes
`checkpoint_fps`; `speed_scale` multiplies it again in the executor). Pick one.

---

## 8. Reading the telemetry

At rollout start the planner logs one INFO line, `B-spline timing contract`
([runners/bspline.py:232-260](src/flexivtrainer/rollout/runners/bspline.py#L232-L260)),
recording everything that governs timing: `target_hz`, `control_hz`, effective
`denoise_steps` (read off the loaded model, so it reflects the override in §12.3),
`knot_rate_hz`, `playback_speed`, `speed_scale`, the resulting `source_rate`,
`predict_before_end_s` and `handoff_blend_s`. Read it first — it is there so a
timing failure is never again diagnosed as a model failure.

Every planner tick appends a metrics row
([runners/bspline.py:370-391](src/flexivtrainer/rollout/runners/bspline.py#L370-L391)):

| field | healthy | what it means when it isn't |
|---|---|---|
| `send_hz` | ≈ `control_hz` (200) | well below ⇒ RDK calls are blocking or GIL contention. **This is the one thing the offline probe cannot see**, so it is the prime suspect for roughness that survives the C¹ handoff |
| `missed_deadlines` | grows slowly / not at all | rapid growth ⇒ executor is starved |
| `spline_remaining_s` | deep sawtooth: full domain (~1.9 s) down to `predict_before_end_s`, repeating | **pinned at 0** ⇒ inference can't keep up; the robot is holding the old curve's end pose. A deep sawtooth is *normal* — plans are seconds long by design (§12.2), not a sign of trouble |
| `infer_ms` | < `predict_before_end_s × 1000` (60 at the default) | larger ⇒ raise `predict_before_end_s` or cut `num_denoise_steps`. Inference **compute** only — the age of the observation the plan used is larger, and is what alignment consumes (§5) |
| `alignment_error` | ~0.008–0.011 | this is the gap the blend absorbs, not a fault in itself, and it is mostly *curve disagreement between consecutive diffusion samples* rather than phase error — §12.6 measured the phase search as buying 0 in the median case. Consistently *high* ⇒ plans disagree badly; suspect OOD observations. Varies run to run by more than most changes you will make, so never A/B on this alone |
| `handoff_blend_s` | ≈ `handoff_blend_s` config (0.15), longer when the gap is big | **`0` means the blend is not engaging** — check the config wired through, because the handoff is then stepping (§12.4) |
| `align_searched` | **true on every install except the first** (the first plan has no predecessor to align to, so false is correct there) | false *later* ⇒ the search window collapsed to a point, i.e. `observation_age_s` reached the executor as 0 — suspect the observation-timestamp plumbing. Before §12.6 this was effectively always false and nothing said so |
| `align_capped` | occasionally true | persistently true ⇒ the outgoing plan is running more than 20% of a curve ahead of where the new one starts, so alignment is clamping. Check `infer_ms` and `send_hz` first; only then consider raising `time_align_max_fraction` |
| `handoff_warnings` | 0, or rare | 0 is expected and no longer implies anything about smoothness: the threshold (0.1 = 100 mm) is ~12× the residual the blend handles. Judge smoothness from `handoff_blend_s` + `send_hz`, not this |
| `fresh` | true at each replan | never true ⇒ replan never completing |

The tell-tale triple for "policy is fine, timing is not": `spline_remaining_s ==
0`, `infer_ms` large, `alignment_error` low. The tell-tale for "policy is
struggling": `alignment_error` high while `infer_ms` is small.

---

## 9. Failure handling and shutdown

A single `threading.Event` (`stop_event`) is the shared kill switch. Anything
that raises — executor exception, gripper error, robot `fault()`, non-finite
spline sample — sets it, and all three threads unwind
([runners/bspline.py:367-391](src/flexivtrainer/rollout/runners/bspline.py#L367-L391)).

Non-finite values are rejected at two points, deliberately: on decode (a
malformed action never becomes a plan,
[executors/bspline.py:254-255](src/flexivtrainer/rollout/executors/bspline.py#L254-L255))
and on sample (a numerically bad evaluation never reaches the robot,
[executors/bspline.py:389-390](src/flexivtrainer/rollout/executors/bspline.py#L389-L390)).
Knots that arrive slightly out of order are repaired rather than rejected —
`_repair_knots` nudges each violator to `previous + 1e-6`
([executors/bspline.py:89-94](src/flexivtrainer/rollout/executors/bspline.py#L89-L94)) —
because a diffusion model will occasionally emit a knot vector that is
monotone-violating by float noise, and killing the rollout for that would be
absurd.

Shutdown order matters: `inference_pool.shutdown(cancel_futures=True)` first (so
no new plan lands mid-teardown), then join the executor, then stop the gripper,
then release the robots. If the executor won't join, `stop_robots()` is called to
break it out of a blocking RDK call, then joined again with a shorter timeout.

One more rollout-specific detail: on start, the service drops any depth-alignment
preview lease ([service.py:~300](src/flexivtrainer/rollout/service.py#L300)),
because depth→color alignment holds the GIL and was the actual cause of slow
rollout in an earlier investigation.

---

## 10. Running your checkpoint

```
Rollout page → checkpoint:
  .local/training/merged_20260728_163313_nodepth_bspline-bspline_diffusion_20260728_171259/checkpoints/last
```

Requirements this specific checkpoint imposes:

- **Both arms active**, named `left_arm` and `right_arm`, in that order, each
  with a follower serial. Anything else fails preflight.
- **Three cameras**: `ego`, `left_wrist`, `right_wrist`, each fed at 240×320.
- **38-dim `observation.state`** — must match what the recording pipeline built.
- **Teleop stopped** — RDK and TDK can't share the follower connection.
- No gripper configuration needed (no gripper channel predicted).

Then watch, in order: `send_hz` ≈ 200 → `handoff_blend_s` ≈ 0.15 (not 0) →
`spline_remaining_s` sawtoothing above 0 → `alignment_error` around 0.01. If those
hold, the plumbing is healthy and anything you don't like about the motion is the
policy, not the executor.

### Checking it offline first

`scripts/probe_bspline_rollout.py` reproduces the whole command path with no
robot: it drives the real `BSplineExecutor` over an injectable virtual clock,
replanning exactly when production would, against the dataset's own observations.

```bash
.venv/bin/python scripts/probe_bspline_rollout.py \
  --checkpoint .local/training/<run>/checkpoints/last \
  --dataset    .local/datasets/<bspline-dataset> \
  --episode 0 --probes 8 --replay-seconds 12 --out /tmp/probe
```

Part A compares predicted spline parameters against ground truth, separating
*geometry* error (curves sampled at matched normalized phase) from *timing* error
(domain length in seconds) — which distinguishes "went the wrong way" from "went
the right way too slowly". Part B logs commanded vs demonstrated pose per control
tick and reports position step, heading change, and acceleration, split by whether
the tick straddles a replan. `--handoff-blend-s 0` gives a direct A/B against the
old stepping handoff.

**Do not run it during a live rollout** — it loads the policy onto the same GPU
and decodes video on the CPU, so it competes for exactly the GPU and GIL the
rollout needs, manufacturing the jitter you would be trying to measure.

---

## 11. Contrast with the waypoint path, in one table

| | waypoint (ACT / diffusion) | B-spline |
|---|---|---|
| policy output | list of poses (chunk) | knots + control points (one curve) |
| `n_action_steps` | how many poses to consume | always 1 — one *plan* |
| what's commanded | pose (+ formerly raw twist) | pose sampled from `spline(t)`, zero wrench |
| command rate | planner-tied | independent 200 Hz thread |
| time base | loop iterations | wall clock × `source_rate` |
| replan continuity | position fade toward new chunk (velocity kinks) | phase alignment + C¹ offset/rate decay (§5) |
| late inference | queue drains, stale actions | `t` clips; holds final pose |
| rotation | quaternion | rotation-6D, projected at sample time |

---

## 12. Investigation log — "wrong trajectory", then "zigzag" (2026-07-29)

Two symptoms, chased in order:

1. The checkpoint executed smoothly on hardware — no faults, healthy `send_hz` —
   but the arms went to the wrong places. → §12.1 (still open: retrain needed).
2. After the position step was fixed, the arms zigzagged at a constant frequency.
   → §12.4 diagnosis, §12.5 fix (landed).

| § | finding | status |
|---|---|---|
| 12.1 | random-crop augmentation silently disabled in training | fixed (`crop_ratio` 0.9) — **needs retrain** |
| 12.2 | three suspected bugs that are faithful ports of the reference | not bugs — do not "fix" |
| 12.3 | secondary config divergences from the reference | open, low confidence |
| 12.4 | replan handoff stepped commanded pose *and* heading | fixed by 12.5 |
| 12.5 | C¹ handoff (offset + rate decay) | **landed** |
| 12.6 | phase alignment never ran — the warning threshold gated the search | landed, but measured impact small |
| 12.7 | tail control rows held a repeated constant (23.7% of target) | fixed — **needs re-convert + retrain** |

Everything below was checked against the reference implementation at
`~/pycheng/bspline-policy` (the B-spline Policy repo this feature was ported
from). **Read §12.2 before re-deriving any of this**: three plausible-looking
"bugs" turned out to be faithful ports of the reference, and confirming that took
longer than finding the real one.

### 12.1 Confirmed bug — random-crop augmentation is silently disabled

`TrainingConfig.crop_shape = (216, 288)`
([policies/bspline_diffusion.py:35](src/flexivtrainer/policies/bspline_diffusion.py#L35),
and the same field in [policies/diffusion.py:56](src/flexivtrainer/policies/diffusion.py#L56))
**never takes effect.** LeRobot's `DiffusionConfig.__post_init__` overwrites it:

```python
if self.resize_shape is not None:
    if self.crop_ratio < 1.0:
        self.crop_shape = (int(resize[0] * ratio), int(resize[1] * ratio))
    else:
        self.crop_shape = None      # <-- discards the caller's crop_shape
```

Because we always set `resize_shape` and never set `crop_ratio` (which defaults
to `1.0`), `crop_shape` is forced to `None`. Reproduced directly:

```
BSplineDiffusionConfig(resize_shape=(240,320), crop_shape=(216,288))
  -> crop_shape = None,  crop_ratio = 1.0
BSplineDiffusionConfig(resize_shape=(240,320), crop_shape=(216,288), crop_ratio=0.9)
  -> crop_shape = (216, 288)
```

The trained checkpoint confirms the outcome: `crop_shape: null`,
`crop_ratio: 1.0`, `resize_shape: [240, 320]`.

**`crop_ratio` is not exposed anywhere in the flexivtrainer schema** — `grep -rn
"crop_ratio" src/flexivtrainer/policies/` returns nothing — so cropping cannot be
switched on from the training form at all. `crop_shape` is a field that silently
does nothing whenever `resize_shape` is set.

Reference for comparison: `crop_shape: [76, 76]` on `[84, 84]` images (ratio
0.905) plus `eval_fixed_crop: True`
(`bspline_policy/config/clean_bspline_policy_unet_bspline.yaml`).

**Why this produces exactly this symptom.** Training ran 172,500 steps at batch
64 over 110,381 frames — **100 epochs with zero image augmentation**. Worse, ~5.2
consecutive frames share an identical action target (see §12.2), so it is ~520
passes over only ~21k distinct targets. Random crop is Diffusion Policy's
principal visual regularizer; without it the ResNet keys on exact pixel
alignment. The policy stays confident and smooth — the curve is still a valid
smooth spline, so motion quality is unaffected — while the visual features it
relies on don't transfer to the live scene. Smooth execution toward the wrong
target is the signature of visual overfitting, not of an executor fault.

**Fixed.** `crop_ratio` is now exposed in `TrainingConfig` for both
`bspline_diffusion` and `diffusion`, defaulting to **0.9**, so `crop_shape`
resolves to `(216, 288)` instead of `None`. `crop_shape` remains in the schema but
is documented as derived — it is never read when `resize_shape` is set.
`test_crop_ratio_actually_enables_random_crop` pins the default below 1.0 so this
cannot silently regress. **Requires a retrain to take effect.**

### 12.2 Not bugs — verified against the reference, do not "fix" these

| suspicion | verdict |
|---|---|
| `tied_stats` collapses action stats across the 16 rows, so all rows share one min/max per channel ([convert_bspline_dataset.py:340-364](src/flexivtrainer/jobs/convert_bspline_dataset.py#L340-L364)) | **Faithful port.** The reference `get_normalizer` does the same: `min` over the row axis, `max` over rows, `mean` of means, `mean` of stds, broadcast back across rows. Operation for operation identical. |
| One plan covers median **1.87 s** of motion, and `predict_before_end_s = 0.06` means replanning only in the last 60 ms — an effective closed-loop rate near 0.75 Hz | **By design, not a defect.** The reference replans identically (`_request_spline_if_needed` fires only when `time_remaining < predict_before_end`). Its `chunk_size=10` at 10 Hz gives a ≳1.0 s domain — comparable. Long open-loop plans are the entire point of the representation: temporal compression. |
| 76% of consecutive frames carry a duplicate target (~5.2 frames per distinct chunk); within a shared chunk only the knot column shifts by 1, ≈0.5% of normalized output range | **Also matches the reference** — same `stride=1` and the same `while local_idx <= t[degree]` assignment loop. Not a divergence. |

Also worth recording: **the 304-dim action is not the problem.** Per-channel
normalized std is 0.23–0.48 across all 19 channels, so the network uses its
output range healthily. The reference's own stack-cube config is structurally
identical — `chunk_size: 10`, `degree: 3`, `max_error: 0.002`, 16 rows.

### 12.7 Fixed — tail control rows now carry real lookahead

`_chunk_fitted_spline` sliced only 12 control rows and appended four copies of row
11 unconditionally. Confirmed on the converted dataset: rows 12–15 were exact
copies of row 11 for **every frame**, i.e. **72 of 304 dims (23.7%) of the
diffusion target was a constant**. The loss covers all 16 rows
(`DiffusionModel.compute_loss` reduces over the whole `(B, 16, C)` tensor with no
row mask), so roughly a quarter of the model's output capacity was spent
regressing "copy row 11", and the horizon axis the temporal U-Net convolves over
carried a five-long artificial plateau.

The reference slices control points over the *same window as the knots* and
repeat-pads only when the coefficient array genuinely ends. This was the largest
training-target divergence, and §12.2 originally recorded it as "the same as
ours" — that was wrong.

**Fix.** Slice `expected_rows` (16) control rows; pad only the shortfall.
Verified arithmetic: `len(coeffs) == len(knots) - degree - 1`, the first 12 rows
are always available, and only the **last four windows (~1.4%)** need padding
(amounts 1, 2, 3, 4). Nothing about decoding changes — every consumer already
slices `[:-(degree + 1)]`, and the existing reconstruction test still confirms a
decoded chunk reproduces the global fit.

Side effect to expect: `tied_stats` computes min/max/mean/std over all 16 rows, so
real tail values shift the tied per-channel statistics. Converted datasets are
therefore **not** numerically comparable across this change, and old checkpoints'
normalizer stats no longer match newly converted data. `format_version` is bumped
**2 → 3** for exactly this reason — `train_policy` hard-rejects mismatches, so a
v2 dataset can no longer be silently trained on.

**Requires a re-convert and a retrain to take effect.**

### 12.3 Secondary divergences from the reference (lower confidence)

- `use_group_norm = False` vs the reference's `obs_encoder_group_norm: True`. The
  usual argument for GroupNorm is that BatchNorm's running stats interact badly
  with EMA — but LeRobot's diffusion implementation has no EMA, so that
  reasoning does not apply here. Worth aligning; not demonstrated as causal.
- The reference has a `relative_knots` encoding (knots as first-knot plus
  successive deltas, `bspline_policy/common/knots.py`) with no equivalent here.
  Their stack-cube config leaves it `false`, so it isn't required — but its
  existence suggests knot conditioning was a real concern for them.

**Correction — `num_inference_steps` was listed here and is not a divergence.**
At rollout the model already runs **16** steps, the same as the reference.
`RolloutConfig.num_denoise_steps` defaults to 16
([policies/diffusion.py:67](src/flexivtrainer/policies/diffusion.py#L67)) and
`apply_rollout_overrides` assigns it onto `diffusion.num_inference_steps`
([policies/diffusion.py:79-99](src/flexivtrainer/policies/diffusion.py#L79-L99)),
so the checkpoint's stored `8` is overridden on load. Read the effective value
off the model, not the checkpoint — the startup contract log (§8) now prints it.

### 12.4 Fixed bug — the replan handoff stepped the commanded pose and heading

**Status: fixed in §12.5.** Kept for the diagnosis, because the measurements are
what justify the blend and two of them are counter-intuitive.

Measured with `scripts/probe_bspline_rollout.py`, driving the real
`BSplineExecutor` over an injectable virtual clock (episode 0, 12 s, 200 Hz):

| commanded position step, per 5 ms control tick | median | max |
|---|---|---|
| within a plan | **0.45 mm** | — |
| across a replan | **6.2 mm** | **34.4 mm** |

A 14× step-up in position, which finite-differences to a ~1300–1600× spike in
commanded acceleration (0.17 m/s² within a plan → 260 m/s² at the boundary; a
6 mm step in one 5 ms tick is an instantaneous 1.2 m/s, a 34 mm step is 6.9 m/s).
Verified not to be a measurement artifact: the recorded tick spacing is uniform
at exactly 0.005 s, so no time gap inflates the difference.

**Cause.** `_align()` finds the parameter `t*` that best matched the previous
commanded pose, but "best" is not "exact" — the new curve generally does not pass
through that pose. `minimize_scalar` returned the closest approach and the
leftover residual was applied as a jump on the next tick. Observed residual
(`alignment_error`) is 0.008–0.010, i.e. 8–10 mm. That residual is still there
today; §12.5 absorbs it instead of stepping it.

**Why nothing warned.** A warning fires only above `time_align_error_threshold`,
default **0.1** — about 12× the residual that actually caused the jitter. So
`handoff_warnings` read 0/14 while every single replan stepped the pose. The guard
is calibrated for gross misalignment and is blind to this, which is why §8 now
says not to judge smoothness from that counter.

**Symptom match.** The jump recurs once per replan, so the disturbance appears at
a *constant frequency* equal to the replan rate — ≈1.2 Hz at
`predict_before_end_s = 0.06`. That matches reported rollout jitter, and it is
distinct from the position-fade velocity kinks of the old waypoint path (see
[[diffusion-rollout-smoothness]]): this is a step in *position*, not in velocity.

**Consequence for the reactivity knob — *before* the fix.** Raising
`predict_before_end_s` to replan sooner made jitter *worse*, because the per-event
magnitude barely changed while the event rate climbed:

| `predict_before_end_s` | replans/s | step across replan (median) |
|---|---|---|
| 0.06 | 1.17 | 6.2 mm |
| 0.30 | 2.08 | 5.7 mm |
| 0.60 | 6.50 | ~6 mm |

Roughly 3× more jitter events per second at 0.60 — reactivity and smoothness
traded off directly. **§12.5 breaks that trade-off**, which is why
`predict_before_end_s` is now listed in §7 as safe to raise.

**What the jitter actually was.** Measuring the *heading* of the commanded motion
(not its position or speed) found the real magnitude of the problem:

| commanded heading change per tick | within a plan | across a replan |
|---|---|---|
| left arm | 0.49° median | **93.9° median, 170.8° max** |
| right arm | 0.38° median | **74.3° median, 175.1° max** |

The arm was being told to reverse direction — up to a near-complete U-turn —
at every single replan. That, twice a second, is the zigzag.

### 12.5 Fixed — C¹ handoff (offset + rate decay)

Landed in [executors/bspline.py](src/flexivtrainer/rollout/executors/bspline.py).
On install, capture *both* mismatches against the outgoing plan evaluated at the
install instant, and retire them over `handoff_blend_s`:

```
commanded(t) = new(t) + p·W0(u) + v·T·W1(u),      u = (t − T0)/T
p = old(T0) − new(T0)                position gap
v = old'(T0) − new'(T0)              velocity gap
W0(u) = 1 − (10u³ − 15u⁴ + 6u⁵)      1→0, flat at both ends
W1(u) = u(1 − u)³                    0→0, unit slope at u=0
```

`W0` alone gives C⁰; the `W1` term is what makes it C¹, because alignment matches
pose and never rate. This is *not* the old position-fade — that blended toward
the plan linearly and injected a velocity step; these profiles enter and leave at
zero slope.

Two traps found the hard way:

- **Evaluate the outgoing plan at the install instant, not `_last_raw_command`.**
  The stale value is up to one control period old, so replaying it commanded a
  literal **dead stop** — measured commanded speed went 0.055 → **0.000** →
  0.033 m/s at every handoff. A one-tick stall twice a second reads as stutter,
  which is how the first version of this fix traded a position step for a
  velocity notch.
- **Store the pose actually sent**, blend included, since the next handoff aligns
  against it.

Measured result (episode 0, 12 s, 200 Hz, `predict_before_end_s = 0.3`):

| | handoff off | C¹ handoff |
|---|---|---|
| position step across replan | 4.8 mm (max 20.2) | **0.67 mm** (max 1.1) |
| heading change across replan | 93.9° median | **1.09°** (max 17.5°) |
| acceleration at replan | 194 m/s² (max 833) | **0.44 m/s²** (max 4.4) |
| boundary / interior ratio | 1098× | **2.0×** |
| within-plan accel (unchanged) | 0.18 (p99 3.3) | 0.22 (p99 2.4) |

At-replan acceleration is now *below* the within-plan p99, i.e. the handoff is no
longer distinguishable from ordinary motion. Knobs: `handoff_blend_s` (0.15,
`0` restores the old stepping behaviour) and `handoff_max_accel` (2.0), which
stretches the window for a large gap so peak added acceleration stays bounded.

Note this also makes `predict_before_end_s` safe to raise for reactivity — the
per-replan disturbance no longer scales with the replan rate. And
`time_align_error_threshold` was deliberately **not** tightened: it also drives
how hard `_align` searches, and with blending a small residual is harmless.

### 12.6 Fixed bug — phase alignment never ran

**Mechanism.** `_align` seeded `best_error` with the alignment error *at
`min_time`* and then entered its search loop only
`while best_error > self._alignment_threshold`. The threshold defaults to `0.1`,
which in these units is ≈100 mm. So whenever the naive start was already within
10 cm of where the outgoing plan had reached, the loop body never executed,
`minimize_scalar` was never called, and `_align` returned `min_time` unchanged.
The whole phase search was dead code on the path that matters.

**Evidence.** Across four `scripts/probe_bspline_rollout.py` runs, **88 of 88
replans** reported `alignment_error` between 0.0000 and 0.0399 — every single one
below the 0.1 threshold. The optimizer therefore ran **zero times**, and every
replan started the new curve at its own left boundary. Nothing in the telemetry
said so, because `alignment_error` is reported whether or not a search produced
it; `align_searched` exists now so this cannot recur invisibly.

**Why §12.5 masked it, and made it worse to notice.** Before the C¹ blend, a
mis-phased start showed up as a visible position jump and a heading snap — ugly,
but legible. With the blend, the arm leaves the outgoing pose at the outgoing
velocity and *glides smoothly onto the wrong phase* of the new curve. Motion
quality looks excellent while the trajectory is wrong, which is harder to spot and
worse for task success than the jitter it replaced. It also reframes §12.4's
measurements: the 6 mm gap and ≈90° heading change per handoff were recorded with
alignment effectively disabled, so starting each fresh curve at a boundary
velocity unrelated to the old plan is a large part of why the heading snapped.

**Fix.** The threshold now governs widening and warning only:

- always run at least one bounded `minimize_scalar` pass;
- keep `min_time` as a competing candidate, accepting the optimizer's result only
  when strictly better, so a non-unimodal objective can never make the choice
  worse than doing nothing;
- check the `time_align_max_fraction` rail *before* the ×1.5 widening budget. The
  old ordering (`while … and scale <= 20`) exited one widening step short of the
  rail at realistic window sizes, so the 20% cap was unreachable; it now binds and
  reports itself through `align_capped`.

Paired with it, the search window is seeded from **observation age** rather than
inference compute time (§5, item 1) — the observation the plan used was enqueued
up to one 30 Hz period before inference even started, so the old window was
roughly 2× too small even on the rare occasions the search ran.

**Measured impact: small. This did not fix the wrong trajectories.** Re-running the
probe after the fix (episode 0, 12 s, `predict_before_end_s = 0.3`):

| | before | after |
|---|---|---|
| replans that searched | 0 / 88 | **25 / 26** (first install has no predecessor) |
| `alignment_error` median | 0.0102 / 0.0145 (two runs) | 0.0075 / 0.0109 (two runs) |
| search gain over `min_time` | — | **median 0.000000**, max 0.0108, improved **10/25** |
| `align_capped` | — | 0 / 26 |
| heading change across replan | 1.09° | 1.56–1.98° |
| at-replan acceleration | 0.44 m/s² | 0.63 m/s² |

Two before-runs differ from each other by more (0.0102 vs 0.0145) than before
differs from after, because diffusion sampling is stochastic — so the handoff
metrics above are within run-to-run noise and no improvement can be claimed from
them. `search gain` is the noise-free measurement, and it says the search buys
**nothing in the median case** and helps in only 40% of replans.

The reason is the local-time knot convention (§1): each plan's domain starts at
roughly "now", so `min_time` usually *is* the correct phase. The residual ~7–11 mm
is not phase error at all — it is disagreement between what two consecutive
diffusion samples predict for the same trajectory. No amount of phase search
removes that; the C¹ blend absorbing it is the right treatment.

Keep the fix regardless: it was genuinely dead code, it helps in 40% of handoffs
by up to 11 mm, and it would matter a great deal under higher inference latency or
for a checkpoint whose knots drift from the local-time convention. But it means
**the wrong-trajectory cause is still open**, and the remaining suspects are the
degenerate lookahead targets and the disabled crop augmentation (§12.1) — both
training-side, both requiring a re-convert and retrain.

---

## Source map

| concern | file |
|---|---|
| curve fitting, chunking, rot6d ↔ quat, feature names | [src/flexivtrainer/data/bspline.py](src/flexivtrainer/data/bspline.py) |
| dataset conversion job | [src/flexivtrainer/jobs/convert_bspline_dataset.py](src/flexivtrainer/jobs/convert_bspline_dataset.py) |
| policy config + validation | [.../configuration_bspline_diffusion.py](src/flexivtrainer/policies/lerobot_plugins/configuration_bspline_diffusion.py) |
| policy model, obs queue, `predict_action_chunk` | [.../modeling_bspline_diffusion.py](src/flexivtrainer/policies/lerobot_plugins/modeling_bspline_diffusion.py) |
| training/rollout config schema | [src/flexivtrainer/policies/bspline_diffusion.py](src/flexivtrainer/policies/bspline_diffusion.py) |
| decode, align, C¹ handoff, sample, 200 Hz send loop | [src/flexivtrainer/rollout/executors/bspline.py](src/flexivtrainer/rollout/executors/bspline.py) |
| thread orchestration, metrics, teardown | [src/flexivtrainer/rollout/runners/bspline.py](src/flexivtrainer/rollout/runners/bspline.py) |
| preflight, wiring, target_hz resolution | [src/flexivtrainer/rollout/service.py](src/flexivtrainer/rollout/service.py) |
| offline probe (no robot): plan comparison + virtual-clock replay | [scripts/probe_bspline_rollout.py](scripts/probe_bspline_rollout.py) |
| tests | [tests/test_bspline_rollout.py](tests/test_bspline_rollout.py), [tests/test_bspline.py](tests/test_bspline.py) |
