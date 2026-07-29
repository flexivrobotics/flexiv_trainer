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
matches the pose it just commanded — so the handoff has no positional jump and
no velocity kink.

Two things make this different from the waypoint (ACT / plain diffusion) path:

1. **Time is continuous.** The commanded pose is `spline(t)` where `t` advances
   with real elapsed time, not with loop iterations. Jitter in the Python loop
   changes *when* you sample, not *what* the trajectory is.
2. **Replanning is phase-aligned, not position-faded.** The old smoothing hack
   (blend from current pose toward the new plan) injected velocity
   discontinuities at every replan. Here the executor searches for the spline
   parameter `t*` on the *new* curve whose pose matches the last command, and
   starts there.

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

Only `rows - (degree + 1)` = 16 − 4 = **12 control rows are active**. The
remaining 4 are padding that repeats the last control point — it exists so the
policy sees a fixed rectangular tensor without inflating the boundary-knot
multiplicity ([data/bspline.py:378-388](src/flexivtrainer/data/bspline.py#L378-L388)).

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
│   → executor.install(flat_304_vector, inference_latency_s)           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ EXECUTOR THREAD "rollout-bspline-executor"   control_hz = 200 Hz     │
│   t = start_time + (now - installed_at) * source_rate                │
│   raw = spline(t)                       (18 values, in 6D)           │
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
([executors/bspline.py:370-377](src/flexivtrainer/rollout/executors/bspline.py#L370-L377)):

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
([executors/bspline.py:392-410](src/flexivtrainer/rollout/executors/bspline.py#L392-L410)):

```python
position   = raw[position_indices]                       # 3
quaternion = rotation_6d_to_quaternion_wxyz(raw[rot_i])  # Gram-Schmidt → wxyz
robot.SendCartesianMotionForce(position + quaternion,
                               [0]*6,        # zero wrench
                               [0]*6,        # zero stiffness override
                               max_lin_vel, max_ang_vel,
                               max_lin_acc, max_ang_acc)
```

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

`install()` ([executors/bspline.py:331-368](src/flexivtrainer/rollout/executors/bspline.py#L331-L368))
handles it in two cases:

**First plan** — nothing has been commanded yet, so start at
`clip(0.0, min_time, max_time)`; local-time convention says 0 is "now".
Alignment error is 0 by definition.

**Every later plan** — call `_align()`
([executors/bspline.py:275-329](src/flexivtrainer/rollout/executors/bspline.py#L275-L329)),
which answers: *at what parameter `t*` on the new curve does the pose most
closely match `_last_raw_command` — the pose the executor commanded microseconds
ago?*

```
old curve  ────────────●━━━━━━ (t saturating / still running)
                       │ last_raw_command
                       │  ↓ find matching phase
new curve  ─────────────●──────────────────────────────────►
           min_time     t*                          max_time
                        └─ start_time; plan runs forward from here
```

Mechanics:

1. Seed the search window from inference latency:
   `initial_max = min_time + inference_latency_s * source_rate`. Intuition — the
   policy planned from a stale observation, so the matching phase is roughly
   *latency × rate* knot-units in. A 50 ms inference at 30 Hz ⇒ ~1.5 knot-units.
2. Hard-cap the window at `min_time + (max_time - min_time) * time_align_max_fraction`
   (default 0.2). **This is a safety rail**: it forbids skipping more than 20% of
   the new curve, so a bad match can never fast-forward the robot deep into the
   plan.
3. `scipy.optimize.minimize_scalar(..., method="bounded")` on the L1 pose
   difference over the aligned channels (positions + rotation-6D; gripper is
   excluded from alignment).
4. If the resulting max-abs error still exceeds `time_align_error_threshold`
   (default 0.1), widen the window ×1.5 and retry — up to 20× or until the cap
   is hit.
5. If it *still* exceeds threshold, install anyway but emit a warning, bump
   `handoff_warnings`, and surface it in the UI log
   ([runners/bspline.py:276-284](src/flexivtrainer/rollout/runners/bspline.py#L276-L284)).

Two design choices worth calling out. Alignment matches **pose, not velocity** —
continuity is C⁰ in the commanded signal, and smoothness beyond that comes from
the fact that both curves are cubic and the matched phase generally has similar
tangent. And the target is **`_last_raw_command`, not the measured robot pose** —
matching the *commanded* signal keeps the command stream continuous even if the
robot is lagging under load; a match against measurement would bake tracking
error into the plan.

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

From [policies/bspline_diffusion.py:39-44](src/flexivtrainer/policies/bspline_diffusion.py#L39-L44),
plus the inherited diffusion knobs:

| knob | default | what it does | when to change it |
|---|---|---|---|
| `control_hz` | 200 | executor send rate | lower if the robot's RDK link can't take 200 Hz; it does not change the path, only its sampling density |
| `speed_scale` | 1.0 | time warp on the curve | <1 to slow a too-aggressive policy; >1 hits velocity limits |
| `predict_before_end_s` | 0.06 | replan trigger — replan when this much curve remains | raise if `infer_ms` > 60 and you see `spline_remaining_s` pinned at 0 |
| `time_align_error_threshold` | 0.1 | acceptable handoff mismatch (max-abs over pos+rot6d) | lower to force harder alignment searching; raise to quiet spurious warnings |
| `time_align_max_fraction` | 0.2 | max fraction of the new curve alignment may skip | raise cautiously if latency is high and alignment keeps saturating the cap |
| `playback_speed` | 1.0 | scales `target_hz` **before** it becomes `checkpoint_fps` | logs a warning; overlaps with `speed_scale`, prefer one |
| `num_inference_steps` | 8 (in ckpt) | DDIM steps | the main lever on `infer_ms` |

`playback_speed` and `speed_scale` multiply into the same `source_rate` from
different directions (`_apply_playback_speed` scales `target_hz`, which becomes
`checkpoint_fps`; `speed_scale` multiplies it again in the executor). Pick one.

---

## 8. Reading the telemetry

Every planner tick appends a metrics row
([runners/bspline.py:324-342](src/flexivtrainer/rollout/runners/bspline.py#L324-L342)):

| field | healthy | what it means when it isn't |
|---|---|---|
| `send_hz` | ≈ `control_hz` (200) | well below ⇒ RDK calls are blocking or GIL contention |
| `missed_deadlines` | grows slowly / not at all | rapid growth ⇒ executor is starved |
| `spline_remaining_s` | deep sawtooth: full domain (~1.9 s) down to `predict_before_end_s`, repeating | **pinned at 0** ⇒ inference can't keep up; the robot is holding the old curve's end pose. A deep sawtooth is *normal* — plans are seconds long by design (§12.2), not a sign of trouble |
| `infer_ms` | < `predict_before_end_s × 1000` (60) | larger ⇒ raise `predict_before_end_s` or cut `num_inference_steps` |
| `alignment_error` | below `time_align_error_threshold` | consistently high ⇒ plans disagree; suspect OOD observations or stale state |
| `handoff_warnings` | 0, or rare | steady growth ⇒ every replan is a visible jump |
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

Then watch, in order: `send_hz` ≈ 200 → `spline_remaining_s` oscillating above 0
→ `alignment_error` staying under 0.1. If those three hold, the plumbing is
healthy and anything you don't like about the motion is the policy, not the
executor.

---

## 11. Contrast with the waypoint path, in one table

| | waypoint (ACT / diffusion) | B-spline |
|---|---|---|
| policy output | list of poses (chunk) | knots + control points (one curve) |
| `n_action_steps` | how many poses to consume | always 1 — one *plan* |
| what's commanded | pose (+ formerly raw twist) | pose sampled from `spline(t)`, zero wrench |
| command rate | planner-tied | independent 200 Hz thread |
| time base | loop iterations | wall clock × `source_rate` |
| replan continuity | position fade toward new chunk (velocity kinks) | phase alignment to `_last_raw_command` |
| late inference | queue drains, stale actions | `t` clips; holds final pose |
| rotation | quaternion | rotation-6D, projected at sample time |

---

## 12. Bugs found investigating "runs fine, wrong trajectory" (2026-07-29)

Symptom: the `merged_20260728_163313_nodepth_bspline` checkpoint executes
smoothly on hardware — no faults, no jerk, healthy `send_hz` — but the arms go to
the wrong places.

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

**Fix.** Expose `crop_ratio` in `TrainingConfig` (default ~0.9) and retrain.
Either drop `crop_shape` from the schema or document it as inert when
`resize_shape` is set.

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

### 12.3 Secondary divergences from the reference (lower confidence)

- `num_inference_steps = 8` vs the reference's `16`. Halving DDIM steps on a
  304-dim action costs sample fidelity. Testable at rollout without retraining.
- `use_group_norm = False` vs the reference's `obs_encoder_group_norm: True`. The
  usual argument for GroupNorm is that BatchNorm's running stats interact badly
  with EMA — but LeRobot's diffusion implementation has no EMA, so that
  reasoning does not apply here. Worth aligning; not demonstrated as causal.
- The reference has a `relative_knots` encoding (knots as first-knot plus
  successive deltas, `bspline_policy/common/knots.py`) with no equivalent here.
  Their stack-cube config leaves it `false`, so it isn't required — but its
  existence suggests knot conditioning was a real concern for them.

---

## Source map

| concern | file |
|---|---|
| curve fitting, chunking, rot6d ↔ quat, feature names | [src/flexivtrainer/data/bspline.py](src/flexivtrainer/data/bspline.py) |
| dataset conversion job | [src/flexivtrainer/jobs/convert_bspline_dataset.py](src/flexivtrainer/jobs/convert_bspline_dataset.py) |
| policy config + validation | [.../configuration_bspline_diffusion.py](src/flexivtrainer/policies/lerobot_plugins/configuration_bspline_diffusion.py) |
| policy model, obs queue, `predict_action_chunk` | [.../modeling_bspline_diffusion.py](src/flexivtrainer/policies/lerobot_plugins/modeling_bspline_diffusion.py) |
| training/rollout config schema | [src/flexivtrainer/policies/bspline_diffusion.py](src/flexivtrainer/policies/bspline_diffusion.py) |
| decode, align, sample, 200 Hz send loop | [src/flexivtrainer/rollout/executors/bspline.py](src/flexivtrainer/rollout/executors/bspline.py) |
| thread orchestration, metrics, teardown | [src/flexivtrainer/rollout/runners/bspline.py](src/flexivtrainer/rollout/runners/bspline.py) |
| preflight, wiring, target_hz resolution | [src/flexivtrainer/rollout/service.py](src/flexivtrainer/rollout/service.py) |
| tests | [tests/test_bspline_rollout.py](tests/test_bspline_rollout.py), [tests/test_bspline.py](tests/test_bspline.py) |
