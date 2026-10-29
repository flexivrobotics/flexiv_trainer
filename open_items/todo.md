# ACT with and without relative pose: notes and to do

Goal: train ACT with and without chunk-relative poses, on teleop recordings
and on UMI exports from flexiv_gdk's `03_convert_to_lerobot.py`, and roll
each out.

Chunk-relative SE(3) pose actions (UMI style) are implemented in
`src/flexivtrainer/policies/lerobot_plugins/relative_pose.py`. A training
request with `relative_pose: true` trains each action pose as
`T_cur⁻¹ · T_action`, where `T_cur` is the newest observed pose. Rollout
converts each chunk back with `T_cur · T_rel`.

| | Absolute pose | Relative pose |
|---|---|---|
| Teleop | Works (the original path) | Should work; not yet run end to end on a real teleop dataset (item 1) |
| UMI | Refused (see Done) | Works; run end to end on the `screw` export |

## Open items, in priority order

1. **Teleop with relative pose: verify, and decide on twist and wrench.**
   - Train and roll out a real teleop dataset with relative pose on. The
     plugin finds teleop pose names (`tcp_pose.*`, `tcp_rotation_6d.*`) and
     the unit tests cover that layout, but no real run has been done.
   - A full teleop recording has 23 state values and a 22-value action
     (pose, twist, wrench, target width). Only the pose becomes relative, so
     the twist and wrench actions stay in the robot base frame. That's
     workable on the same robot, but not frame-independent.
   - To match UMI (pose and width only), either record teleop with only the
     pose and gripper entries, or add a training-time column filter that
     drops twist and wrench from an existing dataset.

2. **Normalization of the extra rotation columns.** ACT normalizes state
   with mean/std. Some columns are nearly constant across a dataset:
   - the diagonal entries (`r1_x`, `r2_y`) of `tcp_motion_<ms>ms`, which
     stay near 1;
   - `tcp_rotation_wrt_start`, when the demos barely rotate.

   Such columns blow up into large z-scores at rollout. Check their std in
   the dataset's `meta/stats.json`. If they're tiny, drop those columns or
   skip normalizing them, as UMI does for rotations.

3. **Temporal ensembling with relative poses.** Rollout refuses a
   relative-pose ACT policy until `disable_temporal_ensemble` is set in the
   ACT rollout config. LeRobot's ACT ensembles inside `select_action`, before
   the postprocessor makes poses absolute, so it would average offsets from
   different anchors. To support it, convert each chunk to absolute poses
   with its own anchor when it is predicted, ensemble in the absolute frame,
   and re-orthonormalize the averaged rotation. That replaces LeRobot's
   ensembler in the rollout runner. Only worth doing if chunked execution
   (e.g. `chunk_size` 30, `n_action_steps` 15) jerks between chunks.

4. **Tactile grip force: training and rollout sensors differ.** Training
   uses the fingertips' recorded |Fz| with no tare, and rollout uses the
   electric gripper's motor force estimate. They can differ in offset and
   scale; the fingertips' open-gripper reading drifts by about 3 N over 30 s
   on the `screw` captures, as large as the grasp signal. The
   gripper's force limit is set by hand from the range the export prints.
   Options:
   - a calibration that maps the gripper's force to the fingertip force,
     from a few grasps measured with both;
   - a rollout start check that the gripper reads near 0 N while open.

5. **Practice step skills.** `practice/runner.py` predicts one action per
   tick through `predict_action`. That re-anchors every tick, so step skills
   with relative poses are refused. Supporting them means holding the anchor
   there the same way `rollout/observations.py` does. Only needed if
   practice step skills are used with these policies.

6. **Closer to UMI's inputs.** Not done:
   - random noise on the start pose (σ 0.05, per training sample), for
     `--rotation-wrt-start`;
   - UMI's per-sensor latency shifts. Training poses are SLAM poses at the
     image time, while rollout poses are about 1 ms old and images one camera
     latency old. A constant delay mostly cancels in the motion;
   - the previous gripper width as an input (only its change is given).

7. **Roll out absolute UMI policies.** Needs a base←map calibration applied
   to state and actions. Until then, they are refused (see Done).

8. **Lower priority limitations.**
   - Relative pose is refused for bspline_diffusion, smolvla and pi0. The
     supported families are act, diffusion and multi_task_dit.
   - The pre- and post-processor steps share the anchor through a
     module-level slot (`_ANCHORS`) keyed by `pair_id`. Two rollouts of the
     same checkpoint in one process would share that slot.

9. **Housekeeping.**
   - Commit the uncommitted work:
     - in flexiv_trainer, everything after `65b38d0` (state names, pose
       history, tactile, the training checkbox, the UMI absolute-pose
       refusal, docs);
     - in flexiv_gdk, the exporter changes.
   - Optional: a `--no-imu` exporter flag. ACT ignores `observation.imu`,
     but it is still exported whenever every capture has IMU data.

## Before rolling out a relative-pose policy

- **Turn off temporal ensembling** in the ACT rollout config
  (`disable_temporal_ensemble`), or rollout refuses the policy (item 3).
- **The robot's TCP is assumed to be the export's gripper tip.** Training
  poses are tip poses, `T_w_tip = Twb · T_b_tip`, from `imu_to_tcp.yaml`.
  Nothing checks that the robot's configured TCP matches; a different TCP
  would offset every relative action.
- **Name the cameras `left` and `right`.** The UMI export calls the images
  `observation.images.left` / `.right`, and rollout reads cameras by their
  configured `camera_names`. Give the robot's cameras those names in the
  robot config.
- **With tactile grip force:** set the gripper's force limit within the
  range the export printed, and check its open-gripper reading is near 0 N
  before trusting the policy (item 4).

## Done

- **Action names in the UMI export.** By default, flexiv_gdk's
  `03_convert_to_lerobot.py` writes the names the rollout executor expects
  for `right_arm`: `right_arm.tcp_pose.*` and
  `right_arm.tcp_rotation_6d.r1_*/r2_*` (rows of R) for the pose, and
  `right_arm.gripper.width` in metres. The width is the normalized opening ×
  0.094 m, assuming the opening is linear in width.
  - **Options:** `--arm-side SIDE` names another arm, and
    `--gripper-max-width-m M` changes the 94 mm width scale.
    `--arm-side none` writes the earlier UMI layout exactly.
  - **Re-export older datasets.** Datasets exported before this change use
    the old layout and need exporting again to roll out.

- **Rollout feeds the trained state columns.** Training writes the
  dataset's `observation.state` names beside each checkpoint as
  `state_names.json`. Rollout reads every metric of each arm, which is 23
  values: pose 9, twist 6, wrench 6, gripper width and force 2. It then feeds
  the policy only the named columns, in their trained order
  (`rollout/observations.py:resolve_state_layout`). The full state is still
  what the rollout recording and live checks see.
  - **Where the names come from.** A checkpoint without the sidecar takes
    the names from its training dataset's `meta/info.json`.
  - **Older checkpoints.** If neither the sidecar nor the dataset has names,
    rollout feeds the full state and matches only its width, as before.
  - **Retrain older UMI checkpoints.** UMI checkpoints trained before this
    change predate the sidecar, so they need retraining, or their dataset
    present, to roll out.

- **Tactile grip force, on or off at export.** The exporter writes no
  tactile data by default. With `--tactile-grip-force`, it adds one state
  value, `right_arm.gripper.force` in N. That is the mean of the two
  fingertips' Paxini |Fz| as recorded, with no tare. At rollout, that
  column is fed the electric gripper's measured `force`. The action never
  includes the force.
  - **A gripper with no force reading** makes rollout refuse such a
    checkpoint, with an error that names `gripper.force`.

- **Recent motion and rotation since start, as optional inputs.** The pose
  in `observation.state` is replaced by the identity (see Design notes), so
  the exporter can add UMI's two pose inputs as extra state columns, and
  rollout computes the same values from the robot's TCP poses
  (`rollout/pose_history.py`):
  - **`--recent-motion-ms MS`** (UMI uses about 50) adds
    `tcp_motion_<MS>ms.*`: the pose MS ms earlier in the current TCP frame,
    `T_cur⁻¹ · T_past`. It also adds `gripper.width_change_<MS>ms`. The delay
    is in the name, so rollout reads it from the checkpoint.
  - **`--rotation-wrt-start`** adds `tcp_rotation_wrt_start.*`, the rotation
    of `T_start⁻¹ · T_cur`. In training, `T_start` is the episode's first
    frame. At rollout, it is the pose at rollout start, and a takeover
    doesn't reset it.

- **Relative-pose checkbox on the training form.** The Training
  Configuration panel (`web/app.js`) shows `relative_pose` for act,
  diffusion and multi_task_dit, and sends it to `POST /training/start`. It
  is hidden when fine-tuning, which follows its checkpoint. With temporal
  ensembling also on, its hint warns that rollout will refuse ensembling.

- **UMI with absolute pose is refused.** UMI poses are in the SLAM map frame
  (IMU frame of the first keyframe, not gravity-aligned). A policy trained
  on them as absolute poses would drive the arm to map coordinates. A
  dataset counts as a UMI export when its `meta/flexiv_extras.json` has a
  `world_frame` (`data/lerobot_io.py:slam_world_frame`).
  - Training refuses it without relative pose, including fine-tuning an
    absolute checkpoint on it (`jobs/train_policy.py`).
  - Rollout refuses an absolute checkpoint whose training dataset is a UMI
    export, for checkpoints trained before this check
    (`rollout/checkpoint.py:checkpoint_slam_world_frame`). It can only tell
    when a local copy of that dataset exists.

## Design notes

- **The state pose is the identity.** The pose in `observation.state` is
  replaced by the identity rather than made relative to the newest frame. A
  policy with an observation history queues each frame after preprocessing,
  so a true pose history can't be fed. The recent motion and rotation since
  start inputs (see Done) stand in for it.
