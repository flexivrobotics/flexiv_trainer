# Relative pose actions: open items

Chunk-relative SE(3) pose actions (UMI style) are implemented in
`src/flexivtrainer/policies/lerobot_plugins/relative_pose.py`. A training
request with `relative_pose: true` trains each action pose as
`T_cur⁻¹ · T_action`, where `T_cur` is the newest observed pose. Rollout
converts each chunk back with `T_cur · T_rel`. The work is drafted, tested
and still uncommitted.

## Open items

1. **Done: action names in the UMI export.** By default, flexiv_gdk's
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
   - **Name the cameras `left` and `right`.** The export calls the images
     `observation.images.left` / `.right`, and rollout reads cameras by their
     configured `camera_names`. Give the robot's cameras those names in the
     robot config.

2. **Rollout refuses the UMI export's state width.** Rollout builds
   `observation.state` from every recorded metric of each arm: pose (9),
   twist (6), wrench (6), gripper width and force (2), which is 23 values. It
   only checks that width against the checkpoint
   (`rollout/observations.py:resolve_state_pose_format`). A UMI export has
   only pose and gripper width, so its checkpoints are refused. There are two
   possible fixes:
   - **Rollout picks the trained subset (preferred).** Rollout already knows
     each value's name (`rollout/recorder.py:recorded_state_names`). It would
     keep only the state names the checkpoint was trained on, which means
     saving those names with the checkpoint, the way `action_names.json` is
     saved now.
   - **The exporter pads the state to 23 values.** Twist could come from
     differentiating the SLAM poses, but there is no wrench or gripper force,
     so they would be zeros in training and real readings on the robot. That
     makes the policy's inputs out of distribution.

3. **Turn off ACT temporal ensembling for rollout.** Ensembling would average
   chunks predicted from different poses, so the rollout service refuses a
   relative-pose ACT policy until `disable_temporal_ensemble` is set in the ACT
   rollout config.

4. **Practice step skills are refused.** `practice/runner.py` predicts one
   action per tick through `predict_action`. That re-anchors every tick, so
   step skills with relative poses are refused for now. Supporting them means
   holding the anchor there the same way `rollout/observations.py` does.

5. **No UI toggle.** `relative_pose` is only a field on the
   `POST /training` request (`api/routes/training.py`). The training form in
   `web/app.js` needs a checkbox that sends it.

6. **The robot's TCP must be the same gripper tip.** Training poses are tip
   poses (`T_w_tip = Twb · T_b_tip`, from `imu_to_tcp.yaml`). Rolling out with
   a different TCP offsets every relative action.

## Known limitations of the design

- **No pose in the observation.** The pose in `observation.state` is replaced
  by the identity. A policy with an observation history queues each frame
  after preprocessing, so at rollout an older frame can't be expressed
  relative to the newest pose the way UMI does in training. The policy
  therefore sees no proprioceptive pose; motion comes only from the images.
  Restoring a pose history would mean loading one extra state frame per
  sample, through a dataset patch, and feeding the policy per-frame deltas
  `T_{k-1}⁻¹ · T_k`.
- **Some policy families aren't supported.** bspline_diffusion, smolvla and
  pi0 are refused. The supported families are act, diffusion and
  multi_task_dit.
- **One anchor slot per checkpoint.** The pre- and post-processor steps share
  the anchor through a module-level slot keyed by `pair_id`. Two rollouts of
  the same checkpoint in one process would share that slot.
