# Flexiv Trainer

Flexiv Trainer is an all-in-one, local-first software platform for training physical AI skills on Flexiv robots. It takes you through the full imitation-learning workflow — collecting teleoperation demonstrations, saving them as LeRobot episodes, merging episodes into training datasets, and launching policy training runs — from a single web UI on your own machine, with no cloud services required.

The steps to set up and use this software can be summarized as:

1. Install the software.
2. Start the server.
3. Set up robots and cameras in the UI.
4. Teleoperate and record demonstration episodes.
5. Review and merge episodes into one training dataset.
6. Choose a policy and start training.

## Supported Platforms

| OS            | CPU             |
| ------------- | --------------- |
| Ubuntu 22.04+ | x86-64, aarch64 |

## Software Requirements

1. Python 3.12 or 3.14. Orbbec cameras require 3.12; on 3.14 only RealSense
   cameras are supported.
2. GPU acceleration (optional but recommended).
3. System config for realtime scheduling (see below).

### Grant realtime privileges to non-root users (Linux only)

This is required to run the teleoperation module:

```bash
echo "${USER}    -   rtprio    99" | sudo tee -a /etc/security/limits.conf
echo "${USER}    -   nice     -20" | sudo tee -a /etc/security/limits.conf
echo "${USER} soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "${USER} hard memlock unlimited" | sudo tee -a /etc/security/limits.conf
```

> [!IMPORTANT]
> Restart your device for the changes to take effect.

## Hardware Requirements

1. A single- or dual-arm teleoperation setup eligible for high-transparency teleoperation.
2. Supported cameras connected to your computer. A single-arm setup uses up to
   3 cameras and a dual-arm setup up to 5. The default layout is 1 egocentric
   plus 1 in-hand camera per follower robot; the remaining slots are spare and
   start out inactive.

### Supported robots

- Flexiv Rizon series

### Supported cameras

- Intel RealSense cameras
- Orbbec cameras (Gemini 305/330 series and other OrbbecSDK v2 devices)

Both vendors can be used at the same time; each camera location is assigned a
device serial independently, so a setup may mix RealSense and Orbbec cameras.

Orbbec support needs the `pyorbbecsdk2` SDK, installed separately with
`--no-deps`:

```bash
pip install --no-deps 'pyorbbecsdk2>=2.1'
```

The `--no-deps` is required, not optional. `pyorbbecsdk2` declares hard pins for
its bundled *example* scripts (`av==12.3.0`, `open3d`, `pygame`, …) that
conflict with LeRobot's `av>=15,<16`; a plain install fails to resolve. The
binding itself does not import those packages and works against LeRobot's own
`av`.

The wheel is published for CPython 3.8–3.13 on Linux (x64/ARM64), Windows x64
and Apple Silicon — there is no 3.14 wheel and no source distribution, so
**Orbbec cameras need a Python 3.12 environment**. When the SDK is absent the
app still runs, reporting Orbbec as unavailable while RealSense cameras keep
working.

On Linux, install the Orbbec udev rules once so cameras are reachable without
root. Skipping this is the usual cause of a camera that appears in `lsusb` but
never connects in the app:

```bash
sudo ./scripts/install-orbbec-udev-rules.sh
```

Replug the camera afterwards so the new permissions apply.

The rules cover both the raw USB node and the V4L2 `/dev/videoN` nodes. Without
the V4L2 rule the camera is still *discovered* but fails to open with
`uvc_open failed ... Return Code: -6`, because those nodes default to
`root:video` and most accounts are not in the `video` group.

Use a USB 3.0 port; the Gemini 305 falls back to reduced frame rates on USB 2.0.

## Install

1. Create a Python 3.12+ virtual environment:

   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

3. Upgrade pip and install the package:

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

To also install development dependencies (for running tests and linting):

```bash
pip install -e ".[dev]"
```

### PyTorch on NVIDIA Jetson

The standard PyPI PyTorch package for `aarch64` does not support CUDA (GPU acceleration) and will fall back to CPU-only execution. To enable GPU acceleration on NVIDIA Jetson devices, you must install the Jetson-compatible PyTorch wheels and their corresponding CUDA dependencies.

For JetPack 7 / CUDA 13, you can replace the default CPU PyTorch after installation:

```bash
pip install --no-deps --force-reinstall \
  --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 \
  torch==2.10.0 torchvision==0.25.0
```

Refer to NVIDIA's official Jetson documentation for more details.

## Start Flexiv Trainer

Start the backend server from the same environment where you installed the package:

```bash
source .venv/bin/activate
flexiv-trainer-server
```

After startup, the terminal prints the URL for the web UI. By default this is:

```text
http://127.0.0.1:8000/
```

Open that URL in your browser to use the UI.

If the port is still occupied after closing a previous server session, clear leftover processes and start again:

```bash
source .venv/bin/activate
scripts/cleanup-server.sh
flexiv-trainer-server
```

## First-Time Setup In The UI

When the UI opens, you will start on the home page.

1. Enter the serial numbers for the two leader robots.
2. Enter the serial numbers for the two follower robots.
3. In the Camera Setup tile, choose how many cameras to use and name each one
   (see [Camera Setup](#camera-setup) below).
4. In the System Status tile, connect:
   - Teleop Service
   - Robot Data Service
   - Cameras
5. Confirm the system shows healthy status before moving on.

### Camera Setup

The `Camera Setup` tile on the home page controls how many cameras the rig uses
and what each one is called.

- **Count.** Pick 1 to 3 cameras in single-arm mode, or 1 to 5 in dual-arm mode.
  Only the first N slots are active; the rest are ignored everywhere.
- **Names.** Every slot is renameable. `ego`, `wrist`, `left_wrist`, and
  `right_wrist` are only what an unconfigured slot starts as, and spare slots
  start as `aux_1`, `aux_2`, and so on. Use letters, digits, and underscores
  (up to 32 characters). An empty, duplicate, or `*_depth` name is rejected and
  the slot reverts to its default.
- **Device assignment.** Each slot is assigned a device serial independently in
  the camera tiles, so renaming a slot keeps the camera already bound to it.
- **Per-mode layouts.** Counts and names are stored separately for single-arm
  and dual-arm mode, so switching modes back and forth preserves both layouts.

A camera named `X` becomes the dataset feature `observation.images.X`, with its
depth map (when enabled) as `observation.images.X_depth`. Renaming a camera
therefore changes the feature names in every episode recorded afterwards — a
policy trained on the old names expects a rig that still uses them.

The home page also shows the storage locations used by the app for episodes, merged datasets, and training outputs.

## Collect Demonstrations In Data Collection

Open `Data Collection` from the navigation bar.

On this page you can verify robot and camera health, view the camera feeds, monitor telemetry, run teleoperation, and record episodes.

### Recommended workflow

1. Check `System Status` and make sure teleoperation, robot data, and cameras are all connected.
2. Confirm that every active camera feed is updating at around 30 FPS.
3. If needed, use `Home All Robots` before starting teleoperation.
4. In `Teleoperation Control`, click `Start` to begin teleoperation.
5. In `Episode Recording`, choose which entries you want to record.
   - The default selection includes all available camera observations, robot states, and robot actions.
   - Use `Select All` or `Deselect All` if needed.
6. Click the recording `Start` button when you are ready to capture a demonstration.
7. Perform the task through teleoperation.
8. Click the recording `Stop` button when one episode is complete.
9. Click `Save Episode` to keep the recording, or `Discard Episode` to throw it away.

Each episode stores the follower stiffness it was recorded at. Set it with
`FLEXIV_TRAINER_TELEOP__FOLLOWER_CARTESIAN_STIFFNESS`; see [Stiffness](#stiffness).

Saved episodes are written to the episode storage directory shown on the home page. By default, episodes are stored under:

```text
.local/episodes/
```

## Load And Merge Episodes In Data Processing

Open `Data Processing` from the navigation bar.

### Step 1: Load episodes

1. Click the add button in `Load Episodes`.
2. In the episode browser, select one or more saved episode datasets.
3. Use `Select All` or `Deselect All` if you want to quickly toggle the full list.
4. Click `Load`.

### Step 2: Review selected episodes

1. Click an episode in the list to review it.
2. Use the checkboxes to choose which episodes should be merged.
3. Use `Select All` or `Deselect All` to toggle the current list.
4. Click `Merge Selected Episodes`.

### Step 3: Review the merged dataset

The merging will take some time depending on the actual dataset size. When done:

1. Review the merged dataset.
2. Click `Next` if the merged dataset looks correct.

Merged datasets are stored under:

```text
.local/datasets/
```

## Convert TCP Actions To B-Spline Training Targets

The experimental B-spline converter creates a new LeRobot dataset while
leaving the source dataset unchanged. Observations and videos are copied
without re-encoding. The copied dataset's discrete TCP `action` is replaced by
one flattened B-spline parameter segment per frame.

When **B-Spline Diffusion** is selected in Policy Training, the Start button
runs this conversion automatically in the background and launches training
only after the converted dataset validates successfully. Other policy types
continue to train directly from the selected dataset. The command below remains
available when a reusable converted dataset is wanted ahead of time.

With the default cubic configuration, one arm is represented by XYZ plus
rotation-6D. The parameter matrix has shape:

```text
(chunk_size + 2 * degree, 1 knot + 9 Cartesian controls)
= (16, 10)
```

Run the converter on a merged dataset:

```bash
flexivtrainer-convert-bspline \
  .local/datasets/my_task \
  .local/datasets/my_task_bspline
```

TCP pose axes are detected from the action names. To select arms explicitly:

```bash
flexivtrainer-convert-bspline \
  .local/datasets/my_task \
  .local/datasets/my_task_bspline \
  --side left_arm \
  --side right_arm
```

Useful fitting options include:

```text
--degree 3
--chunk-size 10
--stride 1
--max-error 0.002
--smoothing 1e-12
```

The converted dataset stores its representation and per-episode reconstruction
report in `meta/bspline.json`. With the default settings, every episode must
contain at least 12 frames.

> [!IMPORTANT]
> This command prepares trainable spline-parameter targets. A B-spline-aware
> policy/rollout adapter is still required to reshape, validate, align, and
> evaluate predicted parameters. Do not send a checkpoint trained on this
> converted action feature through the existing Cartesian waypoint executor.

## Prepare Demonstrations For A Step Skill

A step skill makes one small, bounded move per step, for example 10 times a
second, and can practise on the robot. Its **practice space** says what it sees
and what one unit of its action means. The space is stored with the skill, in
`practice_space.json`:

```json
{
  "rate_hz": 10,
  "step_m": {"x": 0.002, "y": 0.002},
  "step_deg": {"x": 0.5, "y": 0.5, "z": 0.5},
  "force_n": {"z": 80},
  "views": ["wrist_1", "wrist_2"],
  "crops": {"wrist_1": [[140, 40, 160, 256], [360, 40, 160, 256]]},
  "image_size": [128, 176],
  "last_actions": 3,
  "wrench_samples": 10,
  "sense_hz": 30
}
```

- **The action** is a vector in [-1, 1]. Its columns are the moves (`step_m`,
  largest move per step, in metres), then the rotations (`step_deg`, in
  degrees), then the force targets (`force_n`, in newtons), each in tool-axis
  order x, y, z. A move is relative to where the tool is when the step starts,
  in the tool frame. A force axis is held where it is and pressed with its
  target. An axis in none of the three is held where it is.
- **What it sees of each camera:** the frame, cut to its `crops` (strips given
  as `[x, y, width, height]` in frame pixels, placed side by side, left to
  right) and resized to `image_size` (height, width). Both are optional. They
  are applied by the space, not by the model, because practice keeps every
  image it learns from in memory: two whole camera frames are about 7 MB a
  step. `frame_size` records the size the strips refer to. The converter fills
  it in from the recording, and on the robot each frame is first resized to it.
- **What else it sees:** the tool's pose and velocity
  (`observation.state`); the last `wrench_samples` force readings, taken at
  `sense_hz`, as the force the tool applies, in the tool frame
  (`observation.wrench`); and its last `last_actions` commands, re-expressed
  relative to where the tool is now (`observation.last_actions`).

The converter turns a teleop recording into demonstrations in that space. You
choose nothing per column: everything is computed from the recorded pose,
velocity, wrench and commanded pose, so the recording must include those
entries. `sense_hz` must match the recording's rate.

```bash
flexivtrainer-practice-demos \
  .local/datasets/my_task \
  .local/datasets/my_task_steps \
  --space practice_space.json
```

At each step, the action heads for the pose the operator commanded one step
later, and the force target is the force the tool applied one step later. Both
are clipped to the bounds. Every episode counts as a success that ends on its
last step. With `--check NAME`, each episode instead ends at the step that
reaches that check's first labelled frame
([learned checks](#learned-checks)); episodes labelled `--never`, or not
labelled, are left out.

The new dataset carries the space, the source's stiffness (`control.json`) and
a report (`meta/practice_demos.json`). Read the report's `clipped` first: for
each action column, the share of steps on which the demonstrations outran the
bound, and `covers_95_*`, the bound under which 95% of them fit. A bound that
clips more than 5% of steps is too small for these demonstrations, and the
command warns about it.

The observation is built by the same code that steps the skill on the robot,
so a skill trained on these demonstrations sees there the numbers it was
trained on.

### Roll a step skill out

A checkpoint with a `practice_space.json` beside its weights is a step skill,
and Policy Rollout runs it as one; nothing else needs choosing. Its policy
type must be importable (see [Policy Types From Other Packages](#policy-types-from-other-packages)).

- **One arm**, in single-arm mode, with every camera the space lists. The
  skill does not command the gripper.
- **One command per step.** At each step the skill sees the arm and the
  cameras and sends one bounded move. Between steps the arm is read at
  `sense_hz`, which also feeds the force window. `rollout.max_steps` counts
  the skill's steps.
- **Force axes.** The space's force axes are put under force control in the
  tool's own frame before the first step, and released when the rollout ends
  or an operator takes over. A force target is sent as the force the sensor
  should read, which is the negative of the force the tool applies.
  `rollout.force_axis_max_speed_m_s` (default 0.02) caps the speed along a
  force axis when contact is lost.
- **Everything else works as for any policy**: the stiffness the checkpoint
  stores, the safety envelope (each step's target is checked before it is
  sent), taking over with the leader arm, learned checks scored on every tick
  (one with `end_on_pass` ends the skill), and recording.
- **A recorded step rollout is a teleop recording.** Each tick stores the
  teleop state and, as its action, the pose the controller was given: the
  step's target, or the leader's while an operator drives. Under the
  operator, each step is also measured the way the converter labels a
  demonstration, and becomes one of the skill's last commands.

### Practise a step skill on the robot

Practice improves a step skill on the robot, with an operator who can take
over. The trainer acts; a learner learns. The learner is LeRobot's own
(`python -m lerobot.rl.learner`), running on a GPU machine, with the skill's
checkpoint as its starting policy and the converted demonstrations as its
dataset. The trainer reaches it over LeRobot's actor-learner link (gRPC), so
the trainer holds none of the skill's own training code.

Each episode:

1. **Reset.** The tool backs off along its own axis, then goes slowly to a
   start drawn at random in the plan's box. All axes are motion-controlled
   on the way. Without a box, an episode starts where the arm is.
2. **Act.** The skill steps as in a rollout, but samples its actions to
   explore. Take over with the leader arm whenever it needs help: your steps
   are measured like demonstrations and marked as interventions, and the
   learner keeps them among its demonstrations too.
3. **Judge.** The plan's learned check is scored on every tick. The step at
   which it passes earns the reward and ends the episode; otherwise the
   episode is cut short at `max_episode_steps`.
4. **Learn.** The episode's steps go to the learner, and the newest weights
   it has sent are loaded before the next episode.

After a success, practice waits until you have done what the reset cannot,
such as unlatching a part, and send `POST /rollout/practice/continue`.

Start it with `POST /rollout/practice`:

```json
{
  "checkpoint_path": "<the skill's checkpoint>",
  "plan": {
    "reward": {"seated": {"kind": "learned", "model": "checks/seated", "hold_s": 0.2}},
    "max_episode_steps": 100,
    "learner_host": "192.168.1.20",
    "learner_port": 50051,
    "episodes": 0,
    "start": {
      "min_m": [0.40, -0.02, 0.30],
      "max_m": [0.44, 0.02, 0.32],
      "yaw_deg": 6,
      "retreat_m": 0.03,
      "speed_m_s": 0.05
    },
    "wait_after_success": true,
    "seed": 0
  }
}
```

`episodes` 0 practises until stopped. `GET /rollout/status` reports practice
under `practice`: its state, the totals (episodes, successes, steps, steps
under the operator, weights loaded) and the last episodes. Stop practice as
you stop a rollout. There is no practice page in the UI yet.

## Start Training

Open `Policy Training` from the navigation bar, then work through the steps.

### Step 1: Load the training dataset

1. Click the add button to browse merged datasets.
2. Select the dataset you want to train on.
3. Click `Next`.

### Step 2: Review the dataset

1. Review the dataset preview (whole dataset or a single episode).
2. Click `Next`.

### Step 3: Choose a policy

1. Select a training policy.
2. The training output directory is generated automatically from the dataset name, policy, and a timestamp, and is shown under `Training Output Directory`.
3. Click `Next`.

### Step 4: Run training

1. Under `Computation Device`, leave the device on `auto` or pick a specific device. Use the reload button to re-evaluate available devices.
2. Click the start button in `Training Control`.
3. Watch progress in the progress bar and the terminal log pane. Use `Pause`/`Resume` while a run is live.

Training outputs are written to the output directory shown in Step 3. By default these live under:

```text
.local/training/
```

## Checkpoint Compatibility

Recorded TCP orientation moved from a quaternion to a rotation-6D
representation, because the driver reports `q` and `-q` interchangeably for the
same rotation and the sign flips mid-trajectory.

Rollout accepts checkpoints in either representation. Which one a checkpoint
uses is detected when it loads:

- **Actions** are read from the checkpoint's axis names, recovered from its
  `action_names.json` sidecar, its policy config, the training dataset, or the
  dataset's Hub metadata. When no names are recoverable, the layout is inferred
  from the action width instead.
- **Observations** are matched against the width the checkpoint declares for
  `observation.state`, so a pre-rotation-6D checkpoint is fed the driver
  quaternion it was trained on. Rollout logs `Legacy quaternion state layout`
  when this applies.

Datasets are only ever recorded in rotation-6D, so this affects rollout of older
checkpoints, not new recordings. Datasets recorded before the change are
converted on read when they are used as B-spline training targets.

## Policy Types From Other Packages

A policy family maintained outside this repository registers its LeRobot policy
type when its package is imported. List the package so rollout imports it
before loading a checkpoint. Give such types their own rollout settings, which
otherwise default to chunked execution:

```bash
export FLEXIV_TRAINER_POLICIES__PLUGIN_PACKAGES='["my_models.family_a"]'
# For a policy that blends its chunks itself when stepped every tick:
export FLEXIV_TRAINER_POLICIES__PLUGIN_ROLLOUT__N_ACTION_STEPS=1
export FLEXIV_TRAINER_POLICIES__PLUGIN_ROLLOUT__REPLAN_STEPS=0
```

The package must be installed in the same environment as the trainer, and its
checkpoints must carry the standard `policy_preprocessor.json` and
`policy_postprocessor.json`.

## Record And Evaluate Rollouts

A training loss says how well a policy imitates its demonstrations, not how
often the robot finishes the task. The `Recording & Evaluation` panel on the
`Policy Rollout` page measures that.

### Record a rollout

Tick `Record single rollouts` before `Start`. The rollout is saved under
`.local/rollouts/adhoc/<timestamp>_rollout/` as a LeRobot dataset with the same
`observation.state` and `action` columns as a teleoperated episode, so it can
be reviewed and merged like any demonstration. Two files sit beside the data:

- `meta/rollout_ticks.jsonl`: one line per control tick, with the measured
  pose, force, joint positions and torques, inference time, and whether the
  plan was fresh.
- `meta/rollout_outcome.json`: the checkpoint, why the rollout stopped, any
  safety stop, and your verdict.

The images are the frames the policy saw, at the resolution it was given them.
B-spline rollouts cannot be recorded yet.

When the rollout ends, the panel asks how it went: `Success`, `Failure` (with
what went wrong), or `Can't tell`. A new rollout cannot start until the last
recorded one has a verdict. A verdict given later from memory is worth less
than one given now.

### Run a planned evaluation session

To compare checkpoints, write a plan before running anything, at
`.local/rollouts/<session>/session.json`:

```json
{
  "about": "new defaults against old",
  "arms": {
    "A": {"checkpoint": "training/run_a/checkpoints/100000/pretrained_model"},
    "B": {"checkpoint": "training/run_b/checkpoints/100000/pretrained_model"}
  },
  "conditions": [
    {"id": "home", "description": "start from HOME"},
    {"id": "offset", "description": "part 5 cm to the left of its usual place"}
  ],
  "trials_per_condition": 10,
  "seed": 1,
  "blind": true,
  "failure_modes": ["missed the target", "stalled", "dropped the part"]
}
```

Choose the session in the panel. `Start trial` runs the next planned trial and
tells you which start condition to set up first. Trials are shuffled by the
seed, in blocks that cover every arm and condition once before any repeats.
With `"blind": true` the page shows only the arm's label, never its
checkpoint. An arm may also set `"send_twist"` and `"task"`, or use
`"source": "hub"` with a `"repo_id"`.

The panel shows each arm's success rate with a 95% Wilson interval.
`GET /rollout/evaluations/<session>` also returns paired comparisons between
arms on the same conditions, with an exact p-value. As a guide, telling 50%
from 80% takes about 33 paired trials, or about 39 per arm unpaired. Telling
60% from 80% takes about 80 per arm.

### Automatic checks

A session can also declare checks: yes/no questions the robot's own
measurements can answer, scored when each trial's recording is saved. There
are two kinds. `settled_in_box`, scored from the trial's `rollout_ticks.jsonl`:
the tool comes to rest inside a box, in the arm's base frame, for at least
`hold_s`. And `learned` (next section), scored from the recorded frames.

```json
"checks": {
  "placed": {
    "kind": "settled_in_box",
    "box_min_m": [0.40, -0.05, 0.10],
    "box_max_m": [0.44, -0.01, 0.12],
    "hold_s": 0.5,
    "max_speed_m_s": 0.003,
    "when": "any"
  }
}
```

`when` is `any` (at any point in the rollout) or `end` (still there when the
rollout ends). `arm` picks an arm by its position in the tick log; the default
is the first. A box check can also act while a trial runs, with `live` and
`end_on_pass` as a learned check does (next section); it is then scored from
the measured pose on every tick.

Your verdict stays the measurement. The panel shows how often each check
agrees with the verdicts, how many failures it passed and how many successes
it failed. It never shows a check's answer for the trial you are judging,
because a judge who sees the machine's answer first is no longer independent.
A check that agrees with you over enough trials can then stand in for you, for
example as the reward for learning on the robot.

To tune a check's thresholds on trials you have already judged, without
running the robot again:

```bash
python -m flexivtrainer.evaluation.rescore <session>                       # the plan's checks
python -m flexivtrainer.evaluation.rescore <session> --checks other.json   # try other values
python -m flexivtrainer.evaluation.rescore <session> --checks other.json --write
```

Without `--write` nothing on disk changes. With it, each trial's outcome gets
the new results; verdicts are never touched.

### Learned checks

Some states are not settled by where the tool is: a latch has closed, a part
sits flush, a cap is on. A learned check is a small classifier you train from
labelled frames of your own recordings, demonstrations or rollouts.

**Label.** For each episode, give the first frame at which the state holds, or
`--never` when it never does. Frames from that one on are examples of "yes",
frames before it of "no", and a short margin just before it is left out.
Near misses labelled `--never` are the most useful labels you can give.
Labels are stored in the dataset's `meta/check_labels.json` and follow their
episodes through a merge.

```bash
python -m flexivtrainer.evaluation.learned_check label <dataset> --check closed --episode 3 --first-frame 412
python -m flexivtrainer.evaluation.learned_check label <dataset> --check closed --episode 4 --never
python -m flexivtrainer.evaluation.learned_check labels <dataset>
```

**Train.** Choose what the check looks at: cameras, each optionally cut to a
box given as fractions of the frame (`x0,y0,x1,y1`), and `observation.state`
columns such as a force axis. Each frame goes through a frozen, pretrained
image network; only a small head is trained, which takes minutes.

```bash
python -m flexivtrainer.evaluation.learned_check train --check closed \
    --data <dataset> --data <another dataset> --out <storage>/checks/closed \
    --camera wrist --crop wrist=0.2,0.4,0.8,1.0 \
    --state single_arm.tcp_wrench.fz
```

`--data` also takes a session folder. With `--verdicts`, a judged trial
without a label counts too, when its verdict is about the check's state: a
failure gives "no" frames throughout and a success gives "yes" frames at the
end.

Training holds out whole episodes (`--holdout`, default 20%) and prints, for
each one, whether the check would have passed and whether that was right. Read
`false_passes` first. A false pass is a pass that begins on a frame labelled
"no": on an episode that never gets there, or before the labelled first frame
of one that does. Whatever runs after the check would then start from a
failure. `sweep` repeats the totals for thresholds from 0.1 to 0.9.

**Use.** Name the folder in a session plan, as an absolute path or relative to
the storage root:

```json
"checks": {
  "closed": {"kind": "learned", "model": "checks/closed", "hold_s": 0.2}
}
```

It passes when its probability stays at or above `threshold` (default: the
one it was saved with, 0.5) for `hold_s`, at any time or, with `"when":
"end"`, at the end. Like any check, it earns trust by agreeing with your
verdicts.

Two options make it act while a trial runs:

- `"live": true` also scores it on every tick. Each tick of
  `rollout_ticks.jsonl` records its newest probability, about a tick behind,
  because a worker thread scores the newest tick while the control loop
  carries on. The outcome records whether and when it passed (`live_checks`).
- `"end_on_pass": true` (implies `live`) ends the trial when it passes, under
  the policy or under the operator after a takeover. The stop reason is
  `check <name> passed`. A trial whose live check cannot be loaded does not
  start.

The trial's verdict and the check's agreement still come from scoring the
saved recording, which sees every frame. To score recordings outside a
session:

```bash
python -m flexivtrainer.evaluation.learned_check score <dataset> --model <storage>/checks/closed
```

### Velocity feedforward and the safety envelope

`Send velocity feedforward` streams the policy's predicted twist to the
controller along with the pose target. Switching it off sends zero twist, which
stops a policy that stalls in contact from pushing on the part. The default is
`FLEXIV_TRAINER_ROLLOUT__SEND_TWIST` (on).

A command with a non-finite value always stops the rollout before it reaches
the arm. These bounds are off until you set them for your cell:

| Setting | Stops the rollout when |
|---|---|
| `FLEXIV_TRAINER_ROLLOUT__WORKSPACE_MIN_M`, `..._WORKSPACE_MAX_M` (e.g. `[0.3,-0.2,0.1]`) | a commanded position leaves the box, in each arm's base frame |
| `FLEXIV_TRAINER_ROLLOUT__MAX_REFERENCE_OFFSET_M` | a new plan's first target is further than this from the arm |
| `FLEXIV_TRAINER_ROLLOUT__MAX_FORCE_N` | the measured external force exceeds this |
| `FLEXIV_TRAINER_ROLLOUT__STALE_PLAN_S` | no fresh plan arrives within this many seconds |

A pose target is also a force target. Under impedance control, a target 2 cm
away pushes with stiffness times 2 cm, so set the reference offset above the
largest offset your task presses with. Safety stops are recorded in the
outcome. The envelope applies to waypoint policies, not yet to B-spline ones.

### Stiffness

A pose-target policy learns targets that sit beyond where the arm ends up, by
force divided by stiffness. Run it at the stiffness its demonstrations were
recorded at, or the same targets press with a different force. A stiffer
spring presses harder. So the stiffness goes with the data, from the
recording to the rollout:

1. **Teleoperation sets it.** `FLEXIV_TRAINER_TELEOP__FOLLOWER_CARTESIAN_STIFFNESS`
   (`[kx,ky,kz,krx,kry,krz]`, N/m and N·m/rad) is sent to the follower each
   time teleoperation starts, after its initialization and before its loop
   starts. Each value must be between 0 and the robot's nominal stiffness.
   Unset, the teleop controller keeps its own default.
2. **Each recording stores it.** Every episode has `meta/control.json`: the
   follower stiffness, the contact-wrench limit and the force feedback scale it
   was recorded under. A value the trainer did not set reads
   `"controller default"`. Recorded rollouts store theirs the same way.
3. **Merging checks it.** Episodes recorded under different settings are not
   merged; merge them into separate datasets. A policy runs at one stiffness,
   so data recorded at another would teach it the wrong force. Episodes
   recorded before `control.json` existed count as recorded at the teleop
   controller's default.
4. **Training copies it.** The checkpoint gets the dataset's settings as
   `pretrained_model/control.json`. Fine-tuning a checkpoint on data recorded
   under different settings is refused, for the same reason.
5. **The rollout applies it.** In order:
   - `FLEXIV_TRAINER_ROLLOUT__CARTESIAN_STIFFNESS`, when set. If the checkpoint
     was trained at a different stiffness, the log and the page warn
     "Trained at X, running at Y".
   - Otherwise, the stiffness the checkpoint stores.
   - Otherwise the rollout does not start. This protects you from running at
     the controller's default, which can be several times stiffer than the
     demonstrations. Set `FLEXIV_TRAINER_ROLLOUT__ALLOW_CONTROLLER_DEFAULT_STIFFNESS=true`
     to run at that default anyway. A checkpoint trained on data recorded at
     the teleop default also needs one of these, because the value of that
     default is not known.

The page shows the stiffness the rollout uses and where it came from, and
each rollout's outcome stores it under `stiffness`. When you take over with the
leader arm, the follower uses the same stiffness. Set the rollout's
contact-wrench ceiling with `FLEXIV_TRAINER_ROLLOUT__MAX_CONTACT_WRENCH` (N and
N·m); unset, the controller's default applies.

A rollout zeroes the follower's force/torque sensor when it starts, unless the
tool already measures more than `FLEXIV_TRAINER_ROLLOUT__FT_ZERO_MAX_FORCE_N`
(default 1.5 N). Above that, the tool is touching something, and zeroing would
offset every later force reading by that force. That includes the force limit,
a force-aware policy's input and a learned check. The log then says "F/T sensor
not zeroed" with the force it measured. If that appears with the tool in free
air, the sensor has drifted: zero it from teleoperation, or raise the
threshold. Unset, the sensor is always zeroed.

### Taking over with the leader arm

You can take the follower from a running policy with the leader arm, the same
way you teleoperate it to collect data, and give it back later. Nothing is
disconnected.

1. Connect teleoperation on `Home` before you start the rollout. The rollout
   then stops the teleop loop and drives the followers through the same
   connection. (Without it, the rollout connects as before and cannot be taken
   over; the page says why.)
2. Press `Take over` on the `Policy Rollout` page. The policy stops at once.
   Teleoperation starts without zeroing the force sensors, because the tool may
   be touching the part, and the leader is engaged. The follower copies the
   leader's motion from where both arms are now, so you do not need to move the
   leader onto the follower first. The gripper keeps its state until you press
   the leader's gripper trigger.
3. Press `Hand back`. The leader disengages, teleoperation stops and the robots
   hold. The rollout takes the follower back with its own settings, and the
   policy starts again with a fresh plan from where you left the arm. Nothing
   it planned before the takeover is sent.

The page shows who has control and how long the last switch took. If a switch
fails, the robots hold and the policy stays stopped: press `Hand back` again or
stop the trial. The safety envelope keeps watching force and the workspace box
while you drive; a safety stop disengages the leader and ends the trial. The
stale-plan check rests while you drive. `MAX_CONTACT_WRENCH` also limits the
follower while you drive. After the rollout, teleoperation is still connected
but stopped; start it again as usual.

To switch from the leader arm itself, set a free digital input of the leader:
`FLEXIV_TRAINER_ROLLOUT__TAKEOVER_TRIGGER_CHANNEL` (the input),
`..._TAKEOVER_TRIGGER_ARM` (which arm's leader, default `0`),
`..._TAKEOVER_TRIGGER_ACTIVE` (`high` or `low`) and
`..._TAKEOVER_TRIGGER_DEBOUNCE_S` (default `0.05`). Each press toggles control.
An input that a gripper trigger already uses is refused.

In a recorded rollout, every line of `rollout_ticks.jsonl` says who had
control (`policy`, `operator` or `switching`) and whether it was an
intervention, and the tick where a switch finished carries its duration. While
you drive, the `action` column holds what teleoperation records, so a corrected
part reads like a demonstration. `rollout_outcome.json` counts the takeovers,
the ticks and seconds you drove, and every switch's duration. B-spline policies
cannot be taken over yet.

## Run A Task As A Chain Of Steps

A longer task can be split into short steps, each a skill or a generic
primitive, joined by checks. Each step may have a start check, which must pass
before it runs, and an end check, which ends it when it passes. The checks are
declared as in an evaluation session: `settled_in_box` or `learned`.

```json
{
  "steps": [
    {"name": "approach", "skill": "skills/approach", "end": "ready", "max_s": 30},
    {
      "name": "insert",
      "skill": "skills/insert",
      "start": "ready",
      "end": "home",
      "max_s": 20,
      "tries": 3,
      "back_off": {
        "moves": [
          {"tool_m": [0, 0, -0.02]},
          {"box_min_m": [0.40, -0.01, 0.31], "box_max_m": [0.42, 0.01, 0.33]}
        ],
        "max_force_n": 25
      }
    },
    {"name": "lift", "primitive": {"moves": [{"base_m": [0, 0, 0.05]}]}}
  ],
  "checks": {
    "ready": {"kind": "settled_in_box", "box_min_m": [0.40, -0.01, 0.31], "box_max_m": [0.42, 0.01, 0.33], "hold_s": 0.3},
    "home": {"kind": "learned", "model": "checks/home", "hold_s": 0.2}
  }
}
```

- **A skill step** names a checkpoint (absolute, or relative to the storage
  root) and runs as a rollout until its end check passes, or until `max_s`
  seconds are up. It needs an end check: nothing else says it is done.
- **A primitive step** moves the tool in straight lines, slowly, with every
  axis motion-controlled. A move is one of `tool_m` (an offset in the tool's
  own frame), `base_m` (an offset in the base frame), or a box
  (`box_min_m`, `box_max_m`: to the point of the box nearest the tool). A move
  may also set `rotation` ([qw, qx, qy, qz]) and `speed_m_s` (default 0.02).
  The step is done when its moves are, and its end check, if any, has passed.

**When a step fails**, the chain backs off and tries again:

1. The step's `back_off` primitive takes the arm back towards where the step
   starts, and the step's start check must pass again.
2. The step runs again, up to `tries` times in all.

After its last try, the chain stops and calls you. It also calls you at once
when a back-off meets more force than its `max_force_n` (a half-latched part
needs a person, not more force), when a start check does not pass, and when
the safety envelope stopped a try. Do what the chain could not, with the leader
arm through teleoperation or by hand, then send
`POST /rollout/chain/continue` with `{"action": "retry"}` to run the step again
from its start check, or `{"action": "skip"}` to count it as done.

Start a chain with `POST /rollout/chain` and `{"plan": {...}}`; stop it as you
stop a rollout. Each step runs as a rollout of its own, so the leader arm can
take over a skill as in any rollout; primitives cannot be taken over. A
back-off and a start check move at their skill's stiffness. A primitive step
moves at the plan's `stiffness`, or `ROLLOUT__CARTESIAN_STIFFNESS`, which wins
over both as it does for a skill. `GET /rollout/status` reports the chain under
`chain`: its state, the step and try, and why it waits for you. Every run is
written as it goes to `<storage>/rollouts/chains/<time>.json`: each try, how it
ended and in what time, its back-offs, your takeovers and your calls. A chain
drives one arm. There is no chain page in the UI yet.

## Typical End-To-End Session

For a normal session, the order is:

1. Start `flexiv-trainer-server`.
2. Open the web UI.
3. Configure robot serial numbers and the camera layout on `Home`.
4. Connect teleop service, robot data service, and cameras.
5. Open `Data Collection`.
6. Start teleoperation and record one or more demonstration episodes.
7. Save the episodes you want to keep.
8. Open `Data Processing`.
9. Load and merge the saved episodes into one training dataset.
10. Open `Policy Training`.
11. Select the merged dataset and choose a policy.
12. Pick a computation device and start training.

## Open UI From Another Device

If you want to use the UI from another device on the same LAN, set a public base URL before starting the server:

```bash
export FLEXIV_TRAINER_PUBLIC_BASE_URL=http://<backend-host-ip>:8000
flexiv-trainer-server
```

Then open that public URL in a browser on the other device.

## API Documentation

The backend API doc can be accessed at:

```text
http://127.0.0.1:8000/docs
```
