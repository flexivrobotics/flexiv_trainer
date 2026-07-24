# Rollout Runtime Refactor Plan

## Summary

Refactor `src/flexivtrainer/rollout` into a small public service, policy-family
runners, shared executors, and focused checkpoint/observation/hardware helpers.
Keep the existing rollout API and robot behavior unchanged.

The refactor is structural only. It must not change policy outputs, command
frequencies, B-spline alignment, motion limits, hardware modes, metrics, or
shutdown behavior.

Use `runners/` instead of `inference_loop/`: each loop owns observation
acquisition, inference scheduling, replanning, fault monitoring, and executor
lifecycle, so it is broader than inference alone.

Keep gripper control under `executors/` instead of creating a one-file
`gripper/` package. A separate gripper package is warranted only if it later
grows device adapters or multiple control strategies.

## Goals

- Reduce `rollout/service.py` to the public lifecycle facade.
- Give each policy execution strategy one clearly owned runner.
- Keep time-sensitive robot commands isolated in executor threads.
- Make `GripperExecutor` reusable across policy families.
- Preserve one universal rollout API.
- Preserve fail-closed checkpoint and hardware initialization ordering.
- Make future policy integrations additive rather than expanding
  `RolloutService`.

## Non-Goals

- Do not change API routes or response schemas.
- Do not change training, recording, teleoperation, cameras, or robot config.
- Do not change ACT, Diffusion, or B-spline model behavior.
- Do not enable ACT/Diffusion gripper actuation in this refactor.
- Do not change dataset action contracts.
- Do not change NRT Cartesian mode, control rates, motion limits, or
  `speed_scale`.
- Do not introduce separate public rollout scripts.
- Do not introduce a policy registry beyond the two runner selections currently
  required.

## Current Problems

`rollout/service.py` currently owns:

- checkpoint path and metadata resolution;
- LeRobot plugin and policy loading;
- robot connection, F/T zeroing, mode switching, and shutdown;
- observation acquisition and preprocessing;
- the ACT/Diffusion waypoint planner loop;
- the B-spline asynchronous inference loop;
- executor construction and lifecycle;
- logging, metrics, status, and failure propagation.

The two planner loops are materially different, but both mutate private service
state and depend on many service helpers. Adding another policy execution model
would grow the same file and increase lifecycle coupling.

## Target Structure

```text
src/flexivtrainer/rollout/
├── __init__.py
├── service.py
├── checkpoint.py
├── hardware.py
├── observations.py
├── runners/
│   ├── __init__.py
│   ├── waypoint.py
│   └── bspline.py
└── executors/
    ├── __init__.py
    ├── waypoint.py
    ├── bspline.py
    └── gripper.py
```

### `service.py`

Own only:

- the public `start`, `stop`, `shutdown`, and `status` methods;
- teleoperation exclusion;
- checkpoint preparation;
- follower selection;
- runner selection;
- top-level robot and runner lifecycle;
- aggregation of service and runner status.

It must contain no policy inference loop and no direct Cartesian or gripper
command.

### `checkpoint.py`

Move the pure checkpoint and policy-loading functions:

- checkpoint path resolution;
- model directory resolution;
- plugin import and policy loading;
- checkpoint policy type, task, and rate resolution;
- dataset metadata fallback;
- scheduler override selection;
- B-spline layout, degree, rate, and gripper-contract preflight.

Return a small immutable prepared-checkpoint value containing the loaded policy,
processors, policy type, target rate, task requirement, rollout config, and
validated B-spline layout when applicable.

All validation that can fail without hardware must complete before the robot
factory is called.

### `hardware.py`

Move the RDK lifecycle helpers:

- default robot construction;
- fault clearing and enabling;
- F/T sensor zeroing;
- Cartesian mode entry;
- follower stop and cleanup.

Preserve B-spline gripper initialization ordering:

```text
connect and enable in IDLE
    → initialize configured grippers and switch tools
    → zero F/T sensors
    → enter NRT_CARTESIAN_MOTION_FORCE
```

Use functions and explicit robot lists initially. Do not add a hardware class
unless resource ownership cannot remain clear with functions.

### `observations.py`

Move shared observation operations:

- camera capture and BGR-to-RGB conversion;
- robot state snapshot construction;
- optional cached gripper telemetry insertion;
- LeRobot observation feature construction;
- policy preprocessing;
- action tensor-to-list conversion where it is policy-independent.

Observation helpers must not depend on `RolloutService` or own threads.

### `runners/waypoint.py`

Add `WaypointRunner` for ACT, standard Diffusion, and other policies that emit
discrete action chunks.

Own:

- policy reset;
- checkpoint-FPS observation cadence;
- action queue refresh and chunk inference;
- `n_action_steps`, replan, and anchor semantics;
- waypoint target-time construction;
- `WaypointExecutor` lifecycle;
- waypoint timing metrics and command-versus-measured logging;
- runner error and completion state.

This runner must reproduce the current `_policy_planner_loop` behavior without
algorithmic cleanup during extraction.

### `runners/bspline.py`

Add `BSplineRunner`.

Own:

- policy reset and observation-only queue updates;
- checkpoint-FPS observation cadence;
- the single-worker asynchronous inference pool;
- postprocessing and flat action validation;
- atomic spline installation and handoff-warning reporting;
- `BSplineExecutor` lifecycle;
- optional `GripperExecutor` lifecycle;
- B-spline metrics and error propagation.

This runner must preserve:

- 200 Hz default Cartesian sampling;
- execution of the old spline during inference;
- companion L1 alignment and maximum-coordinate warning behavior;
- `predict_before_end_s` timing;
- observation updates while inference is running;
- latest-value gripper targets and cached telemetry;
- fail-closed handling of invalid predictions.

### `executors/`

Move existing executors without changing their behavior:

- `waypoint_executor.py` → `executors/waypoint.py`;
- `bspline_executor.py` → `executors/bspline.py`;
- `gripper_executor.py` → `executors/gripper.py`.

Executor responsibilities remain narrow:

- `WaypointExecutor`: dispatch discrete Cartesian waypoints at target times.
- `BSplineExecutor`: decode, align, sample, and dispatch continuous splines.
- `GripperExecutor`: execute latest-value width commands without blocking arm
  dispatch.

Executors must not import policy implementations, processors, cameras,
`RolloutService`, or application settings.

## Runner Contract

Use a small structural interface rather than an inheritance hierarchy:

```python
class RolloutRunner(Protocol):
    def start(self) -> None: ...
    def stop(self, timeout: float = 2.0) -> None: ...
    def status(self) -> RunnerStatus: ...
```

Both runners should receive their dependencies explicitly. A frozen context
dataclass is acceptable for the shared immutable inputs:

- policy and processors;
- robots and semantic side ordering;
- cameras;
- target rate, device, task, and motion limits;
- rollout configuration;
- stop event;
- bounded logging and metric callbacks.

The context must not expose `RolloutService` itself. Runners must not mutate
service-private fields.

`RunnerStatus` should expose only the state needed to preserve the current
public status response:

- error;
- stop reason;
- logs;
- metrics;
- running state.

## Control Flow

### Start

```text
RolloutService.start
    → reject active rollout or active teleoperation
    → resolve and validate checkpoint
    → load policy and processors
    → determine target rate and runner type
    → complete policy-specific preflight
    → connect follower robots in IDLE
    → construct selected runner
    → initialize runner-owned grippers, if any
    → zero F/T sensors and enter Cartesian mode
    → start runner
```

### B-spline runtime

```text
BSplineRunner
├── observation/planning thread at checkpoint FPS
├── inference worker with at most one in-flight request
├── BSplineExecutor thread at configured control_hz
└── GripperExecutor thread at up to 30 Hz, when configured
```

### Waypoint runtime

```text
WaypointRunner
├── observation/planning thread at checkpoint FPS
└── WaypointExecutor thread scheduled by waypoint timestamps
```

### Stop

```text
RolloutService.stop
    → signal runner stop
    → stop time-sensitive executors
    → stop gripper worker
    → join planner and inference work
    → issue robot Stop
    → release robot references
    → publish final status
```

Cleanup must remain unconditional when a worker times out or hardware I/O
raises.

## File Changes

### Add

- `src/flexivtrainer/rollout/checkpoint.py`
- `src/flexivtrainer/rollout/hardware.py`
- `src/flexivtrainer/rollout/observations.py`
- `src/flexivtrainer/rollout/runners/__init__.py`
- `src/flexivtrainer/rollout/runners/waypoint.py`
- `src/flexivtrainer/rollout/runners/bspline.py`
- `src/flexivtrainer/rollout/executors/__init__.py`
- `src/flexivtrainer/rollout/executors/waypoint.py`
- `src/flexivtrainer/rollout/executors/bspline.py`
- `src/flexivtrainer/rollout/executors/gripper.py`
- `tests/test_waypoint_runner.py`
- `tests/test_bspline_runner.py`
- focused tests for checkpoint, observation, and hardware helpers if existing
  service tests cannot cover them cleanly.

### Modify

- `src/flexivtrainer/rollout/service.py`
- rollout imports in existing tests.
- `tests/test_rollout_service.py`
- `tests/test_rollout_waypoint_executor.py`
- `tests/test_bspline_rollout.py`
- `tests/test_gripper_executor.py`

`src/flexivtrainer/runtime/manager.py` should not require a behavioral change;
it should continue constructing the same `RolloutService` facade.

### Move

- `src/flexivtrainer/rollout/waypoint_executor.py`
- `src/flexivtrainer/rollout/bspline_executor.py`
- `src/flexivtrainer/rollout/gripper_executor.py`

No compatibility modules are needed unless these internal module paths are
confirmed to be consumed outside this repository.

## Implementation Phases

### Phase 0: Freeze behavior

- Run and record the current focused rollout test baseline.
- Add any missing lifecycle assertions before moving code.
- Verify no executor or planner threads remain after each stop test.

Verification:

```text
tests/test_rollout_service.py
tests/test_rollout_waypoint_executor.py
tests/test_bspline_rollout.py
tests/test_gripper_executor.py
```

### Phase 1: Move executors

- Create `executors/`.
- Move the three executor modules.
- Update imports only.
- Do not rename public classes or alter behavior.

Verification:

- executor tests pass unchanged except imports;
- service tests pass;
- no old executor imports remain.

### Phase 2: Extract checkpoint, hardware, and observation helpers

- Move pure functions first.
- Preserve dependency-injection hooks used by tests.
- Keep service method signatures and status output unchanged.
- Verify B-spline preflight still fails before robot construction.

Verification:

- path security and checkpoint metadata tests pass;
- malformed B-spline checkpoints never call the robot factory;
- gripper initialization still precedes Cartesian mode entry;
- observation keys and tensor shapes remain identical.

### Phase 3: Extract `WaypointRunner`

- Move the existing waypoint planner loop without rewriting it.
- Move its timing and command logging with it.
- Make the runner own `WaypointExecutor`.
- Have the service read runner status instead of runner internals.

Verification:

- ACT and Diffusion command sequences are unchanged;
- chunk refresh, anchoring, and replanning tests pass;
- checkpoint FPS remains the waypoint spacing;
- standard policies never construct `BSplineExecutor`.

### Phase 4: Extract `BSplineRunner`

- Move the B-spline planner and inference worker without algorithm changes.
- Make the runner own both executors.
- Preserve warning, invalid-plan, and shutdown behavior.

Verification:

- old-plan execution continues during blocked inference;
- observations continue at checkpoint FPS during inference;
- alignment and deadline metrics are unchanged;
- flattened spline parameters never reach the robot directly;
- gripper failures stop Cartesian execution and cleanup remains unconditional.

### Phase 5: Simplify the service

- Remove extracted helpers and loops.
- Select the runner based on prepared checkpoint policy type.
- Keep only facade state and top-level resource cleanup.
- Remove imports made unused by this refactor.

Verification:

- the rollout API and status dictionary are unchanged;
- `service.py` contains no planner loop and no RDK command call;
- standard and B-spline service integration tests pass;
- RuntimeManager tests pass unchanged.

## Test Plan

### Unit tests

- checkpoint path, plugin, policy type, rate, task, and B-spline preflight;
- observation image conversion, robot snapshot, gripper telemetry, and processor
  shapes;
- hardware initialization ordering and cleanup on partial failure;
- each executor's existing timing and decoding behavior;
- each runner's inference, replanning, metrics, and stop behavior.

### Integration tests

- service selects `WaypointRunner` for ACT and standard Diffusion;
- service selects `BSplineRunner` for B-spline Diffusion;
- malformed checkpoints fail before robot construction;
- each runner owns and stops only its expected executors;
- public status keys and metric fields remain compatible;
- a blocked inference does not stop executor command delivery;
- a blocked executor or gripper cannot skip robot cleanup.

### Commands

Run after each phase:

```bash
python -m pytest -q \
  tests/test_rollout_service.py \
  tests/test_rollout_waypoint_executor.py \
  tests/test_bspline_rollout.py \
  tests/test_gripper_executor.py

python -m ruff check src/flexivtrainer/rollout tests/test_rollout*
git diff --check
```

Run the complete test suite after the final phase. Report unrelated baseline
failures separately rather than changing recording, API, or teleoperation code.

## Success Criteria

- `RolloutService` has no policy planner implementation.
- `RolloutService` sends no robot or gripper command directly.
- Each runner owns its planner thread and executor lifecycle.
- All executors are policy-independent.
- B-spline and waypoint behavior remain byte-for-behavior equivalent where
  observable by tests.
- All checkpoint validation that does not require hardware happens before robot
  connection.
- Existing rollout API and status consumers require no changes.
- Focused rollout tests, scoped Ruff, compilation, and diff checks pass.
- Hardware acceptance is repeated at `speed_scale=1.0` after the structural
  refactor before testing higher speeds.

## Deferred Follow-Up

After this refactor is stable, define an explicit commanded-gripper action
contract for ACT and standard Diffusion. Then reuse `GripperExecutor` from
`WaypointRunner`. Do not infer that contract from the currently recorded
measured gripper width/force fields.
