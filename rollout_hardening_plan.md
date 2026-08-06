# Industrial Rollout Hardening Plan

## Summary

Keep the existing asynchronous architecture: one policy/planner thread, one Cartesian executor, plus bounded gripper and safety workers. Harden it through independently shippable phases addressing problems 1, 2, 3, 4, and 6.

## Phased Implementation

### Phase 1 — Explicit preparation lifecycle (Problem 1)

- Make `WaypointRunner` and `BSplineRunner` constructors side-effect free.
- Add explicit lifecycle states: `NEW → PREPARED → RUNNING → STOPPING → STOPPED`.
- Add `prepare()` to initialize grippers while robots are IDLE, prepare Cartesian motion, and construct policy-specific executors.
- Require `prepare()` before `start()`; make `stop()` safe from every lifecycle state.
- Check cancellation between hardware operations and roll back partially prepared grippers, executors, and robots on failure.
- Log preparation phases and durations: policy setup, robot connection, gripper initialization, Cartesian-mode transition.

### Phase 2 — Shared runner lifecycle (Problem 6)

- Introduce a shallow internal managed-runner base implementing planner thread creation, error translation, stop/join behavior, callback-once guarantees, and cleanup ordering.
- Keep waypoint and B-spline inference/control loops in their existing runner classes; only lifecycle mechanics move into the shared implementation.
- Standardize cleanup: signal cancellation, stop command workers, stop grippers, call `Robot.Stop()` on every robot, join the planner, release policy/GPU state, then report completion.
- Make cleanup idempotent and collect/log every cleanup failure without preventing remaining resources from being stopped.

### Phase 3 — Atomic service state machine (Problem 2)

- Replace `_running`/`_stopping` booleans with `idle`, `starting`, `running`, `stopping`, and `failed`.
- Reserve `starting` under the service lock before checkpoint loading or hardware access.
- Give every rollout a monotonically increasing `run_id`, private stop event, runner, robots, and bootstrap thread.
- Perform loading, connection, `prepare()`, and `start()` in the bootstrap thread; `POST /rollout/start` returns HTTP 202 with `starting`.
- Reject concurrent starts before any second policy load or robot connection occurs.
- Allow `stop()` during `starting`; it cancels preparation and cleans any resources already acquired.
- Bind callbacks to `run_id` so a late callback from an old planner cannot alter or stop a newer run.
- Update the UI so `starting` is visible, configuration is locked while active, Stop is available during startup, and `stopping` cannot trigger another start.

### Phase 4 — Deadline-aware waypoint dispatch (Problem 4)

- Give `WaypointExecutor` an injectable monotonic clock for deterministic testing.
- At each wake-up, collect all due waypoints and send only the newest due waypoint; discard superseded commands instead of replaying them in a burst.
- Preserve future waypoints and keep pose, wrench, twist, and gripper width from the selected action together.
- Publish pending count, total overdue drops, last/max dispatch lateness, last command time, and plan-valid-until time.
- Log rate-limited warnings when deadline drops begin or exceed operational thresholds.

### Phase 5 — Independent command-freshness watchdog (Problem 3)

- Add a shared safety watchdog thread that does not depend on the planner or inference future making progress.
- Have both executors publish a thread-safe execution lease containing plan installation time, valid-until time, last successful command time, and executor heartbeat.
- Trip once when the first plan is not acquired in time, the active plan expires without replacement, or the executor heartbeat stops.
- On trip, immediately latch a failure code, set the run stop event, call `Robot.Stop()` on every follower, and let normal lifecycle cleanup stop the executors and grippers.
- Use derived defaults:
  - Plan-expiry grace: two action periods, clamped to 0.10–0.50 seconds.
  - Executor heartbeat timeout: two action periods, clamped to 0.25–0.50 seconds.
  - Initial-plan timeout: twice the expected plan horizon, clamped to 5–30 seconds; compiled policies use the 30-second default.
- Add optional per-policy overrides for initial-plan timeout and plan-expiry grace.
- Report `plan_remaining_ms`, `command_age_ms`, `heartbeat_age_ms`, and the latched watchdog reason in status/metrics.

### Phase 6 — Industrial qualification

- Run fault-injection tests after every phase before continuing.
- Add structured lifecycle logs containing `run_id`, phase, elapsed time, policy type, active sides, and failure code.
- Preserve readable `error` details while adding stable failure codes such as `startup_failed`, `plan_expired`, `executor_unresponsive`, and `cleanup_failed`.
- Document lifecycle transitions, watchdog semantics, and the distinction between application-level `Robot.Stop()` and the robot’s independent safety system.
- Complete bounded-workspace single-arm and dual-arm hardware soak tests before enabling the watchdog defaults in production.

## Interface Changes

- Runner contract: `prepare()`, `start()`, `stop(timeout)`, `is_alive()`, and lifecycle status.
- Rollout API status adds `starting`, `run_id`, `phase`, and `failure_code`; existing fields remain available.
- `POST /rollout/start` becomes asynchronous and returns HTTP 202.
- Policy rollout configuration gains optional watchdog timeout overrides.
- Checkpoint formats and waypoint/B-spline action schemas remain unchanged.

## Test and Acceptance Criteria

- Runner construction performs no robot, gripper, thread, or GPU operations.
- Partial preparation failure stops every acquired resource exactly once.
- Two simultaneous start requests produce exactly one bootstrap, policy load, and robot connection set.
- Stop during every startup phase leaves no executor, planner, gripper, or bootstrap thread behind.
- Late callbacks from an earlier `run_id` cannot modify the current rollout.
- A delayed waypoint executor sends one newest-due action and never emits a catch-up burst; its gripper target matches that action.
- A deliberately blocked inference call causes `Robot.Stop()` no later than the lease deadline plus one watchdog polling interval.
- Fresh replacement plans prevent false watchdog trips for both waypoint and B-spline policies.
- Existing rollout, API, schema, Ruff, and full pytest checks pass after every phase.

## Assumptions

- An expired plan is a failure and causes an immediate safe stop, as selected.
- `starting` is a public API/UI state, as selected.
- Timeout defaults are derived from action timing and may be overridden, as selected.
- Python threads remain best-effort; certified robot safety, limits, and emergency-stop functionality remain the hardware controller’s responsibility.
