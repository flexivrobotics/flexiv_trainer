# Copyright 2026 Flexiv Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Practise a step skill on the robot, episode by episode, with a learner.

Each episode:

1. **Reset.** The arm goes to a random start in the plan's box (`reset.py`),
   with every axis motion-controlled; without a box it starts where it is.
2. **Act.** The skill steps (`env.py`), sampling its actions: practice
   explores. An operator may take over with the leader arm at any time; their
   steps are measured like demonstrations and marked as interventions.
3. **Judge.** The plan's learned check is scored on every tick. The step at
   which it passes earns the reward and ends the episode; at the step limit
   the episode is cut short instead.
4. **Learn.** The episode's steps go to the learner (`learner.py`), and the
   newest weights it has pushed are loaded before the next episode.

After a success the loop can wait for the operator, for whatever the reset
cannot undo (a part latched home), until they say it may go on.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from flexivtrainer.evaluation.checks import LearnedCheck, parse_checks
from flexivtrainer.practice.env import POLICY, StepEnv
from flexivtrainer.practice.learner import LearnerLink, apply_weights, transition
from flexivtrainer.practice.reset import ResetInterrupted, StartBox, go_to_start
from flexivtrainer.practice.runner import StepRunner
from flexivtrainer.rollout.safety import SafetyStop

__all__ = ["PracticePlan", "PracticeRunner"]

#: Episode summaries kept for the status.
_KEPT_EPISODES = 200


@dataclass(frozen=True)
class PracticePlan:
    """How a practice session runs."""

    #: The episode's reward: one learned check, which ends the episode when it
    #: passes. Given as a session plan's `checks` entry, {name: {...}}.
    reward: Mapping[str, Any]
    #: The step at which an episode is cut short.
    max_episode_steps: int
    #: Where LeRobot's learner listens.
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    #: How many episodes; 0 runs until stopped.
    episodes: int = 0
    #: Where episodes start; None starts each where the arm is.
    start: StartBox | None = None
    #: After a success, wait for the operator before the next reset.
    wait_after_success: bool = True
    #: Seeds the starts.
    seed: int = 0
    check_name: str = field(init=False, default="")
    check: LearnedCheck = field(init=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        checks = parse_checks(dict(self.reward))
        if len(checks) != 1:
            raise ValueError(f"practice is rewarded by one check, not {len(checks)}")
        ((name, check),) = checks.items()
        if not isinstance(check, LearnedCheck):
            raise ValueError(f"the reward {name!r} must be a learned check")
        # Scored on every tick, and the end of the episode when it passes.
        object.__setattr__(self, "check_name", name)
        object.__setattr__(self, "check", replace(check, live=True, end_on_pass=True))
        for key in ("max_episode_steps", "learner_port"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a whole number >= 1, got {value!r}")
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, int):
            raise ValueError(f"episodes must be a whole number, got {self.episodes!r}")
        if self.episodes < 0:
            raise ValueError("episodes must be 0 (until stopped) or more")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PracticePlan:
        values = dict(data)
        known = {
            "reward",
            "max_episode_steps",
            "learner_host",
            "learner_port",
            "episodes",
            "start",
            "wait_after_success",
            "seed",
        }
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown practice settings {unknown}")
        for key in ("reward", "max_episode_steps"):
            if key not in values:
                raise ValueError(f"a practice plan needs {key}")
        if values.get("start") is not None:
            values["start"] = StartBox.from_dict(values["start"])
        return cls(**values)


class PracticeRunner(StepRunner):
    """Practise a step skill: episodes of reset, act, judge, learn."""

    def __init__(
        self,
        *,
        plan: PracticePlan,
        link: LearnerLink,
        **runner: Any,
    ) -> None:
        if runner.get("live_checks") is None:
            raise ValueError("practice needs its reward check, loaded")
        super().__init__(**runner)
        self._plan = plan
        self._link = link
        self._rng = np.random.default_rng(plan.seed)
        self._continue = threading.Event()
        self._lock = threading.Lock()
        self._state = "starting"
        self._episodes: deque[dict[str, Any]] = deque(maxlen=_KEPT_EPISODES)
        self._totals = {
            "episodes": 0,
            "successes": 0,
            "steps": 0,
            "intervention_steps": 0,
            "weights_loaded": 0,
        }
        # Practice explores: a policy that can act its centre is told to sample.
        config = getattr(self._policy, "config", None)
        if config is not None and hasattr(config, "sample_actions"):
            config.sample_actions = True

    # --- the service's side -------------------------------------------------

    def practice_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "learner": self._link.address,
                "reward": self._plan.check_name,
                **self._totals,
                "recent": list(self._episodes)[-20:],
            }

    def go_on(self) -> None:
        """The operator has done what the reset cannot: go on."""
        self._continue.set()

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state
        self._append_log("INFO", "PRACTICE", state, "")

    # --- the loop ---------------------------------------------------------------

    def _run(self) -> None:
        env: StepEnv | None = None
        total = 0
        try:
            self._set_state("connecting to the learner")
            self._link.connect(self._stop_event)
            self._link.start()
            env = StepEnv(
                robot=self._robot,
                cameras=self._cameras,
                camera_names=self._camera_names,
                space=self._space,
                motion_limits=self._motion_limits,
                stop_event=self._stop_event,
                safety=self._safety,
                image_resolutions=self._image_resolutions,
                control=self._control,
                on_tick=self._on_tick,
            )
            self._env = env
            # The operator may take the arm from here on, resets included.
            self._ready = True
            rotation = (
                self._plan.start.reference(env.read()) if self._plan.start else None
            )
            episode = 0
            while not self._stop_event.is_set():
                if self._plan.episodes and episode >= self._plan.episodes:
                    self._stop_reason = "episodes done"
                    break
                if self._link.error is not None:
                    raise RuntimeError(f"the learner link failed: {self._link.error}")
                self._reset(env, rotation)
                if self._stop_event.is_set():
                    break
                summary = self._episode(env, episode)
                total += summary["steps"]
                episode += 1
                if summary["success"] and self._plan.wait_after_success:
                    self._set_state("waiting for the operator")
                    self._continue.clear()
                    while not self._stop_event.is_set():
                        if self._continue.wait(0.1):
                            break
            self._stop_reason = self._stop_reason or "stopped"
        except SafetyStop as exc:
            if self._takeover is not None:
                self._takeover.close()
            self._stop_reason = "safety"
            self.safety_stop = {"code": exc.code, "detail": exc.detail}
            self._error = str(exc)
            self._on_error(str(exc))
        except Exception as exc:
            if self._takeover is not None:
                self._takeover.close()
            self._error = f"{type(exc).__name__}: {exc}"
            self._on_error(self._error)
        finally:
            self._set_state("finished")
            self._link.close()
            if self._takeover is not None:
                self._takeover.close()
            self._stop_event.set()
            try:
                self._clear_force()
            except Exception as exc:
                self._stop_robots(self._robots)
                self._on_cleanup_error(f"{type(exc).__name__}: {exc}")
            self._policy = self._preprocessor = self._postprocessor = None
            self._release_robots()
            self._on_finished(self._stop_reason, total)

    def _reset(self, env: StepEnv, rotation: np.ndarray | None) -> None:
        """To a start, with every axis motion-controlled on the way. Retried
        once the operator hands back, when they took the arm meanwhile."""
        box = self._plan.start
        if box is None:
            return
        self._set_state("resetting")
        while not self._stop_event.is_set():
            if self._control() != POLICY:
                self._stop_event.wait(0.1)
                continue
            self._clear_force()
            try:
                go_to_start(
                    box,
                    read=env.read,
                    send=env.send_pose,
                    stop=self._stop_event,
                    rng=self._rng,
                    rotation=rotation,
                    safety=self._safety,
                )
            except ResetInterrupted:
                continue
            finally:
                if self._control() == POLICY:
                    self._set_force()
            return

    def _episode(self, env: StepEnv, number: int) -> dict[str, Any]:
        self._set_state(f"episode {number + 1}")
        self._live_checks.reset()
        self._policy.reset()
        observation = env.reset()
        steps: list[dict[str, Any]] = []
        interventions = 0
        success = False
        limit = self._plan.max_episode_steps
        started = time.monotonic()
        for k in range(limit):
            tick = time.monotonic()
            if self._control() == POLICY:
                if self._reset_pending:
                    self._reset_pending = False
                    self._policy.reset()
                epoch = env.epoch
                action = self._act(observation)
                self._infer_ms = (time.monotonic() - tick) * 1000.0
                result = env.step(action, epoch=epoch)
            else:
                self._infer_ms = 0.0
                result = env.step(None)
            self._append_metric(
                {
                    "t": round(tick, 3),
                    "step": env.steps,
                    "infer_ms": round(self._infer_ms, 1),
                    "fresh": bool(result.sent),
                    "intervention": bool(result.intervention),
                }
            )
            interventions += int(result.intervention)
            success = self._live_checks.ended() is not None
            last = success or k == limit - 1 or self._stop_event.is_set()
            steps.append(
                transition(
                    observation,
                    result.action,
                    1.0 if success else 0.0,
                    result.observation,
                    done=success,
                    truncated=last and not success,
                    intervention=result.intervention,
                )
            )
            observation = result.observation
            if last:
                break
        self._link.send_episode(steps)
        with self._lock:
            totals = self._totals
            totals["episodes"] += 1
            totals["successes"] += int(success)
            totals["steps"] += len(steps)
            totals["intervention_steps"] += interventions
            interaction_step = totals["steps"]
        self._link.send_stats(
            {
                "Episodic reward": 1.0 if success else 0.0,
                "Interaction step": interaction_step,
                "Episode intervention": int(interventions > 0),
                "Intervention rate": interventions / len(steps) if steps else 0.0,
            }
        )
        weights = self._link.latest_weights()
        if weights is not None:
            apply_weights(self._policy, weights)
            with self._lock:
                self._totals["weights_loaded"] += 1
        summary = {
            "episode": number + 1,
            "steps": len(steps),
            "success": success,
            "interventions": interventions,
            "seconds": round(time.monotonic() - started, 1),
        }
        with self._lock:
            self._episodes.append(summary)
        self._append_log(
            "INFO",
            "PRACTICE",
            f"episode {number + 1}",
            f"{'success' if success else 'no success'} in {len(steps)} steps, "
            f"{interventions} by the operator",
        )
        return summary
