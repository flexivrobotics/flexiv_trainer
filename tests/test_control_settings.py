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

"""The follower stiffness travels from teleoperation to the rollout.

Teleop sets it, each recording stores it (meta/control.json), a merge refuses
episodes recorded under different settings, training copies it beside the
checkpoint, and the rollout picks it up (tests/test_rollout_takeover.py drives
the rollout and the takeover against the fake hardware).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_teleop_service import FakeTeleopController

from flexivtrainer.config import AppSettings, StorageConfig, TeleopRobotPair
from flexivtrainer.data import remove_action_wrench
from flexivtrainer.data.control_settings import (
    CONTROL_RELATIVE_PATH,
    ControlSettings,
    dataset_control_settings,
    effective_control_settings,
    read_control_settings,
    write_control_settings,
)
from flexivtrainer.evaluation.outcome import RolloutOutcome
from flexivtrainer.jobs import merge_episodes, train_policy
from flexivtrainer.jobs.train_policy import TrainingService
from flexivtrainer.rollout.stiffness import (
    StiffnessRefused,
    choose_rollout_stiffness,
)
from flexivtrainer.teleop.service import TeleopService

K = [2600.0, 2600.0, 2900.0, 150.0, 150.0, 150.0]
OTHER_K = [5000.0, 5000.0, 5000.0, 300.0, 300.0, 300.0]


# --- the file -------------------------------------------------------------------------


def test_control_json_round_trips_and_spells_out_the_controller_default(
    tmp_path,
) -> None:
    settings = ControlSettings(follower_cartesian_stiffness=K)
    path = write_control_settings(tmp_path, settings)

    assert path == tmp_path / "meta" / "control.json"
    assert json.loads(path.read_text()) == {
        "format_version": 1,
        "recorded_in": "teleop",
        "follower_cartesian_stiffness": K,
        "follower_max_contact_wrench": "controller default",
        "wrench_feedback_scale": "controller default",
    }
    assert read_control_settings(path) == settings
    assert settings.stiffness == K


@pytest.mark.parametrize(
    "payload",
    [
        {"format_version": 2},
        {"format_version": 1, "recorded_in": "elsewhere"},
        {"format_version": 1, "follower_cartesian_stiffness": [1.0, 2.0, 3.0]},
        {"format_version": 1, "follower_cartesian_stiffness": [-1, 0, 0, 0, 0, 0]},
        {"format_version": 1, "follower_cartesian_stiffness": "stiff"},
        {"format_version": 1, "wrench_feedback_scale": float("inf")},
    ],
)
def test_an_invalid_control_json_is_refused(tmp_path, payload) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid control settings"):
        read_control_settings(path)


def test_two_defaults_match_only_when_the_same_controller_recorded_both() -> None:
    teleop_default = ControlSettings(recorded_in="teleop")
    rollout_default = ControlSettings(recorded_in="rollout")
    assert teleop_default.matches(ControlSettings(recorded_in="teleop"))
    assert not teleop_default.matches(rollout_default)
    # A set stiffness is the same spring whoever set it.
    assert ControlSettings("teleop", K).matches(ControlSettings("rollout", K))
    assert not ControlSettings("teleop", K).matches(ControlSettings("teleop", OTHER_K))
    assert not ControlSettings("teleop", K).matches(teleop_default)
    assert not ControlSettings("teleop", K).matches(
        ControlSettings("teleop", K, follower_max_contact_wrench=[1] * 6)
    )


def test_datasets_without_control_json_read_as_what_recorded_them(tmp_path) -> None:
    teleop = tmp_path / "old_episode"
    (teleop / "meta").mkdir(parents=True)
    assert dataset_control_settings(teleop) is None
    assert effective_control_settings(teleop) == ControlSettings(recorded_in="teleop")

    rollout = tmp_path / "old_rollout"
    (rollout / "meta").mkdir(parents=True)
    (rollout / "meta" / "rollout_outcome.json").write_text(
        json.dumps({"safety": {"cartesian_stiffness": K, "max_contact_wrench": None}})
    )
    assert effective_control_settings(rollout) == ControlSettings("rollout", K)

    write_control_settings(rollout, ControlSettings("rollout", OTHER_K))
    assert effective_control_settings(rollout) == ControlSettings("rollout", OTHER_K)


def test_an_outcome_written_before_stiffness_was_recorded_still_reads() -> None:
    outcome = RolloutOutcome.from_dict(
        {
            "episode": "e1",
            "checkpoint": "c",
            "task": None,
            "started_at": "2026-10-16T10:00:00",
            "safety": {"cartesian_stiffness": None},
            "schema_version": RolloutOutcome(
                episode="", checkpoint="", task=None, started_at=""
            ).schema_version,
        }
    )
    assert outcome.stiffness == {}


# --- settings -------------------------------------------------------------------------


def test_the_settings_read_from_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env
    defaults = AppSettings()
    assert defaults.teleop.follower_cartesian_stiffness is None
    assert defaults.rollout.allow_controller_default_stiffness is False

    monkeypatch.setenv(
        "FLEXIV_TRAINER_TELEOP__FOLLOWER_CARTESIAN_STIFFNESS", json.dumps(K)
    )
    monkeypatch.setenv(
        "FLEXIV_TRAINER_ROLLOUT__ALLOW_CONTROLLER_DEFAULT_STIFFNESS", "1"
    )
    settings = AppSettings()
    assert list(settings.teleop.follower_cartesian_stiffness) == K
    assert settings.rollout.allow_controller_default_stiffness is True

    monkeypatch.setenv(
        "FLEXIV_TRAINER_TELEOP__FOLLOWER_CARTESIAN_STIFFNESS", "[-1,0,0,0,0,0]"
    )
    with pytest.raises(ValidationError):
        AppSettings()


# --- teleoperation sets it ------------------------------------------------------------


class _LoggingTDK(FakeTeleopController):
    """The TDK fake of test_teleop_service, logging calls in order."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[tuple] = []

    def Init(self, zero_ft_sensor=None) -> None:  # noqa: N802
        super().Init(zero_ft_sensor)
        self.log.append(("Init",))

    def SetFollowerCartStiff(self, idx, values) -> None:  # noqa: N802
        self.log.append(("SetFollowerCartStiff", idx, list(values)))

    def Start(self) -> None:  # noqa: N802
        super().Start()
        self.log.append(("Start",))


class _OldTDK(FakeTeleopController):
    """A TDK build without SetFollowerCartStiff."""


def _teleop(tmp_path, controller, stiffness=None) -> TeleopService:
    settings = AppSettings(storage=StorageConfig(root=tmp_path))
    settings.teleop.follower_cartesian_stiffness = stiffness
    pairs = [
        TeleopRobotPair(leader_serial="L1", follower_serial="F1"),
        TeleopRobotPair(leader_serial="L2", follower_serial="F2"),
    ]
    service = TeleopService(settings, get_robot_pairs=lambda: pairs)
    service._controller = controller
    service._initialized = True
    return service


def test_teleop_sets_the_follower_stiffness_after_init_and_before_start(
    tmp_path,
) -> None:
    tdk = _LoggingTDK()
    service = _teleop(tmp_path, tdk, stiffness=tuple(K))

    assert service.start().started
    service.stop()
    assert service.start().started

    once = [
        ("Init",),
        ("SetFollowerCartStiff", 0, K),
        ("SetFollowerCartStiff", 1, K),
        ("Start",),
    ]
    assert tdk.log == once + once, "every start must set it again after Init"
    assert service.control_settings() == ControlSettings("teleop", K)


def test_unset_teleop_keeps_the_controller_default_and_records_it(tmp_path) -> None:
    tdk = _LoggingTDK()
    service = _teleop(tmp_path, tdk)

    assert service.start().started
    assert tdk.log == [("Init",), ("Start",)]
    assert service.control_settings() == ControlSettings(recorded_in="teleop")


def test_teleop_does_not_start_when_the_tdk_cannot_set_the_stiffness(
    tmp_path,
) -> None:
    tdk = _OldTDK()
    service = _teleop(tmp_path, tdk, stiffness=tuple(K))

    snapshot = service.start()
    assert not snapshot.started
    assert "SetFollowerCartStiff" in snapshot.error
    assert tdk.start_calls == 0


# --- merging checks it ----------------------------------------------------------------


def _episode(root: Path, name: str, control: ControlSettings | None) -> Path:
    path = root / name
    (path / "meta").mkdir(parents=True)
    features = {"observation.state": {}, "action": {"shape": [1], "names": ["a.x"]}}
    (path / "meta" / "info.json").write_text(json.dumps({"features": features}))
    if control is not None:
        write_control_settings(path, control)
    return path


@pytest.fixture
def fake_merge(monkeypatch):
    """LeRobot's merge, replaced by one that only makes the output directory."""
    pytest.importorskip("lerobot")
    merged: list[Path] = []

    def merge_datasets(datasets, output_repo_id, output_dir):
        (Path(output_dir) / "meta").mkdir(parents=True)
        merged.append(Path(output_dir))

    monkeypatch.setattr("lerobot.datasets.dataset_tools.merge_datasets", merge_datasets)
    monkeypatch.setattr(
        "lerobot.datasets.lerobot_dataset.LeRobotDataset",
        lambda repo_id, root: SimpleNamespace(repo_id=repo_id, root=root),
    )
    monkeypatch.setattr(
        merge_episodes,
        "_load_manifest",
        lambda root: SimpleNamespace(repo_id=f"local/{root.name}", root=root),
    )
    return merged


def test_a_merge_of_episodes_recorded_at_one_stiffness_keeps_it(
    tmp_path, fake_merge
) -> None:
    first = _episode(tmp_path, "e1", ControlSettings("teleop", K))
    second = _episode(tmp_path, "e2", ControlSettings("teleop", K))

    merge_episodes.merge_episode_datasets([first, second], tmp_path / "out", "merged")

    stored = dataset_control_settings(tmp_path / "out" / "merged")
    assert stored == ControlSettings("teleop", K)


def test_a_merge_of_episodes_recorded_at_different_stiffness_is_refused(
    tmp_path, fake_merge
) -> None:
    first = _episode(tmp_path, "e1", ControlSettings("teleop", K))
    second = _episode(tmp_path, "e2", ControlSettings("teleop", OTHER_K))

    with pytest.raises(ValueError, match="different follower control settings.*e2"):
        merge_episodes.merge_episode_datasets(
            [first, second], tmp_path / "out", "merged"
        )
    assert fake_merge == [], "nothing may be merged"
    assert not (tmp_path / "out" / "merged").exists()


def test_an_episode_at_the_default_does_not_join_one_at_a_set_stiffness(
    tmp_path, fake_merge
) -> None:
    # The episode recorded before control.json existed was at the default.
    old = _episode(tmp_path, "old", None)
    new = _episode(tmp_path, "new", ControlSettings("teleop", K))

    with pytest.raises(ValueError, match="different follower control settings"):
        merge_episodes.merge_episode_datasets([old, new], tmp_path / "out", "m")


def test_old_and_new_episodes_at_the_teleop_default_merge(tmp_path, fake_merge) -> None:
    old = _episode(tmp_path, "old", None)
    new = _episode(tmp_path, "new", ControlSettings(recorded_in="teleop"))

    merge_episodes.merge_episode_datasets([old, new], tmp_path / "out", "m")
    assert dataset_control_settings(tmp_path / "out" / "m") == ControlSettings()

    # Older episodes only: the merge claims nothing they did not store.
    merge_episodes.merge_episode_datasets(
        [old, _episode(tmp_path, "old2", None)], tmp_path / "out", "m2"
    )
    assert dataset_control_settings(tmp_path / "out" / "m2") is None


def test_a_rollout_at_the_default_does_not_join_teleop_at_the_default(
    tmp_path, fake_merge
) -> None:
    teleop = _episode(tmp_path, "demo", ControlSettings(recorded_in="teleop"))
    rollout = _episode(tmp_path, "trial", ControlSettings(recorded_in="rollout"))

    with pytest.raises(ValueError, match="different follower control settings"):
        merge_episodes.merge_episode_datasets([teleop, rollout], tmp_path / "out", "m")


def test_removing_wrench_axes_keeps_control_json(tmp_path) -> None:
    source = _episode(tmp_path, "source", ControlSettings("teleop", K))
    staging = tmp_path / "staging"
    (staging / "meta").mkdir(parents=True)

    remove_action_wrench._copy_sidecars(source, staging)

    assert read_control_settings(staging / CONTROL_RELATIVE_PATH) == ControlSettings(
        "teleop", K
    )


# --- training copies it ---------------------------------------------------------------


def _training_dataset(tmp_path: Path, control: ControlSettings | None) -> Path:
    root = tmp_path / "datasets" / "demos"
    (root / "meta").mkdir(parents=True)
    info = {
        "fps": 10,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7], "names": None},
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["single_arm.tcp_pose.x", "single_arm.tcp_pose.y"],
            },
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    if control is not None:
        write_control_settings(root, control)
    return root


def _source_checkpoint(tmp_path: Path, control: ControlSettings | None) -> Path:
    step = tmp_path / "training" / "source" / "checkpoints" / "005000"
    model = step / "pretrained_model"
    model.mkdir(parents=True)
    config = {
        "type": "diffusion",
        "input_features": {"observation.state": {"type": "STATE", "shape": [7]}},
        "output_features": {"action": {"type": "ACTION", "shape": [2]}},
    }
    (model / "config.json").write_text(json.dumps(config))
    (model / "model.safetensors").write_bytes(b"weights")
    if control is not None:
        write_control_settings(model, control, checkpoint=True)
    return step


class _Process:
    pid = 4321
    returncode = None

    def __init__(self) -> None:
        self.stdout: list[str] = []

    def wait(self, timeout=None) -> int:
        return 0

    def poll(self):
        return None


class _NoThread:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass


@pytest.fixture
def training(tmp_path, monkeypatch):
    monkeypatch.setattr(train_policy.shutil, "which", lambda _: "/bin/lerobot-train")
    monkeypatch.setattr(train_policy.subprocess, "Popen", lambda *a, **k: _Process())
    monkeypatch.setattr(train_policy.threading, "Thread", _NoThread)
    monkeypatch.setattr(
        train_policy,
        "Pulse",
        lambda *a, **k: SimpleNamespace(start=lambda: None),
    )
    monkeypatch.setattr(train_policy, "resolve_training_device", lambda _: "cpu")
    return TrainingService(AppSettings(storage=StorageConfig(root=tmp_path)))


def _finish_training(service: TrainingService) -> Path:
    """Save one checkpoint as lerobot-train would, then let the job complete."""
    job = service._job
    model = job.output_dir / "checkpoints" / "000100" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    job.pulse = None
    service._collect_logs(job)
    assert job.status == "completed"
    return model


def test_training_writes_the_datasets_settings_beside_the_checkpoint(
    tmp_path, training
) -> None:
    dataset = _training_dataset(tmp_path, ControlSettings("teleop", K))

    snapshot = training.start(dataset, tmp_path / "training" / "run", "act")
    assert snapshot["control_settings"]["follower_cartesian_stiffness"] == K
    model = _finish_training(training)

    assert read_control_settings(model / "control.json") == ControlSettings("teleop", K)


def test_training_on_an_older_dataset_writes_no_settings(tmp_path, training) -> None:
    dataset = _training_dataset(tmp_path, None)

    training.start(dataset, tmp_path / "training" / "run", "act")
    model = _finish_training(training)

    assert not (model / "control.json").exists()


def test_fine_tuning_on_data_recorded_at_another_stiffness_is_refused(
    tmp_path, training
) -> None:
    dataset = _training_dataset(tmp_path, ControlSettings("teleop", OTHER_K))
    checkpoint = _source_checkpoint(tmp_path, ControlSettings("teleop", K))

    with pytest.raises(ValueError, match="different follower control settings"):
        training.start(
            dataset,
            tmp_path / "training" / "tuned",
            "act",
            training_mode="fine_tune",
            checkpoint_path=checkpoint,
        )
    assert training._job is None


def test_fine_tuning_at_the_same_stiffness_keeps_it(tmp_path, training) -> None:
    dataset = _training_dataset(tmp_path, ControlSettings("teleop", K))
    checkpoint = _source_checkpoint(tmp_path, ControlSettings("teleop", K))

    training.start(
        dataset,
        tmp_path / "training" / "tuned",
        "act",
        training_mode="fine_tune",
        checkpoint_path=checkpoint,
    )
    model = _finish_training(training)
    assert read_control_settings(model / "control.json") == ControlSettings("teleop", K)


def test_fine_tuning_an_older_checkpoint_on_data_at_a_set_stiffness_is_refused(
    tmp_path, training
) -> None:
    # The older checkpoint was trained at the teleop default.
    dataset = _training_dataset(tmp_path, ControlSettings("teleop", K))
    checkpoint = _source_checkpoint(tmp_path, None)

    with pytest.raises(ValueError, match="different follower control settings"):
        training.start(
            dataset,
            tmp_path / "training" / "tuned",
            "act",
            training_mode="fine_tune",
            checkpoint_path=checkpoint,
        )


# --- the rollout's precedence ---------------------------------------------------------


def test_the_setting_wins_and_warns_when_the_checkpoint_differs() -> None:
    choice = choose_rollout_stiffness(
        OTHER_K, ControlSettings("teleop", K), allow_controller_default=False
    )
    assert (choice.values, choice.source, choice.trained) == (OTHER_K, "setting", K)
    assert "Trained at stiffness [2600, 2600, 2900, 150, 150, 150]" in choice.warning
    assert "running at [5000, 5000, 5000, 300, 300, 300]" in choice.warning


def test_the_setting_equal_to_the_checkpoint_does_not_warn() -> None:
    choice = choose_rollout_stiffness(
        K, ControlSettings("teleop", K), allow_controller_default=False
    )
    assert (choice.source, choice.warning) == ("setting", None)


def test_without_the_setting_the_checkpoint_applies() -> None:
    choice = choose_rollout_stiffness(
        None, ControlSettings("teleop", K), allow_controller_default=True
    )
    assert (choice.values, choice.source, choice.warning) == (K, "checkpoint", None)


@pytest.mark.parametrize(
    ("checkpoint", "why"),
    [
        (None, "stores no stiffness"),
        (ControlSettings(recorded_in="teleop"), "teleop controller's default"),
    ],
)
def test_without_any_stiffness_the_rollout_is_refused_unless_allowed(
    checkpoint, why
) -> None:
    with pytest.raises(StiffnessRefused, match=why) as refused:
        choose_rollout_stiffness(None, checkpoint, allow_controller_default=False)
    assert "ROLLOUT__CARTESIAN_STIFFNESS" in str(refused.value)
    assert "ROLLOUT__ALLOW_CONTROLLER_DEFAULT_STIFFNESS" in str(refused.value)

    choice = choose_rollout_stiffness(None, checkpoint, allow_controller_default=True)
    assert (choice.values, choice.source) == (None, "controller default")
    assert "controller's default" in choice.warning
