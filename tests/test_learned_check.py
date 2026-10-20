"""Learned checks: labels, training on held-out episodes, scoring recordings."""

import json
import shutil

import numpy as np
import pytest
from test_rollout_evaluation import (
    PLAN,
    _action,
    _run_ticks,
    _service,
    _wait_for_verdict_slot,
    fake_loop,  # noqa: F401 - a fixture
)
from test_rollout_service import _checkpoint, _FakePolicy, _FakeRobot

from flexivtrainer.evaluation.check_labels import (
    IGNORED,
    LABELS_FILE,
    frame_targets,
    labels_for,
    merge_labels,
    read_labels,
    remove_label,
    verdict_targets,
    write_label,
)
from flexivtrainer.evaluation.checks import (
    LearnedCheck,
    SettledInBox,
    parse_checks,
    run_checks,
)
from flexivtrainer.evaluation.outcome import (
    RolloutOutcome,
    Verdict,
    read_outcome,
    write_outcome,
)

FPS = 10
LENGTH = 24
SIZE = 32
FZ = "single_arm.tcp_wrench.fz"
Z = "single_arm.tcp_pose.z"
#: First frame the state holds, per episode; None never; "near" a near miss.
EPISODES = [8, 12, None, 6, "near", 10, None, 14, "near", 9]


def _frame(index: int, first, rng) -> tuple[np.ndarray, np.ndarray]:
    image = rng.integers(0, 40, (SIZE, SIZE, 3), dtype=np.uint8)
    fz, z = 2.0, 0.0
    if first == "near" and index >= 8:
        image[10:22, 4:16] = 90  # something there, but not the state
        fz = 12.0
    elif isinstance(first, int) and index >= first:
        image[10:22, 10:22] = 255
        fz, z = 30.0, 3.0
    state = np.array([fz + rng.normal(0, 0.5), z], dtype=np.float32)
    return image, state


def _dataset(root, episodes, *, seed=0):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from flexivtrainer.data.lerobot_io import resolve_recording_vcodec

    dataset = LeRobotDataset.create(
        repo_id=f"local/{root.name}",
        fps=FPS,
        root=root,
        features={
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (SIZE, SIZE, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": [FZ, Z],
            },
            "action": {"dtype": "float32", "shape": (1,), "names": ["a"]},
        },
        use_videos=True,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec=resolve_recording_vcodec("h264")),
    )
    rng = np.random.default_rng(seed)
    for first in episodes:
        for index in range(LENGTH):
            image, state = _frame(index, first, rng)
            dataset.add_frame(
                {
                    "observation.images.wrist": image,
                    "observation.state": state,
                    "action": np.zeros(1, dtype=np.float32),
                    "task": "t",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


def _label_all(root, episodes, check="closed"):
    for episode, first in enumerate(episodes):
        write_label(root, check, episode, first if isinstance(first, int) else None)


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    pytest.importorskip("lerobot")
    root = _dataset(tmp_path_factory.mktemp("learned") / "demos", EPISODES)
    _label_all(root, EPISODES)
    return root


def _inputs(**overrides):
    from flexivtrainer.evaluation.learned_check import CheckInputs

    values = {
        "cameras": ("wrist",),
        "image_size": (16, 16),
        "encoder": "pixels",
        "grid": 4,
    }
    return CheckInputs(**{**values, **overrides})


def _options(**overrides):
    from flexivtrainer.evaluation.learned_check import TrainingOptions

    values = {"holdout": 0.3, "epochs": 40, "margin_s": 0.2, "device": "cpu"}
    return TrainingOptions(**{**values, **overrides})


# --- labels -------------------------------------------------------------------------


def test_a_label_is_one_first_frame_per_episode_and_survives_a_reload(demos) -> None:
    labels = labels_for(demos, "closed")
    assert labels[0] == 8 and labels[2] is None and len(labels) == len(EPISODES)
    stored = json.loads((demos / LABELS_FILE).read_text())
    assert stored["format_version"] == 1 and stored["checks"]["closed"]["0"] == 8
    assert labels_for(demos, "other") == {}


def test_a_label_for_a_frame_or_episode_the_dataset_lacks_is_refused(demos) -> None:
    with pytest.raises(ValueError, match="episode 99"):
        write_label(demos, "closed", 99, 1)
    with pytest.raises(ValueError, match="outside episode 0"):
        write_label(demos, "closed", 0, LENGTH)
    with pytest.raises(ValueError, match="check name"):
        write_label(demos, "no spaces", 0, 1)


def test_relabelling_and_removing_touch_only_that_episode(tmp_path, demos) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(demos, copy)
    write_label(copy, "closed", 1, 3)
    write_label(copy, "seated", 1, None)
    assert labels_for(copy, "closed")[1] == 3 and labels_for(copy, "closed")[0] == 8
    assert remove_label(copy, "closed", 1) is True
    assert remove_label(copy, "closed", 1) is False
    assert 1 not in labels_for(copy, "closed") and labels_for(copy, "seated") == {
        1: None
    }
    assert not list((copy / "meta").glob(".check-labels-*"))  # no temp files left


def test_frames_from_the_first_are_positive_and_a_margin_before_it_is_left_out() -> (
    None
):
    targets = frame_targets(10, 6, margin=2)
    assert targets.tolist() == [0, 0, 0, 0, IGNORED, IGNORED, 1, 1, 1, 1]
    assert frame_targets(5, None, margin=2).tolist() == [0] * 5
    assert frame_targets(5, 1, margin=3).tolist() == [IGNORED, 1, 1, 1, 1]
    assert frame_targets(5, 0, margin=3).tolist() == [1] * 5


def test_a_verdict_stands_in_for_a_label_only_where_it_says_something() -> None:
    def outcome(verdict):
        return RolloutOutcome(
            episode="e",
            checkpoint="c",
            task=None,
            started_at="t",
            verdict=None
            if verdict is None
            else Verdict(
                outcome=verdict,
                failure_mode="stalled" if verdict == "failure" else None,
            ),
        )

    assert verdict_targets(6, outcome("failure"), tail=2).tolist() == [0] * 6
    success = verdict_targets(6, outcome("success"), tail=2).tolist()
    assert success == [IGNORED] * 4 + [1, 1]
    assert verdict_targets(6, outcome("unknown"), tail=2) is None
    assert verdict_targets(6, outcome(None), tail=2) is None
    assert verdict_targets(6, None, tail=2) is None


def test_merged_labels_are_renumbered_as_the_merge_numbers_episodes(tmp_path) -> None:
    first, second, merged = tmp_path / "a", tmp_path / "b", tmp_path / "m"
    for root in (first, second):
        (root / "meta").mkdir(parents=True)
    (first / LABELS_FILE).write_text(
        json.dumps({"format_version": 1, "checks": {"closed": {"1": 4}}})
    )
    (second / LABELS_FILE).write_text(
        json.dumps(
            {"format_version": 1, "checks": {"closed": {"0": None}, "seated": {"2": 7}}}
        )
    )
    merge_labels([first, second], merged, [3, 5])
    assert read_labels(merged) == {"closed": {1: 4, 3: None}, "seated": {5: 7}}
    only_never = tmp_path / "o"
    (only_never / "meta").mkdir(parents=True)
    (only_never / LABELS_FILE).write_text(
        json.dumps({"format_version": 1, "checks": {"closed": {"0": None}}})
    )
    merge_labels([only_never], tmp_path / "m2", [1])
    assert read_labels(tmp_path / "m2") == {"closed": {0: None}}
    nothing = tmp_path / "n"
    (nothing / "meta").mkdir(parents=True)
    assert merge_labels([nothing], tmp_path / "out", [2]) is None
    assert not (tmp_path / "out" / LABELS_FILE).exists()


def test_merging_recorded_episodes_carries_their_labels(tmp_path) -> None:
    pytest.importorskip("lerobot")
    from flexivtrainer.jobs.merge_episodes import merge_episode_datasets

    first = _dataset(tmp_path / "one", [5, None])
    second = _dataset(tmp_path / "two", ["near", 7], seed=1)
    write_label(first, "closed", 0, 5)
    write_label(second, "closed", 1, 7)
    merge_episode_datasets([first, second], tmp_path / "merged", "all")
    assert labels_for(tmp_path / "merged" / "all", "closed") == {0: 5, 3: 7}


# --- training -----------------------------------------------------------------------


def test_inputs_that_cannot_mean_what_they_say_are_refused() -> None:
    from flexivtrainer.evaluation.learned_check import CheckInputs

    with pytest.raises(ValueError, match="at least one camera"):
        CheckInputs()
    with pytest.raises(ValueError, match="not an input camera"):
        _inputs(crops={"ego": (0, 0, 1, 1)})
    with pytest.raises(ValueError, match="x0 < x1"):
        _inputs(crops={"wrist": (0.5, 0, 0.4, 1)})
    with pytest.raises(ValueError, match="encoder"):
        _inputs(encoder="vit")
    with pytest.raises(ValueError, match="listed twice"):
        _inputs(cameras=("wrist", "wrist"))
    round_trip = CheckInputs.from_dict(
        _inputs(crops={"wrist": (0, 0.1, 1, 0.9)}).to_dict()
    )
    assert round_trip == _inputs(crops={"wrist": (0, 0.1, 1, 0.9)})


def test_a_check_learns_from_labels_and_reports_on_episodes_it_never_saw(
    tmp_path, demos
) -> None:
    from flexivtrainer.evaluation.learned_check import LearnedCheckModel, train_check

    report = train_check([demos], "closed", _inputs(), tmp_path / "check", _options())
    held = report["held_out"]
    assert report["trained"]["episodes"] == {"train": 7, "held_out": 3}
    assert held["false_passes"] == 0 and held["misses"] == 0
    assert {e["result"] for e in held["episodes"]} <= {"passed", "no pass"}
    for episode in held["episodes"]:
        if episode["reached"]:
            assert episode["first_at_s"] >= episode["labelled_at_s"] - 0.2
    assert [row["threshold"] for row in held["sweep"]][0] == 0.1

    model = LearnedCheckModel.load(tmp_path / "check")
    assert model.name == "closed" and model.inputs == _inputs()
    rng = np.random.default_rng(5)
    shown = [_frame(20, 8, rng)[0], _frame(2, 8, rng)[0], _frame(20, "near", rng)[0]]
    p = model.probabilities({"wrist": np.stack(shown)})
    assert p[0] > 0.5 > p[1] and p[2] < 0.5
    with pytest.raises(ValueError, match=r"no frames for camera \['wrist'\]"):
        model.probabilities({"ego": np.stack(shown)})


def test_held_out_episodes_are_whole_and_drawn_from_both_kinds(demos) -> None:
    from flexivtrainer.evaluation.learned_check import _split, collect_episodes

    episodes = collect_episodes([demos], "closed", _options())
    train, held = _split(episodes, 0.3, seed=3)
    assert len(train) + len(held) == len(EPISODES)
    assert any(e.reached for e in held) and any(not e.reached for e in held)
    assert any(e.reached for e in train) and any(not e.reached for e in train)
    assert _split(episodes, 0.3, seed=3) == (train, held)
    train, held = _split(episodes[1:3], 0.9, seed=0)  # one of each kind
    assert len(held) == 0


def test_a_check_built_on_force_alone_needs_no_camera(tmp_path, demos) -> None:
    from flexivtrainer.evaluation.learned_check import (
        CheckInputs,
        LearnedCheckModel,
        train_check,
    )

    inputs = CheckInputs(state=(FZ,))
    report = train_check([demos], "closed", inputs, tmp_path / "force", _options())
    assert report["held_out"]["false_passes"] == 0
    model = LearnedCheckModel.load(tmp_path / "force")
    p = model.probabilities(state=np.array([[30.0], [2.0]]))
    assert p[0] > 0.5 > p[1]
    # Standardised by the training frames' statistics, kept with the check.
    assert 2.0 < float(model.state_mean[0]) < 30.0 and float(model.state_std[0]) > 1
    centred = model._standardise(np.array([[float(model.state_mean[0])]]))
    assert abs(float(centred[0, 0])) < 1e-5


def test_training_refuses_labels_with_one_side_only(tmp_path) -> None:
    pytest.importorskip("lerobot")
    from flexivtrainer.evaluation.learned_check import train_check

    root = _dataset(tmp_path / "one_sided", [0, 0])
    _label_all(root, [0, 0])
    with pytest.raises(ValueError, match="needs frames where its state holds"):
        train_check([root], "closed", _inputs(), tmp_path / "x", _options())
    with pytest.raises(ValueError, match="needs frames"):
        train_check([root], "unlabelled", _inputs(), tmp_path / "x", _options())


def test_a_dataset_without_an_input_is_named_not_guessed(tmp_path, demos) -> None:
    from flexivtrainer.evaluation.learned_check import train_check

    with pytest.raises(ValueError, match=r"no RGB camera \['ego'\]"):
        train_check([demos], "closed", _inputs(cameras=("ego",)), tmp_path / "x")
    with pytest.raises(ValueError, match="no state column"):
        train_check(
            [demos], "closed", _inputs(state=("single_arm.tcp_wrench.mx",)), tmp_path
        )


def test_an_early_pass_on_an_episode_that_gets_there_is_a_false_pass() -> None:
    from flexivtrainer.evaluation.learned_check import LabelledEpisode, _report

    times = np.arange(10) / 10.0
    reached = LabelledEpisode(None, 0, frame_targets(10, 6, margin=1), "label")
    never = LabelledEpisode(None, 1, np.zeros(10, dtype=np.int8), "label")
    early = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    on_time = np.array([0.1] * 5 + [0.9] * 5)  # starts in the margin: fine
    report = _report(
        [reached, reached, never],
        [(times, early), (times, on_time), (times, np.full(10, 0.2))],
        threshold=0.5,
        hold_s=0.15,
    )
    assert [e["result"] for e in report["episodes"]] == [
        "false pass",
        "passed",
        "no pass",
    ]
    assert report["false_passes"] == 1 and report["misses"] == 0
    loose = {row["threshold"]: row for row in report["sweep"]}[0.1]
    assert loose["false_passes"] == 3 and loose["misses"] == 0


# --- as a check in a session -------------------------------------------------------


@pytest.fixture(scope="module")
def trained(demos, tmp_path_factory):
    from flexivtrainer.evaluation.learned_check import train_check

    out = tmp_path_factory.mktemp("models") / "closed"
    train_check([demos], "closed", _inputs(), out, _options(holdout=0.0))
    return out


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"kind": "learned"}, "needs a model"),
        ({"kind": "learned", "model": "m", "threshold": 1.5}, "between 0 and 1"),
        ({"kind": "learned", "model": "m", "hold_s": -1}, "0 or more"),
        ({"kind": "learned", "model": "m", "when": "start"}, "'any' or 'end'"),
        ({"kind": "learned", "model": "m", "box_min_m": [0, 0, 0]}, "unknown keys"),
    ],
)
def test_a_learned_check_that_cannot_mean_what_it_says_is_refused(spec, match) -> None:
    with pytest.raises(ValueError, match=match):
        parse_checks({"c": spec})


def test_a_relative_model_is_found_under_the_storage_root(tmp_path) -> None:
    check = parse_checks({"c": {"kind": "learned", "model": "checks/c"}})["c"]
    assert isinstance(check, LearnedCheck)
    assert check.model_path(tmp_path) == tmp_path / "checks" / "c"
    with pytest.raises(ValueError, match="no storage root"):
        check.model_path(None)
    absolute = LearnedCheck(model=str(tmp_path / "m"))
    assert absolute.model_path(None) == tmp_path / "m"


def _trial(root, first, verdict, *, seed):
    """One recorded trial: a single-episode dataset with an outcome."""
    _dataset(root, [first], seed=seed)
    write_outcome(
        root,
        RolloutOutcome(
            episode=root.name,
            checkpoint="x",
            task=None,
            started_at="t",
            session="cmp",
            trial=int(root.name[1:]),
            verdict=Verdict(
                outcome=verdict,
                failure_mode=None if verdict == "success" else "stalled",
            ),
        ),
    )
    return root


def test_a_recorded_trial_is_scored_from_its_frames(tmp_path, trained) -> None:
    pytest.importorskip("lerobot")
    closed = _trial(tmp_path / "e0", 8, "success", seed=3)
    missed = _trial(tmp_path / "e1", "near", "failure", seed=4)
    checks = {
        "closed": LearnedCheck(model="closed", hold_s=0.3),
        "placed": SettledInBox(box_min_m=(0, 0, 0), box_max_m=(1, 1, 1)),
    }
    yes = run_checks(checks, [], episode_dir=closed, storage_root=trained.parent)
    assert yes["closed"]["passed"] is True and yes["closed"]["frames"] == LENGTH
    assert 0.6 <= yes["closed"]["first_at_s"] <= 1.0
    assert yes["placed"]["error"] == "no tick log"  # unchanged for tick checks
    no = run_checks(checks, [], episode_dir=missed, storage_root=trained.parent)
    assert no["closed"]["passed"] is False and no["closed"]["max_p"] < 0.5
    strict = LearnedCheck(model=str(trained), threshold=0.25)
    assert run_checks({"c": strict}, [], episode_dir=closed)["c"]["threshold"] == 0.25


def test_a_learned_check_passes_on_a_long_enough_stretch_or_only_at_the_end() -> None:
    from flexivtrainer.evaluation.learned_check import judge

    times = [i / 10 for i in range(10)]
    middle = [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1]
    assert judge(times, middle, 0.5, 0.3) == {
        "passed": True,
        "held_s": 0.3,
        "first_at_s": 0.1,
        "max_p": 0.9,
    }
    assert judge(times, middle, 0.5, 0.3, when="end")["passed"] is False
    assert judge(times, middle, 0.5, 0.4)["passed"] is False
    assert judge(times[:5], middle[:5], 0.5, 0.3, when="end")["passed"] is True
    assert judge([], [], 0.5, 0.0) == {
        "passed": False,
        "held_s": 0.0,
        "first_at_s": None,
        "max_p": 0.0,
    }


def test_a_learned_check_without_a_recording_is_unscored(tmp_path, trained) -> None:
    check = {"closed": LearnedCheck(model=str(trained))}
    assert run_checks(check, [{"t": 0.0}])["closed"]["error"] == "no recording"
    empty = tmp_path / "e0"
    (empty / "meta").mkdir(parents=True)
    result = run_checks(check, [{"t": 0.0}], episode_dir=empty)
    assert result["closed"] == {"kind": "learned", "error": "no recording"}
    gone = {"closed": LearnedCheck(model=str(tmp_path / "missing"))}
    closed = _trial(tmp_path / "e1", 8, "success", seed=3)
    error = run_checks(gone, [], episode_dir=closed)["closed"]["error"]
    assert "FileNotFoundError" in error  # unscored, never a failure


def test_rescoring_a_session_with_a_learned_check_and_training_on_verdicts(
    tmp_path, trained
) -> None:
    from flexivtrainer.evaluation.learned_check import collect_episodes
    from flexivtrainer.evaluation.rescore import rescore

    session = tmp_path / "cmp"
    session.mkdir()
    plan = {**PLAN, "checks": {"closed": {"kind": "learned", "model": "closed"}}}
    (session / "session.json").write_text(json.dumps(plan))
    _trial(session / "e0", 6, "success", seed=6)
    _trial(session / "e1", "near", "failure", seed=7)
    _trial(session / "e2", 9, "failure", seed=8)  # judged a failure for another reason

    report = rescore(tmp_path, "cmp", storage_root=trained.parent)
    agreement = report["agreement"]["closed"]
    assert agreement["trials"] == 3 and agreement["passed_but_failure"] == 1
    assert agreement["agree"] == 2

    episodes = collect_episodes([session], "closed", _options(use_verdicts=True))
    assert [(e.root.name, e.source, e.reached) for e in episodes] == [
        ("e0", "verdict", True),
        ("e1", "verdict", False),
        ("e2", "verdict", False),
    ]
    assert collect_episodes([session], "closed", _options()) == []
    write_label(session / "e2", "closed", 0, 9)  # a label wins over the verdict
    labelled = collect_episodes([session], "closed", _options(use_verdicts=True))
    assert [(e.root.name, e.source) for e in labelled][-1] == ("e2", "label")


def test_a_trial_scores_its_learned_check_before_it_waits_for_its_verdict(
    tmp_path,
    fake_loop,  # noqa: F811
) -> None:
    pytest.importorskip("lerobot")
    from flexivtrainer.evaluation.learned_check import CheckInputs, train_check

    # The fake robot holds its tool at z = 3: a state-only check on the height.
    demos = _dataset(tmp_path / "demos", EPISODES)
    _label_all(demos, EPISODES, check="high")
    train_check(
        [demos],
        "high",
        CheckInputs(state=(Z,)),
        tmp_path / "checks" / "high",
        _options(holdout=0.0),
    )
    robot = _FakeRobot("F1")
    service = _service(tmp_path, _FakePolicy(_action()), robot)
    checkpoint = _checkpoint(tmp_path)
    session = tmp_path / "rollouts" / "cmp"
    session.mkdir(parents=True)
    plan = {
        **PLAN,
        "arms": {"A": {"checkpoint": checkpoint}, "B": {"checkpoint": checkpoint}},
        "checks": {"high": {"kind": "learned", "model": "checks/high", "hold_s": 0.0}},
    }
    (session / "session.json").write_text(json.dumps(plan))
    _run_ticks(service, robot, lambda: service.start_next_trial("cmp"))
    status = _wait_for_verdict_slot(service)
    scored = read_outcome(session / status["awaiting_verdict"]["episode"]).checks
    assert scored["high"]["passed"] is True, scored


# --- command line -------------------------------------------------------------------


def test_the_command_labels_trains_and_scores(tmp_path, demos, capsys) -> None:
    from flexivtrainer.evaluation.learned_check import main

    copy = tmp_path / "copy"
    shutil.copytree(demos, copy)
    assert (
        main(
            [
                "label",
                str(copy),
                "--check",
                "lid",
                "--episode",
                "0",
                "--first-frame",
                "8",
            ]
        )
        == 0
    )
    assert (
        main(["label", str(copy), "--check", "lid", "--episode", "2", "--never"]) == 0
    )
    assert (
        main(["label", str(copy), "--check", "lid", "--episode", "4", "--never"]) == 0
    )
    assert (
        main(
            [
                "label",
                str(copy),
                "--check",
                "lid",
                "--episode",
                "3",
                "--first-frame",
                "6",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["labels", str(copy)]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["lid"]["summary"] == {"episodes": 4, "reached": 2, "never": 2}

    out = tmp_path / "lid"
    assert (
        main(
            [
                "train",
                "--check",
                "lid",
                "--data",
                str(copy),
                "--out",
                str(out),
                "--camera",
                "wrist",
                "--encoder",
                "pixels",
                "--grid",
                "4",
                "--image-size",
                "16x16",
                "--holdout",
                "0",
                "--device",
                "cpu",
                "--crop",
                "wrist=0,0,1,1",
            ]
        )
        == 0
    )
    trained = json.loads(capsys.readouterr().out)
    assert trained["held_out"] is None and (out / "check.json").is_file()

    assert (
        main(
            [
                "score",
                str(copy),
                "--model",
                str(out),
                "--episode",
                "1",
                "--episode",
                "6",
                "--frames",
            ]
        )
        == 0
    )
    scored = json.loads(capsys.readouterr().out)
    assert [e["passed"] for e in scored["episodes"]] == [True, False]
    assert len(scored["episodes"][0]["p"]) == LENGTH

    assert (
        main(["label", str(copy), "--check", "lid", "--episode", "0", "--remove"]) == 0
    )
    assert (
        main(["label", str(copy), "--check", "lid", "--episode", "99", "--never"]) == 2
    )
    assert "episode 99" in capsys.readouterr().err
