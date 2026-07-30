import os
from pathlib import Path

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")
av = pytest.importorskip("av")

from lerobot.datasets.video_utils import FrameTimestampError  # noqa: E402

from flexivtrainer.config import TRAIN_LOAD_DEPTH_ENV  # noqa: E402
from flexivtrainer.policies.lerobot_plugins.dataset_patches import (  # noqa: E402
    depth_filter_requested,
    is_depth_feature,
    original_decode_video_frames_pyav,
    robust_decode_video_frames_pyav,
)

# Not a from-import of video_utils: another test module may have installed the
# patch already, making that name resolve to the patched function.
decode_video_frames_pyav = original_decode_video_frames_pyav()

FPS = 30
# Episode 185's first frame, on a segment boundary upstream's seek cannot reach.
KNOWN_BAD_VIDEO = Path(
    ".local/datasets/merged_20260728_163313/videos/"
    "observation.images.ego_depth/chunk-000/file-021.mp4"
)
KNOWN_BAD_TS = 3149 / FPS


@pytest.fixture
def rgb_video(tmp_path: Path) -> Path:
    """Short h264 clip with a keyframe every other frame, as recorded."""

    path = tmp_path / "clip.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.gop_size = 2
        for index in range(30):
            array = np.full((48, 64, 3), index * 8 % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return path


class TestDepthFilterRequested:
    def test_absent_variable_leaves_metadata_alone(self, monkeypatch):
        # Must be a no-op in the app process, which needs depth for previews.
        monkeypatch.delenv(TRAIN_LOAD_DEPTH_ENV, raising=False)
        assert depth_filter_requested() is False

    def test_zero_drops_depth(self, monkeypatch):
        monkeypatch.setenv(TRAIN_LOAD_DEPTH_ENV, "0")
        assert depth_filter_requested() is True

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_truthy_keeps_depth(self, monkeypatch, value):
        monkeypatch.setenv(TRAIN_LOAD_DEPTH_ENV, value)
        assert depth_filter_requested() is False


class TestIsDepthFeature:
    def test_matches_lerobot_predicate(self):
        assert is_depth_feature({"info": {"is_depth_map": True}})
        assert is_depth_feature({"info": {"video.is_depth_map": True}})
        assert is_depth_feature({"video_info": {"video.is_depth_map": True}})
        assert not is_depth_feature({"info": {"is_depth_map": False}})
        assert not is_depth_feature({"info": {}})
        assert not is_depth_feature({})

    def test_agrees_with_installed_lerobot(self):
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        assert hasattr(LeRobotDatasetMetadata, "depth_keys")


class TestRobustDecodeEquivalence:
    """The retry must not change results where upstream already succeeds."""

    @pytest.mark.parametrize("frame_index", [0, 1, 7, 14, 29])
    def test_matches_upstream_frame_for_frame(self, rgb_video, frame_index):
        timestamps = [frame_index / FPS]
        expected = decode_video_frames_pyav(rgb_video, timestamps, 1e-4)
        actual = robust_decode_video_frames_pyav(rgb_video, timestamps, 1e-4)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_matches_upstream_for_a_two_frame_window(self, rgb_video):
        timestamps = [9 / FPS, 10 / FPS]
        expected = decode_video_frames_pyav(rgb_video, timestamps, 1e-4)
        actual = robust_decode_video_frames_pyav(rgb_video, timestamps, 1e-4)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_still_raises_when_no_frame_is_close_enough(self, rgb_video):
        with pytest.raises(FrameTimestampError):
            robust_decode_video_frames_pyav(rgb_video, [99.0], 1e-4)


@pytest.mark.skipif(
    not KNOWN_BAD_VIDEO.exists(),
    reason=f"regression fixture {KNOWN_BAD_VIDEO} is not present",
)
class TestKnownBoundaryFrame:
    """Regression guard for the frame that aborted a training run."""

    def test_upstream_still_misses_it(self):
        with pytest.raises(FrameTimestampError):
            decode_video_frames_pyav(
                KNOWN_BAD_VIDEO, [KNOWN_BAD_TS], 1e-4, is_depth=True
            )

    def test_retry_recovers_it(self):
        frames = robust_decode_video_frames_pyav(
            KNOWN_BAD_VIDEO, [KNOWN_BAD_TS], 1e-4, is_depth=True
        )
        assert frames.shape[0] == 1
        assert frames.shape[1] == 1
        assert frames.numpy().any()

    def test_retry_recovers_the_two_frame_observation_window(self):
        # How training hit this: n_obs_steps=2 clamps both requests onto the
        # boundary at an episode's first frame.
        frames = robust_decode_video_frames_pyav(
            KNOWN_BAD_VIDEO, [KNOWN_BAD_TS, KNOWN_BAD_TS], 1e-4, is_depth=True
        )
        assert frames.shape[0] == 2


def test_patches_are_installed_on_import():
    from lerobot.datasets import video_utils

    assert video_utils.decode_video_frames_pyav is robust_decode_video_frames_pyav


def test_apply_patches_is_idempotent():
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    from flexivtrainer.policies.lerobot_plugins.dataset_patches import apply_patches

    before = LeRobotDatasetMetadata.__init__
    apply_patches()
    apply_patches()
    assert LeRobotDatasetMetadata.__init__ is before


@pytest.mark.skipif(
    not KNOWN_BAD_VIDEO.exists(),
    reason="depth-bearing dataset is not present",
)
def test_metadata_drops_depth_only_when_requested(monkeypatch):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    root = KNOWN_BAD_VIDEO.parents[3]  # <root>/videos/<key>/chunk-000/file.mp4

    monkeypatch.delenv(TRAIN_LOAD_DEPTH_ENV, raising=False)
    assert LeRobotDatasetMetadata("local/x", root=root).depth_keys

    monkeypatch.setenv(TRAIN_LOAD_DEPTH_ENV, "1")
    assert LeRobotDatasetMetadata("local/x", root=root).depth_keys

    monkeypatch.setenv(TRAIN_LOAD_DEPTH_ENV, "0")
    meta = LeRobotDatasetMetadata("local/x", root=root)
    assert meta.depth_keys == []
    assert meta.video_keys
    assert all("depth" not in key for key in meta.video_keys)


def test_import_does_not_set_the_env_var():
    assert TRAIN_LOAD_DEPTH_ENV not in os.environ or os.environ[
        TRAIN_LOAD_DEPTH_ENV
    ] in {"0", "1", "true", "false", "yes", "no", "on", "off"}
