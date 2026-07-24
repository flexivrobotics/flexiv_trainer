from __future__ import annotations

import subprocess
import sys
from collections import deque
from types import SimpleNamespace

import pytest

lerobot = pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
)

from flexivtrainer.policies.lerobot_plugins import (  # noqa: E402
    BSplineDiffusionConfig,
    BSplineDiffusionPolicy,
    make_bspline_diffusion_pre_post_processors,
)


def _action_names(rows: int, channels: int) -> list[str]:
    suffixes = ["knot", *(f"control_{index}" for index in range(channels - 1))]
    return [
        f"bspline.row_{row:02d}.{suffix}"
        for row in range(rows)
        for suffix in suffixes
    ]


def _config(channels: int = 11, **overrides) -> BSplineDiffusionConfig:
    rows = overrides.pop("horizon", 16)
    values = {
        "horizon": rows,
        "n_obs_steps": 2,
        "down_dims": (8, 16),
        "n_groups": 4,
        "diffusion_step_embed_dim": 16,
        "num_train_timesteps": 2,
        "num_inference_steps": 1,
        "device": "cpu",
        "input_features": {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (3,)),
            OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (2,)),
        },
        "output_features": {
            ACTION: PolicyFeature(FeatureType.ACTION, (rows * channels,))
        },
        "action_feature_names": _action_names(rows, channels),
    }
    values.update(overrides)
    return BSplineDiffusionConfig(**values)


def _batch(channels: int) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(2, 2, 3),
        OBS_ENV_STATE: torch.randn(2, 2, 2),
        ACTION: torch.randn(2, 1, 16 * channels),
        "action_is_pad": torch.zeros(2, 1, dtype=torch.bool),
    }


def test_plugin_discovery_in_fresh_process() -> None:
    code = """
from lerobot.configs.parser import load_plugin
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
load_plugin("flexivtrainer.policies.lerobot_plugins")
assert "bspline_diffusion" in PreTrainedConfig.get_known_choices()
assert get_policy_class("bspline_diffusion").name == "bspline_diffusion"
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("channels", [10, 11, 19, 21])
def test_config_accepts_supported_dynamic_action_widths(channels: int) -> None:
    config = _config(channels)

    assert config.action_delta_indices == [0]
    assert config.logical_action_shape() == (16, channels)
    assert config.n_action_steps == 1
    assert config.drop_n_last_frames == 0
    assert config.do_mask_loss_for_padding is False
    assert config.spline_degree == 3
    assert config.knot_rate_hz is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"spline_degree": 0}, "spline_degree"),
        ({"spline_degree": 2.5}, "spline_degree"),
        ({"knot_rate_hz": 0}, "knot_rate_hz"),
        ({"knot_rate_hz": float("inf")}, "knot_rate_hz"),
    ],
)
def test_config_rejects_invalid_spline_metadata(change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**change)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n_action_steps": 2}, "n_action_steps=1"),
        ({"drop_n_last_frames": 1}, "drop_n_last_frames=0"),
        ({"do_mask_loss_for_padding": True}, "padded-loss masking"),
    ],
)
def test_config_rejects_unsupported_diffusion_modes(change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**change)


def test_config_rejects_malformed_or_inconsistent_names() -> None:
    malformed = _config()
    malformed.action_feature_names[0] = "action.0"
    with pytest.raises(ValueError, match="Malformed"):
        malformed.logical_action_shape()

    inconsistent = _config()
    inconsistent.action_feature_names[-1] = "bspline.row_15.other"
    with pytest.raises(ValueError, match="identical channel layouts"):
        inconsistent.logical_action_shape()

    missing_row = _config()
    missing_row.action_feature_names = missing_row.action_feature_names[:-11]
    missing_row.output_features[ACTION] = PolicyFeature(
        FeatureType.ACTION, (15 * 11,)
    )
    with pytest.raises(ValueError, match="expected horizon=16"):
        missing_row.logical_action_shape()


@pytest.mark.parametrize("channels", [11, 21])
def test_forward_reshapes_flat_plan_and_backpropagates(
    channels: int,
    monkeypatch,
) -> None:
    policy = BSplineDiffusionPolicy(_config(channels))
    core_action_shape = None
    compute_loss = policy.diffusion.compute_loss

    def capture_core_shape(batch):
        nonlocal core_action_shape
        core_action_shape = tuple(batch[ACTION].shape)
        return compute_loss(batch)

    monkeypatch.setattr(policy.diffusion, "compute_loss", capture_core_shape)

    loss, output = policy(_batch(channels))

    assert output is None
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
    assert core_action_shape == (2, 16, channels)
    assert policy.diffusion.config.action_feature.shape == (channels,)


def test_inference_returns_one_flat_spline_plan() -> None:
    channels = 11
    policy = BSplineDiffusionPolicy(_config(channels))
    policy.eval()
    noise = torch.randn(2, 16, channels)

    action = policy.select_action(
        {
            OBS_STATE: torch.randn(2, 3),
            OBS_ENV_STATE: torch.randn(2, 2),
        },
        noise=noise,
    )

    assert action.shape == (2, 16 * channels)


def test_observations_can_be_enqueued_without_running_inference(
    monkeypatch,
) -> None:
    policy = BSplineDiffusionPolicy(_config())
    calls = 0
    sample = policy.diffusion.conditional_sample

    def count_samples(*args, **kwargs):
        nonlocal calls
        calls += 1
        return sample(*args, **kwargs)

    monkeypatch.setattr(policy.diffusion, "conditional_sample", count_samples)
    observation = {
        OBS_STATE: torch.randn(1, 3),
        OBS_ENV_STATE: torch.randn(1, 2),
    }

    policy.enqueue_observation(observation)
    policy.enqueue_observation(observation)

    assert calls == 0
    assert len(policy._queues[OBS_STATE]) == 2
    action = policy.predict_action_chunk(noise=torch.randn(1, 16, 11))
    assert calls == 1
    assert action.shape == (1, 1, 176)


def test_logical_core_config_does_not_mutate_flat_policy_config() -> None:
    config = _config()

    policy = BSplineDiffusionPolicy(config)

    assert config.output_features[ACTION].shape == (176,)
    assert policy.config.output_features[ACTION].shape == (176,)
    assert policy.diffusion.config.output_features[ACTION].shape == (11,)


def test_current_images_gain_time_dimension_only_when_queued() -> None:
    policy = BSplineDiffusionPolicy.__new__(BSplineDiffusionPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        image_features={"camera_a": object(), "camera_b": object()},
        n_obs_steps=2,
    )
    batch = {
        "camera_a": torch.randn(2, 3, 12, 16),
        "camera_b": torch.randn(2, 3, 12, 16),
    }

    prepared = policy._prepare_observations(
        batch,
        ensure_time_dimension=False,
    )
    queues = {OBS_IMAGES: deque(maxlen=2)}
    populate_queues(queues, prepared)
    stacked = torch.stack(list(queues[OBS_IMAGES]), dim=1)

    assert prepared[OBS_IMAGES].shape == (2, 2, 3, 12, 16)
    assert stacked.shape == (2, 2, 2, 3, 12, 16)


def test_policy_save_reload_preserves_logical_shape(tmp_path) -> None:
    policy = BSplineDiffusionPolicy(_config())
    policy.save_pretrained(tmp_path)

    restored = BSplineDiffusionPolicy.from_pretrained(tmp_path)
    restored.eval()
    action = restored.select_action(
        {
            OBS_STATE: torch.randn(1, 3),
            OBS_ENV_STATE: torch.randn(1, 2),
        },
        noise=torch.randn(1, 16, 11),
    )

    assert restored.parameter_rows == 16
    assert restored.parameter_channels == 11
    assert restored.config.action_feature_names == _action_names(16, 11)
    assert restored.config.spline_degree == 3
    assert restored.config.knot_rate_hz is None
    assert action.shape == (1, 176)


def test_processors_are_standard_diffusion_processors() -> None:
    config = _config()

    preprocessor, postprocessor = make_bspline_diffusion_pre_post_processors(
        config
    )

    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"


def test_default_rollout_loader_restores_plugin_checkpoint(tmp_path) -> None:
    from flexivtrainer.rollout.service import _default_policy_loader

    config = _config()
    policy = BSplineDiffusionPolicy(config)
    preprocessor, postprocessor = make_bspline_diffusion_pre_post_processors(
        config
    )
    policy.save_pretrained(tmp_path)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    restored, restored_pre, restored_post = _default_policy_loader(
        str(tmp_path), "cpu"
    )

    assert isinstance(restored, BSplineDiffusionPolicy)
    assert restored.config.action_feature_names == _action_names(16, 11)
    assert restored_pre.name == "policy_preprocessor"
    assert restored_post.name == "policy_postprocessor"


def test_tied_action_statistics_round_trip_through_standard_processors() -> None:
    config = _config()
    channel_min = torch.arange(11, dtype=torch.float32)
    action_min = channel_min.repeat(16)
    action_max = action_min + 2.0

    def stats(minimum, maximum):
        return {
            "min": minimum,
            "max": maximum,
            "mean": (minimum + maximum) / 2,
            "std": torch.ones_like(minimum),
            "count": torch.tensor([1]),
        }

    dataset_stats = {
        ACTION: stats(action_min, action_max),
        OBS_STATE: stats(torch.zeros(3), torch.ones(3)),
        OBS_ENV_STATE: stats(torch.zeros(2), torch.ones(2)),
    }
    preprocessor, postprocessor = make_bspline_diffusion_pre_post_processors(
        config,
        dataset_stats,
    )
    action = action_min + 0.75
    processed = preprocessor(
        {
            ACTION: action,
            OBS_STATE: torch.zeros(3),
            OBS_ENV_STATE: torch.zeros(2),
        }
    )

    restored = postprocessor(processed[ACTION])

    torch.testing.assert_close(restored.squeeze(0), action)
