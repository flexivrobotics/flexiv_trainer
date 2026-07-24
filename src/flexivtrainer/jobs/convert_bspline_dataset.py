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

"""Convert physical TCP actions in a LeRobot dataset to B-spline parameters."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from flexivtrainer.data.bspline import (
    TCPActionLayout,
    build_episode_spline_targets,
    detect_tcp_action_layouts,
    extract_cartesian_controls,
    parameter_feature_names,
    parameter_matrix_shape,
    rotation_6d_to_matrix,
    validate_parameter_matrix_shape,
)

_INFO_PATH = Path("meta/info.json")
_BSPLINE_METADATA_PATH = Path("meta/bspline.json")


@dataclass(frozen=True, slots=True)
class _RecordedFrame:
    episode_index: int
    frame_index: int
    action: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_action_names(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    action = features.get("action") if isinstance(features, dict) else None
    names = action.get("names") if isinstance(action, dict) else None
    if not isinstance(names, list) or not names or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError(
            "The source dataset's action feature must have named axes in "
            "meta/info.json. Re-record or repair metadata before conversion."
        )
    return names


def _load_recorded_frames(root: Path, action_dim: int) -> list[_RecordedFrame]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - installed with LeRobot
        raise RuntimeError(
            "pandas/pyarrow are required for dataset conversion"
        ) from exc

    data_paths = sorted((root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise ValueError(f"No LeRobot data parquet files found under {root / 'data'}")

    frames: list[_RecordedFrame] = []
    seen: set[tuple[int, int]] = set()
    for data_path in data_paths:
        table = pd.read_parquet(data_path)
        missing = {
            "action",
            "episode_index",
            "frame_index",
        } - set(table.columns)
        if missing:
            raise ValueError(
                f"{data_path} is missing required columns: {sorted(missing)}"
            )
        for action, episode_index, frame_index in zip(
            table["action"],
            table["episode_index"],
            table["frame_index"],
            strict=True,
        ):
            key = (int(episode_index), int(frame_index))
            if key in seen:
                raise ValueError(f"Duplicate episode/frame key in dataset: {key}")
            seen.add(key)
            action_array = np.asarray(action, dtype=np.float64).reshape(-1)
            if len(action_array) != action_dim:
                raise ValueError(
                    f"Action {key} has {len(action_array)} values, "
                    f"expected {action_dim}"
                )
            frames.append(
                _RecordedFrame(
                    episode_index=key[0],
                    frame_index=key[1],
                    action=action_array,
                )
            )
    return frames


def _episode_error_report(
    controls: np.ndarray,
    reconstructed: np.ndarray,
    layouts: Sequence[TCPActionLayout],
) -> tuple[float, float]:
    max_translation_error = 0.0
    max_rotation_error = 0.0
    start = 0
    for layout in layouts:
        translation_error = np.linalg.norm(
            reconstructed[:, start : start + 3] - controls[:, start : start + 3],
            axis=1,
        )
        max_translation_error = max(
            max_translation_error,
            float(np.max(translation_error)),
        )

        actual_matrix = rotation_6d_to_matrix(controls[:, start + 3 : start + 9])
        fitted_matrix = rotation_6d_to_matrix(
            reconstructed[:, start + 3 : start + 9]
        )
        difference = Rotation.from_matrix(fitted_matrix) * Rotation.from_matrix(
            actual_matrix
        ).inv()
        max_rotation_error = max(
            max_rotation_error,
            float(np.max(difference.magnitude())),
        )
        start += len(layout.control_names)
    return max_translation_error, math.degrees(max_rotation_error)


def _build_targets(
    frames: Sequence[_RecordedFrame],
    layouts: Sequence[TCPActionLayout],
    *,
    degree: int,
    chunk_size: int,
    stride: int,
    max_error: float,
    smoothing: float,
    max_knots: int | None,
) -> tuple[dict[tuple[int, int], np.ndarray], list[dict[str, Any]]]:
    grouped: dict[int, list[_RecordedFrame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.episode_index].append(frame)

    targets: dict[tuple[int, int], np.ndarray] = {}
    reports: list[dict[str, Any]] = []
    for episode_index in sorted(grouped):
        episode_frames = sorted(
            grouped[episode_index],
            key=lambda frame: frame.frame_index,
        )
        actions = np.stack([frame.action for frame in episode_frames])
        controls = extract_cartesian_controls(actions, layouts)
        result = build_episode_spline_targets(
            controls,
            degree=degree,
            chunk_size=chunk_size,
            stride=stride,
            max_error=max_error,
            smoothing=smoothing,
            max_knots=max_knots,
        )
        validate_parameter_matrix_shape(
            result.parameters,
            layouts,
            parameter_rows=chunk_size + 2 * degree,
        )

        sample_times = np.arange(len(controls), dtype=np.float64)
        reconstructed = np.asarray(result.fit.spline(sample_times))
        translation_error, rotation_error_deg = _episode_error_report(
            controls,
            reconstructed,
            layouts,
        )
        for local_index, frame in enumerate(episode_frames):
            targets[(frame.episode_index, frame.frame_index)] = result.parameters[
                local_index
            ].reshape(-1)

        reports.append(
            {
                "episode_index": episode_index,
                "frames": len(episode_frames),
                "knot_count": len(result.fit.knots),
                "control_point_count": len(result.fit.control_points),
                "max_component_error": result.fit.max_abs_error,
                "max_translation_error_m": translation_error,
                "max_rotation_error_deg": rotation_error_deg,
                "tolerance_reached": result.fit.tolerance_reached,
            }
        )
    return targets, reports


def _write_data_parquet(
    frame: Any,
    path: Path,
    features: dict[str, Any],
) -> None:
    """Replace a Parquet action column while preserving all other schemas."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - installed with LeRobot
        raise RuntimeError("pyarrow is required for dataset conversion") from exc

    table = pq.read_table(path)
    flat_actions = np.stack(frame["action"].to_numpy()).astype(
        np.float32,
        copy=False,
    )
    action_dim = int(features["action"]["shape"][0])
    if flat_actions.shape != (len(frame), action_dim):
        raise ValueError(
            f"Action table has shape {flat_actions.shape}, expected "
            f"({len(frame)}, {action_dim})"
        )
    values = pa.array(flat_actions.reshape(-1), type=pa.float32())
    action_column = pa.FixedSizeListArray.from_arrays(values, action_dim)
    action_index = table.column_names.index("action")
    table = table.set_column(action_index, "action", action_column)

    schema_metadata = dict(table.schema.metadata or {})
    huggingface_raw = schema_metadata.get(b"huggingface")
    if huggingface_raw:
        huggingface = json.loads(huggingface_raw)
        huggingface["info"]["features"]["action"] = {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": action_dim,
            "_type": "List",
        }
        schema_metadata[b"huggingface"] = json.dumps(
            huggingface,
            separators=(",", ":"),
        ).encode()
        table = table.replace_schema_metadata(schema_metadata)

    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(
        table,
        temporary,
        compression="snappy",
        use_dictionary=True,
    )
    temporary.replace(path)


def _replace_action_data(
    root: Path,
    targets: dict[tuple[int, int], np.ndarray],
    features: dict[str, Any],
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - installed with LeRobot
        raise RuntimeError(
            "pandas/pyarrow are required for dataset conversion"
        ) from exc

    replaced: set[tuple[int, int]] = set()
    for data_path in sorted((root / "data").glob("*/*.parquet")):
        frame = pd.read_parquet(data_path)
        new_actions: list[np.ndarray] = []
        for episode_index, frame_index in zip(
            frame["episode_index"],
            frame["frame_index"],
            strict=True,
        ):
            key = (int(episode_index), int(frame_index))
            try:
                new_actions.append(targets[key].astype(np.float32, copy=False))
            except KeyError as exc:
                raise ValueError(
                    f"No B-spline target generated for frame {key}"
                ) from exc
            replaced.add(key)
        frame["action"] = new_actions
        _write_data_parquet(frame, data_path, features)

    missing = set(targets) - replaced
    if missing:
        preview = sorted(missing)[:5]
        raise ValueError(f"Failed to replace {len(missing)} action rows: {preview}")


def _refresh_action_statistics(
    root: Path,
    targets: dict[tuple[int, int], np.ndarray],
    action_feature: dict[str, Any],
    *,
    parameter_rows: int,
) -> None:
    try:
        import pandas as pd
        from lerobot.datasets.dataset_tools import (
            compute_episode_stats,
            write_stats,
        )
    except ImportError as exc:  # pragma: no cover - installed with LeRobot
        raise RuntimeError("LeRobot 0.6 is required for statistics conversion") from exc

    by_episode: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (episode_index, frame_index), target in targets.items():
        by_episode[episode_index].append((frame_index, target))

    def tied_stats(ordered: np.ndarray) -> dict[str, Any]:
        if ordered.shape[1] % parameter_rows:
            raise ValueError(
                f"Action width {ordered.shape[1]} is not divisible by "
                f"{parameter_rows} parameter rows"
            )
        channel_count = ordered.shape[1] // parameter_rows
        row_stats = compute_episode_stats(
            {"action": ordered},
            {"action": action_feature},
        )["action"]
        tied: dict[str, Any] = {"count": row_stats["count"]}
        for stat_name, value in row_stats.items():
            if stat_name == "count":
                continue
            values = np.asarray(value).reshape(parameter_rows, channel_count)
            if stat_name == "min":
                channel_values = np.min(values, axis=0)
            elif stat_name == "max":
                channel_values = np.max(values, axis=0)
            else:
                channel_values = np.mean(values, axis=0)
            tied[stat_name] = np.tile(channel_values, parameter_rows)
        return tied

    episode_stats: dict[int, dict[str, Any]] = {}
    for episode_index, entries in by_episode.items():
        ordered = np.stack([value for _, value in sorted(entries)])
        episode_stats[episode_index] = tied_stats(ordered)

    stats_path = root / "meta" / "stats.json"
    existing_stats = _read_json(stats_path) if stats_path.exists() else {}
    all_targets = np.stack(
        [
            target
            for _key, target in sorted(
                targets.items(),
                key=lambda item: item[0],
            )
        ]
    )
    existing_stats["action"] = tied_stats(all_targets)
    write_stats(existing_stats, root)

    updated_episodes: set[int] = set()
    for metadata_path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        metadata = pd.read_parquet(metadata_path)
        for stat_name in next(iter(episode_stats.values())):
            column = f"stats/action/{stat_name}"
            if column not in metadata:
                metadata[column] = None
        for row_index, episode_index in metadata["episode_index"].items():
            index = int(episode_index)
            if index not in episode_stats:
                continue
            for stat_name, value in episode_stats[index].items():
                serialized = value.tolist() if hasattr(value, "tolist") else value
                metadata.at[row_index, f"stats/action/{stat_name}"] = serialized
            updated_episodes.add(index)
        temporary = metadata_path.with_suffix(".tmp.parquet")
        metadata.to_parquet(temporary, index=False)
        temporary.replace(metadata_path)

    missing = set(episode_stats) - updated_episodes
    if missing:
        raise ValueError(
            "Episode metadata is missing rows for converted episodes: "
            f"{sorted(missing)}"
        )


def _validate_output(
    root: Path,
    expected_frames: int,
    expected_action_dim: int,
) -> None:
    try:
        from datasets import config as datasets_config
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - installed with project
        raise RuntimeError(
            "LeRobot 0.6 is required to validate converted data"
        ) from exc

    original_cache = datasets_config.HF_DATASETS_CACHE
    with tempfile.TemporaryDirectory(prefix="flexivtrainer-hf-") as cache_dir:
        datasets_config.HF_DATASETS_CACHE = cache_dir
        try:
            dataset = LeRobotDataset(
                repo_id=f"local/{root.name}",
                root=root,
                download_videos=False,
            )
        finally:
            datasets_config.HF_DATASETS_CACHE = original_cache
        if len(dataset) != expected_frames:
            raise ValueError(
                f"Converted dataset has {len(dataset)} frames, "
                f"expected {expected_frames}"
            )
        action_shape = tuple(dataset.features["action"]["shape"])
        if action_shape != (expected_action_dim,):
            raise ValueError(
                f"Converted action shape is {action_shape}, expected "
                f"({expected_action_dim},)"
            )


def convert_lerobot_tcp_actions_to_bspline(
    source_root: Path,
    output_root: Path,
    *,
    sides: Sequence[str] | None = None,
    degree: int = 3,
    chunk_size: int = 10,
    stride: int = 1,
    max_error: float = 0.002,
    smoothing: float = 1e-12,
    max_knots: int | None = None,
) -> dict[str, Any]:
    """Create a LeRobot copy whose action is a flattened B-spline segment.

    The source dataset is never modified. The output is assembled in a temporary
    sibling directory and moved into place only after LeRobot validates it.
    """

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if not (source / _INFO_PATH).is_file():
        raise FileNotFoundError(f"No LeRobot metadata found under: {source}")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    if output.is_relative_to(source):
        raise ValueError("Output dataset cannot be placed inside the source dataset")

    info = _read_json(source / _INFO_PATH)
    action_names = _load_action_names(info)
    layouts = detect_tcp_action_layouts(action_names, sides=sides)
    frames = _load_recorded_frames(source, len(action_names))
    if not frames:
        raise ValueError("Source dataset contains no frames")

    targets, episode_reports = _build_targets(
        frames,
        layouts,
        degree=degree,
        chunk_size=chunk_size,
        stride=stride,
        max_error=max_error,
        smoothing=smoothing,
        max_knots=max_knots,
    )

    parameter_rows = chunk_size + 2 * degree
    control_names = [name for layout in layouts for name in layout.control_names]
    matrix_shape = list(
        parameter_matrix_shape(layouts, parameter_rows=parameter_rows)
    )
    flattened_action_dim = math.prod(matrix_shape)
    flattened_names = parameter_feature_names(
        layouts,
        parameter_rows=parameter_rows,
    )
    if len(flattened_names) != flattened_action_dim:
        raise AssertionError("Generated action names do not match flattened shape")

    converted_info = json.loads(json.dumps(info))
    action_feature = converted_info["features"]["action"]
    action_feature.update(
        {
            "dtype": "float32",
            "shape": [flattened_action_dim],
            "names": flattened_names,
        }
    )

    metadata = {
        "format_version": 2,
        "source_dataset": str(source),
        "source_action_names": action_names,
        "selected_sides": [layout.side for layout in layouts],
        "control_names": control_names,
        "parameter_channel_names": ["knot", *control_names],
        "gripper_width_sides": [
            layout.side
            for layout in layouts
            if layout.gripper_width_index is not None
        ],
        "rotation_representation": "rotation_6d_rows",
        "degree": degree,
        "chunk_size": chunk_size,
        "stride": stride,
        "max_error": max_error,
        "smoothing": smoothing,
        "max_knots": max_knots,
        "knot_units": "source_frames",
        "parameter_matrix_shape": matrix_shape,
        "flatten_order": "row_major",
        "active_control_rows": parameter_rows - (degree + 1),
        "flattened_action_dim": flattened_action_dim,
        "normalization_mode": "tied_per_semantic_channel",
        "episodes": episode_reports,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.bspline-", dir=output.parent)
    )
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        (staging / _INFO_PATH).write_text(
            json.dumps(converted_info, indent=4) + "\n",
            encoding="utf-8",
        )
        _replace_action_data(staging, targets, converted_info["features"])
        _refresh_action_statistics(
            staging,
            targets,
            action_feature,
            parameter_rows=parameter_rows,
        )
        (staging / _BSPLINE_METADATA_PATH).write_text(
            json.dumps(metadata, indent=4) + "\n",
            encoding="utf-8",
        )
        _validate_output(staging, len(frames), flattened_action_dim)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "source_root": str(source),
        "output_root": str(output),
        "frames": len(frames),
        "episodes": len(episode_reports),
        "sides": [layout.side for layout in layouts],
        "parameter_matrix_shape": matrix_shape,
        "flattened_action_dim": flattened_action_dim,
        "all_tolerances_reached": all(
            report["tolerance_reached"] for report in episode_reports
        ),
        "max_translation_error_m": max(
            report["max_translation_error_m"] for report in episode_reports
        ),
        "max_rotation_error_deg": max(
            report["max_rotation_error_deg"] for report in episode_reports
        ),
    }
