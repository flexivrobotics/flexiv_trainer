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

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from flexivtrainer.jobs.convert_bspline_dataset import (
    convert_lerobot_tcp_actions_to_bspline,
)

app = typer.Typer(add_completion=False)


@app.command()
def run(
    source: Annotated[
        Path,
        typer.Argument(help="Source LeRobot v3 dataset directory."),
    ],
    output: Annotated[
        Path,
        typer.Argument(help="New LeRobot dataset directory to create."),
    ],
    side: Annotated[
        list[str] | None,
        typer.Option(
            "--side",
            help=(
                "Arm side to include; repeat for multiple arms. "
                "Auto-detected by default."
            ),
        ),
    ] = None,
    degree: Annotated[
        int,
        typer.Option(min=1, help="B-spline polynomial degree."),
    ] = 3,
    chunk_size: Annotated[
        int,
        typer.Option(min=2, help="Unique-knot segment size before boundary support."),
    ] = 10,
    stride: Annotated[
        int,
        typer.Option(min=1, help="Stride between parameter segments in knot space."),
    ] = 1,
    max_error: Annotated[
        float,
        typer.Option(min=1e-12, help="Maximum component-wise fitting error."),
    ] = 0.002,
    smoothing: Annotated[
        float,
        typer.Option(min=0.0, help="SciPy knot-generator smoothing factor."),
    ] = 1e-12,
    max_knots: Annotated[
        int | None,
        typer.Option(
            min=2,
            help="Optional cap on fitted knot count; omit for interpolation fallback.",
        ),
    ] = None,
) -> None:
    """Replace copied TCP actions with trainable B-spline parameters."""

    result = convert_lerobot_tcp_actions_to_bspline(
        source,
        output,
        sides=side,
        degree=degree,
        chunk_size=chunk_size,
        stride=stride,
        max_error=max_error,
        smoothing=smoothing,
        max_knots=max_knots,
    )
    typer.echo(json.dumps(result, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
