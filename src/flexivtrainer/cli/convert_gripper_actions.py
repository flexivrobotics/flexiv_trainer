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

from flexivtrainer.jobs.convert_gripper_actions import (
    convert_legacy_gripper_actions,
)

app = typer.Typer(add_completion=False)


@app.command()
def run(
    source: Annotated[Path, typer.Argument(help="Source LeRobot dataset.")],
    output: Annotated[Path, typer.Argument(help="New converted dataset.")],
    open_width_m: Annotated[
        float,
        typer.Option(min=0.0, help="Target width written for inferred Open states."),
    ],
    close_width_m: Annotated[
        float,
        typer.Option(min=0.0, help="Target width written for inferred Close states."),
    ],
    velocity_m_s: Annotated[
        float,
        typer.Option(min=1e-9, help="Gripper velocity bound to the dataset."),
    ],
    force_limit_n: Annotated[
        float,
        typer.Option(min=1e-9, help="Gripper force limit bound to the dataset."),
    ],
    motion_threshold_m: Annotated[
        float,
        typer.Option(min=1e-9, help="Cumulative width movement for a transition."),
    ] = 0.0002,
    force_threshold_n: Annotated[
        float,
        typer.Option(min=0.0, help="Force used to infer each initial state."),
    ] = 5.0,
    initial_state_manifest: Annotated[
        Path | None,
        typer.Option(help="JSON mapping episode indices and sides to open/close."),
    ] = None,
) -> None:
    result = convert_legacy_gripper_actions(
        source,
        output,
        open_width_m=open_width_m,
        close_width_m=close_width_m,
        velocity_m_s=velocity_m_s,
        force_limit_n=force_limit_n,
        motion_threshold_m=motion_threshold_m,
        force_threshold_n=force_threshold_n,
        initial_state_manifest=initial_state_manifest,
    )
    typer.echo(json.dumps(result, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
