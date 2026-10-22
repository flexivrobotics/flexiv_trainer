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

from flexivtrainer.practice.demos import convert_demos
from flexivtrainer.practice.space import read_space

app = typer.Typer(add_completion=False)


@app.command()
def run(
    source: Annotated[Path, typer.Argument(help="Teleop recording (LeRobot).")],
    output: Annotated[Path, typer.Argument(help="New recording, in the space.")],
    space: Annotated[
        Path,
        typer.Option(help="The skill's practice space (practice_space.json)."),
    ],
    check: Annotated[
        str | None,
        typer.Option(help="End each episode at this check's labelled frame."),
    ] = None,
    arm: Annotated[
        str | None,
        typer.Option(help="The arm that does the skill, when there are several."),
    ] = None,
    episode: Annotated[
        list[int] | None,
        typer.Option(help="Convert only this episode; repeat for more."),
    ] = None,
    video_codec: Annotated[
        str, typer.Option(help="Video codec of the new recording.")
    ] = "h264",
) -> None:
    result = convert_demos(
        source,
        output,
        read_space(space),
        check=check,
        arm=arm,
        episodes=episode or None,
        video_codec=video_codec,
    )
    result.pop("episodes")
    typer.echo(json.dumps(result, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
