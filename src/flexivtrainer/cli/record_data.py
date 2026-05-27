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

import time

import typer

from flexivtrainer.runtime.manager import get_runtime_manager

app = typer.Typer(add_completion=False)


@app.command()
def run(
    task: str = "Dual-arm Flexiv teleoperation demonstration",
    duration_s: int = 10,
    fps: int = 30,
    save: bool = True,
) -> None:
    runtime = get_runtime_manager()
    runtime.bootstrap_teleop_module()
    typer.echo(runtime.recording.start(task=task, fps=fps))
    time.sleep(max(duration_s, 1))
    typer.echo(runtime.recording.stop())
    typer.echo(runtime.recording.save() if save else runtime.recording.discard())


def main() -> None:
    app()
