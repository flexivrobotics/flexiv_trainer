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
from pathlib import Path

import typer

from flexivtrainer.runtime.manager import get_runtime_manager

app = typer.Typer(add_completion=False)


@app.command()
def run(
    dataset_path: str,
    output_dir: str,
    policy_type: str = "diffusion",
) -> None:
    runtime = get_runtime_manager()
    typer.echo(
        runtime.training.start(
            dataset_root=Path(dataset_path).resolve(),
            output_dir=Path(output_dir).resolve(),
            policy_type=policy_type,
        )
    )
    while True:
        snapshot = runtime.training.status()
        typer.echo(snapshot)
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(2)


def main() -> None:
    app()
