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
