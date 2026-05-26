from __future__ import annotations

import time

import typer

from flexiv_trainer.runtime.manager import get_runtime_manager

app = typer.Typer(add_completion=False)


@app.command()
def run(start_immediately: bool = True) -> None:
    runtime = get_runtime_manager()
    typer.echo(runtime.bootstrap_teleop_module())
    if start_immediately:
        typer.echo(runtime.teleop.start().__dict__)
    typer.echo("Teleoperation CLI is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        typer.echo(runtime.teleop.stop().__dict__)


def main() -> None:
    app()
