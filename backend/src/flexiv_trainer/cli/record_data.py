from __future__ import annotations

import time

import typer

from flexiv_trainer.runtime.manager import get_runtime_manager

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
