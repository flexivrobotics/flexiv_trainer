from __future__ import annotations

import typer

from flexiv_trainer.runtime.manager import get_runtime_manager

app = typer.Typer(add_completion=False)


@app.command()
def run(episode_paths: list[str], output_name: str) -> None:
    runtime = get_runtime_manager()
    typer.echo(runtime.combine_episodes(episode_paths, output_name))


def main() -> None:
    app()
