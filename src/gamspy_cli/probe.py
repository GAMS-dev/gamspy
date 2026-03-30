from __future__ import annotations

import os
import subprocess

import typer

app = typer.Typer(
    short_help="To probe a node's information.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy probe -j <output_path.json>",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _probe(json_out: str | None) -> None:
    import gamspy_base

    process = subprocess.run(
        [os.path.join(gamspy_base.directory, "gamsprobe")],
        text=True,
        capture_output=True,
    )

    if process.returncode:
        typer.echo(process.stderr)
        raise typer.Exit(code=1)

    print(process.stdout)

    if json_out:
        with open(json_out, "w") as file:
            file.write(process.stdout)

    raise typer.Exit()


@app.callback()
def callback(
    json_out: str | None = typer.Option(
        None,
        "--json-out",
        "-j",
        help="Output path for the JSON file.",
        callback=_probe,
    ),
) -> None: ...


if __name__ == "__main__":
    app()
