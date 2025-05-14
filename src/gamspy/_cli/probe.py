from __future__ import annotations

import os
import subprocess
from typing import Optional

import typer

from gamspy.exceptions import ValidationError
import gamspy.utils as utils

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To probe a node's information.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy probe -j <output_path.json>",
    context_settings={"help_option_names": ["-h", "--help"]},
)

def _probe(json_out: Optional[str]) -> None:
    gamspy_base_dir = utils._get_gamspy_base_directory()
    process = subprocess.run(
        [os.path.join(gamspy_base_dir, "gamsprobe")],
        text=True,
        capture_output=True,
    )

    if process.returncode:
        raise ValidationError(process.stderr)

    print(process.stdout)

    if json_out:
        with open(json_out, "w") as file:
            file.write(process.stdout)

    raise typer.Exit()

@app.callback()
def callback(
    json_out: Optional[str] = typer.Option(
        None,
        "--json-out", "-j",
        help="Output path for the JSON file.",
        callback=_probe
    ),
) -> None:
    ...

if __name__ == "__main__":
    app()
