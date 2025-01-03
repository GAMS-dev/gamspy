from __future__ import annotations

import os
import subprocess
from typing import Annotated, Union

import typer

import gamspy.utils as utils
from gamspy.exceptions import ValidationError


def probe(
    json_out: Annotated[
        Union[str, None], typer.Option(help="Output path for the json file.")
    ] = None,
):
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


app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To probe node information.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy probe --json-out <output_path>.json",
    context_settings={"help_option_names": ["-h", "--help"]},
    callback=probe,
    invoke_without_command=True,
)

if __name__ == "__main__":
    app()
