from __future__ import annotations

import os
import subprocess
from typing import Annotated, Union

import typer

import gamspy.utils as utils
from gamspy.exceptions import ValidationError

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To retrieve a license with another node's information.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy retrieve license <access_code> [--input <input_path>.json] [--output <output_path>.json]",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy retrieve license <access_code> [--input <input_path>.json] [--output <output_path>.json]"
)
def license(
    access_code: Annotated[
        str,
        typer.Argument("--access-code", help="Access code of the license."),
    ],
    input: Annotated[
        Union[str, None],
        typer.Option(
            "--input",
            "-i",
            help="Input json file path to retrieve the license based on the node information.",
        ),
    ] = None,
    output: Annotated[
        Union[str, None],
        typer.Option(
            "--output", "-o", help="Output path for the license file."
        ),
    ] = None,
) -> None:
    if input is None or not os.path.isfile(input):
        raise ValidationError(
            f"Given path `{input}` is not a json file. Please use `gamspy retrieve license <access_code> -i <json_file_path>`"
        )

    if access_code is None:
        raise ValidationError(
            f"Given licence id `{access_code}` is not valid!"
        )

    gamspy_base_dir = utils._get_gamspy_base_directory()
    process = subprocess.run(
        [
            os.path.join(gamspy_base_dir, "gamsgetkey"),
            access_code,
            "-i",
            input,
        ],
        text=True,
        capture_output=True,
    )

    if process.returncode:
        raise ValidationError(process.stderr)

    print(process.stdout)
    if output:
        with open(output, "w") as file:
            file.write(process.stdout)


if __name__ == "__main__":
    app()
