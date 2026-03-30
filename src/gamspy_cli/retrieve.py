from __future__ import annotations

import os
import subprocess

import certifi
import typer

app = typer.Typer(
    short_help="To retrieve a license with another node's information.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy retrieve license <access_code> [--input <input_path>.json] [--output <output_path>.json]",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command(
    short_help="Retrives the license with the given node information.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy retrieve license <access_code> [--input <input_path>.json] [--output <output_path>.json]",
)
def license(
    access_code: str = typer.Argument(..., help="Access code of the license."),
    input: str = typer.Option(
        None,
        "--input",
        "-i",
        help="Input json file path to retrieve the license based on the node information.",
    ),
    output: str = typer.Option(
        None, "--output", "-o", help="Output path for the license file."
    ),
    checkout_duration: int | None = typer.Option(
        None,
        "--checkout-duration",
        "-c",
        help="Specifies a duration in hours to checkout a session.",
    ),
) -> None:
    import gamspy_base

    if input is None or not os.path.isfile(input):
        typer.echo(
            f"Given path `{input}` is not a json file. Please use `gamspy retrieve license <access_code> -i <json_file_path>`"
        )
        raise typer.Exit(code=1)

    if access_code is None:
        typer.echo(f"Given licence id `{access_code}` is not valid!")
        raise typer.Exit(code=1)

    environment_variables = os.environ.copy()
    if "CURL_CA_BUNDLE" not in environment_variables:
        environment_variables["CURL_CA_BUNDLE"] = certifi.where()

    command = [
        os.path.join(gamspy_base.directory, "gamsgetkey"),
        access_code,
        "-i",
        input,
    ]
    if checkout_duration:
        command.append("-c")
        command.append(str(checkout_duration))

    process = subprocess.run(
        command, text=True, capture_output=True, env=environment_variables
    )

    if process.returncode:
        typer.echo(process.stderr)
        raise typer.Exit(code=1)

    print(process.stdout)
    if output:
        with open(output, "w") as file:
            file.write(process.stdout)


if __name__ == "__main__":
    app()
