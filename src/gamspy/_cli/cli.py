from __future__ import annotations

import os
import subprocess
from typing import Optional

import typer

from gamspy.exceptions import ValidationError
import gamspy.utils as utils

from . import install, list, retrieve, run, show, uninstall

app = typer.Typer(
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(install.app, name="install")
app.add_typer(list.app, name="list")
app.add_typer(retrieve.app, name="retrieve")
app.add_typer(run.app, name="run")
app.add_typer(show.app, name="show")
app.add_typer(uninstall.app, name="uninstall")


def version_callback(value: bool):
    if value:
        import gams

        import gamspy

        print(f"GAMSPy version: {gamspy.__version__}")
        print(f"GAMS version: {gams.__version__}")

        try:
            import gamspy_base

            print(f"gamspy_base version: {gamspy_base.__version__}")
        except ModuleNotFoundError:
            ...

        raise typer.Exit()


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None,
        "-v",
        "--version",
        help="Shows the version of gamspy, gamsapi, and gamspy_base.",
        callback=version_callback,
    ),
) -> None:
    """
    GAMSPy CLI - The [bold]gamspy[/bold] command line app. 😎

    Install solvers and licenses, run MIRO apps, and more.

    Read more in the docs: [link=https://gamspy.readthedocs.io/en/latest/cli/index.html]https://gamspy.readthedocs.io/en/latest/cli/index.html[/link].
    """
    ...

@app.command(short_help="To probe node information.")
def probe(
    json_out: Optional[str] = typer.Option(
        None,
        "--json-out", "-j",
        help="Output path for the JSON file."
    ),
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

def main():
    """
    Entry point for gamspy command line application.
    """
    app()
