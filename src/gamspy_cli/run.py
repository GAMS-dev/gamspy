from __future__ import annotations

import os
import platform
import subprocess
import sys
from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import Annotated, Optional

import typer

from .util import cli_error

app = typer.Typer(
    short_help="To run your model with GAMS MIRO.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy run miro [--path <path_to_miro>] [--model <path_to_model>]",
    context_settings={"help_option_names": ["-h", "--help"]},
)


class ModeEnum(StrEnum):
    config = "config"
    base = "base"
    deploy = "deploy"


def run_miro(
    model: str | os.PathLike[str],
    *,
    mode: ModeEnum | str = ModeEnum.base,
    path: str | os.PathLike[str] | None = None,
    skip_execution: bool = False,
    args: list[str] | None = None,
) -> None:
    """
    Runs a GAMSPy model with GAMS MIRO. Equivalent of ``gamspy run miro``.

    Parameters
    ----------
    model : str | os.PathLike
        Path to the GAMSPy model.
    mode : str, optional
        Execution mode of MIRO: "base", "config" or "deploy".
    path : str | os.PathLike, optional
        Path to the MIRO executable (.exe, .app, or .AppImage). By default, the
        MIRO_PATH environment variable or the standard installation locations
        are used.
    skip_execution : bool, optional
        Whether to skip model execution.
    args : list[str], optional
        Arguments to the GAMSPy model.

    Raises
    ------
    CLIError
        In case MIRO cannot be found or the model fails.

    Examples
    --------
    >>> from gamspy import cli
    >>> cli.run_miro("model.py", mode="config")  # doctest: +SKIP

    """
    model_path = os.path.abspath(model)
    execution_mode = ModeEnum(mode).value
    miro_path = os.getenv("MIRO_PATH", path)

    if miro_path is None:
        miro_path = miro_path if miro_path is not None else discover_miro()

    if miro_path is None:
        raise cli_error("--path must be provided to run MIRO")

    if platform.system() == "Darwin" and os.path.splitext(miro_path)[1] == ".app":
        miro_path = os.path.join(miro_path, "Contents", "MacOS", "GAMS MIRO")

    # Initialize MIRO
    if not skip_execution:
        subprocess_env = os.environ.copy()
        subprocess_env["MIRO"] = "1"
        command = [sys.executable, model_path]
        if args is not None:
            command += args

        process = subprocess.run(
            command,
            env=subprocess_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if process.returncode != 0:
            raise cli_error(process.stderr, process.returncode)

    # Run MIRO
    subprocess_env = os.environ.copy()
    if execution_mode == "deploy":
        subprocess_env["MIRO_BUILD"] = "true"
        execution_mode = "base"

    subprocess_env["MIRO_MODEL_PATH"] = model_path
    subprocess_env["MIRO_MODE"] = execution_mode
    subprocess_env["MIRO_DEV_MODE"] = "true"
    subprocess_env["MIRO_USE_TMP"] = "false"
    subprocess_env["PYTHON_EXEC_PATH"] = sys.executable

    subprocess.run([miro_path], env=subprocess_env, check=True)


@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy run miro [--path <path_to_miro>] [--model <path_to_model>]",
    short_help="Runs a GAMSPY model with GAMS MIRO app.",
)
def miro(
    mode: Annotated[
        ModeEnum,
        typer.Option("--mode", "-m", help="Execution mode of MIRO"),
    ] = ModeEnum.base,
    path: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(
            "--path",
            "-p",
            exists=True,
            help="Path to the MIRO executable (.exe, .app, or .AppImage)",
        ),
    ] = None,
    skip_execution: Annotated[
        bool,
        typer.Option("--skip-execution", help="Whether to skip model execution."),
    ] = False,
    model: Annotated[
        Optional[Path],  # noqa: UP045,
        typer.Option("--model", "-g", exists=True, help="Path to the GAMSPy model."),
    ] = None,
    args: Annotated[
        list[str] | None,
        typer.Argument(help="Arguments to the GAMSPy model."),
    ] = None,
) -> None:
    if model is None:
        typer.echo("--model must be provided to run MIRO", file=sys.stderr)
        sys.exit(1)

    run_miro(model, mode=mode, path=path, skip_execution=skip_execution, args=args)


def discover_miro() -> str | None:
    system = platform.system()
    if system == "Linux":
        return None

    home = os.path.expanduser("~")
    standard_locations = {
        "Darwin": [
            os.path.join(
                "/",
                "Applications",
                "GAMS MIRO.app",
                "Contents",
                "MacOS",
                "GAMS MIRO",
            ),
            os.path.join(
                home,
                "Applications",
                "GAMS MIRO.app",
                "Contents",
                "MacOS",
                "GAMS MIRO",
            ),
        ],
        "Windows": [
            os.path.join("C:\\", "Program Files", "GAMS MIRO", "GAMS MIRO.exe"),
            os.path.join(
                home,
                "AppData",
                "Local",
                "Programs",
                "GAMS MIRO",
                "GAMS MIRO.exe",
            ),
        ],
    }

    if system in ("Darwin", "Windows"):
        for location in standard_locations[system]:
            if os.path.isfile(location):
                return location

    return None


if __name__ == "__main__":
    app()
