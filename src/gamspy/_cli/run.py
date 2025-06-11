from __future__ import annotations
import os
import platform
import subprocess
import sys
from enum import Enum
from typing import List

import typer

from gamspy.exceptions import GamspyException, ValidationError

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To run your model with GAMS MIRO.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy run miro [--path <path_to_miro>] [--model <path_to_model>]",
    context_settings={"help_option_names": ["-h", "--help"]},
)

class ModeEnum(Enum):
    config = "config"
    base = "base"
    deploy = "deploy"

@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy run miro [--path <path_to_miro>] [--model <path_to_model>]", 
    short_help="Runs a GAMSPY model with GAMS MIRO app."
)
def miro(
    mode: ModeEnum = typer.Option(
        "base",
        "--mode",
        "-m",
        help="Execution mode of MIRO"
    ),
    path: str = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to the MIRO executable (.exe on Windows, .app on macOS or .AppImage on Linux"
    ),
    skip_execution: bool = typer.Option(
        False,
        "--skip-execution",
        help="Whether to skip model execution."
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-g",
        help="Path to the GAMSPy model."
    ),
    args: List[str] = typer.Argument(
        None,
        help="Arguments to the GAMSPy model."
    ),
) -> None:
    if model is None:
        raise ValidationError("--model must be provided to run MIRO")
    
    model = os.path.abspath(model)
    execution_mode = mode.value
    path = os.getenv("MIRO_PATH", None)

    if path is None:
        path = path if path is not None else discover_miro()

    if path is None:
        raise GamspyException(
            "--path must be provided to run MIRO"
        )

    if (
        platform.system() == "Darwin"
        and os.path.splitext(path)[1] == ".app"
    ):
        path = os.path.join(path, "Contents", "MacOS", "GAMS MIRO")

    # Initialize MIRO
    if not skip_execution:
        subprocess_env = os.environ.copy()
        subprocess_env["MIRO"] = "1"
        command = [sys.executable, model]
        if args is not None:
            command += args

        try:
            subprocess.run(command, env=subprocess_env, check=True)
        except subprocess.CalledProcessError:
            return

    # Run MIRO
    subprocess_env = os.environ.copy()
    if execution_mode == "deploy":
        subprocess_env["MIRO_BUILD"] = "true"
        execution_mode = "base"

    subprocess_env["MIRO_MODEL_PATH"] = model
    subprocess_env["MIRO_MODE"] = execution_mode
    subprocess_env["MIRO_DEV_MODE"] = "true"
    subprocess_env["MIRO_USE_TMP"] = "false"
    subprocess_env["PYTHON_EXEC_PATH"] = sys.executable

    subprocess.run([path], env=subprocess_env, check=True)

def discover_miro():
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
            os.path.join(
                "C:\\", "Program Files", "GAMS MIRO", "GAMS MIRO.exe"
            ),
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
