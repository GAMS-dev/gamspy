from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path  # noqa: TC003
from typing import Annotated

import gamspy_base
import typer

# Valid values for categorical parameters based on help string
VALID_YN = {"0", "N", "1", "Y"}
VALID_DUPLICATES = {"NOCHECK", "ADD", "IGNORE", "ERROR"}
VALID_ORIGNAMES = {"NO", "MODIFIED", "ALL"}
VALID_CONVERTSENSE = {"0", "N", "1", "Y", "MIN", "-1", "MAX"}


# Completion functions for Typer
def complete_yn(ctx: typer.Context, incomplete: str):
    return [v for v in VALID_YN if v.lower().startswith(incomplete.lower())]


def complete_duplicates(ctx: typer.Context, incomplete: str):
    return [v for v in VALID_DUPLICATES if v.lower().startswith(incomplete.lower())]


def complete_orignames(ctx: typer.Context, incomplete: str):
    return [v for v in VALID_ORIGNAMES if v.lower().startswith(incomplete.lower())]


def complete_convertsense(ctx: typer.Context, incomplete: str):
    return [v for v in VALID_CONVERTSENSE if v.lower().startswith(incomplete.lower())]


def mps2gms(
    input_file: Annotated[
        Path, typer.Argument(help="MPS or LP file to translate.", exists=True)
    ],
    gdx_file: Annotated[
        Path | None, typer.Argument(help="Name of GDX output file.")
    ] = None,
    gms_file: Annotated[
        Path | None, typer.Argument(help="Name of GAMS program output file.")
    ] = None,
    py_file: Annotated[
        str | None, typer.Option("--py", help="Name of GAMSPy program output file.")
    ] = None,
    dec_file: Annotated[
        str | None, typer.Option("--dec", help="DEC file for decomposition info.")
    ] = None,
    column_int_vars_binary: Annotated[
        str | None,
        typer.Option(
            "--columnintvarsarebinary",
            help="Integer variables appearing first are binary.",
            autocompletion=complete_yn,
        ),
    ] = None,
    duplicates: Annotated[
        str | None,
        typer.Option(
            help="How to handle multiple coefficients.",
            autocompletion=complete_duplicates,
        ),
    ] = None,
    orignames: Annotated[
        str | None,
        typer.Option(
            help="Whether to make original names available.",
            autocompletion=complete_orignames,
        ),
    ] = None,
    stageshift: Annotated[
        int | None, typer.Option(help="Shift block numbers by this integer.")
    ] = None,
    convertsense: Annotated[
        str | None,
        typer.Option(
            help="Convert the objective function sense.",
            autocompletion=complete_convertsense,
        ),
    ] = None,
):
    """
    Translates an MPS or LP file into equivalent generic GAMS and GAMSPy programs.
    Defaults to writing .py and .gdx files if no outputs are specified.
    """
    binary_name = "mps2gms.exe" if platform.system() == "Windows" else "mps2gms"
    MPS2GMS_PATH = os.path.join(gamspy_base.directory, binary_name)

    if not os.path.exists(MPS2GMS_PATH):
        typer.echo(f"Binary not found: {MPS2GMS_PATH}", err=True)
        raise typer.Exit(code=1)

    # Determine default names based on input stem
    input_name = input_file.with_suffix("")
    actual_gdx = gdx_file if gdx_file else f"{input_name}.gdx"
    actual_py = py_file if py_file else f"{input_name}.py"

    # mps2gms <input> <gdx>
    cmd = [MPS2GMS_PATH, str(input_file), str(actual_gdx)]

    # If gms_file is provided, add it as the third positional;
    # otherwise, explicitly disable GMS output to prioritize py/gdx.
    if gms_file:
        cmd.append(str(gms_file))
    else:
        cmd.append("GMS=")

    # Construct key=value parameters for the binary call
    params = {
        "PY": actual_py,
        "DEC": dec_file,
        "COLUMNINTVARSAREBINARY": column_int_vars_binary,
        "DUPLICATES": duplicates,
        "ORIGNAMES": orignames,
        "STAGESHIFT": stageshift,
        "CONVERTSENSE": convertsense,
    }

    for key, value in params.items():
        if value is not None:
            cmd.append(f"{key}={value}")

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    raise typer.Exit(code=result.returncode)
