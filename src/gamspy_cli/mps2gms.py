from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from .util import cli_error

# Valid values for categorical parameters based on help string
VALID_YN = {"0", "N", "1", "Y"}
VALID_DUPLICATES = {"NOCHECK", "ADD", "IGNORE", "ERROR"}
VALID_ORIGNAMES = {"NO", "MODIFIED", "ALL"}
VALID_CONVERTSENSE = {"0", "N", "1", "Y", "MIN", "-1", "MAX"}


# Completion functions for Typer
def complete_yn(ctx: typer.Context, incomplete: str) -> list[str]:
    return [v for v in VALID_YN if v.lower().startswith(incomplete.lower())]


def complete_duplicates(ctx: typer.Context, incomplete: str) -> list[str]:
    return [v for v in VALID_DUPLICATES if v.lower().startswith(incomplete.lower())]


def complete_orignames(ctx: typer.Context, incomplete: str) -> list[str]:
    return [v for v in VALID_ORIGNAMES if v.lower().startswith(incomplete.lower())]


def complete_convertsense(ctx: typer.Context, incomplete: str) -> list[str]:
    return [v for v in VALID_CONVERTSENSE if v.lower().startswith(incomplete.lower())]


def mps2gms(
    input_file: str | os.PathLike[str],
    *,
    gdx_file: str | None = None,
    gms_file: str | None = None,
    py_file: str | None = None,
    dec_file: str | None = None,
    column_int_vars_binary: str | None = None,
    duplicates: str | None = None,
    orignames: str | None = None,
    stageshift: int | None = None,
    compress_gdx: int | None = None,
    convertsense: str | None = None,
) -> None:
    """
    Translates an MPS or LP file into equivalent generic GAMS and GAMSPy programs.
    Defaults to writing .py and .gdx files next to the input file if no outputs
    are specified. Equivalent of ``gamspy mps2gms``. Run ``gamspy mps2gms --help``
    for the description of the options.

    Parameters
    ----------
    input_file : str | os.PathLike
        MPS or LP file to translate.
    gdx_file : str, optional
        Name of GDX output file.
    gms_file : str, optional
        Name of GAMS program output file.
    py_file : str, optional
        Name of GAMSPy program output file.

    Raises
    ------
    CLIError
        In case the input file does not exist or the translation fails.

    Examples
    --------
    >>> from gamspy import cli
    >>> cli.mps2gms("model.mps", py_file="model.py")  # doctest: +SKIP

    """
    import gamspy_base

    binary_name = "mps2gms.exe" if platform.system() == "Windows" else "mps2gms"
    MPS2GMS_PATH = os.path.join(gamspy_base.directory, binary_name)

    if not os.path.exists(MPS2GMS_PATH):
        raise cli_error(f"Binary not found: {MPS2GMS_PATH}")

    input_file = Path(input_file)
    if not input_file.is_file():
        raise cli_error(f"File not found: {input_file}")

    # Determine default names based on input stem
    stem = input_file.name.removesuffix("".join(input_file.suffixes))
    stem_path = os.path.join(input_file.parent.absolute(), stem)
    actual_gdx = gdx_file if gdx_file else f"{stem_path}.gdx"
    actual_py = py_file if py_file else f"{stem_path}.py"

    # mps2gms <input> <gdx>
    cmd = [MPS2GMS_PATH, str(input_file), str(actual_gdx)]

    # Whether to compress output gdx
    env = os.environ.copy()
    if compress_gdx is not None:
        env["GDXCOMPRESS"] = "1" if compress_gdx else "0"

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
    result = subprocess.run(cmd, text=True, env=env)
    if result.returncode:
        raise cli_error(
            f"mps2gms failed with return code {result.returncode}.", result.returncode
        )


def mps2gms_command(
    input_file: Annotated[
        Path, typer.Argument(help="MPS or LP file to translate.", exists=True)
    ],
    gdx_file: Annotated[
        str | None, typer.Option("--gdx", help="Name of GDX output file.")
    ] = None,
    gms_file: Annotated[
        str | None, typer.Option("--gms", help="Name of GAMS program output file.")
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
    compress_gdx: Annotated[
        int | None, typer.Option(help="Whether to compress output gdx.")
    ] = None,
    convertsense: Annotated[
        str | None,
        typer.Option(
            help="Convert the objective function sense.",
            autocompletion=complete_convertsense,
        ),
    ] = None,
) -> None:
    """
    Translates an MPS or LP file into equivalent generic GAMS and GAMSPy programs.
    Defaults to writing .py and .gdx files if no outputs are specified.
    """
    mps2gms(
        input_file,
        gdx_file=gdx_file,
        gms_file=gms_file,
        py_file=py_file,
        dec_file=dec_file,
        column_int_vars_binary=column_int_vars_binary,
        duplicates=duplicates,
        orignames=orignames,
        stageshift=stageshift,
        compress_gdx=compress_gdx,
        convertsense=convertsense,
    )
