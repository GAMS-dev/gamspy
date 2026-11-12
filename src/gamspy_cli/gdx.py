"""
This module provides a wrapper around GDX tools of GAMS to expose them in GAMSPy CLI.
Run `gamspy gdx -h` to see available gdx tools in GAMSPy CLI.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

import typer

from .util import cli_error

if TYPE_CHECKING:
    from collections.abc import Iterable

app = typer.Typer(
    short_help="To dump and compare GDX files.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy gdx dump gdxfile.gdx | gamspy gdx diff gdxfile.gdx gdxfile2.gdx",
    context_settings={"help_option_names": ["-h", "--help"]},
)

VALID_DELIMS = {"period", "comma", "tab", "blank", "semicolon"}
VALID_DECIMAL_SEP = {"period", "comma"}
VALID_FORMATS = {"normal", "gamsbas", "csv"}
VALID_DFORMATS = {"normal", "hexponential", "hexBytes"}
VALID_YN = {"Y", "N"}
VALID_FIELDS = {"L", "M", "Up", "Lo", "Prior", "Scale", "All"}
VALID_SETDESC = {"Y", "N"}


def _option_args(options: list[tuple[str, str | float | bool | None]]) -> list[str]:
    args: list[str] = []
    for name, value in options:
        if value is True:
            args.append(name)
        elif value is not False and value is not None and value != "":
            args.append(f"{name}={value}")

    return args


def _get_gdx_symbol_names(filename: str) -> set[str]:
    import gams.core.gdx as gdx
    import gamspy_base

    import gamspy._gdx as gdxio

    symbol_names = set()
    with gdxio.open_gdx(gamspy_base.directory, filename) as handle:
        _, number_of_symbols, _ = gdx.gdxSystemInfo(handle)

        # Symbol 0 is the universe, hence the symbols are numbered from 1.
        for symbol_number in range(1, number_of_symbols + 1):
            _, symbol_name, _, _ = gdx.gdxSymbolInfo(handle, symbol_number)
            symbol_names.add(symbol_name)

    return symbol_names


def complete_delim(ctx: typer.Context, incomplete: str) -> list[str]:
    return [d for d in VALID_DELIMS if d.startswith(incomplete)]


def complete_decimalsep(ctx: typer.Context, incomplete: str) -> list[str]:
    return [s for s in VALID_DECIMAL_SEP if s.startswith(incomplete)]


def complete_format(ctx: typer.Context, incomplete: str) -> list[str]:
    return [f for f in VALID_FORMATS if f.startswith(incomplete)]


def complete_dformat(ctx: typer.Context, incomplete: str) -> list[str]:
    return [f for f in VALID_DFORMATS if f.startswith(incomplete)]


def complete_yes_no(ctx: typer.Context, incomplete: str) -> list[str]:
    return [v for v in VALID_YN if v.startswith(incomplete)]


def complete_field(ctx: typer.Context, incomplete: str) -> list[str]:
    return [f for f in VALID_FIELDS if f.startswith(incomplete)]


def gdx_dump(
    filename: str,
    *,
    version: bool = False,
    output: str | None = None,
    symb: str | None = None,
    ueltable: str | None = None,
    delim: str | None = None,
    decimalsep: str | None = None,
    noheader: bool = False,
    nodata: bool = False,
    csvallfields: bool = False,
    csvsettext: bool = False,
    symbols: bool = False,
    domaininfo: bool = False,
    symbolsasset: bool = False,
    symbolsassetdi: bool = False,
    settext: bool = False,
    format: str | None = None,
    dformat: str | None = None,
    cdim: str | None = None,
    filterdef: str | None = None,
    epsout: str | None = None,
    naout: str | None = None,
    pinfout: str | None = None,
    minfout: str | None = None,
    undfout: str | None = None,
    zeroout: str | None = None,
    header: str | None = None,
) -> str:
    """
    Dumps the content of a GDX file in ASCII format. Equivalent of ``gamspy gdx dump``.
    Run ``gamspy gdx dump --help`` for the description of the options.

    Parameters
    ----------
    filename : str
        Input GDX filename.

    Returns
    -------
    str
        The dump of the GDX file. Empty if ``output`` is given.

    Raises
    ------
    CLIError
        In case the file does not exist, an option is invalid or gdxdump fails.

    Examples
    --------
    >>> import gamspy as gp
    >>> from gamspy import cli
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=["i1", "i2"])
    >>> m.write("data.gdx")
    >>> print(cli.gdx_dump("data.gdx", symb="i", format="csv"), end="")
    "Dim1"
    "i1"
    "i2"

    """
    import gamspy_base

    GDXDUMP_PATH = os.path.join(gamspy_base.directory, "gdxdump")
    if platform.system() == "Windows":
        GDXDUMP_PATH = f"{GDXDUMP_PATH}.exe"

    if not filename.endswith(".gdx"):
        filename += ".gdx"

    if not os.path.exists(filename):
        raise cli_error(f"File not found: {filename}")

    # Validation
    for name, value, valid in (
        ("Delim", delim, VALID_DELIMS),
        ("DecimalSep", decimalsep, VALID_DECIMAL_SEP),
        ("Format", format, VALID_FORMATS),
        ("dFormat", dformat, VALID_DFORMATS),
    ):
        if value and value not in valid:
            raise cli_error(
                f"Invalid {name}: {value}. Must be one of: {', '.join(valid)}"
            )
    for name, value in (("CDim", cdim), ("FilterDef", filterdef)):
        if value and value not in VALID_YN:
            raise cli_error(f"Invalid {name}: {value}. Must be Y or N")

    cmd = [GDXDUMP_PATH, filename]
    cmd += _option_args(
        [
            ("-V", version),
            ("Output", output),
            ("Symb", symb),
            ("UelTable", ueltable),
            ("Delim", delim),
            ("DecimalSep", decimalsep),
            ("NoHeader", noheader),
            ("NoData", nodata),
            ("CSVAllFields", csvallfields),
            ("CSVSetText", csvsettext),
            ("Symbols", symbols),
            ("DomainInfo", domaininfo),
            ("SymbolsAsSet", symbolsasset),
            ("SymbolsAsSetDI", symbolsassetdi),
            ("SetText", settext),
            ("Format", format),
            ("dFormat", dformat),
            ("CDim", cdim),
            ("FilterDef", filterdef),
            ("EpsOut", epsout),
            ("NaOut", naout),
            ("PinfOut", pinfout),
            ("MinfOut", minfout),
            ("UndfOut", undfout),
            ("ZeroOut", zeroout),
            ("Header", header),
        ]
    )

    process = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if process.returncode:
        raise cli_error(process.stdout + process.stderr, process.returncode)

    return process.stdout


@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy gdx dump gdxfile.gdx",
    short_help="To dump the content of a GDX file in ASCII format.",
)
def dump(
    filename: str = typer.Argument(..., help="Input GDX filename"),
    version: bool = typer.Option(
        False, "--version", "-v", help="Write version info of input file only"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write output to file"
    ),
    symb: str | None = typer.Option(
        None, "--symb", "-s", help="Select a single identifier"
    ),
    ueltable: str | None = typer.Option(
        None, "--ueltable", "-u", help="Include all unique elements"
    ),
    delim: str | None = typer.Option(
        None,
        "--delim",
        "-d",
        help="Dimension delimiter",
        autocompletion=complete_delim,
    ),
    decimalsep: str | None = typer.Option(
        None,
        "--decimalsep",
        "-p",
        help="Decimal separator",
        autocompletion=complete_decimalsep,
    ),
    noheader: bool = typer.Option(
        False, "--noheader", help="Suppress writing of headers"
    ),
    nodata: bool = typer.Option(False, "--nodata", help="Write headers only; no data"),
    csvallfields: bool = typer.Option(
        False,
        "--csvallfields",
        help="Write all variable/equation fields in CSV",
    ),
    csvsettext: bool = typer.Option(
        False, "--csvsettext", help="Write set element text in CSV"
    ),
    symbols: bool = typer.Option(
        False, "--symbols", "-S", help="Get list of all symbols"
    ),
    domaininfo: bool = typer.Option(
        False, "--domaininfo", help="Show domain information"
    ),
    symbolsasset: bool = typer.Option(False, "--symbolsasset", help="Symbols as set"),
    symbolsassetdi: bool = typer.Option(
        False, "--symbolsassetdi", help="Symbols as set incl. domain info"
    ),
    settext: bool = typer.Option(False, "--settext", help="Show set text"),
    format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format: normal, gamsbas, csv",
        autocompletion=complete_format,
    ),
    dformat: str | None = typer.Option(
        None,
        "--dformat",
        "-F",
        help="Data format: normal, hexponential, hexBytes",
        autocompletion=complete_dformat,
    ),
    cdim: str | None = typer.Option(
        None,
        "--cdim",
        help="Use last dim as column headers (Y/N)",
        autocompletion=complete_yes_no,
    ),
    filterdef: str | None = typer.Option(
        None,
        "--filterdef",
        "-x",
        help="Filter default values (Y/N)",
        autocompletion=complete_yes_no,
    ),
    epsout: str | None = typer.Option(None, "--epsout", help="String for EPS"),
    naout: str | None = typer.Option(None, "--naout", help="String for NA"),
    pinfout: str | None = typer.Option(None, "--pinfout", help="String for +Inf"),
    minfout: str | None = typer.Option(None, "--minfout", help="String for -Inf"),
    undfout: str | None = typer.Option(None, "--undfout", help="String for Undefined"),
    zeroout: str | None = typer.Option(None, "--zeroout", help="String for Zero"),
    header: str | None = typer.Option(
        None, "--header", help="New header for CSV output"
    ),
) -> None:
    content = gdx_dump(
        filename,
        version=version,
        output=output,
        symb=symb,
        ueltable=ueltable,
        delim=delim,
        decimalsep=decimalsep,
        noheader=noheader,
        nodata=nodata,
        csvallfields=csvallfields,
        csvsettext=csvsettext,
        symbols=symbols,
        domaininfo=domaininfo,
        symbolsasset=symbolsasset,
        symbolsassetdi=symbolsassetdi,
        settext=settext,
        format=format,
        dformat=dformat,
        cdim=cdim,
        filterdef=filterdef,
        epsout=epsout,
        naout=naout,
        pinfout=pinfout,
        minfout=minfout,
        undfout=undfout,
        zeroout=zeroout,
        header=header,
    )
    typer.echo(content, nl=False)


@dataclass(frozen=True)
class GdxDiffResult:
    """
    Result of :func:`gdx_diff`.

    Attributes
    ----------
    has_differences : bool
        Whether the GDX files differ.
    output : str
        Output of gdxdiff that summarizes the differences.
    """

    has_differences: bool
    output: str


def gdx_diff(
    file1: str,
    file2: str,
    diffile: str | None = None,
    *,
    eps: float | None = None,
    releps: float | None = None,
    field: str | None = None,
    fldonly: bool = False,
    diffonly: bool = False,
    cmpdefaults: bool = False,
    cmpdomains: bool = False,
    matrixfile: bool = False,
    ignoreorder: bool = False,
    setdesc: str | None = None,
    id: Iterable[str] | None = None,
    skipid: Iterable[str] | None = None,
    skip_regex: str | None = None,
) -> GdxDiffResult:
    """
    Compares two GDX files. Equivalent of ``gamspy gdx diff``. Run
    ``gamspy gdx diff --help`` for the description of the options.

    Parameters
    ----------
    file1 : str
        First input GDX file.
    file2 : str
        Second input GDX file.
    diffile : str, optional
        Output GDX file for the differences. Defaults to diffile.gdx.

    Returns
    -------
    GdxDiffResult

    Raises
    ------
    CLIError
        In case a file does not exist, an option is invalid or gdxdiff fails.

    Examples
    --------
    >>> import gamspy as gp
    >>> from gamspy import cli
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=["i1", "i2"])
    >>> m.write("data1.gdx")
    >>> i.setRecords(["i1", "i3"])
    >>> m.write("data2.gdx")
    >>> cli.gdx_diff("data1.gdx", "data2.gdx").has_differences
    True

    """
    import gamspy_base

    GDXDIFF_PATH = os.path.join(gamspy_base.directory, "gdxdiff")
    if platform.system() == "Windows":
        GDXDIFF_PATH = f"{GDXDIFF_PATH}.exe"

    if not file1.endswith(".gdx"):
        file1 += ".gdx"

    if not os.path.exists(file1):
        raise cli_error(f"File not found: {file1}")

    if not file2.endswith(".gdx"):
        file2 += ".gdx"

    if not os.path.exists(file2):
        raise cli_error(f"File not found: {file2}")

    if diffile and not diffile.endswith(".gdx"):
        diffile += ".gdx"

    # Validate restricted strings
    if field and field not in VALID_FIELDS:
        raise cli_error(
            f"Invalid field: '{field}'. Must be one of: {', '.join(VALID_FIELDS)}"
        )

    if setdesc and setdesc not in VALID_SETDESC:
        raise cli_error(f"Invalid SetDesc value: '{setdesc}'. Must be Y or N")

    skipid_list = list(skipid) if skipid else []

    if skip_regex:
        pattern = re.compile(skip_regex)

        file1_symbol_names = _get_gdx_symbol_names(file1)
        file2_symbol_names = _get_gdx_symbol_names(file2)
        union = file1_symbol_names.union(file2_symbol_names)
        matched_symbol_names = [sym for sym in union if pattern.search(sym)]
        skipid_list.extend(matched_symbol_names)

    # gdxdiff writes the differences to a temporary file in the working directory
    # and renames it to diffile, which fails if they are on different file systems.
    cwd = None
    if diffile:
        file1, file2, diffile = (
            os.path.abspath(path) for path in (file1, file2, diffile)
        )
        cwd = os.path.dirname(diffile)

    # Build the command
    cmd = [GDXDIFF_PATH, file1, file2]
    if diffile:
        cmd.append(diffile)
    cmd += _option_args(
        [
            ("Eps", eps),
            ("RelEps", releps),
            ("Field", field),
            ("FldOnly", fldonly),
            ("DiffOnly", diffonly),
            ("CmpDefaults", cmpdefaults),
            ("CmpDomains", cmpdomains),
            ("MatrixFile", matrixfile),
            ("IgnoreOrder", ignoreorder),
            ("SetDesc", setdesc),
        ]
    )
    if id:
        for i in id:
            cmd.append(f"ID={i}")
    if skipid_list:
        for s in set(skipid_list):
            cmd.append(f"SkipID={s}")

    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )
    # gdxdiff returns 1 if the files differ.
    if process.returncode > 1:
        raise cli_error(process.stdout + process.stderr, process.returncode)

    return GdxDiffResult(has_differences=process.returncode == 1, output=process.stdout)


@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy gdx diff gdxfile.gdx gdxfile2.gdx",
    short_help="To show the differences between two GDX files.",
)
def diff(
    file1: str = typer.Argument(..., help="First input GDX file"),
    file2: str = typer.Argument(..., help="Second input GDX file"),
    diffile: str | None = typer.Argument(
        None, help="Optional output GDX file for differences"
    ),
    eps: float | None = typer.Option(
        None, "--eps", "-e", help="Epsilon for comparison"
    ),
    releps: float | None = typer.Option(
        None, "--releps", "-r", help="Relative epsilon for comparison"
    ),
    field: str | None = typer.Option(
        None,
        "--field",
        "-f",
        help="Field to compare: L, M, Up, Lo, Prior, Scale, All",
        autocompletion=complete_field,
    ),
    fldonly: bool = typer.Option(
        False, "--fldonly", "-o", help="Write only selected field"
    ),
    diffonly: bool = typer.Option(
        False,
        "--diffonly",
        "-d",
        help="Write only differences with field dimension",
    ),
    cmpdefaults: bool = typer.Option(
        False, "--cmpdefaults", "-c", help="Compare default values"
    ),
    cmpdomains: bool = typer.Option(
        False, "--cmpdomains", "-m", help="Compare domain information"
    ),
    matrixfile: bool = typer.Option(
        False, "--matrixfile", "-x", help="Compare GAMS matrix files"
    ),
    ignoreorder: bool = typer.Option(
        False, "--ignoreorder", help="Ignore UEL order of input files"
    ),
    setdesc: str | None = typer.Option(
        None,
        "--setdesc",
        help="Compare set element descriptions (Y/N)",
        autocompletion=complete_yes_no,
    ),
    id: Annotated[
        list[str] | None,
        typer.Option("--id", "-i", help="One or more identifiers to include"),
    ] = None,
    skipid: Annotated[
        list[str] | None,
        typer.Option("--skipid", "-s", help="One or more identifiers to skip"),
    ] = None,
    skip_regex: Annotated[
        str | None,
        typer.Option(
            "--skip-regex", help="Regular expression pattern to skip identifiers"
        ),
    ] = None,
) -> None:
    result = gdx_diff(
        file1,
        file2,
        diffile,
        eps=eps,
        releps=releps,
        field=field,
        fldonly=fldonly,
        diffonly=diffonly,
        cmpdefaults=cmpdefaults,
        cmpdomains=cmpdomains,
        matrixfile=matrixfile,
        ignoreorder=ignoreorder,
        setdesc=setdesc,
        id=id,
        skipid=skipid,
        skip_regex=skip_regex,
    )
    typer.echo(result.output, nl=False)
    if result.has_differences:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
