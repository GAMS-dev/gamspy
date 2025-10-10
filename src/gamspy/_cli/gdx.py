"""
This module provides a wrapper around GDX tools of GAMS to expose them in GAMSPy CLI.
Run `gamspy gdx -h` to see available gdx tools in GAMSPy CLI.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Annotated

import gamspy_base
import typer

app = typer.Typer(
    rich_markup_mode="rich",
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


def complete_delim(ctx: typer.Context, incomplete: str):
    return [d for d in VALID_DELIMS if d.startswith(incomplete)]


def complete_decimalsep(ctx: typer.Context, incomplete: str):
    return [s for s in VALID_DECIMAL_SEP if s.startswith(incomplete)]


def complete_format(ctx: typer.Context, incomplete: str):
    return [f for f in VALID_FORMATS if f.startswith(incomplete)]


def complete_dformat(ctx: typer.Context, incomplete: str):
    return [f for f in VALID_DFORMATS if f.startswith(incomplete)]


def complete_yes_no(ctx: typer.Context, incomplete: str):
    return [v for v in VALID_YN if v.startswith(incomplete)]


def complete_field(ctx: typer.Context, incomplete: str):
    return [f for f in VALID_FIELDS if f.startswith(incomplete)]


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
):
    GDXDUMP_PATH = os.path.join(gamspy_base.directory, "gdxdump")
    if platform.system() == "Windows":
        GDXDUMP_PATH = f"{GDXDUMP_PATH}.exe"

    if not filename.endswith(".gdx"):
        filename += ".gdx"

    if not os.path.exists(filename):
        typer.echo(f"File not found: {filename}", err=True)
        raise typer.Exit(code=1)

    # Validation
    if delim and delim not in VALID_DELIMS:
        typer.echo(
            f"Invalid Delim: {delim}. Must be one of: {', '.join(VALID_DELIMS)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if decimalsep and decimalsep not in VALID_DECIMAL_SEP:
        typer.echo(
            f"Invalid DecimalSep: {decimalsep}. Must be one of: {', '.join(VALID_DECIMAL_SEP)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if format and format not in VALID_FORMATS:
        typer.echo(
            f"Invalid Format: {format}. Must be one of: {', '.join(VALID_FORMATS)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if dformat and dformat not in VALID_DFORMATS:
        typer.echo(
            f"Invalid dFormat: {dformat}. Must be one of: {', '.join(VALID_DFORMATS)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if cdim and cdim not in VALID_YN:
        typer.echo(f"Invalid CDim: {cdim}. Must be Y or N", err=True)
        raise typer.Exit(code=1)
    if filterdef and filterdef not in VALID_YN:
        typer.echo(f"Invalid FilterDef: {filterdef}. Must be Y or N", err=True)
        raise typer.Exit(code=1)

    cmd = [GDXDUMP_PATH, filename]
    if version:
        cmd.append("-V")
    if output:
        cmd.append(f"Output={output}")
    if symb:
        cmd.append(f"Symb={symb}")
    if ueltable:
        cmd.append(f"UelTable={ueltable}")
    if delim:
        cmd.append(f"Delim={delim}")
    if decimalsep:
        cmd.append(f"DecimalSep={decimalsep}")
    if noheader:
        cmd.append("NoHeader")
    if nodata:
        cmd.append("NoData")
    if csvallfields:
        cmd.append("CSVAllFields")
    if csvsettext:
        cmd.append("CSVSetText")
    if symbols:
        cmd.append("Symbols")
    if domaininfo:
        cmd.append("DomainInfo")
    if symbolsasset:
        cmd.append("SymbolsAsSet")
    if symbolsassetdi:
        cmd.append("SymbolsAsSetDI")
    if settext:
        cmd.append("SetText")
    if format:
        cmd.append(f"Format={format}")
    if dformat:
        cmd.append(f"dFormat={dformat}")
    if cdim:
        cmd.append(f"CDim={cdim}")
    if filterdef:
        cmd.append(f"FilterDef={filterdef}")
    if epsout:
        cmd.append(f"EpsOut={epsout}")
    if naout:
        cmd.append(f"NaOut={naout}")
    if pinfout:
        cmd.append(f"PinfOut={pinfout}")
    if minfout:
        cmd.append(f"MinfOut={minfout}")
    if undfout:
        cmd.append(f"UndfOut={undfout}")
    if zeroout:
        cmd.append(f"ZeroOut={zeroout}")
    if header:
        cmd.append(f"Header={header}")

    result = subprocess.run(cmd, text=True)
    raise typer.Exit(code=result.returncode)


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
):
    GDXDIFF_PATH = os.path.join(gamspy_base.directory, "gdxdiff")
    if platform.system() == "Windows":
        GDXDIFF_PATH = f"{GDXDIFF_PATH}.exe"

    if not file1.endswith(".gdx"):
        file1 += ".gdx"

    if not os.path.exists(file1):
        typer.echo(f"File not found: {file1}", err=True)
        raise typer.Exit(code=1)

    if not file2.endswith(".gdx"):
        file2 += ".gdx"

    if not os.path.exists(file2):
        typer.echo(f"File not found: {file2}", err=True)
        raise typer.Exit(code=1)

    if diffile and not diffile.endswith(".gdx"):
        diffile += ".gdx"

    # Validate restricted strings
    if field and field not in VALID_FIELDS:
        typer.echo(
            f"Invalid field: '{field}'. Must be one of: {', '.join(VALID_FIELDS)}",
            err=True,
        )
        raise typer.Exit(code=1)

    if setdesc and setdesc not in VALID_SETDESC:
        typer.echo(f"Invalid SetDesc value: '{setdesc}'. Must be Y or N", err=True)
        raise typer.Exit(code=1)

    # Build the command
    cmd = [GDXDIFF_PATH, file1, file2]
    if diffile:
        cmd.append(diffile)
    if eps is not None:
        cmd.append(f"Eps={eps}")
    if releps is not None:
        cmd.append(f"RelEps={releps}")
    if field:
        cmd.append(f"Field={field}")
    if fldonly:
        cmd.append("FldOnly")
    if diffonly:
        cmd.append("DiffOnly")
    if cmpdefaults:
        cmd.append("CmpDefaults")
    if cmpdomains:
        cmd.append("CmpDomains")
    if matrixfile:
        cmd.append("MatrixFile")
    if ignoreorder:
        cmd.append("IgnoreOrder")
    if setdesc:
        cmd.append(f"SetDesc={setdesc}")
    if id:
        for i in id:
            cmd.append(f"ID={i}")
    if skipid:
        for s in skipid:
            cmd.append(f"SkipID={s}")

    result = subprocess.run(cmd, text=True)
    raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    app()
