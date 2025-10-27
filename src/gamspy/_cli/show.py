from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import rich
import rich.table
import typer

import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To show your license and gamspy_base directory.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy show license | gamspy show base",
    context_settings={"help_option_names": ["-h", "--help"]},
)

LICENSE_TYPE_MAP = {
    "00": "Demo license. All solvers available with size limitations.\nSee https://www.gams.com/latest/docs/UG_License.html#UG_License_Additional_Solver_Limits",
    "05": "Community license. All solvers available with size limitations.\nSee https://www.gams.com/latest/docs/UG_License.html#UG_License_Additional_Solver_Limits",
    "07": "GAMSPy license",
    "08": "GAMSPy++ license",
    "09": "GAMS & GAMSPy++ license",
}

SOLVER_PAGE_MAP = {
    "ANTIGONE": "https://www.gams.com/latest/docs/S_ANTIGONE.html",
    "BARON": "https://www.gams.com/latest/docs/S_BARON.html",
    "CBC": "https://www.gams.com/latest/docs/S_CBC.html",
    "CONOPT": "https://www.gams.com/latest/docs/S_CONOPT4.html",
    "CONVERT": "https://www.gams.com/latest/docs/S_CONVERT.html",
    "COPT": "https://www.gams.com/latest/docs/S_COPT.html",
    "CPLEX": "https://www.gams.com/latest/docs/S_CPLEX.html",
    "DE": "https://www.gams.com/latest/docs/S_DE.html",
    "DECISC": "https://www.gams.com/latest/docs/S_DECIS.html",
    "DECISM": "https://www.gams.com/latest/docs/S_DECIS.html",
    "DICOPT": "https://www.gams.com/latest/docs/S_DICOPT.html",
    "EXAMINER": "https://www.gams.com/latest/docs/S_EXAMINER.html",
    "GAMSCHK": "https://www.gams.com/latest/docs/S_GAMSCHK.html",
    "GUROBI": "https://www.gams.com/latest/docs/S_GUROBI.html",
    "HIGHS": "https://www.gams.com/latest/docs/S_HIGHS.html",
    "IPOPT": "https://www.gams.com/latest/docs/S_IPOPT.html",
    "IPOPTH": "https://www.gams.com/latest/docs/S_IPOPT.html",
    "JAMS": "https://www.gams.com/latest/docs/S_JAMS.html",
    "KESTREL": "https://www.gams.com/latest/docs/S_KESTREL.html",
    "KNITRO": "https://www.gams.com/latest/docs/S_KNITRO.html",
    "LINDO": "https://www.gams.com/latest/docs/S_LINDO.html",
    "LINDOGLOBAL": "https://www.gams.com/latest/docs/S_LINDO.html",
    "LOGMIP": "https://www.gams.com/latest/docs/S_JAMS.html",
    "MILES": "https://www.gams.com/latest/docs/S_MILES.html",
    "MINOS": "https://www.gams.com/latest/docs/S_MINOS.html",
    "MOSEK": "https://www.gams.com/latest/docs/S_MOSEK.html",
    "MPSGE": "https://www.gams.com/50/docs/UG_MPSGE_Intro.html",
    "NLPEC": "https://www.gams.com/latest/docs/S_NLPEC.html",
    "ODHCPLEX": "https://www.gams.com/latest/docs/S_ODHCPLEX.html",
    "PATH": "https://www.gams.com/latest/docs/S_PATH.html",
    "QUADMINOS": "https://www.gams.com/latest/docs/S_MINOS.html",
    "RESHOP": "https://www.gams.com/latest/docs/S_RESHOP.html",
    "SBB": "https://www.gams.com/latest/docs/S_SBB.html",
    "SCIP": "https://www.gams.com/latest/docs/S_SCIP.html",
    "SHOT": "https://www.gams.com/latest/docs/S_SHOT.html",
    "SNOPT": "https://www.gams.com/latest/docs/S_SNOPT.html",
    "SOPLEX": "https://www.gams.com/latest/docs/S_SOPLEX.html",
    "XPRESS": "https://www.gams.com/latest/docs/S_XPRESS.html",
}

# cm: component_map, ia: is_academic
EVALUATIONS: dict[str, Callable[[dict[str, bool], bool], bool]] = {
    "ANTIGONE": lambda cm, ia: cm["AT"]
    and (cm["CP"] or cm["CL"])
    and (cm["CO"] or cm["SN"]),
    "BARON": lambda cm, ia: cm["BA"],
    "CBC": lambda cm, ia: True,
    "CONOPT": lambda cm, ia: cm["CO"],
    "CONVERT": lambda cm, ia: True,
    "COPT": lambda cm, ia: cm["CT"] or cm["CK"],
    "CPLEX": lambda cm, ia: cm["CP"] or cm["CL"],
    "DE": lambda cm, ia: True,
    "DECISC": lambda cm, ia: cm["DE"] and (cm["CP"] or cm["CL"]),
    "DECISM": lambda cm, ia: cm["DE"] and cm["M5"],
    "DICOPT": lambda cm, ia: cm["DI"],
    "EXAMINER": lambda cm, ia: True,
    "GAMSCHK": lambda cm, ia: True,
    "GUROBI": lambda cm, ia: cm["GU"] or cm["GL"],
    "HIGHS": lambda cm, ia: cm["HI"] or ia,
    "IPOPT": lambda cm, ia: True,
    "IPOPTH": lambda cm, ia: cm["IP"] or ia,
    "JAMS": lambda cm, ia: True,
    "KESTREL": lambda cm, ia: True,
    "KNITRO": lambda cm, ia: cm["KN"],
    "LINDO": lambda cm, ia: cm["LD"],
    "LINDOGLOBAL": lambda cm, ia: cm["LD"] or cm["LI"],
    "LOGMIP": lambda cm, ia: True,
    "MILES": lambda cm, ia: True,
    "MINOS": lambda cm, ia: cm["M5"],
    "MOSEK": lambda cm, ia: cm["MB"] or cm["ML"],
    "MPSGE": lambda cm, ia: cm["GE"],
    "NLPEC": lambda cm, ia: True,
    "ODHCPLEX": lambda cm, ia: (ia or cm["OD"]) and (cm["CP"] or cm["CL"]),
    "PATH": lambda cm, ia: cm["PT"],
    "QUADMINOS": lambda cm, ia: cm["M5"],
    "RESHOP": lambda cm, ia: True,
    "SBB": lambda cm, ia: cm["SB"],
    "SCIP": lambda cm, ia: ia or cm["SC"],
    "SHOT": lambda cm, ia: True,
    "SNOPT": lambda cm, ia: cm["SN"],
    "SOPLEX": lambda cm, ia: ia or cm["SC"],
    "XPRESS": lambda cm, ia: cm["XP"] or cm["XL"] or cm["XS"] or cm["XX"] or cm["XG"],
}

LICENSE_COMPONENTS = {
    "AT": "ANTIGONE                 https://www.gams.com/latest/docs/S_ANTIGONE.html",
    "BA": "BARON                    https://www.gams.com/latest/docs/S_BARON.html",
    "CL": "CPLEX (Link only)        https://www.gams.com/latest/docs/S_CPLEX.html",
    "CO": "CONOPT                   https://www.gams.com/latest/docs/S_CONOPT4.html",
    "CP": "CPLEX                    https://www.gams.com/latest/docs/S_CPLEX.html",
    "CT": "COPT                     https://www.gams.com/latest/docs/S_COPT.html",
    "CK": "COPT (Link only)         https://www.gams.com/latest/docs/S_COPT.html",
    "DE": "DECIS                    https://www.gams.com/latest/docs/S_DECIS.html",
    "DI": "DICOPT                   https://www.gams.com/latest/docs/S_DICOPT.html",
    "EC": "ALPHAECP                 https://www.gams.com/latest/docs/S_ALPHAECP.html",
    "GE": "MPSGE                    https://www.gams.com/latest/docs/UG_MPSGE.html",
    "GL": "GUROBI (Link only)       https://www.gams.com/latest/docs/S_GUROBI.html",
    "GU": "GUROBI                   https://www.gams.com/latest/docs/S_GUROBI.html",
    "HI": "HIGHS                    https://www.gams.com/latest/docs/S_HIGHS.html",
    "IP": "IPOPT/IPOPTH             https://www.gams.com/latest/docs/S_IPOPT.html",
    "KN": "KNITRO                   https://www.gams.com/latest/docs/S_KNITRO.html",
    "LD": "LINDO                    https://www.gams.com/latest/docs/S_LINDO.html",
    "LI": "LINDOGlobal              https://www.gams.com/latest/docs/S_LINDO.html",
    "M5": "MINOS/QUADMINOS          https://www.gams.com/latest/docs/S_MINOS.html",
    "MB": "MOSEK                    https://www.gams.com/latest/docs/S_MOSEK.html",
    "ML": "MOSEK (Link only)        https://www.gams.com/latest/docs/S_MOSEK.html",
    "OD": "ODHCPLEX                 https://www.gams.com/latest/docs/S_ODHCPLEX.html",
    "PT": "PATH/PATHNLP             https://www.gams.com/latest/docs/S_PATH.html",
    "SB": "SBB                      https://www.gams.com/latest/docs/S_SBB.html",
    "SC": "SCIP/SOPLEX              https://www.gams.com/latest/docs/S_SCIP.html",
    "SN": "SNOPT                    https://www.gams.com/latest/docs/S_SNOPT.html",
    "XL": "XPRESS (Link only)       https://www.gams.com/latest/docs/S_XPRESS.html",
    "XP": "XPRESS LP/MIP            https://www.gams.com/latest/docs/S_XPRESS.html",
    "XS": "XPRESS LP/SLP            https://www.gams.com/latest/docs/S_XPRESS.html",
    "XX": "XPRESS LP/SLP/MIP        https://www.gams.com/latest/docs/S_XPRESS.html",
    "XG": "XPRESS LP/SLP/MIP/Global https://www.gams.com/latest/docs/S_XPRESS.html",
}


def print_expiration_date(lines: list[str]) -> None:
    month = lines[3][0:2]
    m = month[1] if month[0] == "0" else month[0]
    mval = 0
    if "1" <= m <= "9":
        mval = ord(m) - ord("0")
    elif "A" <= m <= "Z":
        mval = ord(m) - ord("A") + 10
    elif "a" <= m <= "z":
        mval = ord(m) - ord("a") + 36

    assert mval > 0
    expiration = datetime.strptime(lines[0][48:54], "%y%m%d") + timedelta(
        days=mval * 30
    )
    if month[0] == "0":
        rich.print(
            f"\n[bold]License expiration date[/bold]: {expiration.strftime('%Y-%m-%d')}"
        )
    else:
        rich.print(
            f"\n[bold]License M&S expiration date[/bold]: {expiration.strftime('%Y-%m-%d')}"
        )


def print_licensed_solvers(lines: list[str], verbose: bool) -> None:
    component_map: dict[str, bool] = dict.fromkeys(LICENSE_COMPONENTS.keys(), False)

    is_academic = lines[0][59] == "A"
    license_number = lines[2][0:2]
    rich.print(f"[bold]License type[/bold]: {LICENSE_TYPE_MAP[license_number]}")

    if license_number not in ["00", "05"]:
        licensed_components = []
        for i in range(2, len(lines[2]), 2):
            c = lines[2][i : i + 2]
            if c == "__":
                break
            component_map[c] = True
            licensed_components.append(c)

        if verbose and licensed_components:
            rich.print("\n[bold]Licensed components:[/bold]")
            for c in licensed_components:
                print(f"  {LICENSE_COMPONENTS[c]}")

        licensed_solvers = []
        for solver, is_licensed_func in EVALUATIONS.items():
            if is_licensed_func(component_map, is_academic):
                licensed_solvers.append(solver)

        if licensed_solvers:
            print()
            table = rich.table.Table(title="Licensed Solvers")
            table.add_column("Solver Name", style="green")
            table.add_column("Solver Manual", style="magenta")
            for solver in licensed_solvers:
                table.add_row(solver, SOLVER_PAGE_MAP[solver])

            rich.print(table)


@app.command(short_help="Shows the license content.")
def license(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Shows more information about the license.",
    ),
) -> None:
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    license_path = utils._get_license_path(gamspy_base.directory)
    rich.print(f"[bold]License found at[/bold]: {license_path}\n")
    rich.print("[bold]License Content[/bold]")
    print("=" * 15)
    with open(license_path, encoding="utf-8") as license_file:
        lines = [line.strip() for line in license_file.readlines()]

    print("\n".join(lines))

    print_expiration_date(lines)
    print_licensed_solvers(lines, verbose)


@app.command(short_help="Shows the path of gamspy_base.")
def base():
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    print(gamspy_base.directory)


if __name__ == "__main__":
    app()
