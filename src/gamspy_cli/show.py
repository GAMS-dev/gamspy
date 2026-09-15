from __future__ import annotations

from datetime import datetime, timedelta

import typer

from .license import EVALUATIONS, LICENSE_COMPONENTS

app = typer.Typer(
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
    "DECISC": "https://www.gams.com/latest/docs/S_DECIS.html",
    "DECISM": "https://www.gams.com/latest/docs/S_DECIS.html",
    "DICOPT": "https://www.gams.com/latest/docs/S_DICOPT.html",
    "EXAMINER": "https://www.gams.com/latest/docs/S_EXAMINER.html",
    "GUROBI": "https://www.gams.com/latest/docs/S_GUROBI.html",
    "HIGHS": "https://www.gams.com/latest/docs/S_HIGHS.html",
    "IPOPT": "https://www.gams.com/latest/docs/S_IPOPT.html",
    "IPOPTH": "https://www.gams.com/latest/docs/S_IPOPT.html",
    "JAMS": "https://www.gams.com/latest/docs/S_JAMS.html",
    "KESTREL": "https://www.gams.com/latest/docs/S_KESTREL.html",
    "KNITRO": "https://www.gams.com/latest/docs/S_KNITRO.html",
    "LINDO": "https://www.gams.com/latest/docs/S_LINDO.html",
    "LINDOGLOBAL": "https://www.gams.com/latest/docs/S_LINDO.html",
    "MILES": "https://www.gams.com/latest/docs/S_MILES.html",
    "MINOS": "https://www.gams.com/latest/docs/S_MINOS.html",
    "MOSEK": "https://www.gams.com/latest/docs/S_MOSEK.html",
    "MPSGE": "https://www.gams.com/50/docs/UG_MPSGE_Intro.html",
    "NLPEC": "https://www.gams.com/latest/docs/S_NLPEC.html",
    "PATH": "https://www.gams.com/latest/docs/S_PATH.html",
    "PATHNLP": "https://gams.com/latest/docs/S_PATHNLP.html",
    "QUADMINOS": "https://www.gams.com/latest/docs/S_MINOS.html",
    "RESHOP": "https://www.gams.com/latest/docs/S_RESHOP.html",
    "SBB": "https://www.gams.com/latest/docs/S_SBB.html",
    "SCIP": "https://www.gams.com/latest/docs/S_SCIP.html",
    "SHOT": "https://www.gams.com/latest/docs/S_SHOT.html",
    "SNOPT": "https://www.gams.com/latest/docs/S_SNOPT.html",
    "SOPLEX": "https://www.gams.com/latest/docs/S_SOPLEX.html",
    "XPRESS": "https://www.gams.com/latest/docs/S_XPRESS.html",
}


def print_expiration_date(lines: list[str]) -> None:
    import rich

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
    import rich
    import rich.table

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
    import gamspy_base
    import rich

    import gamspy.utils as utils

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
    import gamspy_base

    print(gamspy_base.directory)


if __name__ == "__main__":
    app()
