from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# cm: component_map, ia: is_academic
EVALUATIONS: dict[str, Callable[[dict[str, bool], bool], bool]] = {
    "ANTIGONE": lambda cm, ia: (
        cm["AT"] and (cm["CP"] or cm["CL"]) and (cm["CO"] or cm["SN"])
    ),
    "BARON": lambda cm, ia: cm["BA"],
    "CBC": lambda cm, ia: True,
    "CONOPT": lambda cm, ia: cm["CO"],
    "CONVERT": lambda cm, ia: True,
    "COPT": lambda cm, ia: cm["CT"] or cm["CK"],
    "CPLEX": lambda cm, ia: cm["CP"] or cm["CL"],
    "DECISC": lambda cm, ia: cm["DE"] and (cm["CP"] or cm["CL"]),
    "DECISM": lambda cm, ia: cm["DE"] and cm["M5"],
    "DICOPT": lambda cm, ia: cm["DI"],
    "EXAMINER": lambda cm, ia: True,
    "GUROBI": lambda cm, ia: cm["GU"] or cm["GL"],
    "HIGHS": lambda cm, ia: cm["HI"] or ia,
    "IPOPT": lambda cm, ia: True,
    "IPOPTH": lambda cm, ia: cm["IP"] or ia,
    "KESTREL": lambda cm, ia: True,
    "KNITRO": lambda cm, ia: cm["KN"],
    "LINDO": lambda cm, ia: cm["LD"],
    "LINDOGLOBAL": lambda cm, ia: cm["LD"] or cm["LI"],
    "MILES": lambda cm, ia: True,
    "MINOS": lambda cm, ia: cm["M5"],
    "MOSEK": lambda cm, ia: cm["MB"] or cm["ML"],
    "MPSGE": lambda cm, ia: cm["GE"],
    "NLPEC": lambda cm, ia: True,
    "PATH": lambda cm, ia: cm["PT"],
    "PATHNLP": lambda cm, ia: cm["PT"],
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

# License numbers for which all solvers are available (with size limits), so
# there is no per-component breakdown to parse out of the license file.
_UNRESTRICTED_LICENSE_NUMBERS = {"00", "05"}


def get_licensed_solvers(license_path: str) -> list[str]:
    with open(license_path, encoding="utf-8") as license_file:
        lines = [line.strip() for line in license_file.readlines()]

    license_number = lines[2][0:2]
    if license_number in _UNRESTRICTED_LICENSE_NUMBERS:
        import gamspy_base

        return sorted(gamspy_base.available_solvers)

    is_academic = lines[0][59] == "A"
    component_map: dict[str, bool] = dict.fromkeys(LICENSE_COMPONENTS.keys(), False)
    for i in range(2, len(lines[2]), 2):
        component = lines[2][i : i + 2]
        if component == "__":
            break
        component_map[component] = True

    return sorted(
        solver
        for solver, is_licensed in EVALUATIONS.items()
        if is_licensed(component_map, is_academic)
    )
