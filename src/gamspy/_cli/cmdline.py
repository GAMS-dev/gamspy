import subprocess
import os
import importlib
import shutil
import argparse
import gamspy.utils as utils
from gamspy.exceptions import GamspyException
from typing import Dict


SOLVERS = [
    "cbc",
    "soplex",
    "highs",
    "copt",
    "cplex",
    "gurobi",
    "xpress",
    "odhcplex",
    "mosek",
    "path",
    "mpsge",
    "miles",
    "nlpec",
    "knitro",
    "ipopt",
    "minos",
    "snopt",
    "conopt",
    "ipopth",
    "sbb",
    "dicopt",
    "alphaecp",
    "shot",
    "octeract",
    "scip",
    "antigone",
    "baron",
    "lindo",
    "decis",
]


def get_args():
    parser = argparse.ArgumentParser(
        prog="gamspy",
        description="A script for installing solvers and licenses",
    )
    parser.add_argument("command", choices=["install"], type=str)
    parser.add_argument("component", choices=["license", "solver"], type=str)
    parser.add_argument("name", type=str)

    return vars(parser.parse_args())


def install_license(args: Dict[str, str]):
    minigams_dir = utils._getMinigamsDirectory()
    shutil.copy(args["name"], minigams_dir + os.sep + "gamslice.txt")


def install_solver(args: Dict[str, str]):
    solver_name = args["name"]
    if solver_name not in SOLVERS:
        raise Exception(
            f"Solver name is not valid. Possible solver names: {SOLVERS}"
        )

    # install specified solver
    try:
        _ = subprocess.run(["pip", "install", f"gamspy_{solver_name}"])
    except subprocess.CalledProcessError as e:
        raise GamspyException(
            f"Could not install gamspy_{solver_name}: {e.output}"
        )

    # copy solver files to minigams
    minigams_dir = utils._getMinigamsDirectory()
    solver_lib = importlib.import_module(f"gamspy_{solver_name}")
    for file in solver_lib.file_paths:
        shutil.copy(file, minigams_dir)

    update_dist_info(solver_lib, minigams_dir)


def update_dist_info(solver_lib, minigams_dir: str):
    """Updates dist-info/RECORD in site-packages for pip uninstall"""
    import gamspy as gp

    gamspy_path: str = gp.__path__[0]
    dist_info_path = f"{gamspy_path}-{gp.__version__}.dist-info"

    with open(dist_info_path + os.sep + "RECORD", "a") as record:
        minigams_relative_path = os.sep.join(minigams_dir.split(os.sep)[-3:])

        lines = []
        for file in solver_lib.files:
            line = f"{minigams_relative_path}{os.sep}{file},,"
            lines.append(line)

        record.write("\n".join(lines))


def install(args: Dict[str, str]):
    if args["component"] == "license":
        install_license(args)
    elif args["component"] == "solver":
        install_solver(args)


def main():
    """
    Entry point for gamspy command line application.
    """
    args = get_args()

    if args["command"] == "install":
        install(args)
