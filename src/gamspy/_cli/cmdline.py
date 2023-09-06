import os
import importlib
import shutil
import argparse
import gamspy.utils as utils


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


def install_license(args: dict):
    minigams_dir = utils._getMinigamsDirectory()
    shutil.copy(args["name"], minigams_dir)


def install_solver(args: dict):
    solver_name = args["name"]
    if solver_name not in SOLVERS:
        raise Exception(
            f"Solver name is not valid. Possible solver names: {SOLVERS}"
        )

    # install specified solver
    # _ = subprocess.run(["pip", "install", f"gamspy_{solver_name}"])

    # move solver files to minigams
    minigams_dir = utils._getMinigamsDirectory()
    solver_lib = importlib.import_module(f"gamspy_{solver_name}")
    for file in solver_lib.file_paths:
        shutil.copy(file, minigams_dir)

    # update gamspy egg for uninstall
    import gamspy as gp

    gamspy_path: str = gp.__path__[0]
    dist_info_path = f"{gamspy_path}-{gp.__version__}.dist-info"

    with open(dist_info_path + os.sep + "RECORD", "a") as record:
        for file in solver_lib.files:
            record.write(f"\n{minigams_dir}{os.sep}{file},,")


def install(args: dict):
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
