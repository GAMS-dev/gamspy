#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import argparse
import importlib
import os
import shutil
import subprocess
from typing import Dict

import gamspy.utils as utils
from gamspy.exceptions import GamspyException


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
