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
import sys
from typing import Dict

import gamspy.utils as utils
from .util import add_solver_entry
from .util import remove_solver_entry
from gamspy.exceptions import GamspyException


SOLVERS = [
    "cbc",
    "soplex",
    "highs",
    "copt",
    "gurobi",
    "xpress",
    "odhcplex",
    "mosek",
    "mpsge",
    "miles",
    "knitro",
    "ipopt",
    "minos",
    "snopt",
    "ipopth",
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
    parser.add_argument("command", choices=["install", "uninstall"], type=str)
    parser.add_argument(
        "component", choices=["license", "engine_license", "solver"], type=str
    )
    parser.add_argument("name", type=str, nargs="?", default=None)
    parser.add_argument("--skip-pip-install", action="store_true")
    parser.add_argument("--skip-pip-uninstall", action="store_true")

    res = vars(parser.parse_args())
    if res["name"] is None and not (
        res["command"] == "uninstall"
        and res["component"] in ["license", "engine_license"]
    ):
        sys.stderr.write("name must be specified\n")
        sys.exit(1)

    return res


def install_license(args: Dict[str, str]):
    gamspy_base_dir = utils._getGAMSPyBaseDirectory()
    shutil.copy(args["name"], gamspy_base_dir + os.sep + "gamslice.txt")


def install_engine_license(args: Dict[str, str]):
    gamspy_base_dir = utils._getGAMSPyBaseDirectory()
    shutil.copy(args["name"], gamspy_base_dir + os.sep + "enginelic.txt")
    append_dist_info(["enginelic.txt"], gamspy_base_dir)


def uninstall_license():
    gamspy_base_dir = utils._getGAMSPyBaseDirectory()
    os.unlink(gamspy_base_dir + os.sep + "gamslice.txt")


def uninstall_engine_license():
    gamspy_base_dir = utils._getGAMSPyBaseDirectory()
    os.unlink(gamspy_base_dir + os.sep + "enginelic.txt")


def install_solver(args: Dict[str, str]):
    solver_name = args["name"]
    if solver_name not in SOLVERS:
        raise Exception(
            f"Solver name is not valid. Possible solver names: {SOLVERS}"
        )

    if not args["skip_pip_install"]:
        # install specified solver
        try:
            _ = subprocess.run(
                ["pip", "install", f"gamspy_{solver_name}"], check=True
            )
        except subprocess.CalledProcessError as e:
            raise GamspyException(
                f"Could not install gamspy_{solver_name}: {e.output}"
            )

    # copy solver files to gamspy_base
    gamspy_base_dir = utils._getGAMSPyBaseDirectory()
    solver_lib = importlib.import_module(f"gamspy_{solver_name}")
    for file in solver_lib.file_paths:
        shutil.copy(file, gamspy_base_dir)

    append_dist_info(solver_lib.files, gamspy_base_dir)
    add_solver_entry(gamspy_base_dir, solver_name, solver_lib.verbatim)


def append_dist_info(files, gamspy_base_dir: str):
    """Updates dist-info/RECORD in site-packages for pip uninstall"""
    import gamspy as gp

    gamspy_path: str = gp.__path__[0]
    dist_info_path = f"{gamspy_path}-{gp.__version__}.dist-info"

    with open(dist_info_path + os.sep + "RECORD", "a") as record:
        gamspy_base_relative_path = os.sep.join(
            gamspy_base_dir.split(os.sep)[-3:]
        )

        lines = []
        for file in files:
            line = f"{gamspy_base_relative_path}{os.sep}{file},,"
            lines.append(line)

        record.write("\n".join(lines))


def uninstall_solver(args: Dict[str, str]):
    solver_name = args["name"]
    if solver_name not in SOLVERS:
        raise Exception(
            f"Solver name is not valid. Possible solver names: {SOLVERS}"
        )

    if not args["skip_pip_uninstall"]:
        # uninstall specified solver
        try:
            _ = subprocess.run(
                ["pip", "uninstall", f"gamspy_{solver_name}"], check=True
            )
        except subprocess.CalledProcessError as e:
            raise GamspyException(
                f"Could not uninstall gamspy_{solver_name}: {e.output}"
            )

    # do not delete files from gamspy_base as other solvers might depend on it
    gamspy_base_dir = utils._getGAMSPyBaseDirectory()
    remove_solver_entry(gamspy_base_dir, solver_name)


def install(args: Dict[str, str]):
    if args["component"] == "license":
        install_license(args)
    elif args["component"] == "engine_license":
        install_engine_license(args)
    elif args["component"] == "solver":
        install_solver(args)


def uninstall(args: Dict[str, str]):
    if args["component"] == "license":
        uninstall_license()
    elif args["component"] == "engine_license":
        uninstall_engine_license()
    elif args["component"] == "solver":
        uninstall_solver(args)


def main():
    """
    Entry point for gamspy command line application.
    """
    args = get_args()

    if args["command"] == "install":
        install(args)
    elif args["command"] == "uninstall":
        uninstall(args)
