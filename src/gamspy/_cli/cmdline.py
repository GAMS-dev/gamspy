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
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess

import gamspy.utils as utils
from .util import add_solver_entry
from .util import remove_solver_entry
from gamspy.exceptions import GamspyException
from gamspy.exceptions import ValidationError


def get_args():
    parser = argparse.ArgumentParser(prog="gamspy", description="GAMSPy CLI")

    parser.add_argument(
        "command",
        choices=["install", "list", "update", "uninstall", "version"],
        type=str,
        default=None,
    )
    parser.add_argument(
        "component",
        choices=["license", "solver", "solvers"],
        type=str,
        nargs="?",
        default=None,
    )

    _ = parser.add_argument_group(
        "gamspy install|uninstall license",
        description="Options for installing/uninstalling a license.",
    )

    install_group = parser.add_argument_group(
        "gamspy install|uninstall solver",
        description="Options for installing solvers",
    )
    install_group.add_argument(
        "name", type=str, nargs="?", default=None, help="Solver name"
    )
    install_group.add_argument(
        "--skip-pip-install",
        "-s",
        action="store_true",
        help=(
            "If you already have the solver installed, skip pip install and"
            " update gamspy installed solver list."
        ),
    )
    install_group.add_argument(
        "--skip-pip-uninstall",
        "-u",
        action="store_true",
        help=(
            "If you don't want to uninstall the package of the solver, skip"
            " uninstall and update gamspy installed solver list."
        ),
    )

    list_group = parser.add_argument_group(
        "list solvers", description="`gamspy list solvers` options"
    )
    list_group.add_argument("-a", "--all", action="store_true")

    return parser.parse_args()


def install_license(args: argparse.Namespace):
    gamspy_base_dir = utils._get_gamspy_base_directory()

    if args.name is None or not os.path.exists(args.name):
        raise ValidationError(
            f'Given license path ("{args.name}") is not valid.'
        )

    shutil.copy(args.name, gamspy_base_dir + os.sep + "gamslice.txt")


def uninstall_license():
    gamspy_base_dir = utils._get_gamspy_base_directory()
    os.unlink(gamspy_base_dir + os.sep + "gamslice.txt")


def install_solver(args: argparse.Namespace):
    solver_name = args.name.lower()

    if solver_name.upper() not in utils.getAvailableSolvers():
        raise ValidationError(
            f'Given solver name ("{solver_name}") is not valid. Available'
            f" solvers that can be installed: {utils.getAvailableSolvers()}"
        )

    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    if not args.skip_pip_install:
        # install specified solver
        try:
            _ = subprocess.run(
                [
                    "pip",
                    "install",
                    f"gamspy-{solver_name}=={gamspy_base.__version__}",
                    "--force-reinstall",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise GamspyException(
                f"Could not install gamspy-{solver_name}: {e.output}"
            )
    else:
        try:
            solver_lib = importlib.import_module(f"gamspy_{solver_name}")
        except ModuleNotFoundError as e:
            e.msg = f"You must install gamspy-{solver_name} first!"
            raise e

        if solver_lib.__version__ != gamspy_base.__version__:
            raise ValidationError(
                f"gamspy_base version ({gamspy_base.__version__}) and solver"
                f" version ({solver_lib.__version__}) must match! Run `gamspy"
                " update` to update your solvers."
            )

    # copy solver files to gamspy_base
    gamspy_base_dir = utils._get_gamspy_base_directory()
    solver_lib = importlib.import_module(f"gamspy_{solver_name}")

    file_paths = solver_lib.file_paths
    if solver_name == "scip":
        mosek_lib = importlib.import_module("gamspy_mosek")
        file_paths += mosek_lib.file_paths

    for file in file_paths:
        shutil.copy(file, gamspy_base_dir)

    files = solver_lib.files
    if solver_name == "scip":
        files += mosek_lib.files

    verbatims = [solver_lib.verbatim]
    if solver_name == "scip":
        verbatims.append(mosek_lib.verbatim)
    append_dist_info(files, gamspy_base_dir)
    add_solver_entry(gamspy_base_dir, solver_name, verbatims)


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


def uninstall_solver(args: argparse.Namespace):
    solver_name = args.name.lower()

    if solver_name.upper() not in utils.getInstalledSolvers():
        raise ValidationError(
            f'Given solver name ("{solver_name}") is not valid. Installed'
            " solvers solvers that can be uninstalled:"
            f" {utils.getInstalledSolvers()}"
        )

    if not args.skip_pip_uninstall:
        # uninstall specified solver
        try:
            _ = subprocess.run(
                ["pip", "uninstall", f"gamspy-{solver_name}"], check=True
            )
        except subprocess.CalledProcessError as e:
            raise GamspyException(
                f"Could not uninstall gamspy-{solver_name}: {e.output}"
            )

    # do not delete files from gamspy_base as other solvers might depend on it
    gamspy_base_dir = utils._get_gamspy_base_directory()
    remove_solver_entry(gamspy_base_dir, solver_name)


def install(args: argparse.Namespace):
    if args.component == "license":
        install_license(args)
    elif args.component == "solver":
        install_solver(args)


def update():
    prev_installed_solvers = utils.getInstalledSolvers()

    try:
        _ = subprocess.run(
            [
                "pip",
                "install",
                "gamspy-base",
                "--force-reinstall",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise GamspyException(f"Could not uninstall gamspy-base: {e.output}")

    import gamspy_base

    new_installed_solvers = utils.getInstalledSolvers()

    solvers_to_update = []
    for solver in prev_installed_solvers:
        if solver not in new_installed_solvers:
            solvers_to_update.append(solver)

    for solver in solvers_to_update:
        try:
            _ = subprocess.run(
                [
                    "pip",
                    "install",
                    f"gamspy-{solver.lower()}=={gamspy_base.version}",
                    "--force-reinstall",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise GamspyException(
                "Could not uninstall"
                f" gamspy-{solver.lower()}=={gamspy_base.version}: {e.output}"
            )


def list_solvers(args: argparse.Namespace):
    component = args.component

    if component == "solvers":
        if args.all:
            return utils.getAvailableSolvers()

        return utils.getInstalledSolvers()

    return None


def uninstall(args: argparse.Namespace):
    if args.component == "license":
        uninstall_license()
    elif args.component == "solver":
        uninstall_solver(args)


def main():
    """
    Entry point for gamspy command line application.
    """
    args = get_args()
    if args.command == "version":
        import gamspy

        print(f"GAMSPy version: {gamspy.__version__}")
    elif args.command == "install":
        install(args)
    elif args.command == "update":
        update()
    elif args.command == "list":
        solvers = list_solvers(args)
        if solvers:
            print(solvers)
    elif args.command == "uninstall":
        uninstall(args)
