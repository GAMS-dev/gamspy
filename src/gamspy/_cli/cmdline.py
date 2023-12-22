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
import platform
import shutil
import subprocess
import sys
from typing import Dict

import gamspy.utils as utils
from .util import add_solver_entry
from .util import remove_solver_entry
from gamspy.exceptions import GamspyException


def get_args():
    parser = argparse.ArgumentParser(
        prog="gamspy",
        description="A script for installing solvers and licenses",
    )
    parser.add_argument(
        "command",
        choices=["install", "list", "run", "update", "uninstall"],
        type=str,
    )
    parser.add_argument(
        "component",
        choices=["license", "miro", "solver", "solvers"],
        type=str,
        nargs="?",
        default=None,
    )

    install_group = parser.add_argument_group(
        "install solver", description="`gamspy install solver` options"
    )
    install_group.add_argument(
        "name", type=str, nargs="?", default=None, help="Solver name"
    )
    install_group.add_argument(
        "--skip-pip-install",
        action="store_true",
        help=(
            "If you already have the solver installed, skip pip install and"
            " update gamspy installed solver list."
        ),
    )
    install_group.add_argument(
        "--skip-pip-uninstall",
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

    miro_group = parser.add_argument_group(
        "run miro", description="`gamspy run miro` options"
    )
    miro_group.add_argument(
        "-g",
        "--model",
        type=str,
        help="Path to the gamspy model",
        default=None,
    )
    miro_group.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["config", "base", "deploy"],
        help="Execution mode of MIRO",
        default="base",
    )
    miro_group.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to the MIRO executable (.exe on Windows, .app on macOS or .AppImage on Linux)",
        default=None,
    )
    miro_group.add_argument(
        "-s",
        "--skip-execution",
        help="Whether to skip model execution",
        action="store_true",
    )
    parser.add_argument("--version", action="store_true")

    res = vars(parser.parse_args())

    return res


def install_license(args: Dict[str, str]):
    gamspy_base_dir = utils._get_gamspy_base_directory()
    shutil.copy(args["name"], gamspy_base_dir + os.sep + "gamslice.txt")


def uninstall_license():
    gamspy_base_dir = utils._get_gamspy_base_directory()
    os.unlink(gamspy_base_dir + os.sep + "gamslice.txt")


def install_solver(args: Dict[str, str]):
    solver_name = args["name"]

    try:
        import gamspy_base
    except ModuleNotFoundError:
        raise GamspyException(
            "You must first install gamspy_base to use this functionality"
        )

    if not args["skip_pip_install"]:
        # install specified solver
        try:
            _ = subprocess.run(
                [
                    "pip",
                    "install",
                    f"gamspy_{solver_name}=={gamspy_base.__version__}",
                    "--force-reinstall",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise GamspyException(
                f"Could not install gamspy_{solver_name}: {e.output}"
            )
    else:
        try:
            solver_lib = importlib.import_module(f"gamspy_{solver_name}")
        except ModuleNotFoundError:
            raise GamspyException(
                f"You must install gamspy_{solver_name} first!"
            )

        if solver_lib.__version__ != gamspy_base.__version__:
            raise GamspyException(
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


def uninstall_solver(args: Dict[str, str]):
    solver_name = args["name"]

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
    gamspy_base_dir = utils._get_gamspy_base_directory()
    remove_solver_entry(gamspy_base_dir, solver_name)


def install(args: Dict[str, str]):
    if args["component"] == "license":
        install_license(args)
    elif args["component"] == "solver":
        install_solver(args)


def update():
    prev_installed_solvers = utils.getInstalledSolvers()

    try:
        _ = subprocess.run(
            [
                "pip",
                "install",
                "gamspy_base",
                "--force-reinstall",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise GamspyException(f"Could not uninstall gamspy_base: {e.output}")

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
                    f"gamspy_{solver.lower()}=={gamspy_base.version}",
                    "--force-reinstall",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise GamspyException(
                "Could not uninstall"
                f" gamspy_{solver.lower()}=={gamspy_base.version}: {e.output}"
            )


def list(args: Dict[str, str]):
    component = args["component"]

    if component == "solvers":
        if args["all"]:
            return utils.getAvailableSolvers()

        print(utils.getInstalledSolvers())


def run(args: Dict[str, str]):
    component = args["component"]

    if component == "miro":
        model = os.path.abspath(args["model"])
        mode = args["mode"]
        path = os.getenv("MIRO_PATH", None)

        if args["path"] is not None:
            path = args["path"]

        if model is None or path is None:
            raise GamspyException(
                "--model and --path must be provided to run MIRO"
            )

        if (
            platform.system() == "Darwin"
            and os.path.splitext(path)[1] == ".app"
        ):
            path = os.path.join(path, "Contents", "MacOS", "GAMS MIRO")

        # Initialize MIRO
        if not args["skip_execution"]:
            subprocess_env = os.environ.copy()
            subprocess_env["MIRO"] = "1"
            subprocess.run(
                [sys.executable, model], env=subprocess_env, check=True
            )

        # Run MIRO
        subprocess_env = os.environ.copy()
        if mode == "deploy":
            subprocess_env["MIRO_BUILD"] = "true"
            mode = "base"

        subprocess_env["MIRO_MODEL_PATH"] = model
        subprocess_env["MIRO_MODE"] = mode
        subprocess_env["MIRO_DEV_MODE"] = "true"
        subprocess_env["MIRO_USE_TMP"] = "false"
        subprocess_env["PYTHON_EXEC_PATH"] = sys.executable

        subprocess.run([path], env=subprocess_env, check=True)


def uninstall(args: Dict[str, str]):
    if args["component"] == "license":
        uninstall_license()
    elif args["component"] == "solver":
        uninstall_solver(args)


def main():
    """
    Entry point for gamspy command line application.
    """
    args = get_args()
    if args["version"]:
        import gamspy

        print(f"GAMSPy version: {gamspy.__version__}")
        return

    if args["command"] == "install":
        install(args)
    elif args["command"] == "list":
        list(args)
    elif args["command"] == "run":
        run(args)
    elif args["command"] == "update":
        update()
    elif args["command"] == "uninstall":
        uninstall(args)
