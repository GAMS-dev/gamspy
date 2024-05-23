from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys

import gamspy.utils as utils
from gamspy.exceptions import GamspyException, ValidationError

from .util import add_solver_entry, remove_solver_entry

SOLVER_CAPABILITIES = {
    "BARON": [
        "LP",
        "MIP",
        "NLP",
        "CNS",
        "DNLP",
        "MINLP",
        "QCP",
        "MIQCP",
        "GLOBAL",
    ],
    "CBC": ["LP", "MIP"],
    "CONOPT3": ["LP", "NLP", "CNS", "DNLP", "QCP"],
    "CONOPT": ["LP", "NLP", "CNS", "DNLP", "QCP"],
    "COPT": ["LP", "MIP", "QCP", "MIQCP"],
    "CPLEX": ["LP", "MIP", "QCP", "MIQCP"],
    "DICOPT": ["MINLP", "MIQCP"],
    "GUROBI": ["LP", "MIP", "NLP", "DNLP", "MINLP", "QCP", "MIQCP"],
    "GUSS": [
        "LP",
        "MIP",
        "NLP",
        "MCP",
        "CNS",
        "DNLP",
        "MINLP",
        "QCP",
        "MIQCP",
    ],
    "IPOPT": ["LP", "NLP", "CNS", "DNLP", "QCP"],
    "HIGHS": ["LP", "MIP"],
    "KNITRO": [
        "LP",
        "NLP",
        "MCP",
        "MPEC",
        "CNS",
        "DNLP",
        "MINLP",
        "QCP",
        "MIQCP",
    ],
    "MINOS": ["LP", "NLP", "CNS", "DNLP", "QCP"],
    "MOSEK": ["LP", "MIP", "NLP", "DNLP", "MINLP", "QCP", "MIQCP"],
    "NLPEC": ["MCP", "MPEC"],
    "PATH": ["MCP", "CNS"],
    "SBB": ["MINLP", "MIQCP"],
    "SCIP": ["MIP", "NLP", "CNS", "DNLP", "MINLP", "QCP", "MIQCP", "GLOBAL"],
    "SHOT": ["MINLP", "MIQCP"],
    "SNOPT": ["LP", "NLP", "CNS", "DNLP", "QCP"],
    "XPRESS": ["LP", "MIP", "NLP", "CNS", "DNLP", "MINLP", "QCP", "MIQCP"],
}


def get_args():
    parser = argparse.ArgumentParser(prog="gamspy", description="GAMSPy CLI")

    parser.add_argument(
        "command",
        choices=["install", "list", "run", "show", "update", "uninstall"],
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "component",
        choices=["base", "license", "miro", "solver", "solvers"],
        type=str,
        nargs="?",
        default=None,
    )

    parser.add_argument("-v", "--version", action="store_true")

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
        help=(
            "Path to the MIRO executable (.exe on Windows, .app on macOS or"
            " .AppImage on Linux)"
        ),
        default=None,
    )
    miro_group.add_argument(
        "--skip-execution",
        help="Whether to skip model execution",
        action="store_true",
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

    shutil.copy(args.name, gamspy_base_dir + os.sep + "user_license.txt")


def uninstall_license():
    gamspy_base_dir = utils._get_gamspy_base_directory()
    os.unlink(gamspy_base_dir + os.sep + "user_license.txt")


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
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise GamspyException(
                f"Could not install gamspy-{solver_name}: {e.stderr.decode('utf-8')}"
            ) from e
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

    with open(
        dist_info_path + os.sep + "RECORD", "a", encoding="utf-8"
    ) as record:
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
            ) from e

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
        raise GamspyException(
            f"Could not uninstall gamspy-base: {e.output}"
        ) from e

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
                    "gamspy",
                    "install",
                    "solver",
                    solver.lower(),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise GamspyException(
                "Could not uninstall"
                f" gamspy-{solver.lower()}=={gamspy_base.version}: {e.output}"
            ) from e


def list_solvers(args: argparse.Namespace):
    component = args.component

    if component == "solvers":
        if args.all:
            solvers = utils.getAvailableSolvers()
            print(f"Available solvers: {solvers}\n")
            print("Model types that can be solved with the solver:\n")
            for solver in solvers:
                try:
                    print(f"{solver}: {SOLVER_CAPABILITIES[solver]}")
                except KeyError:
                    ...
            return

        solvers = utils.getInstalledSolvers()
        print(f"Installed solvers: {solvers}\n")
        print("Model types that can be solved with the solver:\n")
        for solver in solvers:
            try:
                print(f"{solver}: {SOLVER_CAPABILITIES[solver]}")
            except KeyError:
                ...
    else:
        raise ValidationError(
            "gamspy list requires a third argument (solvers)."
        )


def run(args: argparse.Namespace):
    component = args.component

    if component == "miro":
        model = os.path.abspath(args.model)
        mode = args.mode
        path = os.getenv("MIRO_PATH", None)

        if args.path is not None:
            path = args.path

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
        if not args.skip_execution:
            subprocess_env = os.environ.copy()
            subprocess_env["MIRO"] = "1"

            try:
                subprocess.run(
                    [sys.executable, model], env=subprocess_env, check=True
                )
            except subprocess.CalledProcessError:
                return

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

    return None


def show(args: argparse.Namespace):
    if args.component == "license":
        show_license()
    elif args.component == "base":
        show_base()
    else:
        raise ValidationError(
            "`gamspy show` requires a third argument (license or base)."
        )


def show_license():
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    userlice_path = os.path.join(gamspy_base.directory, "user_license.txt")
    demolice_path = os.path.join(gamspy_base.directory, "gamslice.txt")
    lice_path = (
        userlice_path if os.path.exists(userlice_path) else demolice_path
    )
    with open(lice_path) as license_file:
        print(license_file.read())


def show_base():
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    print(gamspy_base.directory)


def uninstall(args: argparse.Namespace):
    if args.component == "license":
        uninstall_license()
    elif args.component == "solver":
        uninstall_solver(args)


def print_version():
    import gams

    import gamspy

    print(f"GAMSPy version: {gamspy.__version__}")
    print(f"GAMS version: {gams.__version__}")

    try:
        import gamspy_base

        print(f"gamspy_base version: {gamspy_base.__version__}")
    except ModuleNotFoundError:
        ...


def main():
    """
    Entry point for gamspy command line application.
    """
    args = get_args()
    if args.version:
        print_version()
    elif args.command == "install":
        install(args)
    elif args.command == "run":
        run(args)
    elif args.command == "show":
        show(args)
    elif args.command == "update":
        update()
    elif args.command == "list":
        list_solvers(args)
    elif args.command == "uninstall":
        uninstall(args)
