from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from typing import Iterable

import gamspy.utils as utils
from gamspy.exceptions import GamspyException, ValidationError

from .util import add_solver_entry, remove_solver_entry

USAGE = """gamspy [-h] [-v]
       gamspy install license <access_code> or <path/to/license/file> [--uses-port <port>]
       gamspy uninstall license
       gamspy install solver <solver_name> [--skip-pip-install] [--existing-solvers]
       gamspy uninstall solver <solver_name> [--skip-pip-uninstall]
       gamspy list solvers [--all]
       gamspy show license
       gamspy show base
       gamspy probe [-j <json_output_path>]
       gamspy retrieve license <access_code> [-i <json_file_path>] [-o <output_path>] 
       gamspy run miro [--path <path_to_miro>] [--model <path_to_model>]
"""


def get_args():
    parser = argparse.ArgumentParser(
        prog="gamspy", usage=USAGE, description="GAMSPy CLI"
    )

    parser.add_argument(
        "command",
        choices=[
            "install",
            "list",
            "probe",
            "retrieve",
            "run",
            "show",
            "update",
            "uninstall",
        ],
        type=str,
        nargs="?",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "component", type=str, nargs="?", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("name", type=str, nargs="*", help=argparse.SUPPRESS)
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Shows the version of GAMSPy, GAMS and gamspy_base",
    )

    install_license_group = parser.add_argument_group(
        "gamspy install license <access_code> or <path/to/license/file>",
        description="Options for installing a license.",
    )
    install_license_group.add_argument(
        "--uses-port",
        help="Interprocess communication starting port.",
    )

    _ = parser.add_argument_group(
        "gamspy uninstall license",
        description="Command to uninstall user license.",
    )

    install_solver_group = parser.add_argument_group(
        "gamspy install solver <solver_name>",
        description="Options for installing solvers",
    )
    install_solver_group.add_argument(
        "--skip-pip-install",
        "-s",
        action="store_true",
        help=(
            "If you already have the solver installed, skip pip install and"
            " update gamspy installed solver list."
        ),
    )
    install_solver_group.add_argument(
        "--existing-solvers",
        "-e",
        action="store_true",
        help="Reinstalls previously installed addon solvers.",
    )
    install_solver_group.add_argument(
        "--install-all-solvers",
        action="store_true",
        help="Installs all available addon solvers.",
    )

    uninstall_solver_group = parser.add_argument_group(
        "gamspy uninstall solver <solver_name>",
        description="Options for uninstalling solvers",
    )
    uninstall_solver_group.add_argument(
        "--skip-pip-uninstall",
        "-u",
        action="store_true",
        help=(
            "If you don't want to uninstall the package of the solver, skip"
            " uninstall and update gamspy installed solver list."
        ),
    )
    install_solver_group.add_argument(
        "--uninstall-all-solvers",
        action="store_true",
        help="Uninstalls all installed addon solvers.",
    )

    list_group = parser.add_argument_group(
        "gamspy list solvers", description="`gamspy list solvers` options"
    )
    list_group.add_argument(
        "-a", "--all", action="store_true", help="Shows all available solvers."
    )

    probe_group = parser.add_argument_group(
        "gamspy probe", description="`gamspy probe` options"
    )
    probe_group.add_argument(
        "--json-out", "-j", help="Output path for the json file."
    )

    retrieve_group = parser.add_argument_group(
        "gamspy retrieve license <access_code>",
        description="`gamspy retrieve license` options",
    )
    retrieve_group.add_argument(
        "--output",
        "-o",
        help="Output path for the license file.",
    )
    retrieve_group.add_argument(
        "--input",
        "-i",
        help="json file path to retrieve a license based on node information.",
    )

    miro_group = parser.add_argument_group(
        "gamspy run miro", description="`gamspy run miro` options"
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

    return parser.parse_args()


def install_license(args: argparse.Namespace):
    os.makedirs(utils.DEFAULT_DIR, exist_ok=True)

    if not args.name or len(args.name) > 1:
        raise ValidationError(
            "License is missing: `gamspy install license <access_code> or <path/to/license/file>`"
        )

    license = args.name[0]
    is_alp = not os.path.isfile(license)

    if is_alp and len(license) != 36:
        raise ValidationError(
            f"Access code is a 36 character string or an absolute path to the "
            f"license file but {len(license)} character string ({license}) provided."
        )

    gamspy_base_dir = utils._get_gamspy_base_directory()

    if is_alp:
        command = [os.path.join(gamspy_base_dir, "gamsgetkey"), license]

        if args.uses_port:
            command.append("-u")
            command.append(str(args.uses_port))

        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
        )
        if process.returncode:
            raise ValidationError(process.stderr)

        with open(
            os.path.join(utils.DEFAULT_DIR, "gamspy_license.txt"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(process.stdout)
    else:
        shutil.copy(
            license, os.path.join(utils.DEFAULT_DIR, "gamspy_license.txt")
        )


def uninstall_license():
    try:
        os.unlink(os.path.join(utils.DEFAULT_DIR, "gamspy_license.txt"))
    except FileNotFoundError:
        ...


def install_solver(args: argparse.Namespace):
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    addons_path = os.path.join(utils.DEFAULT_DIR, "solvers.txt")

    def install_addons(addons: Iterable[str]):
        for item in addons:
            solver_name = item.lower()

            if solver_name.upper() not in utils.getAvailableSolvers():
                raise ValidationError(
                    f'Given solver name ("{solver_name}") is not valid. Available'
                    f" solvers that can be installed: {utils.getAvailableSolvers()}"
                )

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
                    solver_lib = importlib.import_module(
                        f"gamspy_{solver_name}"
                    )
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
            for file in file_paths:
                shutil.copy(file, gamspy_base_dir)

            files = solver_lib.files
            verbatims = [solver_lib.verbatim]
            append_dist_info(files, gamspy_base_dir)
            add_solver_entry(gamspy_base_dir, solver_name, verbatims)

            try:
                with open(addons_path) as file:
                    installed = file.read().splitlines()
                    installed = [
                        solver
                        for solver in installed
                        if solver != "" and solver != "\n"
                    ]
            except FileNotFoundError:
                installed = []

            with open(addons_path, "a") as file:
                if solver_name.upper() not in installed:
                    file.write(solver_name.upper() + "\n")

    if args.install_all_solvers:
        available_solvers = utils.getAvailableSolvers()
        installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)
        diff = []
        for solver in available_solvers:
            if solver not in installed_solvers:
                diff.append(solver)

        install_addons(diff)
        return

    if args.existing_solvers:
        try:
            with open(addons_path) as file:
                solvers = file.read().splitlines()
                install_addons(solvers)
                return
        except FileNotFoundError as e:
            raise ValidationError("No existing addon solvers found!") from e

    if not args.name:
        raise ValidationError(
            "Solver name is missing: `gamspy install solver <solver_name>`"
        )

    install_addons(args.name)


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
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    addons_path = os.path.join(utils.DEFAULT_DIR, "solvers.txt")

    def remove_addons(addons: Iterable[str]):
        for item in addons:
            solver_name = item.lower()

            installed_solvers = utils.getInstalledSolvers(
                gamspy_base.directory
            )
            if solver_name.upper() not in installed_solvers:
                raise ValidationError(
                    f'Given solver name ("{solver_name}") is not valid. Installed'
                    f" solvers solvers that can be uninstalled: {installed_solvers}"
                )

            if not args.skip_pip_uninstall:
                # uninstall specified solver
                try:
                    _ = subprocess.run(
                        ["pip", "uninstall", f"gamspy-{solver_name}", "-y"],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    raise GamspyException(
                        f"Could not uninstall gamspy-{solver_name}: {e.output}"
                    ) from e

            # do not delete files from gamspy_base as other solvers might depend on it
            gamspy_base_dir = utils._get_gamspy_base_directory()
            remove_solver_entry(gamspy_base_dir, solver_name)

            try:
                with open(addons_path) as file:
                    installed = file.read().splitlines()
            except FileNotFoundError:
                installed = []

            try:
                installed.remove(solver_name.upper())
            except ValueError as e:
                raise ValidationError(
                    f"Cannot remove `{solver_name}` which was not installed before!"
                ) from e

            with open(addons_path, "w") as file:
                file.write("\n".join(installed) + "\n")

    if args.uninstall_all_solvers:
        try:
            with open(addons_path) as file:
                solvers = file.read().splitlines()
                solvers = [
                    solver
                    for solver in solvers
                    if solver != "" and solver != "\n"
                ]
                remove_addons(solvers)

        except FileNotFoundError as e:
            raise ValidationError("No existing addon solvers found!") from e

        # All addon solvers are gone.
        return

    if not args.name:
        raise ValidationError(
            "Solver name is missing: `gamspy uninstall solver <solver_name>`"
        )

    remove_addons(args.name)


def install(args: argparse.Namespace):
    if args.component == "license":
        install_license(args)
    elif args.component == "solver":
        install_solver(args)
    else:
        raise ValidationError(
            "`gamspy install` requires a third argument (license or solver)."
        )


def update():
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    prev_installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)

    try:
        _ = subprocess.run(
            [
                "pip",
                "install",
                f"gamspy-base=={gamspy_base.__version__}",
                "--force-reinstall",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise GamspyException(
            f"Could not uninstall gamspy-base: {e.output}"
        ) from e

    new_installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)

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
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    component = args.component

    if component == "solvers":
        capabilities = utils.getSolverCapabilities(gamspy_base.directory)
        if args.all:
            solvers = utils.getAvailableSolvers()
            print("Available Solvers")
            print("=" * 17)
            print(", ".join(solvers))
            print(
                "\nModel types that can be solved with the installed solvers:\n"
            )
            for solver in solvers:
                try:
                    print(f"{solver:<10}: {', '.join(capabilities[solver])}")
                except KeyError:
                    ...
            return

        solvers = utils.getInstalledSolvers(gamspy_base.directory)
        print("Installed Solvers")
        print("=" * 17)
        print(", ".join(solvers))

        print("\nModel types that can be solved with the installed solvers")
        print("=" * 57)
        for solver in solvers:
            try:
                print(f"{solver:<10}: {', '.join(capabilities[solver])}")
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

        if path is None:
            path = args.path if args.path is not None else discover_miro()

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


def discover_miro():
    system = platform.system()
    if system == "Linux":
        return None

    home = os.path.expanduser("~")
    standard_locations = {
        "Darwin": [
            os.path.join(
                "/",
                "Applications",
                "GAMS MIRO.app",
                "Contents",
                "MacOS",
                "GAMS MIRO",
            ),
            os.path.join(
                home,
                "Applications",
                "GAMS MIRO.app",
                "Contents",
                "MacOS",
                "GAMS MIRO",
            ),
        ],
        "Windows": [
            os.path.join(
                "C:\\", "Program Files", "GAMS MIRO", "GAMS MIRO.exe"
            ),
            os.path.join(
                home,
                "AppData",
                "Local",
                "Programs",
                "GAMS MIRO",
                "GAMS MIRO.exe",
            ),
        ],
    }

    if system in ["Darwin", "Windows"]:
        for location in standard_locations[system]:
            if os.path.isfile(location):
                return location

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

    license_path = utils._get_license_path(gamspy_base.directory)
    with open(license_path, encoding="utf-8") as license_file:
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
    else:
        raise ValidationError(
            "`gamspy uninstall` requires a third argument (license or solver)."
        )


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


def probe(args: argparse.Namespace):
    gamspy_base_dir = utils._get_gamspy_base_directory()
    process = subprocess.run(
        [os.path.join(gamspy_base_dir, "gamsprobe")],
        text=True,
        capture_output=True,
    )

    if process.returncode:
        raise ValidationError(process.stderr)

    print(process.stdout)

    if args.json_out:
        with open(args.json_out, "w") as file:
            file.write(process.stdout)


def retrieve(args: argparse.Namespace):
    if args.input is None or not os.path.isfile(args.input):
        raise ValidationError(
            f"Given path `{args.input}` is not a json file. Please use `gamspy retrieve license <access_code> -i <json_file_path>`"
        )

    if args.name is None:
        raise ValidationError(f"Given licence id `{args.name}` is not valid!")

    gamspy_base_dir = utils._get_gamspy_base_directory()
    process = subprocess.run(
        [
            os.path.join(gamspy_base_dir, "gamsgetkey"),
            args.name[0],
            "-i",
            args.input,
        ],
        text=True,
        capture_output=True,
    )

    if process.returncode:
        raise ValidationError(process.stderr)

    if args.output:
        with open(args.output, "w") as file:
            file.write(process.stdout)


def main():
    """
    Entry point for gamspy command line application.
    """
    args = get_args()
    if args.version:
        print_version()
    elif args.command == "install":
        install(args)
    elif args.command == "probe":
        probe(args)
    elif args.command == "retrieve":
        retrieve(args)
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
