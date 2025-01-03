from __future__ import annotations
import importlib
import shutil

from typing import Annotated, Iterable, Union, List

import typer
from gamspy.exceptions import GamspyException, ValidationError
import gamspy.utils as utils
import os
import subprocess
from .util import add_solver_entry

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To install licenses and solvers.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy install license <access_code> or <path/to/license/file> | gamspy install solver <solver_name>",
    context_settings={"help_option_names": ["-h", "--help"]},
)

@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy install license <access_code> or <path/to/license/file>",
    short_help="To install a new license"
)
def license(
    license: Annotated[str, typer.Argument(help="access code or path to the license file.")],
    uses_port: Annotated[Union[int, None], typer.Option("--uses-port", help="Interprocess communication starting port.")] = None
):
    import json
    from urllib.parse import urlencode

    import urllib3

    os.makedirs(utils.DEFAULT_DIR, exist_ok=True)

    is_alp = not os.path.isfile(license)

    if is_alp and len(license) != 36:
        raise ValidationError(
            f"Access code is a 36 character string or an absolute path to the "
            f"license file but {len(license)} character string ({license}) provided."
        )

    gamspy_base_dir = utils._get_gamspy_base_directory()
    license_path = os.path.join(utils.DEFAULT_DIR, "gamspy_license.txt")

    if is_alp:
        alp_id = license
        encoded_args = urlencode({"access_token": alp_id})
        request = urllib3.request(
            "GET", "https://license.gams.com/license-type?" + encoded_args
        )
        if request.status != 200:
            raise ValidationError(
                f"License server did not respond in an expected way. Request status: {request.status}. Please try again."
            )

        data = request.data.decode("utf-8", errors="replace")
        cmex_type = json.loads(data)["cmex_type"]
        if not cmex_type.startswith("gamspy"):
            raise ValidationError(
                f"Given access code `{alp_id} ({cmex_type})` is not valid for GAMSPy. "
                "Make sure that you use a GAMSPy license, not a GAMS license."
            )

        command = [os.path.join(gamspy_base_dir, "gamsgetkey"), alp_id]

        if uses_port:
            command.append("-u")
            command.append(str(uses_port))

        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
        )
        if process.returncode:
            raise ValidationError(process.stderr)

        license_text = process.stdout
        lines = license_text.splitlines()
        license_type = lines[0][54]
        if license_type == "+":
            if lines[2][:2] not in ["00", "07", "08", "09"]:
                raise ValidationError(
                    f"Given access code `{alp_id}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )
        else:
            if lines[2][8:10] not in ["00", "07", "08", "09"]:
                raise ValidationError(
                    f"Given access code `{alp_id}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )

        with open(license_path, "w", encoding="utf-8") as file:
            file.write(license_text)
    else:
        with open(license) as file:
            lines = file.read().splitlines()

        license_type = lines[0][54]
        if license_type == "+":
            if lines[2][:2] not in ["00", "07", "08", "09"]:
                raise ValidationError(
                    f"Given license file `{license}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )
        else:
            if lines[2][8:10] not in ["00", "07", "08", "09"]:
                raise ValidationError(
                    f"Given license file `{license}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )

        shutil.copy(license, license_path)

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

@app.command(
    short_help="To install solvers",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy install solver <solver_name>"
)
def solver(
    solver_names: Annotated[
        Union[List[str], None], 
        typer.Argument(default=None, help="solver names to be installed")
    ] = None,
    install_all_solvers: Annotated[
        Union[bool, None], 
        typer.Option("--install-all-solvers", help="Installs all available add-on solvers.")
    ] = None,
    existing_solvers: Annotated[
        Union[bool, None], 
        typer.Option("--existing-solvers", help="Reinstalls previously installed add-on solvers.")
    ] = None,
    skip_pip_install: Annotated[
        Union[bool, None], 
        typer.Option("--skip-pip-install", "-s", help="If you already have the solver installed, skip pip install and update gamspy installed solver list.")
    ] = None
):
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        e.msg = "You must first install gamspy_base to use this functionality"
        raise e

    addons_path = os.path.join(utils.DEFAULT_DIR, "solvers.txt")
    os.makedirs(utils.DEFAULT_DIR, exist_ok=True)

    def install_addons(addons: Iterable[str]):
        for item in addons:
            solver_name = item.lower()

            if solver_name.upper() not in utils.getAvailableSolvers():
                raise ValidationError(
                    f'Given solver name ("{solver_name}") is not valid. Available'
                    f" solvers that can be installed: {utils.getAvailableSolvers()}"
                )

            if not skip_pip_install:
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

            with open(addons_path, "w") as file:
                if solver_name.upper() not in installed:
                    file.write(
                        "\n".join(installed)
                        + "\n"
                        + solver_name.upper()
                        + "\n"
                    )

    if install_all_solvers:
        available_solvers = utils.getAvailableSolvers()
        installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)
        diff = []
        for solver in available_solvers:
            if solver not in installed_solvers:
                diff.append(solver)

        install_addons(diff)
        return

    if existing_solvers:
        try:
            with open(addons_path) as file:
                solvers = file.read().splitlines()
                install_addons(solvers)
                return
        except FileNotFoundError as e:
            raise ValidationError("No existing add-on solvers found!") from e

    if solver_names is None:
        raise ValidationError(
            "Solver name is missing: `gamspy install solver <solver_name>`"
        )

    install_addons(solver_names)

if __name__ == "__main__":
    app()
