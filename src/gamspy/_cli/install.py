from __future__ import annotations
import importlib
import shutil

import sys
from typing import Annotated, Iterable, Optional, Union

import certifi
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
    checkout_duration: Optional[int] = typer.Option(None, "--checkout-duration", "-c", help="Specify a duration in hours to checkout a session."),
    renew: Optional[str] = typer.Option(None, "--renew", "-r", help="Specify a file path to a license file to extend a session."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Specify a file path to write the license file."),
    uses_port: Annotated[Union[int, None], typer.Option("--uses-port", help="Interprocess communication starting port.")] = None,
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
        http = urllib3.PoolManager(
            cert_reqs="CERT_REQUIRED",
            ca_certs=certifi.where()
        )
        encoded_args = urlencode({"access_token": alp_id})
        request = http.request(
            "GET", "https://license.gams.com/license-type?" + encoded_args
        )
        if request.status != 200:
            raise ValidationError(
                f"License server did not respond in an expected way. Request status: {request.status}. Reason: {request.data.decode('utf-8', errors='replace')}"
            )

        data = request.data.decode("utf-8", errors="replace")
        cmex_type = json.loads(data)["cmex_type"]
        if cmex_type not in ("gamspy", "gamspy++", "gamsall"):
            raise ValidationError(
                f"Given access code `{alp_id} ({cmex_type})` is not valid for GAMSPy. "
                "Make sure that you use a GAMSPy license, not a GAMS license."
            )

        command = [os.path.join(gamspy_base_dir, "gamsgetkey"), alp_id]

        if uses_port:
            command.append("-u")
            command.append(str(uses_port))

        if checkout_duration:
            command.append("-c")
            command.append(str(checkout_duration))

        if checkout_duration:
            command.append("-r")
            command.append(str(renew))
        
        if output:
            command.append("-o")
            command.append(output)

        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
        )
        if process.returncode:
            raise ValidationError(process.stderr)

        if output:
            with open(output) as file:
                license_text = file.read()
        else:
            license_text = process.stdout
            
        lines = license_text.splitlines()
        license_type = lines[0][54]
        if license_type == "+":
            if lines[2][:2] not in ("00", "07", "08", "09"):
                raise ValidationError(
                    f"Given access code `{alp_id}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )
        else:
            if lines[2][8:10] not in ("00", "07", "08", "09"):
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
            if lines[2][:2] not in ("00", "07", "08", "09"):
                raise ValidationError(
                    f"Given license file `{license}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )
        else:
            if lines[2][8:10] not in ("00", "07", "08", "09"):
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

        record.write("\n".join(lines) + "\n")

@app.command(
    short_help="To install solvers",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy install solver <solver_name>"
)
def solver(
    solver: list[str] = typer.Argument(
        None,
        help="solver names to be installed",
        autocompletion=lambda: [s.lower() for s in utils.getAvailableSolvers()]
    ),
    install_all_solvers: bool = typer.Option(
        False,
        "--install-all-solvers",
        help="Installs all available add-on solvers."
    ),
    existing_solvers: bool = typer.Option(
        False,
        "--existing-solvers",
        help="Reinstalls previously installed add-on solvers."
    ),
    skip_pip_install: bool = typer.Option(
        False,
        "--skip-pip-install",
        "-s",
        help="If you already have the solver installed, skip pip install and update gamspy installed solver list."
    ),
    use_uv: bool = typer.Option(
        False,
        "--use-uv",
        help="Use uv instead of pip to install solvers."
    ),
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

            if solver_name.upper() in gamspy_base.default_solvers:
                print(f"`{solver_name}` is a default solver, skipping...")
                continue
            installable_solvers = utils.getInstallableSolvers(gamspy_base.directory)
            if solver_name.upper() not in installable_solvers:
                raise ValidationError(
                    f'Given solver name ("{solver_name}") is not valid. Available'
                    f" solvers that can be installed: {installable_solvers}")
            
            installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)
            if solver_name.upper() in installed_solvers:
                print(f"`{solver_name}` is already installed, skipping...")
                continue

            if not skip_pip_install:

                solver_version = gamspy_base.__version__
                # install specified solver
                if use_uv:
                    command = [
                        "uv",
                        "pip",
                        "--python-preference",
                        "only-system",
                        "install",
                        f"gamspy-{solver_name}=={solver_version}",
                        "--force-reinstall",
                    ]
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f"gamspy-{solver_name}=={solver_version}",
                        "--force-reinstall",
                    ]
                try:
                    _ = subprocess.run(command, check=True, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    raise GamspyException(
                        f"Could not install gamspy-{solver_name}. Please check your internet connection. If it's not related to your internet connection, PyPI servers might be down. Please retry it later. Here is the error message of pip:\n\n{e.stderr.decode('utf-8')}"
                    ) from e
            else:
                try:
                    solver_lib = importlib.import_module(
                        f"gamspy_{solver_name}"
                    )
                except ModuleNotFoundError as e:
                    e.msg = f"You must install gamspy-{solver_name} first!"
                    raise e

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
                    file.write("\n".join(installed + [solver_name.upper()]))

    if install_all_solvers:
        available_solvers = utils.getAvailableSolvers()
        installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)
        diff = []
        for available_solver in available_solvers:
            if available_solver not in installed_solvers:
                diff.append(available_solver)

        install_addons(diff)
        return

    if existing_solvers:
        try:
            with open(addons_path) as file:
                solvers = file.read().splitlines()
                solvers = [solver for solver in solvers if solver != "" and solver != "\n"]
                install_addons(solvers)
                return
        except FileNotFoundError as e:
            raise ValidationError("No existing add-on solvers found!") from e

    if solver is None:
        raise ValidationError(
            "Solver name is missing: `gamspy install solver <solver_name>`"
        )

    install_addons(solver)

if __name__ == "__main__":
    app()
