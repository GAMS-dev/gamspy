from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Annotated

import certifi
import typer

from .util import (
    CUOPT,
    cli_error,
    installation_error,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

app = typer.Typer(
    short_help="To install licenses and solvers.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy install license <access_code> or <path/to/license/file> | gamspy install solver <solver_name>",
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Deprecated solvers are only installed when they are asked for by name.
_DEPRECATED_SOLVERS = frozenset({"EXAMINER2"})


def install_license(
    license: str,
    *,
    checkout_duration: int | None = None,
    server: str = "https://license.gams.com",
    port: int | None = None,
    output: str | None = None,
    uses_port: int | None = None,
    use_uv: bool = False,
) -> str:
    """
    Installs a GAMSPy license. Equivalent of ``gamspy install license``.

    Parameters
    ----------
    license : str
        Access code or path to the license file.
    checkout_duration : int, optional
        Duration in hours to checkout a session.
    server : str, optional
        License server address.
    port : int, optional
        Port for the license server connection.
    output : str, optional
        File path to write the license file. By default, the license is written
        to the GAMSPy license path.
    uses_port : int, optional
        Interprocess communication starting port.
    use_uv : bool, optional
        Use uv instead of pip to install the default solvers of the license.

    Returns
    -------
    str
        Path of the installed license file.

    Raises
    ------
    CLIError
        In case the license is invalid or cannot be retrieved.

    Examples
    --------
    >>> from gamspy import cli
    >>> license_path = cli.install_license("<access_code>")  # doctest: +SKIP

    """
    import json

    import requests

    import gamspy.utils as utils
    from gamspy.exceptions import CLIError

    os.makedirs(utils.DEFAULT_DIR, exist_ok=True)

    is_alp = not os.path.isfile(license)

    if is_alp and len(license) != 36:
        raise cli_error(
            f"Invalid access code or license file ({license}). Please enter a 36-character "
            f"access code, or ensure the absolute path to your license file is correct and the file exists."
        )

    gamspy_base_dir = utils._get_gamspy_base_directory()
    license_path = os.path.join(utils.DEFAULT_DIR, "gamspy_license.txt")
    certificate_path = os.path.join(gamspy_base_dir, "gamslice.crt")

    if is_alp:
        alp_id = license

        # Make cmex_type check only for GAMS license server.
        if server == "https://license.gams.com":
            params = {"access_token": alp_id}
            request = requests.get(f"{server}/license-type", params=params)
            if request.status_code != 200:
                raise cli_error(
                    f"License server did not respond in an expected way. Request status: {request.status_code}. Reason: {request.content.decode('utf-8', errors='replace')}"
                )

            data = request.content.decode("utf-8", errors="replace")
            cmex_type = json.loads(data)["cmex_type"]
            if cmex_type not in (
                "demo",
                "community",
                "nosolve",
                "gamspy",
                "gamspy++",
                "gamsall",
            ):
                raise cli_error(
                    f"Given access code `{alp_id} ({cmex_type})` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )

        command = [
            os.path.join(gamspy_base_dir, "gamsgetkey"),
            alp_id,
            "-s",
            server,
        ]

        # On-prem license server licenses start with abcdef12
        if alp_id.startswith("abcdef12"):
            command.append("-g")
            command.append(certificate_path)

        if port:
            command.append("-p")
            command.append(str(port))

        if uses_port:
            command.append("-u")
            command.append(str(uses_port))

        if checkout_duration:
            command.append("-c")
            command.append(str(checkout_duration))

        if output:
            command.append("-o")
            command.append(output)
        else:
            command.append("-o")
            command.append(license_path)

        environment_variables = os.environ.copy()
        if "CURL_CA_BUNDLE" not in environment_variables:
            environment_variables["CURL_CA_BUNDLE"] = certifi.where()

        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment_variables,
        )
        if process.returncode:
            raise cli_error(process.stderr)

        license_file_path = output if output else license_path
        with open(license_file_path) as file:
            license_text = file.read()

        lines = license_text.splitlines()
        try:
            license_type = lines[0][54]
        except (Exception, IndexError) as e:
            raise cli_error(
                f"Invalid license text. \nLicense text: {license_text}\nstdout: {process.stdout}\nstderr: {process.stderr}"
            ) from e

        if license_type == "+":
            if lines[2][:2] not in ("00", "06", "07", "08", "09"):
                raise cli_error(
                    f"Given access code `{alp_id}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )
        else:
            if lines[2][8:10] not in ("00", "06", "07", "08", "09"):
                raise cli_error(
                    f"Given access code `{alp_id}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )
    else:
        with open(license) as file:
            lines = file.read().splitlines()

        license_type = lines[0][54]
        if license_type == "+":
            if lines[2][:2] not in ("00", "06", "07", "08", "09"):
                raise cli_error(
                    f"Given license file `{license}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )
        else:
            if lines[2][8:10] not in ("00", "06", "07", "08", "09"):
                raise cli_error(
                    f"Given license file `{license}` is not valid for GAMSPy. "
                    "Make sure that you use a GAMSPy license, not a GAMS license."
                )

        shutil.copy(license, output if output else license_path)

    # Free personal license has different default solvers. Install them.
    if "FREEPERSONAL" in lines[4]:
        default_solvers = ["HIGHS", "IPOPT", "MILES", "SHOT", "RESHOP", "SCIP"]
        try:
            install_solver(default_solvers, use_uv=use_uv)
        except CLIError as e:
            # The license is installed, only its default solvers are missing.
            names = " ".join(solver.lower() for solver in default_solvers)
            typer.echo(
                "Warning: The default solvers of the license could not be installed: "
                f"{e.message}\nInstall them with `gamspy install solver {names}`.",
                err=True,
            )

    return output if output else license_path


@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy install license <access_code> or <path/to/license/file>",
    short_help="To install a new license",
)
def license(
    license: Annotated[
        str, typer.Argument(help="access code or path to the license file.")
    ],
    checkout_duration: int | None = typer.Option(
        None,
        "--checkout-duration",
        "-c",
        help="Specify a duration in hours to checkout a session.",
    ),
    server: str = typer.Option(
        "https://license.gams.com",
        "--server",
        "-s",
        help="License server address.",
    ),
    port: int | None = typer.Option(
        None, "--port", "-p", help="Port for the license server connection."
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Specify a file path to write the license file.",
    ),
    uses_port: Annotated[
        int | None,
        typer.Option("--uses-port", help="Interprocess communication starting port."),
    ] = None,
    use_uv: bool = typer.Option(
        False, "--use-uv", help="Use uv instead of pip to install solvers."
    ),
) -> None:
    install_license(
        license,
        checkout_duration=checkout_duration,
        server=server,
        port=port,
        output=output,
        uses_port=uses_port,
        use_uv=use_uv,
    )


def _add_installed_addon(addons_path: str, solver_name: str) -> None:
    try:
        with open(addons_path) as file:
            installed = file.read().splitlines()
            installed = [solver for solver in installed if solver not in {"", "\n"}]
    except FileNotFoundError:
        installed = []

    if solver_name.upper() not in installed:
        installed.append(solver_name.upper())

    with open(addons_path, "w") as file:
        file.write("\n".join(installed) + "\n")


def complete_solver_names(ctx: typer.Context, incomplete: str) -> list[str]:
    import gamspy.utils as utils

    return [
        s.lower()
        for s in utils.getInstallableSolvers()
        if s.startswith(incomplete.upper())
    ]


def install_solver(
    solvers: str | Iterable[str] | None = None,
    *,
    install_all_solvers: bool = False,
    existing_solvers: bool = False,
    licensed: bool = False,
    skip_pip_install: bool = False,
    use_uv: bool = False,
) -> None:
    """
    Installs add-on solvers. Equivalent of ``gamspy install solver``.

    Parameters
    ----------
    solvers : str | Iterable[str], optional
        Name(s) of the solvers to be installed.
    install_all_solvers : bool, optional
        Installs all available add-on solvers.
    existing_solvers : bool, optional
        Reinstalls previously installed add-on solvers.
    licensed : bool, optional
        Installs all add-on solvers permitted by the active license that are not yet installed.
    skip_pip_install : bool, optional
        If you already have the solver installed, skip pip install and update gamspy installed solver list.
    use_uv : bool, optional
        Use uv instead of pip to install solvers.

    Raises
    ------
    CLIError
        In case a solver name is invalid or a solver cannot be installed.

    Examples
    --------
    >>> from gamspy import cli
    >>> cli.install_solver("highs")  # doctest: +SKIP
    >>> cli.install_solver(["highs", "scip"])  # doctest: +SKIP
    >>> cli.install_solver(licensed=True)  # doctest: +SKIP

    """
    import gamspy_base

    import gamspy.utils as utils

    solver = [solvers] if isinstance(solvers, str) else list(solvers or [])

    # gamspy_base.install_solver checks for pip (or uv) before it installs anything.
    addons_path = os.path.join(utils.DEFAULT_DIR, "solvers.txt")
    os.makedirs(utils.DEFAULT_DIR, exist_ok=True)

    def install_addons(addons: Iterable[str]) -> None:
        installable_solvers = utils.getInstallableSolvers()
        for item in addons:
            solver_name = item.lower()

            if solver_name.upper() in gamspy_base.default_solvers:
                print(f"`{solver_name}` is a default solver, skipping...")
                continue

            # gamspy_base explains why cuOpt cannot be installed on this platform.
            if solver_name != CUOPT and solver_name.upper() not in installable_solvers:
                raise cli_error(
                    f'Given solver name ("{solver_name}") is not valid. Available'
                    f" solvers that can be installed: {installable_solvers}"
                )

            if solver_name == "examiner2":
                typer.echo(
                    "Warning: `examiner2` solver is deprecated and will be removed in a future release. Use `examiner` instead.",
                    err=True,
                )

            try:
                gamspy_base.install_solver(
                    solver_name,
                    use_uv=use_uv,
                    install_package=not skip_pip_install,
                )
            except (
                gamspy_base.InstallationError,
                ValueError,
                ModuleNotFoundError,
            ) as e:
                raise installation_error(e) from e

            # Remembered for --existing-solvers after an upgrade of gamspy_base
            _add_installed_addon(addons_path, solver_name)

    if install_all_solvers:
        available_solvers = utils.getAvailableSolvers()
        installed_solvers = gamspy_base.get_solver_capabilities()
        diff = []
        for available_solver in available_solvers:
            # cuOpt requires an NVIDIA GPU and downloads more than a gigabyte,
            # hence it is only installed when it is asked for by name.
            if available_solver == "CUOPT" or available_solver in _DEPRECATED_SOLVERS:
                continue

            if available_solver not in installed_solvers:
                diff.append(available_solver)

        install_addons(diff)
        return

    if existing_solvers:
        try:
            with open(addons_path) as file:
                solvers = file.read().splitlines()
                solvers = [solver for solver in solvers if solver not in {"", "\n"}]
                install_addons(solvers)
                return
        except FileNotFoundError as e:
            raise cli_error("No existing add-on solvers found!") from e

    if licensed:
        license_path = utils._get_license_path(gamspy_base.directory)
        licensed_solvers = set(gamspy_base.get_licensed_solvers(license_path))
        installable_solvers = set(utils.getInstallableSolvers())
        installed_solvers = set(gamspy_base.get_solver_capabilities())

        diff = sorted(
            (licensed_solvers & installable_solvers)
            - installed_solvers
            - _DEPRECATED_SOLVERS
        )
        if not diff:
            typer.echo("All licensed solvers are already installed.")
            return

        install_addons(diff)
        return

    if not solver:
        raise cli_error("Solver name is missing: `gamspy install solver <solver_name>`")

    install_addons(solver)


@app.command(
    short_help="To install solvers",
    help=(
        "[bold][yellow]Examples[/yellow][/bold]: gamspy install solver <solver_name> | "
        "gamspy install solver --licensed"
    ),
)
def solver(
    solver: list[str] = typer.Argument(  # noqa: B008
        None,
        help="solver names to be installed",
        autocompletion=complete_solver_names,
    ),
    install_all_solvers: bool = typer.Option(
        False,
        "--install-all-solvers",
        "--all",
        "-a",
        help="Installs all available add-on solvers.",
    ),
    existing_solvers: bool = typer.Option(
        False,
        "--existing-solvers",
        help="Reinstalls previously installed add-on solvers.",
    ),
    licensed: bool = typer.Option(
        False,
        "--licensed",
        help="Installs all add-on solvers permitted by the active license that are not yet installed.",
    ),
    skip_pip_install: bool = typer.Option(
        False,
        "--skip-pip-install",
        "-s",
        help="If you already have the solver installed, skip pip install and update gamspy installed solver list.",
    ),
    use_uv: bool = typer.Option(
        False, "--use-uv", help="Use uv instead of pip to install solvers."
    ),
) -> None:
    install_solver(
        solver,
        install_all_solvers=install_all_solvers,
        existing_solvers=existing_solvers,
        licensed=licensed,
        skip_pip_install=skip_pip_install,
        use_uv=use_uv,
    )


if __name__ == "__main__":
    app()
