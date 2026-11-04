from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING, Annotated

import certifi
import typer

from . import cuopt
from .license import get_licensed_solvers
from .util import cli_error, has_pip, has_uv, remove_solver_entry

if TYPE_CHECKING:
    from collections.abc import Iterable

app = typer.Typer(
    short_help="To uninstall licenses and solvers.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall license | gamspy uninstall solver <solver_name>",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _get_installed_addons(addons_path: str) -> list[str]:
    try:
        with open(addons_path) as file:
            return [solver for solver in file.read().splitlines() if solver]
    except FileNotFoundError:
        return []


def _remove_installed_addon(addons_path: str, solver_name: str) -> None:
    installed = _get_installed_addons(addons_path)
    try:
        installed.remove(solver_name.upper())
    except ValueError:
        return

    with open(addons_path, "w") as file:
        file.write("\n".join(installed) + "\n")


def _get_removable_solvers(addons_path: str) -> set[str]:
    import gamspy_base

    import gamspy.utils as utils

    installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)
    config_solvers = set(utils._get_config_solvers(gamspy_base.directory))
    return (
        set(installed_solvers)
        - set(gamspy_base.default_solvers)
        - (config_solvers - set(_get_installed_addons(addons_path)))
    )


def complete_solver_names(ctx: typer.Context, incomplete: str) -> list[str]:
    import gamspy_base

    import gamspy.utils as utils

    return [
        s.lower()
        for s in utils.getInstalledSolvers(gamspy_base.directory)
        if s.startswith(incomplete.upper())
    ]


def uninstall_license() -> None:
    """
    Uninstalls the current license. Equivalent of ``gamspy uninstall license``.

    Examples
    --------
    >>> from gamspy import cli
    >>> cli.uninstall_license()  # doctest: +SKIP

    """
    import gamspy.utils as utils

    try:
        os.unlink(os.path.join(utils.DEFAULT_DIR, "gamspy_license.txt"))
    except FileNotFoundError:
        ...


@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall license",
    short_help="To uninstall the current license",
)
def license() -> None:
    uninstall_license()


def uninstall_solver(
    solvers: str | Iterable[str] | None = None,
    *,
    uninstall_all_solvers: bool = False,
    unlicensed: bool = False,
    skip_pip_uninstall: bool = False,
    use_uv: bool = False,
) -> None:
    """
    Uninstalls add-on solvers. Equivalent of ``gamspy uninstall solver``.

    Parameters
    ----------
    solvers : str | Iterable[str], optional
        Name(s) of the solvers to be uninstalled.
    uninstall_all_solvers : bool, optional
        Uninstalls all add-on solvers.
    unlicensed : bool, optional
        Uninstalls installed add-on solvers no longer permitted by the active license.
    skip_pip_uninstall : bool, optional
        If you already have the solver uninstalled, skip pip uninstall and update gamspy installed solver list.
    use_uv : bool, optional
        Use uv instead of pip to uninstall solvers.

    Raises
    ------
    CLIError
        In case a solver name is invalid or a solver cannot be uninstalled.

    Examples
    --------
    >>> from gamspy import cli
    >>> cli.uninstall_solver("highs")  # doctest: +SKIP
    >>> cli.uninstall_solver(unlicensed=True)  # doctest: +SKIP

    """
    import gamspy_base

    import gamspy.utils as utils

    solver = [solvers] if isinstance(solvers, str) else list(solvers or [])
    requested = [name.lower() for name in solver]
    if not requested or any(name != cuopt.SOLVER_NAME for name in requested):
        if not use_uv and not has_pip():
            raise cli_error(
                "pip is not installed in your environment. Please install pip first or add --use-uv flag to uninstall solvers with uv."
            )

        if use_uv and not has_uv():
            raise cli_error(
                "uv is not installed in your machine. Please install uv first to uninstall solvers with --use-uv flag."
            )

    addons_path = os.path.join(utils.DEFAULT_DIR, "solvers.txt")
    environment_variables = os.environ.copy()
    if "CURL_CA_BUNDLE" not in environment_variables:
        environment_variables["CURL_CA_BUNDLE"] = certifi.where()

    def remove_addons(addons: Iterable[str]) -> None:
        for item in addons:
            solver_name = item.lower()

            removable_solvers = _get_removable_solvers(addons_path)
            if solver_name.upper() not in removable_solvers:
                raise cli_error(
                    f'Given solver name ("{solver_name}") is not valid. Installed'
                    f" solvers that can be uninstalled: {sorted(removable_solvers)}"
                )

            if solver_name == cuopt.SOLVER_NAME:
                cuopt.uninstall()
                _remove_installed_addon(addons_path, solver_name)
                continue

            if not skip_pip_uninstall:
                # uninstall specified solver
                if use_uv:
                    command = [
                        "uv",
                        "pip",
                        "uninstall",
                        f"gamspy-{solver_name}",
                    ]
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "pip",
                        "uninstall",
                        f"gamspy-{solver_name}",
                        "-y",
                    ]
                try:
                    _ = subprocess.run(
                        command,
                        check=True,
                        encoding="utf-8",
                        stderr=subprocess.PIPE,
                    )
                except subprocess.CalledProcessError as e:
                    raise cli_error(
                        f"Could not uninstall gamspy-{solver_name}: {e.output}"
                    ) from e

            # do not delete files from gamspy_base as other solvers might depend on it
            remove_solver_entry(gamspy_base.directory, solver_name)
            _remove_installed_addon(addons_path, solver_name)

    if uninstall_all_solvers:
        remove_addons(sorted(_get_removable_solvers(addons_path)))

        # All add-on solvers are gone.
        return

    if unlicensed:
        license_path = utils._get_license_path(gamspy_base.directory)
        licensed_solvers = {s.upper() for s in get_licensed_solvers(license_path)}

        diff = sorted(_get_removable_solvers(addons_path) - licensed_solvers)
        if not diff:
            typer.echo("No unlicensed solvers to uninstall.")
            return

        remove_addons(diff)
        return

    if not solver:
        raise cli_error(
            "Solver name is missing: `gamspy uninstall solver <solver_name>`"
        )

    remove_addons(solver)


@app.command(
    help=(
        "[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall solver <solver_name> | "
        "gamspy uninstall solver --unlicensed"
    ),
    short_help="To uninstall solvers",
)
def solver(
    solver: Annotated[
        list[str] | None,
        typer.Argument(
            help="solver names to be uninstalled",
            autocompletion=complete_solver_names,
        ),
    ] = None,
    uninstall_all_solvers: bool = typer.Option(
        False,
        "--uninstall-all-solvers",
        "--all",
        "-a",
        help="Uninstalls all add-on solvers.",
    ),
    unlicensed: bool = typer.Option(
        False,
        "--unlicensed",
        help="Uninstalls installed add-on solvers no longer permitted by the active license.",
    ),
    skip_pip_uninstall: bool = typer.Option(
        False,
        "--skip-pip-install",
        "-s",
        help="If you already have the solver uninstalled, skip pip uninstall and update gamspy installed solver list.",
    ),
    use_uv: bool = typer.Option(
        False, "--use-uv", help="Use uv instead of pip to uninstall solvers."
    ),
) -> None:
    uninstall_solver(
        solver,
        uninstall_all_solvers=uninstall_all_solvers,
        unlicensed=unlicensed,
        skip_pip_uninstall=skip_pip_uninstall,
        use_uv=use_uv,
    )


if __name__ == "__main__":
    app()
