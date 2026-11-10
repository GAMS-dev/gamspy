from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated

import typer

from .util import (
    CUOPT,
    cli_error,
    installation_error,
)

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


def complete_solver_names(ctx: typer.Context, incomplete: str) -> list[str]:
    import gamspy_base

    return [
        s.lower()
        for s in gamspy_base.get_installed_addon_solvers()
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

    # gamspy_base.uninstall_solver checks for pip (or uv) before it uninstalls anything.
    addons_path = os.path.join(utils.DEFAULT_DIR, "solvers.txt")
    removable_solvers = set(gamspy_base.get_installed_addon_solvers())

    def remove_addons(addons: Iterable[str]) -> None:
        for item in addons:
            solver_name = item.lower()

            if solver_name.upper() not in removable_solvers:
                raise cli_error(
                    f'Given solver name ("{solver_name}") is not valid. Installed'
                    f" solvers that can be uninstalled: {sorted(removable_solvers)}"
                )

            # The files of the solver stay in gamspy_base as other solvers might
            # depend on them, apart from the ones of cuOpt.
            try:
                gamspy_base.uninstall_solver(
                    solver_name,
                    use_uv=use_uv,
                    uninstall_package=not skip_pip_uninstall,
                )
            except (gamspy_base.InstallationError, ValueError) as e:
                raise installation_error(e) from e

            removable_solvers.discard(solver_name.upper())
            _remove_installed_addon(addons_path, solver_name)

    if uninstall_all_solvers:
        remove_addons(sorted(removable_solvers))

        # All add-on solvers are gone.
        return

    if unlicensed:
        license_path = utils._get_license_path(gamspy_base.directory)
        licensed_solvers = set(gamspy_base.get_licensed_solvers(license_path))

        # cuOpt is not licensed through the GAMS license.
        licensed_solvers.add(CUOPT.upper())

        diff = sorted(removable_solvers - licensed_solvers)
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
