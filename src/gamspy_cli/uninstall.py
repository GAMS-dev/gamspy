from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING, Annotated

import certifi
import typer

from .util import has_pip, has_uv, remove_solver_entry

if TYPE_CHECKING:
    from collections.abc import Iterable

app = typer.Typer(
    short_help="To uninstall licenses and solvers.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall license | gamspy uninstall solver <solver_name>",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def complete_solver_names(ctx: typer.Context, incomplete: str):
    import gamspy_base

    import gamspy.utils as utils

    return [
        s.lower()
        for s in utils.getInstalledSolvers(gamspy_base.directory)
        if s.startswith(incomplete.upper())
    ]


@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall license",
    short_help="To uninstall the current license",
)
def license():
    import gamspy.utils as utils

    try:
        os.unlink(os.path.join(utils.DEFAULT_DIR, "gamspy_license.txt"))
    except FileNotFoundError:
        ...


@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall solver <solver_name>",
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
    skip_pip_uninstall: bool = typer.Option(
        False,
        "--skip-pip-install",
        "-s",
        help="If you already have the solver uninstalled, skip pip uninstall and update gamspy installed solver list.",
    ),
    use_uv: bool = typer.Option(
        False, "--use-uv", help="Use uv instead of pip to uninstall solvers."
    ),
):
    import gamspy_base

    import gamspy.utils as utils

    if not use_uv and not has_pip():
        typer.echo(
            "pip is not installed in your environment. Please install pip first or add --use-uv flag to uninstall solvers with uv."
        )
        raise typer.Exit(code=1)

    if use_uv and not has_uv():
        typer.echo(
            "uv is not installed in your machine. Please install uv first to uninstall solvers with --use-uv flag."
        )
        raise typer.Exit(code=1)

    addons_path = os.path.join(utils.DEFAULT_DIR, "solvers.txt")
    environment_variables = os.environ.copy()
    if "CURL_CA_BUNDLE" not in environment_variables:
        environment_variables["CURL_CA_BUNDLE"] = certifi.where()

    def remove_addons(addons: Iterable[str]):
        for item in addons:
            solver_name = item.lower()

            installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)
            removable_solvers = set(installed_solvers) - set(
                gamspy_base.default_solvers
            )
            if solver_name.upper() not in removable_solvers:
                typer.echo(
                    f'Given solver name ("{solver_name}") is not valid. Installed'
                    f" solvers that can be uninstalled: {sorted(removable_solvers)}"
                )
                raise typer.Exit(code=1)

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
                    typer.echo(f"Could not uninstall gamspy-{solver_name}: {e.output}")
                    raise typer.Exit(code=1) from e

            # do not delete files from gamspy_base as other solvers might depend on it
            remove_solver_entry(gamspy_base.directory, solver_name)

            try:
                with open(addons_path) as file:
                    installed = file.read().splitlines()
            except FileNotFoundError:
                installed = []

            try:
                installed.remove(solver_name.upper())
            except ValueError:
                ...

            with open(addons_path, "w") as file:
                file.write("\n".join(installed) + "\n")

    if uninstall_all_solvers:
        installed_solvers = utils.getInstalledSolvers(gamspy_base.directory)
        solvers = [
            solver
            for solver in installed_solvers
            if solver not in gamspy_base.default_solvers
        ]
        remove_addons(solvers)

        # All add-on solvers are gone.
        return

    if solver is None:
        typer.echo("Solver name is missing: `gamspy uninstall solver <solver_name>`")
        raise typer.Exit(code=1)

    remove_addons(solver)


if __name__ == "__main__":
    app()
