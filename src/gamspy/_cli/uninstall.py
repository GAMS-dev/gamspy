from __future__ import annotations
import os
import subprocess
from typing import Annotated, Iterable, Union
from gamspy.exceptions import GamspyException, ValidationError
import gamspy.utils as utils
from .util import remove_solver_entry

import typer

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To uninstall licenses and solvers.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall license | gamspy uninstall solver <solver_name>",
    context_settings={"help_option_names": ["-h", "--help"]},
)

@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall license",
    short_help="To uninstall the current license"
)
def license():
    try:
        os.unlink(os.path.join(utils.DEFAULT_DIR, "gamspy_license.txt"))
    except FileNotFoundError:
        ...

@app.command(
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall solver <solver_name>",
    short_help="To uninstall solvers"
)
def solver(
    solver_names: Annotated[
        Union[list[str], None], 
        typer.Argument(help="solver names to be uninstalled")
    ] = None,
    uninstall_all_solvers: Annotated[
        Union[bool, None], 
        typer.Option("--uninstall-all-solvers", help="Uninstalls all add-on solvers.")
    ] = None,
    skip_pip_uninstall: Annotated[
        Union[bool, None], 
        typer.Option("--skip-pip-install", "-s", help="If you already have the solver uninstalled, skip pip uninstall and update gamspy installed solver list.")
    ] = None
):
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

            if not skip_pip_uninstall:
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

    if solver_names is None:
        raise ValidationError(
            "Solver name is missing: `gamspy uninstall solver <solver_name>`"
        )

    remove_addons(solver_names)

if __name__ == "__main__":
    app()
