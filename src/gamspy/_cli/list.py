from __future__ import annotations

from typing import Annotated, Union

import typer
from rich import print
from rich.console import Console
from rich.table import Table

import gamspy.utils as utils
from gamspy.exceptions import ValidationError

console = Console()
app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To list solvers.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy list solvers --all | gamspy list solvers --defaults",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def solvers(
    all: Annotated[
        Union[bool, None],
        typer.Option("--all", "-a", help="Shows all available solvers."),
    ] = None,
    defaults: Annotated[
        Union[bool, None],
        typer.Option("--defaults", "-d", help="Shows default solvers."),
    ] = None,
) -> None:
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    capabilities = utils.getSolverCapabilities(gamspy_base.directory)
    if all:
        solvers = utils.getAvailableSolvers()
        print("Available Solvers")
        print("=" * 17)
        print(", ".join(solvers))
        print("\nModel types that can be solved with the installed solvers:\n")
        table = Table("Solver", "Problem Types")
        for solver in solvers:
            try:
                table.add_row(solver, ", ".join(capabilities[solver]))
            except KeyError:
                ...
        console.print(table)

        print(
            "[bold]Full list can be found here[/bold]: https://www.gams.com/latest/docs/S_MAIN.html#SOLVERS_MODEL_TYPES"
        )
    elif defaults:
        default_solvers = utils.getDefaultSolvers()
        table = Table("Problem", "Solver")
        for problem in default_solvers:
            try:
                table.add_row(problem, default_solvers[problem])
            except KeyError:
                ...

        console.print(table)
    else:
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


if __name__ == "__main__":
    app()
