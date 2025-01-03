from __future__ import annotations

import typer

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To install licenses and solvers.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy install license <access_code> or <path/to/license/file> | gamspy install solver <solver_name>",
    context_settings={"help_option_names": ["-h", "--help"]},
)

if __name__ == "__main__":
    app()
