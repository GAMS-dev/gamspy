from __future__ import annotations

import typer

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To uninstall licenses and solvers.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy uninstall license | gamspy uninstall solver <solver_name>",
    context_settings={"help_option_names": ["-h", "--help"]},
)

if __name__ == "__main__":
    app()
