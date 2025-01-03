from __future__ import annotations

import typer

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To run your model with GAMS MIRO.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy run miro [--path <path_to_miro>] [--model <path_to_model>]",
    context_settings={"help_option_names": ["-h", "--help"]},
)

if __name__ == "__main__":
    app()
