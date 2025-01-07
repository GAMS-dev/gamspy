from __future__ import annotations

import typer

import gamspy.utils as utils
from gamspy.exceptions import ValidationError

app = typer.Typer(
    rich_markup_mode="rich",
    short_help="To show your license and gamspy_base directory.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy show license | gamspy show base",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command(short_help="Shows the license content.")
def license():
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    license_path = utils._get_license_path(gamspy_base.directory)
    print(f"License found at: {license_path}\n")
    print("License Content")
    print("=" * 15)
    with open(license_path, encoding="utf-8") as license_file:
        print(license_file.read().strip())


@app.command(short_help="Shows the path of gamspy_base.")
def base():
    try:
        import gamspy_base
    except ModuleNotFoundError as e:
        raise ValidationError(
            "You must install gamspy_base to use this command!"
        ) from e

    print(gamspy_base.directory)


if __name__ == "__main__":
    app()
