from __future__ import annotations

import os
import subprocess

import typer

from .util import cli_error

app = typer.Typer(
    short_help="To probe a node's information.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy probe -j <output_path.json>",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def probe(json_out: str | None = None) -> str:
    """
    Probes the information of this node to retrieve a license for it on another
    node. Equivalent of ``gamspy probe``.

    Parameters
    ----------
    json_out : str, optional
        Output path for the JSON file.

    Returns
    -------
    str
        Node information in JSON format.

    Raises
    ------
    CLIError
        In case the node cannot be probed.

    Examples
    --------
    >>> import json
    >>> from gamspy import cli
    >>> node_info = json.loads(cli.probe())

    """
    import gamspy_base

    process = subprocess.run(
        [os.path.join(gamspy_base.directory, "gamsprobe")],
        text=True,
        capture_output=True,
    )

    if process.returncode:
        raise cli_error(process.stderr)

    if json_out:
        with open(json_out, "w") as file:
            file.write(process.stdout)

    return process.stdout


def _probe(json_out: str | None) -> None:
    print(probe(json_out))
    raise typer.Exit()


@app.callback()
def callback(
    json_out: str | None = typer.Option(
        None,
        "--json-out",
        "-j",
        help="Output path for the JSON file.",
        callback=_probe,
    ),
) -> None: ...


if __name__ == "__main__":
    app()
