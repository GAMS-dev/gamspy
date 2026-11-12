from __future__ import annotations

import os
import subprocess

import typer

from .util import cli_error

app = typer.Typer(
    short_help="To probe a node's information.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy probe -j <output_path.json> | gamspy probe -p 33333 10 | gamspy probe -n",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def probe(
    json_out: str | None = None,
    *,
    ports: tuple[int, int] | None = None,
    use_port: int | None = None,
    network_interfaces: bool = False,
) -> str:
    """
    Probes the information of this node to retrieve a license for it on another
    node. Equivalent of ``gamspy probe``.

    Parameters
    ----------
    json_out : str, optional
        Output path for the JSON file.
    ports : tuple[int, int], optional
        Starting port and number of ports to search for available ports for
        instantiating a use-limited license.
    use_port : int, optional
        Starting port for instantiating a use-limited license.
    network_interfaces : bool, optional
        Lists all available network interfaces and their hashes instead of the
        node information. Cannot be combined with ``json_out``.

    Returns
    -------
    str
        Node information in JSON format, or the network interface table if
        ``network_interfaces`` is True.

    Raises
    ------
    CLIError
        In case the node cannot be probed.

    Examples
    --------
    >>> import json
    >>> from gamspy import cli
    >>> node_info = json.loads(cli.probe())
    >>> node_info = json.loads(cli.probe(ports=(33333, 10)))

    """
    if network_interfaces and json_out:
        raise cli_error("The network interfaces cannot be written to a JSON file.")

    import gamspy_base

    command = [os.path.join(gamspy_base.directory, "gamsprobe")]
    if ports is not None:
        command += ["-p", str(ports[0]), str(ports[1])]
    if use_port is not None:
        command += ["-u", str(use_port)]
    if network_interfaces:
        command.append("-n")

    process = subprocess.run(command, text=True, capture_output=True)

    if process.returncode:
        raise cli_error(process.stderr)

    if json_out:
        with open(json_out, "w") as file:
            file.write(process.stdout)

    return process.stdout


@app.callback(invoke_without_command=True)
def callback(
    json_out: str | None = typer.Option(
        None,
        "--json-out",
        "-j",
        help="Output path for the JSON file.",
    ),
    ports: tuple[int, int] | None = typer.Option(
        None,
        "--ports",
        "-p",
        help="Starting port and number of ports to search for available ports for instantiating a use-limited license.",
    ),
    use_port: int | None = typer.Option(
        None,
        "--use-port",
        "-u",
        help="Starting port for instantiating a use-limited license.",
    ),
    network_interfaces: bool = typer.Option(
        False,
        "--network-interfaces",
        "-n",
        help="List all available network interfaces and their hashes.",
    ),
) -> None:
    print(
        probe(
            json_out,
            ports=ports,
            use_port=use_port,
            network_interfaces=network_interfaces,
        ),
        end="",
    )


if __name__ == "__main__":
    app()
