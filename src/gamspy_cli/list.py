from __future__ import annotations

import typer

app = typer.Typer(
    short_help="To list solvers.",
    help="[bold][yellow]Examples[/yellow][/bold]: gamspy list solvers --all | gamspy list solvers --defaults",
    context_settings={"help_option_names": ["-h", "--help"]},
)

_SOLVER_TOOLS: dict[str, str] = {
    "CONVERT": "Translates models to other solver input formats",
    "EXAMINER": "Validates solutions and checks model properties",
    "EXAMINER2": "Extended solution validation with detailed diagnostics (deprecated)",
    "KESTREL": "Submits jobs to remote NEOS solvers",
    "MPSGE": "General equilibrium modelling via Arrow-Debreu",
    "NLPEC": "Reformulates MCP/MPEC problems as NLP",
    "RESHOP": "Reformulation and decomposition for MCP/MPEC",
}

_S_GREEN = "[bold bright_green]✓[/bold bright_green]"
_S_YELLOW = "[bold yellow]○[/bold yellow]"
_S_RED = "[bold red]✗[/bold red]"
_S_SIZE_LIMITED = "[bold cyan]*[/bold cyan]"


def _status_cell(
    *, is_licensed: bool, is_installed: bool, is_size_limited: bool = False
) -> str:
    if is_licensed and is_installed:
        icon = _S_GREEN
    elif is_licensed:
        icon = _S_YELLOW
    else:
        return _S_RED
    return f"{icon}{_S_SIZE_LIMITED}" if is_size_limited else icon


def _collect_solver_data(all: bool) -> tuple[list[dict], list[dict], list[str]]:
    import gamspy_base

    import gamspy.utils as utils
    from gamspy_cli import cuopt
    from gamspy_cli.license import get_licensed_solvers, is_size_limited_license

    system_directory = gamspy_base.directory
    installed = set(utils.getInstalledSolvers(system_directory))
    # The capabilities of the system directory only cover installed solvers, hence
    # those of gamspy_base and cuOpt fill in the rest.
    capabilities = {
        **gamspy_base.capabilities,
        "CUOPT": cuopt.MODEL_TYPES,
        **utils.getSolverCapabilities(system_directory),
    }

    license_path = utils._get_license_path(system_directory)
    licensed = {s.upper() for s in get_licensed_solvers(license_path)}

    if "CONOPT" in licensed:
        licensed.add("CONOPT4")
    size_limited_license = is_size_limited_license(license_path)

    tool_names = set(_SOLVER_TOOLS)
    solver_names = [n for n in utils.getAvailableSolvers() if n not in tool_names]

    all_types = sorted({t for caps in capabilities.values() for t in caps})

    solver_rows = []
    for name in solver_names:
        is_inst = name in installed
        # cuOpt is not gated by a GAMS license component — it is either
        # installed or not, so treat it as always "licensed".
        is_lic = name == "CUOPT" or name in licensed
        if not all and not is_inst:
            continue
        solver_rows.append(
            {
                "name": name,
                "licensed": is_lic,
                "installed": is_inst,
                # cuOpt is licensed independent of the GAMS license, so it is
                # never subject to that license's size limit.
                "size_limited": size_limited_license and name != "CUOPT",
                "capabilities": capabilities.get(name, []),
            }
        )

    tool_rows = [
        {
            "name": tool_name,
            "licensed": tool_name in licensed,
            "installed": tool_name in installed,
            "size_limited": size_limited_license,
            "description": desc,
        }
        for tool_name, desc in sorted(_SOLVER_TOOLS.items())
        if all or tool_name in installed
    ]

    return solver_rows, tool_rows, all_types


def _print_solver_tables(all: bool) -> None:
    import sys

    from rich.console import Console
    from rich.table import Table

    solver_rows, tool_rows, all_types = _collect_solver_data(all)

    # Force UTF-8. Windows systems might default to cp1252 encoding.
    # TODO: UTF-8 will be the default on all platforms in Python 3.15. Remove this when we drop support for Python 3.14.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                ...

    console = Console(width=220)

    console.print(
        "\n[bold]Solvers[/bold]:  [dim]Optimization engines by problem type[/dim]"
    )
    solver_table = Table(show_lines=False, padding=(0, 1))
    solver_table.add_column("Name", no_wrap=True)
    solver_table.add_column("Status", justify="center", no_wrap=True)
    for t in all_types:
        solver_table.add_column(t, justify="center", no_wrap=True)

    for row in solver_rows:
        status = _status_cell(
            is_licensed=row["licensed"],
            is_installed=row["installed"],
            is_size_limited=row["size_limited"],
        )
        cap_cells = ["▪" if t in row["capabilities"] else "" for t in all_types]
        solver_table.add_row(row["name"], status, *cap_cells)

    console.print(solver_table)

    if tool_rows:
        console.print(
            "\n[bold]Tools[/bold]:  [dim]Utilities and translators, not standalone optimisers[/dim]"
        )
        tool_table = Table(show_lines=False, padding=(0, 1))
        tool_table.add_column("Name", no_wrap=True)
        tool_table.add_column("Status", justify="center", no_wrap=True)
        tool_table.add_column("Description")

        for row in tool_rows:
            status = _status_cell(
                is_licensed=row["licensed"],
                is_installed=row["installed"],
                is_size_limited=row["size_limited"],
            )
            tool_table.add_row(row["name"], status, row["description"])

        console.print(tool_table)

    console.print(
        f"\n{_S_GREEN} [dim]fully licensed[/dim]   "
        f"{_S_SIZE_LIMITED} [dim]licensed, size limited[/dim]   "
        f"{_S_YELLOW} [dim]licensed; not installed[/dim]   "
        f"{_S_RED} [dim]not licensed[/dim]   "
        "▪ [dim]supported[/dim]\n"
    )


@app.command()
def solvers(
    all: bool = typer.Option(False, "--all", "-a", help="Shows all available solvers."),
    installables: bool = typer.Option(
        False,
        "--installables",
        "-i",
        help="Shows solvers that can be installed. Deprecated, use --all instead.",
    ),
    defaults: bool = typer.Option(
        False, "--defaults", "-d", help="Shows default solvers."
    ),
) -> None:
    import gamspy_base
    from rich.console import Console
    from rich.table import Table

    import gamspy.utils as utils

    console = Console()

    if installables:
        typer.echo(
            "Warning: `--installables` option is deprecated and will be removed in a future release. Use `--all` instead.",
            err=True,
        )
        installable_solvers = utils.getInstallableSolvers()
        console.print(", ".join(installable_solvers))
    elif defaults:
        default_solvers = utils.getDefaultSolvers(gamspy_base.directory)
        table = Table("Problem", "Solver")
        for problem in default_solvers:
            try:
                table.add_row(problem, default_solvers[problem])
            except KeyError:
                ...

        console.print(table)
    else:
        _print_solver_tables(all)


if __name__ == "__main__":
    app()
