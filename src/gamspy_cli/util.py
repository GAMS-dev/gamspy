from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy.exceptions import CLIError

__all__ = [
    "CUOPT",
    "cli_error",
    "installation_error",
]

CUOPT = "cuopt"


def cli_error(message: str, return_code: int = 1) -> CLIError:
    # gamspy is imported lazily since importing it slows down the startup of the CLI.
    from gamspy.exceptions import CLIError

    return CLIError(message, return_code)


def installation_error(error: Exception) -> CLIError:
    """The CLIError for an error of gamspy_base.install_solver or uninstall_solver"""
    if isinstance(error, ModuleNotFoundError) and error.msg:
        return cli_error(error.msg)

    return cli_error(str(error))
