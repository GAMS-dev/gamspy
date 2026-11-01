"""
Programmatic access to the commands of the GAMSPy command line interface.
Each function corresponds to a ``gamspy`` command, e.g. :func:`install_solver`
to ``gamspy install solver``. Instead of printing and exiting, the functions
return their results and raise :class:`CLIError <gamspy.exceptions.CLIError>`
on failure.

Examples
--------
>>> from gamspy import cli
>>> cli.install_solver("highs")  # doctest: +SKIP

"""

from __future__ import annotations

from gamspy_cli.gdx import GdxDiffResult, gdx_diff, gdx_dump
from gamspy_cli.install import install_license, install_solver
from gamspy_cli.list import SolverInfo, list_solvers
from gamspy_cli.mps2gms import mps2gms
from gamspy_cli.probe import probe
from gamspy_cli.retrieve import retrieve_license
from gamspy_cli.run import run_miro
from gamspy_cli.show import LicenseInfo, show_base, show_license
from gamspy_cli.uninstall import uninstall_license, uninstall_solver

__all__ = [
    "GdxDiffResult",
    "LicenseInfo",
    "SolverInfo",
    "gdx_diff",
    "gdx_dump",
    "install_license",
    "install_solver",
    "list_solvers",
    "mps2gms",
    "probe",
    "retrieve_license",
    "run_miro",
    "show_base",
    "show_license",
    "uninstall_license",
    "uninstall_solver",
]
