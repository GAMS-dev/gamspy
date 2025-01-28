from __future__ import annotations

import os
from typing import Any

from gamspy.exceptions import ValidationError

configuration: dict = dict()


def _set_default_options():
    # Set the default for GAMSPY_GAMS_SYSDIR
    try:
        import gamspy_base

        configuration["GAMSPY_GAMS_SYSDIR"] = gamspy_base.directory
    except ModuleNotFoundError:
        ...

    sysdir = os.getenv("GAMSPY_GAMS_SYSDIR", None)
    if sysdir is not None:
        configuration["GAMSPY_GAMS_SYSDIR"] = sysdir

    # Check for domain violation by default
    validate = os.getenv("DOMAIN_VALIDATION", 1)
    if validate:
        configuration["DOMAIN_VALIDATION"] = validate


_set_default_options()


def set_options(options: dict) -> None:
    if not isinstance(options, dict):
        raise ValidationError(
            f"`options` must be a dictionary but found: `{type(options)}`"
        )

    for key, value in options.items():
        configuration[key] = value


def get_option(name: str) -> Any:
    return configuration[name]
