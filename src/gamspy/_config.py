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
    """
    Sets the given configuration options.

    Parameters
    ----------
    options : dict

    Raises
    ------
    ValidationError
        In case the given options are not in dict type.

    Examples
    --------
    >>> import gamspy as gp
    >>> gp.set_options({"DOMAIN_VALIDATION": 1})

    """
    if not isinstance(options, dict):
        raise ValidationError(
            f"`options` must be a dictionary but found: `{type(options)}`"
        )

    for key, value in options.items():
        configuration[key] = value


def get_option(name: str) -> Any:
    """
    Returns the requested option.

    Parameters
    ----------
    name : str
        Option name.

    Returns
    -------
    Any
        The value of the option.

    Raises
    ------
    KeyError
        In case the option is not set.

    Examples
    --------
    >>> import gamspy as gp
    >>> gp.set_options({"DOMAIN_VALIDATION": 0})
    >>> gp.get_option("DOMAIN_VALIDATION")
    0
    >>> gp.set_options({"DOMAIN_VALIDATION": 1})

    """
    return configuration[name]
