from __future__ import annotations

import os
from typing import Any

from gamspy.exceptions import ValidationError

configuration: dict[str, Any] = dict()


def _set_default_options() -> None:
    # Set the default for GAMS_SYSDIR
    try:
        import gamspy_base

        configuration["GAMS_SYSDIR"] = gamspy_base.directory
    except ModuleNotFoundError:
        ...

    sysdir = os.getenv("GAMSPY_GAMS_SYSDIR", None)
    if sysdir is not None:
        configuration["GAMS_SYSDIR"] = sysdir

    # Check for domain violation by default
    validate = os.getenv("GAMSPY_DOMAIN_VALIDATION", 1)
    configuration["DOMAIN_VALIDATION"] = validate

    # Special value mapping
    map_special_values = os.getenv("GAMSPY_MAP_SPECIAL_VALUES", 1)
    configuration["MAP_SPECIAL_VALUES"] = map_special_values

    # Lazy evaluation
    evaluate_lazily = os.getenv("GAMSPY_LAZY_EVALUATION", 0)
    configuration["LAZY_EVALUATION"] = evaluate_lazily


def set_options(options: dict[str, Any]) -> None:
    """
    Sets the given configuration options.

    Parameters
    ----------
    options : dict[str, Any]

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
