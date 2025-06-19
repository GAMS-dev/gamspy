from __future__ import annotations

import os
from typing import Any, Literal

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

    # Enable all validations by default
    validate = int(os.getenv("GAMSPY_VALIDATION", 1))
    configuration["VALIDATION"] = validate

    # Check for domain violation by default
    validate = int(os.getenv("GAMSPY_DOMAIN_VALIDATION", 1))
    configuration["DOMAIN_VALIDATION"] = validate

    # Solver option validation. Enabled by default.
    validate_solver_options = int(
        os.getenv("GAMSPY_SOLVER_OPTION_VALIDATION", 1)
    )
    configuration["SOLVER_OPTION_VALIDATION"] = validate_solver_options

    # Special value mapping
    map_special_values = int(os.getenv("GAMSPY_MAP_SPECIAL_VALUES", 1))
    configuration["MAP_SPECIAL_VALUES"] = map_special_values

    # Lazy evaluation
    evaluate_lazily = int(os.getenv("GAMSPY_LAZY_EVALUATION", 0))
    configuration["LAZY_EVALUATION"] = evaluate_lazily

    # Assume .l or .scale for variables in assignments
    assume_level = int(os.getenv("GAMSPY_ASSUME_VARIABLE_SUFFIX", 1))
    configuration["ASSUME_VARIABLE_SUFFIX"] = assume_level

    # Try to use the Python variable name in case the name is not provided.
    use_py_var_name = os.getenv("GAMSPY_USE_PY_VAR_NAME", "no")
    configuration["USE_PY_VAR_NAME"] = use_py_var_name


def set_options(
    options: dict[
        Literal[
            "GAMS_SYSDIR",
            "VALIDATION",
            "DOMAIN_VALIDATION",
            "SOLVER_OPTION_VALIDATION",
            "MAP_SPECIAL_VALUES",
            "LAZY_EVALUATION",
            "ASSUME_VARIABLE_SUFFIX",
            "USE_PY_VAR_NAME",
        ],
        Any,
    ],
) -> None:
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
