from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from gams.transfer._internals import GAMS_SYMBOL_MAX_LENGTH

import gamspy._symbols as symbols
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._config import get_option
from gamspy._model import Problem, Sense
from gamspy._options import EXECUTION_OPTIONS, MODEL_ATTR_OPTION_MAP, Options
from gamspy._symbols.symbol import Symbol
from gamspy._types import EllipsisType
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Alias, Equation, Model, Parameter, Set, Variable
    from gamspy._algebra.operation import Operation
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
        ImplicitVariable,
    )
    from gamspy._types import EllipsisType

RESERVED_WORDS = [
    "abort",
    "acronym",
    "acronyms",
    "alias",
    "all",
    "and",
    "binary",
    "break",
    "card",
    "continue",
    "diag",
    "display",
    "do",
    "else",
    "elseif",
    "endfor",
    "endif",
    "endloop",
    "endwhile",
    "eps",
    "equation",
    "equations",
    "execute",
    "execute_load",
    "execute_loaddc",
    "execute_loadhandle",
    "execute_loadpoint",
    "execute_unload",
    "execute_unloaddi",
    "execute_unloadidx",
    "file",
    "files",
    "for",
    "free",
    "function",
    "functions",
    "gdxLoad",
    "if",
    "inf",
    "integer",
    "logic",
    "loop",
    "model",
    "models",
    "na",
    "negative",
    "nonnegative",
    "no",
    "not",
    "option",
    "options",
    "or",
    "ord",
    "parameter",
    "parameters",
    "positive",
    "prod",
    "put",
    "put_utility",
    "putclear",
    "putclose",
    "putfmcl",
    "puthd",
    "putheader",
    "putpage",
    "puttitle",
    "puttl",
    "repeat",
    "sameas",
    "sand",
    "scalar",
    "scalars",
    "semicont",
    "semiint",
    "set",
    "sets",
    "singleton",
    "smax",
    "smin",
    "solve",
    "sor",
    "sos1",
    "sos2",
    "sum",
    "system",
    "table",
    "tables",
    "then",
    "undf",
    "until",
    "variable",
    "variables",
    "while",
    "xor",
    "yes",
]


def get_dimension(
    domain: Sequence[Set | Alias | ImplicitSet | str],
):
    dimension = 0

    for elem in domain:
        if type(elem) in (symbols.Set, symbols.Alias, implicits.ImplicitSet):
            dimension += elem.dimension  # type: ignore
        else:
            dimension += 1

    return dimension


def get_domain_path(symbol: Set | Alias | ImplicitSet) -> list[str]:
    path: list[str] = []
    domain = symbol

    while domain != "*":
        if type(domain) is str:
            path.insert(0, domain)
        else:
            path.insert(0, domain.name)

        if type(domain) is symbols.Alias:
            path.insert(0, domain.alias_with.name)

        domain = "*" if type(domain) is str else domain.domain[0]  # type: ignore

    return path


def validate_dimension(
    symbol: Set
    | Parameter
    | Variable
    | Equation
    | ImplicitParameter
    | ImplicitVariable
    | Operation,
    domain: list[Set | Alias | ImplicitSet | str],
):
    dimension = get_dimension(domain)

    if dimension != symbol.dimension:
        raise ValidationError(
            f"The `{symbol}` is referenced with"
            f" {'more' if dimension > symbol.dimension else 'less'} indices"
            f" than declared. Declared dimension is {symbol.dimension} but"
            f" given dimension is {dimension}"
        )


def validate_one_dimensional_sets(
    given: Set | Alias | ImplicitSet,
    actual: str | Set | Alias,
):
    if type(given) is implicits.ImplicitSet:
        return

    given_path = get_domain_path(given)

    if (
        type(actual) is symbols.Set
        and actual.name not in given_path
        or (
            type(actual) is symbols.Alias
            and actual.alias_with.name not in given_path
        )
    ):
        raise ValidationError(
            f"`Given set `{given.name}` is not a valid domain for declared"
            f" domain `{actual.name}`"
        )


def validate_type(domain):
    for given in domain:
        if type(given) not in (
            symbols.Set,
            symbols.Alias,
            symbols.UniverseAlias,
            implicits.ImplicitSet,
            str,
            int,
            EllipsisType,
            slice,
        ):
            raise TypeError(
                "Domain item must be type Set, Alias, ImplicitSet or str but"
                f" found `{type(given)}`"
            )


def _get_ellipsis_range(domain, given_domain):
    start = 0
    end = len(domain)

    for item in given_domain:
        if type(item) is EllipsisType:
            break

        start += 1

    for item in reversed(given_domain):
        if type(item) is EllipsisType:
            break

        end -= 1

    return start, end


def _expand_ellipsis_slice(
    domain: list[Set | Alias | str],
    indices: Sequence[Set | Alias | str | EllipsisType | slice],
) -> Sequence[Set | Alias | str]:
    if len(domain) == 0:
        # If scalar, only correct indexing is [:] or [...]
        if len(indices) != 1:
            raise ValidationError(
                "Scalar values can only be indexed by '[:]' or '[...]'"
            )

        if type(indices[0]) not in (EllipsisType, slice):
            raise ValidationError(
                "Scalar values can only be indexed by '[:]' or '[...]'"
            )

        return []

    if sum(type(item) is EllipsisType for item in indices) > 1:
        raise ValidationError(
            "There cannot be more than one ellipsis in indexing"
        )

    new_domain: list = []
    index = 0
    for item in indices:
        try:
            dimension = item.dimension  # type: ignore
        except AttributeError:
            dimension = 1

        if type(item) is EllipsisType:
            start, end = _get_ellipsis_range(domain, indices)
            new_domain += domain[start:end]
            index = end
        elif type(item) is slice:
            new_domain.append(domain[index])
            index += dimension
        else:
            new_domain.append(item)
            index += dimension

    return new_domain


def validate_domain(
    symbol: Set
    | Parameter
    | Variable
    | Equation
    | ImplicitParameter
    | ImplicitVariable
    | Operation,
    indices: EllipsisType | slice | Set | Alias | str | Iterable | ImplicitSet,
):
    domain = utils._to_list(indices)
    domain = [str(elem) if type(elem) is int else elem for elem in domain]
    domain = _expand_ellipsis_slice(symbol.domain, domain)  # type: ignore
    if not get_option("DOMAIN_VALIDATION"):
        return domain

    validate_type(domain)
    validate_container(symbol, domain)
    validate_dimension(symbol, domain)

    offset = 0
    for given in domain:
        try:
            given_dim = given.dimension
        except AttributeError:
            given_dim = 1
        actual = symbol.domain[offset]

        try:
            actual_dim = actual.dimension
        except AttributeError:
            actual_dim = 1

        if actual_dim == 1 and given_dim == 1:
            if type(given) is str:
                try:
                    if (
                        len(actual.records) < 1000
                        and not actual.records.isin([given]).sum().any()
                    ):
                        raise ValidationError(
                            f"Literal index `{given}` was not found in set"
                            f" `{actual.name}`"
                        )
                except AttributeError:
                    ...
            else:
                validate_one_dimensional_sets(given, actual)

        offset += given_dim

    return domain


def validate_container(
    symbol: Set
    | Parameter
    | Variable
    | Equation
    | ImplicitParameter
    | ImplicitVariable
    | Operation,
    domain: list[str | Set | Alias],
):
    for set in domain:
        if (
            isinstance(set, (symbols.Set, symbols.Alias))
            and set.container != symbol.container
        ):
            raise ValidationError(
                f"`Domain `{set}` must be in the same container"
                f" with `{symbol}`"
            )


def validate_name(word: str) -> str:
    if not isinstance(word, str):
        raise TypeError("Symbol name must be type str")

    if word.lower() in RESERVED_WORDS:
        raise ValidationError(
            "Name cannot be one of the reserved words. List of reserved"
            f" words: {RESERVED_WORDS}"
        )

    return word


def validate_model(
    equations: Iterable[Equation],
    problem: Problem | str,
    sense: str | Sense,
) -> tuple[Problem, Sense]:
    if isinstance(problem, str):
        if problem.upper() not in Problem.values():
            raise ValueError(
                f"Allowed problem types: {Problem.values()} but found"
                f" {problem}."
            )

        problem = Problem(problem.upper())

    if isinstance(sense, str):
        if sense.upper() not in Sense.values():
            raise ValueError(
                f"Allowed sense values: {Sense.values()} but found {sense}."
            )

        sense = Sense(sense.upper())

    if (
        problem not in (Problem.CNS, Problem.MCP)
        and not isinstance(equations, Iterable)
        or any(
            not isinstance(equation, symbols.Equation)
            for equation in equations
        )
    ):
        raise TypeError("`equations` must be an Iterable of Equation objects")

    return problem, sense


def validate_model_name(name: str) -> str:
    if len(name) == 0:
        raise ValueError("Model name must be at least one character.")

    if len(name) > GAMS_SYMBOL_MAX_LENGTH:
        raise ValueError(
            "Model 'name' is too long, "
            f"max is {GAMS_SYMBOL_MAX_LENGTH} characters"
        )

    if name[0] == "_":
        raise ValidationError(
            "Valid GAMS names cannot begin with a '_' character."
        )

    if not all(True if i == "_" else i.isalnum() for i in name):
        raise ValidationError(
            f"`{name}` is an invalid model name. "
            "Model names can only contain alphanumeric characters "
            "(letters and numbers) and the '_' character."
        )

    return name


def validate_solver_args(
    system_directory: str,
    backend: Literal["local", "engine", "neos"],
    solver: str,
    problem: Problem | str,
    options: Options | None,
    output: io.TextIOWrapper | None,
    load_symbols: list[Symbol] | None,
) -> None:
    # Check validity of options
    if options is not None and not isinstance(options, Options):
        raise TypeError(
            f"`options` must be of type Option but found {type(options)}"
        )

    # Check validity of output
    if output is not None and (
        not hasattr(output, "write") or not hasattr(output, "flush")
    ):
        raise ValidationError(
            "`output` must write and flush operations but found"
            f" {type(output)} which does not support them."
        )

    # Check validity of load_symbols
    if load_symbols is not None:
        if not isinstance(load_symbols, list):
            raise ValidationError(
                f"`load_symbols` must be list of Symbol objects. Given type: {type(load_symbols)}"
            )

        for elem in load_symbols:
            if not isinstance(elem, Symbol):
                raise ValidationError(
                    f"Elements of `load_symbols` must be of type Symbol but found {elem}"
                )

    # Check validity of solver
    if not isinstance(solver, str):
        raise TypeError("`solver` argument must be a string.")

    if backend == "neos" and solver.lower() in ["mpsge", "kestrel"]:
        raise ValidationError(
            f"`{solver}` is not a valid solver for NEOS Server."
        )

    if backend == "engine" and solver.lower() in ["mpsge"]:
        raise ValidationError(
            f"`{solver}` is not a valid solver for GAMS Engine."
        )

    if backend == "local":
        # No need to check whether the solver is installed on client's machine for NEOS or ENGINE.
        installed_solvers = utils.getInstalledSolvers(system_directory)
        if solver.upper() not in installed_solvers:
            raise ValidationError(
                f"Provided solver name `{solver}` is not installed on your"
                f" machine. Install `{solver}` with `gamspy install solver"
                f" {solver.lower()}`"
            )

        capabilities = utils.getSolverCapabilities(system_directory)
        if str(problem).upper() not in capabilities[solver.upper()]:
            raise ValidationError(
                f"Given solver `{solver}` is not capable of solving given"
                f" problem type `{problem}`. See capability matrix "
                "(https://www.gams.com/latest/docs/S_MAIN.html#SOLVERS_MODEL_TYPES)"
                " to choose a suitable solver"
            )


def validate_equations(model: Model):
    for equation in model.equations:
        if equation._definition is None:
            raise ValidationError(
                f"`{equation.name}` has been declared as an equation but no equation definition was found."
            )


def validate_global_options(options: Any) -> Options:
    if options is not None and not isinstance(options, Options):
        raise TypeError(
            f"`options` must be of type Option but found {type(options)}"
        )

    if isinstance(options, Options):
        options_dict = options.model_dump(exclude_none=True)
        if any(option in options_dict for option in MODEL_ATTR_OPTION_MAP):
            raise ValidationError(
                f"The following model options cannot be provided at Container creation time: {', '.join(MODEL_ATTR_OPTION_MAP.keys())}."
            )

        if any(option in options_dict for option in EXECUTION_OPTIONS):
            raise ValidationError(
                f"The following runtime options cannot be provided at Container creation time: {', '.join(EXECUTION_OPTIONS.keys())}."
            )

    if options is None:
        return Options()

    return options
