from __future__ import annotations

from typing import List
from typing import TYPE_CHECKING
from typing import Union

import gamspy as gp
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy._symbols.implicits import ImplicitSet, ImplicitParameter
    from gamspy import Alias, Set, Parameter, Equation, Variable


def get_dimension(
    domain: List[Set | Alias | ImplicitSet | str],
):
    dimension = 0

    for elem in domain:
        if isinstance(elem, (gp.Set, gp.Alias, implicits.ImplicitSet)):
            dimension += elem.dimension
        else:
            dimension += 1

    return dimension


def get_domain_path(symbol) -> List[str]:
    path = []
    domain = symbol

    while domain != "*":
        if isinstance(domain, str):
            path.append(domain)
        else:
            path.append(domain.name)

        if isinstance(domain, gp.Alias):
            path.append(domain.alias_with.name)

        if isinstance(domain, str):
            domain = "*"
        else:
            domain = domain.domain[0]

    return path


def validate_dimension(
    symbol: Union[Set, Parameter, Variable, Equation, ImplicitParameter],
    domain: List[Set | Alias | ImplicitSet | str],
):
    dimension = get_dimension(domain)

    if dimension != symbol.dimension:
        raise ValidationError(
            f"The symbol {symbol.name} is referenced with"
            f" {'more' if dimension > symbol.dimension else 'less'} indices"
            f" than declared. Declared dimension is {symbol.dimension} but"
            f" given dimension is {dimension}"
        )


def validate_one_dimensional_sets(
    given: str | Set | Alias | ImplicitSet,
    actual: str | Set | Alias,
):
    if isinstance(given, implicits.ImplicitSet):
        return

    given_path = get_domain_path(given)

    if (
        isinstance(actual, gp.Set)
        and actual.name not in given_path
        or (
            isinstance(actual, gp.Alias)
            and actual.alias_with.name not in given_path
        )
    ):
        raise ValidationError(
            f"`Given set `{given}` is not a valid domain for declared"
            f" domain `{actual}`"
        )


def validate_type(domain):
    for given in domain:
        if not isinstance(
            given,
            (gp.Set, gp.Alias, implicits.ImplicitSet, str, type(...), slice),
        ):
            raise TypeError(
                "Domain item must be type Set, Alias, ImplicitSet or str but"
                f" found `{type(given)}`"
            )


def _get_ellipsis_range(domain, given_domain):
    start = 0
    end = len(domain)

    for item in given_domain:
        if isinstance(item, type(...)):
            break

        start += 1

    for item in reversed(given_domain):
        if isinstance(item, type(...)):
            break

        end -= 1

    return start, end


def _transform_given_indices(
    domain: list[Set | Alias | str],
    indices: Set | Alias | str | tuple | ImplicitSet,
):
    new_domain = []
    given_domain = utils._to_list(indices)
    validate_type(given_domain)

    if len([item for item in given_domain if isinstance(item, type(...))]) > 1:
        raise ValidationError(
            "There cannot be more than one ellipsis in indexing"
        )

    index = 0
    for item in given_domain:
        dimension = (
            1 if isinstance(item, (str, type(...), slice)) else item.dimension
        )
        if isinstance(item, type(...)):
            start, end = _get_ellipsis_range(domain, given_domain)
            new_domain += domain[start:end]
            index = end
        elif isinstance(item, slice):
            new_domain.append(domain[index])
            index += dimension
        else:
            new_domain.append(item)
            index += dimension

    return new_domain


def validate_domain(
    symbol: Union[Set, Parameter, Equation, ImplicitParameter],
    indices: Set | Alias | str | tuple | ImplicitSet,
):
    domain = _transform_given_indices(symbol.domain, indices)
    validate_container(symbol, domain)
    validate_dimension(symbol, domain)

    offset = 0
    for given in domain:
        given_dim = 1 if isinstance(given, str) else given.dimension
        actual = symbol.domain[offset]
        actual_dim = 1 if isinstance(actual, str) else actual.dimension

        if actual_dim == 1 and given_dim == 1:
            if isinstance(given, str):
                if (
                    hasattr(actual, "records")
                    and not actual.records.isin([given]).sum().any()
                ):
                    raise ValidationError(
                        f"Literal index `{given}` was not found in set"
                        f" `{actual}`"
                    )
            else:
                validate_one_dimensional_sets(given, actual)
        offset += given_dim

    return domain


def validate_container(
    self: Set | Parameter | Variable | Equation,
    domain: list[str | Set | Alias],
):
    for set in domain:
        if (
            isinstance(set, (gp.Set, gp.Alias))
            and set.container != self.container
        ):
            raise ValidationError(
                f"`Domain `{set.name}` must be in the same container"
                f" with `{self.name}`"
            )


def validate_name(word: str) -> str:
    reserved_words = [
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

    if word.lower() in reserved_words:
        raise ValidationError(
            "Name cannot be one of the reserved words. List of reserved"
            f" words: {reserved_words}"
        )

    return word
