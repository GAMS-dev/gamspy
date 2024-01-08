from __future__ import annotations

from typing import List
from typing import TYPE_CHECKING
from typing import Union

import gamspy as gp
import gamspy._symbols.implicits as implicits
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
    domain: List[Set | Alias | ImplicitSet | str],
    symbol: Union[Set, Parameter, Equation, ImplicitParameter],
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
    domain: List[Set | Alias | ImplicitSet | str],
    symbol: Union[Set, Parameter, Equation, ImplicitParameter],
):
    index = 0

    for given in domain:
        dimension = 1 if isinstance(given, str) else given.dimension
        if isinstance(given, (str, implicits.ImplicitSet)) or dimension != 1:
            index += dimension
            continue

        given_path = get_domain_path(given)

        actual = symbol.domain[index]
        if actual == "*" or actual.dimension != 1:
            continue

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

        index += 1


def validate_type(domain):
    for given in domain:
        if not isinstance(
            given, (gp.Set, gp.Alias, implicits.ImplicitSet, str)
        ):
            raise TypeError(
                "Domain item must be type Set, Alias, ImplicitSet or str but"
                f" found `{type(given)}`"
            )


def validate_domain(
    domain: List[Set | Alias | ImplicitSet | str],
    symbol: Union[Set, Parameter, Equation, ImplicitParameter],
):
    validate_type(domain)
    validate_dimension(domain, symbol)

    index = 0
    for given in domain:
        dimension = 1 if isinstance(given, str) else given.dimension
        actual = symbol.domain[index : index + dimension]

        if isinstance(given, str) and symbol._records is not None:
            for item in actual:
                if item != "*" and not item.records.isin([given]).sum().any():
                    raise ValidationError(
                        f"Literal index `{given}` was not found in set"
                        f" `{actual}`"
                    )

        index += dimension

    validate_one_dimensional_sets(domain, symbol)


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
