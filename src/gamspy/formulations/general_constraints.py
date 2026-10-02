from __future__ import annotations

import math
import typing

import gamspy as gp
from gamspy._algebra.condition import Condition
from gamspy._algebra.expression import Expression
from gamspy._algebra.operable import Operable
from gamspy._symbols.implicits import ImplicitSet, ImplicitVariable
from gamspy.exceptions import ValidationError
from gamspy.formulations.result import FormulationResult
from gamspy.formulations.utils import _domain_name

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from gamspy._universe import Universe

    DomainElementType: typing.TypeAlias = (
        gp.Set | gp.Alias | gp.UniverseAlias | Universe | str
    )
    DomainType: typing.TypeAlias = list[DomainElementType]
    BigMType: typing.TypeAlias = int | float | Operable
    BinaryType: typing.TypeAlias = gp.Variable | ImplicitVariable


_RELATIONAL_OPERATORS = {"=l=", "=e=", "=g="}


def _is_binary(symbol: typing.Any) -> bool:
    if isinstance(symbol, ImplicitVariable):
        return symbol.parent.type == "binary"

    return isinstance(symbol, gp.Variable) and symbol.type == "binary"


def _domain_names(domain: Sequence[DomainElementType]) -> set[str]:
    return {_domain_name(set_) for set_ in domain}


def _validate_big_m(big_m: BigMType | None, controlled: set[str]) -> None:
    if big_m is None:
        return

    if isinstance(
        big_m, (gp.Variable, ImplicitVariable, gp.Set, gp.Alias, ImplicitSet)
    ):
        raise ValidationError("big_m cannot be a variable or a set")

    if isinstance(big_m, Expression) and big_m.operator in _RELATIONAL_OPERATORS:
        raise ValidationError("big_m cannot be an inequality or equality")

    if isinstance(big_m, Operable):
        if not _domain_names(big_m.domain) <= controlled:  # ty: ignore[unresolved-attribute]
            raise ValidationError(
                "The domain of big_m must be a subset of the domain of the constraint"
            )
        return

    if (
        isinstance(big_m, bool)
        or not isinstance(big_m, (int, float))
        or not math.isfinite(big_m)
        or big_m <= 0
    ):
        raise ValidationError(
            "big_m must be a positive finite number, a parameter expression or None"
        )


def _declaration_domain(index: DomainType) -> DomainType:
    # multidimensional subsets such as ij(i,j) cannot be declaration domains,
    # the symbols are declared over (i,j) and indexed with ij instead
    domain: DomainType = []
    for set_ in index:
        if _is_multidimensional(set_):
            domain.extend(set_.domain)
        else:
            domain.append(set_)

    return domain


def _dimension(set_: DomainElementType) -> int:
    return set_.dimension if isinstance(set_, (gp.Set, gp.Alias)) else 1


def _is_multidimensional(set_: typing.Any) -> typing.TypeGuard[gp.Set | gp.Alias]:
    return isinstance(set_, (gp.Set, gp.Alias)) and set_.dimension > 1


def _control_index(
    domain: DomainType,
) -> tuple[DomainType, list[typing.Any], set[str]]:
    # defining the equations over ij(i,j) instead of ij also controls i and j,
    # e.g. eq(ij(i,j)) .. x(ij) =l= b(i)
    expanded: dict[str, typing.Any] = {}
    components: set[str] = set()
    for set_ in domain:
        if not _is_multidimensional(set_) or not all(
            isinstance(elem, (gp.Set, gp.Alias)) for elem in set_.domain
        ):
            continue

        names = _domain_names(set_.domain)
        if len(names) == set_.dimension and not names & components:
            expanded[_domain_name(set_)] = set_[tuple(set_.domain)]
            components |= names

    # x[ij] <= p[i] has the domain [ij, i] but ij(i,j) already controls i
    index = [
        set_
        for set_ in domain
        if _is_multidimensional(set_) or _domain_name(set_) not in components
    ]
    key = [expanded.get(_domain_name(set_), set_) for set_ in index]

    return index, key, _domain_names(index) | components


def _indicator_positions(
    indicator_var: BinaryType, index: DomainType
) -> tuple[int, ...]:
    # positions of the indices of indicator_var: e.g. (0, 2) for indicator(b[i, k], 1, x[ij, k] <= 1)
    if isinstance(indicator_var, ImplicitVariable) and (
        indicator_var.permutation is not None
        or any(
            not isinstance(set_, (gp.Set, gp.Alias, gp.UniverseAlias, str))
            for set_ in indicator_var.domain
        )
        or sum(_dimension(set_) for set_ in indicator_var.domain)
        != indicator_var.parent.dimension
    ):
        raise ValidationError(
            "indicator_var must be indexed with sets only for native indicator"
            " constraints"
        )

    offsets: dict[str, range] = {}
    position = 0
    for set_ in index:
        dimension = _dimension(set_)
        offsets.setdefault(_domain_name(set_), range(position, position + dimension))
        if dimension > 1:
            for offset, component in enumerate(set_.domain):  # ty: ignore[unresolved-attribute]
                offsets.setdefault(
                    _domain_name(component),
                    range(position + offset, position + offset + 1),
                )
        position += dimension

    return tuple(
        position
        for set_ in indicator_var.domain
        for position in offsets[_domain_name(set_)]
    )


def _add_native_indicator(
    indicator_var: BinaryType,
    indicator_val: typing.Literal[0, 1],
    expr: Expression,
    condition: typing.Any,
    domain: DomainType,
    index: DomainType,
    key: list[typing.Any],
) -> FormulationResult:
    m = indicator_var.container
    positions = _indicator_positions(indicator_var, index)
    binary = (
        indicator_var.parent
        if isinstance(indicator_var, ImplicitVariable)
        else indicator_var
    )

    result = FormulationResult()
    equation = _add_equation(m, domain, key, expr, condition)

    # GAMS generates only the variables that appear in the equations of the
    # model but indicator_var might only appear in the options file
    if key:
        sum_domain = gp.Domain(*key)
        if condition is not None:
            sum_domain = sum_domain.where[condition]
        generation = gp.Sum(sum_domain, indicator_var) >= 0
    else:
        generated = (
            indicator_var if condition is None else indicator_var.where[condition]
        )
        generation = generated >= 0

    result.equations_created["indicator"] = equation
    result.equations_created["binary"] = m.addEquation(definition=generation)

    # Model adds the generation equation if the indicator equation is in the model
    equation._indicator = (
        binary,
        positions,
        indicator_val,
        result.equations_created["binary"],
    )
    return result


def _add_sos1_pair(m: gp.Container, domain: DomainType) -> gp.Variable:
    # at most one of [*domain, "0"] and [*domain, "1"] can be nonzero
    sos_dim = gp.math._generate_dims(m, [2])[0]
    return m.addVariable(domain=[*domain, sos_dim], type="sos1")


def _add_equation(
    m: gp.Container,
    domain: DomainType,
    key: list[typing.Any],
    definition: Expression,
    condition: typing.Any,
) -> gp.Equation:
    equation = m.addEquation(domain=domain)
    indices = key if key else ...
    if condition is None:
        equation[indices] = definition
    else:
        equation[indices].where[condition] = definition

    return equation


def indicator(
    indicator_var: BinaryType,
    indicator_val: typing.Literal[0, 1],
    expr: Expression | Condition,
    *,
    big_m: BigMType | None = None,
    native: bool = False,
) -> FormulationResult:
    """
    Enforces the constraint ``expr`` only when ``indicator_var`` equals
    ``indicator_val``. When the binary variable takes the other value, the
    constraint is relaxed. This corresponds to the indicator constraint
    ``indicator_var == indicator_val  =>  expr``.

    By default, the relationship is modeled with SOS1 variables and therefore
    does not require any bounds. Each ``<=`` or ``>=`` constraint **generates**
    one SOS1 variable and two equations, and an equality constraint generates
    one SOS1 variable and three equations. Usage of SOS1 variables requires a
    MIP solver that supports them.

    If ``big_m`` is provided, a big-M formulation is used instead, which
    **generates** one equation per ``<=`` or ``>=`` constraint and two per
    equality constraint. ``big_m`` must be at least as large as the largest
    possible violation of ``expr``, e.g. ``max(lhs - rhs)`` for a ``<=``
    constraint. Tighter and **correct** values improve the linear relaxation.
    ``big_m`` can also be an expression of parameters and variable bounds,
    e.g. ``x.up - 10``, but must not contain variables.

    If ``native`` is True, the constraint is not reformulated. Instead,
    ``Model.solve`` passes it to the solver as an indicator constraint through
    the solver options file, so that the solver can handle it directly in its
    branch-and-cut algorithm. This is supported by COPT, CPLEX, GUROBI, SCIP
    and XPRESS. Solving a model that contains native indicator constraints
    with any other solver raises an error. Besides the constraint, one
    redundant equation is **generated** to ensure that GAMS generates
    ``indicator_var``.

    The domain of ``indicator_var`` must be a subset of the domain of
    ``expr``. If ``expr`` has additional domains, a single binary variable
    controls every constraint over those domains. If ``expr`` has a
    condition, e.g. ``(x <= 10).where[p > 0]``, the generated equations are
    only defined where the condition holds. Similarly, indexing with a
    multidimensional subset, e.g. ``x[ij] <= 10``, generates equations only
    for the elements of ``ij`` instead of the entire Cartesian product.

    FormulationResult:
        - With SOS1: variables_created: ["sos1"], equations_created: ["slack", "link"]
        - With big-M: equations_created: ["big_m"]
        - With native: equations_created: ["indicator", "binary"]
        - For equality constraints, the ``slack`` and ``big_m`` keys are prefixed with ``le_`` and ``ge_``.

    Parameters
    ----------
    indicator_var : Variable | ImplicitVariable
        Binary variable controlling the constraint.
    indicator_val : Literal[0, 1]
        Value of ``indicator_var`` that activates the constraint.
    expr : Expression | Condition
        Constraint in the form of ``lhs <= rhs``, ``lhs >= rhs`` or ``lhs == rhs``,
        optionally with a condition.
    big_m : int | float | Operable | None, optional
        Big-M value to use instead of the SOS1 formulation.
    native : bool, optional
        Pass the indicator constraint to the solver instead of reformulating it.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> x = gp.Variable(m, "x", domain=i)
    >>> b = gp.Variable(m, "b", type="binary", domain=i)
    >>> p = gp.Parameter(m, "p", domain=i, records=[("0", 1), ("2", 5)])
    >>> res = gp.formulations.indicator(b, 1, x <= 10)
    >>> len(res.equations_created)
    2
    >>> res = gp.formulations.indicator(b, 0, x == 5, big_m=100)
    >>> list(res.equations_created.keys())
    ['le_big_m', 'ge_big_m']
    >>> res = gp.formulations.indicator(b, 1, (x <= 10).where[p > 0], big_m=x.up - 10)
    >>> list(res.equations_created.keys())
    ['big_m']
    >>> res = gp.formulations.indicator(b, 1, x <= 10, native=True)
    >>> list(res.equations_created.keys())
    ['indicator', 'binary']

    """
    if not _is_binary(indicator_var):
        raise ValidationError("indicator_var needs to be a binary variable")

    if isinstance(indicator_val, bool) or indicator_val not in (0, 1):
        raise ValidationError("indicator_val needs to be 1 or 0")

    condition = None
    if isinstance(expr, Condition):
        condition = expr.condition
        expr = expr.conditioning_on  # ty: ignore[invalid-assignment]

    if not isinstance(expr, Expression) or expr.operator not in _RELATIONAL_OPERATORS:
        raise ValidationError("expr needs to be inequality or equality")

    index, key, controlled = _control_index(list(expr.domain))
    if not _domain_names(indicator_var.domain) <= controlled:
        raise ValidationError(
            "The domain of indicator_var must be a subset of the domain of expr"
        )

    if isinstance(condition, (gp.Set, gp.Alias)):
        # a bare s in $(s) does not compile, s(s) or s(i) does
        if condition.name in controlled:
            condition = condition[condition]
        elif _domain_names(condition.domain) <= controlled:
            condition = condition[tuple(condition.domain)]

    if isinstance(condition, (gp.Set, gp.Alias)):
        condition_domain = [condition]
    else:
        condition_domain = getattr(condition, "domain", [])
    if not _domain_names(condition_domain) <= controlled:
        raise ValidationError(
            "The domain of the condition must be a subset of the domain of expr"
        )

    _validate_big_m(big_m, controlled)
    domain = _declaration_domain(index)

    if native:
        if big_m is not None:
            raise ValidationError("big_m cannot be used with native=True")

        return _add_native_indicator(
            indicator_var, indicator_val, expr, condition, domain, index, key
        )

    lhs_minus_rhs = expr.left - expr.right  # ty: ignore[unsupported-operator]
    if expr.operator == "=l=":
        rows = [("", lhs_minus_rhs)]
    elif expr.operator == "=g=":
        rows = [("", -lhs_minus_rhs)]
    else:
        rows = [("le_", lhs_minus_rhs), ("ge_", -lhs_minus_rhs)]

    active = indicator_var if indicator_val == 1 else 1 - indicator_var

    m = indicator_var.container
    result = FormulationResult()
    if big_m is not None:
        for prefix, row in rows:
            # enforces row <= 0 when active == 1
            result.equations_created[f"{prefix}big_m"] = _add_equation(
                m, domain, key, row <= big_m * (1 - active), condition
            )
        return result

    # SOS1 allows either active or the slack to be nonzero, not both
    sos1_var = _add_sos1_pair(m, domain)
    result.variables_created["sos1"] = sos1_var
    for prefix, row in rows:
        result.equations_created[f"{prefix}slack"] = _add_equation(
            m, domain, key, row <= sos1_var[[*index, "1"]], condition
        )
    result.equations_created["link"] = _add_equation(
        m, domain, key, sos1_var[[*index, "0"]] == active, condition
    )

    return result
