from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeGuard

import gamspy as gp
from gamspy._algebra.condition import Condition
from gamspy._algebra.expression import Expression, ShiftExpression
from gamspy._algebra.operable import Operable
from gamspy._symbols.implicits import ImplicitSet, ImplicitVariable
from gamspy._universe import Universe
from gamspy.exceptions import ValidationError
from gamspy.formulations.result import FormulationResult
from gamspy.formulations.utils import _domain_name

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gamspy._algebra.number import Number
    from gamspy._types import ConditionType, DomainElementType, NormalizedDomainType

    BigMType: TypeAlias = int | float | Operable
    BinaryType: TypeAlias = gp.Variable | ImplicitVariable
    OperandType: TypeAlias = int | float | Operable
    OverType: TypeAlias = DomainElementType | Sequence[DomainElementType]
    EquationIndexType: TypeAlias = list[DomainElementType | ImplicitSet]
    WhereType: TypeAlias = ConditionType | Number | bool | int | float | None


_RELATIONAL_OPERATORS = {"=l=", "=e=", "=g="}


def _is_binary(symbol: object) -> bool:
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


def _declaration_domain(var_index: NormalizedDomainType) -> NormalizedDomainType:
    # multidimensional subsets such as ij(i,j) cannot be declaration domains,
    # the symbols are declared over (i,j) and indexed with ij instead
    domain: NormalizedDomainType = []
    for set_ in var_index:
        if _is_multidimensional(set_):
            domain.extend(set_.domain)
        else:
            domain.append(set_)

    return domain


def _index_sets(domain: Sequence[DomainElementType]) -> NormalizedDomainType:
    # a lag or lead such as t - 1 is controlled by t, e.g. x(t) - x(t-1) by t
    sets: NormalizedDomainType = []
    for set_ in domain:
        if isinstance(set_, ShiftExpression):
            set_ = set_.parent
        if _domain_name(set_) not in _domain_names(sets):
            sets.append(set_)

    return sets


def _dimension(set_: DomainElementType) -> int:
    return set_.dimension if isinstance(set_, (gp.Set, gp.Alias)) else 1


def _is_multidimensional(set_: object) -> TypeGuard[gp.Set | gp.Alias]:
    return isinstance(set_, (gp.Set, gp.Alias)) and set_.dimension > 1


def _control_index(
    domain: NormalizedDomainType,
) -> tuple[NormalizedDomainType, EquationIndexType, set[str]]:
    # defining the equations over ij(i,j) instead of ij also controls i and j,
    # e.g. eq(ij(i,j)) .. x(ij) =l= b(i)
    expanded: dict[str, ImplicitSet] = {}
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
    var_index = [
        set_
        for set_ in domain
        if _is_multidimensional(set_) or _domain_name(set_) not in components
    ]
    equation_index = [expanded.get(_domain_name(set_), set_) for set_ in var_index]

    return var_index, equation_index, _domain_names(var_index) | components


def _indicator_positions(
    indicator_var: BinaryType, var_index: NormalizedDomainType
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
    for set_ in var_index:
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
    indicator_val: Literal[0, 1],
    expr: Expression,
    condition: WhereType,
    domain: NormalizedDomainType,
    var_index: NormalizedDomainType,
    equation_index: EquationIndexType,
) -> FormulationResult:
    m = indicator_var.container
    positions = _indicator_positions(indicator_var, var_index)
    binary = (
        indicator_var.parent
        if isinstance(indicator_var, ImplicitVariable)
        else indicator_var
    )

    result = FormulationResult()
    equation = _add_equation(m, domain, equation_index, expr, condition)

    # GAMS generates only the variables that appear in the equations of the
    # model but indicator_var might only appear in the options file
    if equation_index:
        sum_domain = gp.Domain(*equation_index)
        if condition is not None:
            sum_domain = sum_domain.where[condition]
        generation = gp.Sum(sum_domain, indicator_var) >= 0
    else:
        generated = (
            indicator_var if condition is None else indicator_var.where[condition]
        )
        generation = generated >= 0

    result.equations_created["indicator"] = equation
    result.equations_created["binary"] = m.addEquation(
        name=gp.utils._get_unique_name("binary"), definition=generation
    )

    # Model adds the generation equation if the indicator equation is in the model
    equation._indicator = (
        binary,
        positions,
        indicator_val,
        result.equations_created["binary"],
    )
    return result


def _add_sos1_pair(m: gp.Container, domain: NormalizedDomainType) -> gp.Variable:
    # at most one of [*domain, "0"] and [*domain, "1"] can be nonzero
    sos_dim = gp.math._generate_dims(m, [2])[0]
    return m.addVariable(
        name=gp.utils._get_unique_name("sos1"),
        domain=[*domain, sos_dim],
        type="sos1",
    )


def _add_equation(
    m: gp.Container,
    domain: NormalizedDomainType,
    equation_index: EquationIndexType,
    definition: Expression,
    condition: WhereType,
) -> gp.Equation:
    equation = m.addEquation(name=gp.utils._get_unique_name("equation"), domain=domain)
    indices = equation_index if equation_index else ...
    if condition is None:
        equation[indices] = definition
    else:
        equation[indices].where[condition] = definition

    return equation


def indicator(
    indicator_var: BinaryType,
    indicator_val: Literal[0, 1],
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

    var_index, equation_index, controlled = _control_index(_index_sets(expr.domain))
    if not _domain_names(_index_sets(indicator_var.domain)) <= controlled:
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
    domain = _declaration_domain(var_index)

    if native:
        if big_m is not None:
            raise ValidationError("big_m cannot be used with native=True")

        return _add_native_indicator(
            indicator_var,
            indicator_val,
            expr,
            condition,
            domain,
            var_index,
            equation_index,
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
                m, domain, equation_index, row <= big_m * (1 - active), condition
            )
        return result

    # SOS1 allows either active or the slack to be nonzero, not both
    sos1_var = _add_sos1_pair(m, domain)
    result.variables_created["sos1"] = sos1_var
    for prefix, row in rows:
        result.equations_created[f"{prefix}slack"] = _add_equation(
            m, domain, equation_index, row <= sos1_var[[*var_index, "1"]], condition
        )
    result.equations_created["link"] = _add_equation(
        m, domain, equation_index, sos1_var[[*var_index, "0"]] == active, condition
    )

    return result


def _validate_operand(operand: object) -> None:
    if isinstance(operand, (int, float)) and not isinstance(operand, bool):
        if not math.isfinite(operand):
            raise ValidationError("operands must be finite numbers")

        return

    if isinstance(operand, Expression) and operand.operator in _RELATIONAL_OPERATORS:
        raise ValidationError(
            "operands cannot be inequalities or equalities, e.g. use x - 1"
            " instead of x <= 1"
        )

    if not isinstance(operand, Operable) or isinstance(
        operand, (gp.Set, gp.Alias, ImplicitSet)
    ):
        raise ValidationError(
            "operands must be numbers, parameters, variables or expressions"
        )


def _reduction(
    operands: Sequence[OperandType],
    over: OverType | None,
) -> tuple[
    gp.Container,
    NormalizedDomainType,
    list[list[DomainElementType]],
    list[tuple[NormalizedDomainType, EquationIndexType, set[str]]],
]:
    if over is None:
        over = []
    elif isinstance(over, (gp.Set, gp.Alias, gp.UniverseAlias, Universe, str)):
        over = [over]
    reduced = _domain_names(over)

    domains = [_index_sets(getattr(operand, "domain", [])) for operand in operands]
    if not reduced <= _domain_names([d for domain in domains for d in domain]):
        raise ValidationError("over must be a subset of the domain of the operands")

    # the domain of the output is the union of the operand domains except over
    output_domain: NormalizedDomainType = []
    for domain in domains:
        for set_ in domain:
            name = _domain_name(set_)
            if name not in reduced and name not in _domain_names(output_domain):
                output_domain.append(set_)

    containers = [getattr(operand, "container", None) for operand in operands]
    m = next((container for container in containers if container is not None), None)
    if m is None:
        raise ValidationError("At least one operand must be a parameter or a variable")

    reduced_domains = [
        [set_ for set_ in domain if _domain_name(set_) in reduced] for domain in domains
    ]
    indices = [
        _control_index([*output_domain, *reduced_domain])
        for reduced_domain in reduced_domains
    ]

    return m, output_domain, reduced_domains, indices


def _add_output(
    m: gp.Container,
    domain: NormalizedDomainType,
    type: str,
) -> tuple[
    FormulationResult,
    gp.Variable | ImplicitVariable,
    NormalizedDomainType,
    EquationIndexType,
]:
    var_index, equation_index, _ = _control_index(domain)
    declaration = _declaration_domain(var_index)
    output_var = m.addVariable(
        name=gp.utils._get_unique_name("output_var"),
        type=type,
        domain=declaration,
    )

    result = FormulationResult(output_var)
    result.variables_created["output"] = output_var

    return (
        result,
        output_var[var_index] if var_index else output_var,
        declaration,
        equation_index,
    )


def _extremum(
    sense: Literal["min", "max"],
    operands: tuple[OperandType, ...],
    *,
    over: OverType | None,
    big_m: BigMType | None,
    native: bool,
) -> FormulationResult:
    if len(operands) == 0:
        raise ValidationError("At least one operand is required")

    for operand in operands:
        _validate_operand(operand)

    m, output_domain, reduced_domains, indices = _reduction(operands, over)

    for _, _, controlled in indices:
        _validate_big_m(big_m, controlled)

    if native and big_m is not None:
        raise ValidationError("big_m cannot be used with native=True")

    result, output, output_declaration, output_equation_index = _add_output(
        m, output_domain, "free"
    )

    # output is bounded by every operand and equals the selected one
    selections = []
    for k, (operand, reduced_domain, (var_index, equation_index, _)) in enumerate(
        zip(operands, reduced_domains, indices, strict=True)
    ):
        declaration = _declaration_domain(var_index)

        select_var = m.addVariable(
            name=gp.utils._get_unique_name("select_var"),
            type="binary",
            domain=declaration,
        )
        select = select_var[var_index] if var_index else select_var
        result.variables_created[f"select_{k}"] = select_var

        if sense == "max":
            bound, selected = output >= operand, output <= operand
        else:
            bound, selected = output <= operand, output >= operand

        result.equations_created[f"bound_{k}"] = _add_equation(
            m, declaration, equation_index, bound, None
        )
        selection = indicator(select, 1, selected, big_m=big_m, native=native)
        for name, variable in selection.variables_created.items():
            result.variables_created[f"{name}_{k}"] = variable
        for name, equation in selection.equations_created.items():
            result.equations_created[f"{name}_{k}"] = equation

        selections.append(
            gp.Sum(reduced_domain, select)  # ty: ignore[invalid-argument-type]
            if reduced_domain
            else select
        )

    result.equations_created["pick_one"] = _add_equation(
        m,
        output_declaration,
        output_equation_index,
        sum(selections[1:], start=selections[0]) == 1,
        None,
    )

    return result


def minimum(
    *operands: OperandType,
    over: OverType | None = None,
    big_m: BigMType | None = None,
    native: bool = False,
) -> FormulationResult:
    """
    Creates a variable ``y`` that equals the minimum of ``operands``, i.e.
    ``y == min(operands[0], operands[1], ...)``. If ``over`` is given, the
    minimum is also taken over these sets, e.g. ``minimum(x[i, j], over=j)``
    creates ``y[i] == min(x[i, j] for j)``.

    The formulation is the same as for :meth:`maximum
    <gamspy.formulations.maximum>` with the inequalities reversed:

    .. math::
        y \\leq a_k \\quad \\forall{k}

        b_k = 1 \\implies y \\geq a_k \\quad \\forall{k}

        \\sum_{k}{b_k} = 1

        b_k \\in \\{0, 1\\} \\quad \\forall{k}

    The arguments, the domain of ``y``, the generated symbols and the keys of
    the FormulationResult are also the same. The only other difference is that
    ``big_m`` must be at least as large as ``a_k - y`` for every operand.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> j = gp.Set(m, "j", records=range(2))
    >>> x = gp.Variable(m, "x", domain=[i, j])
    >>> z = gp.Variable(m, "z", domain=i)
    >>> y, eqs = gp.formulations.minimum(z, 10)
    >>> [d.name for d in y.domain]
    ['i']
    >>> y, eqs = gp.formulations.minimum(x, over=j, big_m=100)
    >>> [d.name for d in y.domain]
    ['i']

    """
    return _extremum("min", operands, over=over, big_m=big_m, native=native)


def maximum(
    *operands: OperandType,
    over: OverType | None = None,
    big_m: BigMType | None = None,
    native: bool = False,
) -> FormulationResult:
    """
    Creates a variable ``y`` that equals the maximum of ``operands``, i.e.
    ``y == max(operands[0], operands[1], ...)``. If ``over`` is given, the
    maximum is also taken over these sets, e.g. ``maximum(x[i, j], over=j)``
    creates ``y[i] == max(x[i, j] for j)``.

    Each operand can be a number, a parameter, a variable or an expression.
    The domain of ``y`` is the union of the domains of the operands except
    ``over``. If an operand is indexed with a multidimensional subset, e.g.
    ``x[ij]`` with ``ij(i, j)``, ``y`` is declared over ``i`` and ``j`` but is
    only constrained for the elements of ``ij``. Index ``y`` with the subset,
    e.g. ``y[ij]``, since its other elements are free.

    For each operand ``a_k``, the formulation **generates** a binary variable
    ``b_k`` over the domain of ``y`` and the sets of ``over`` that are in the
    domain of the operand. ``b_k = 1`` selects the operand whose value ``y``
    takes:

    .. math::
        y \\geq a_k \\quad \\forall{k}

        b_k = 1 \\implies y \\leq a_k \\quad \\forall{k}

        \\sum_{k}{b_k} = 1

        b_k \\in \\{0, 1\\} \\quad \\forall{k}

    The ``bound_k`` equations keep ``y`` above every operand, the indicator
    constraints keep it below the selected operand and ``pick_one`` selects
    exactly one operand for each element of the domain of ``y``. The
    indicator constraints are modeled with :meth:`indicator
    <gamspy.formulations.indicator>`, which accepts ``big_m`` and ``native``.
    With ``big_m``, they become ``y - a_k <= big_m * (1 - b_k)``, so ``big_m``
    must be at least as large as ``y - a_k`` for every operand, e.g. the
    difference between the largest upper bound and the smallest lower bound
    of the operands.

    FormulationResult:
        - variables_created: ["output", "select_k"] and the variables of the indicator constraints suffixed with ``_k``
        - equations_created: ["bound_k", "pick_one"] and the equations of the indicator constraints suffixed with ``_k``
        - ``k`` is the position of the operand in ``operands``.

    Parameters
    ----------
    *operands : int | float | Operable
        Operands of the maximum.
    over : Set | Alias | Sequence[Set | Alias] | None, optional
        Sets to take the maximum over.
    big_m : int | float | Operable | None, optional
        Big-M value of the indicator constraints.
    native : bool, optional
        Pass the indicator constraints to the solver instead of reformulating
        them.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> j = gp.Set(m, "j", records=range(2))
    >>> x = gp.Variable(m, "x", domain=[i, j])
    >>> z = gp.Variable(m, "z", domain=i)
    >>> res = gp.formulations.maximum(z, 0)
    >>> list(res.variables_created)
    ['output', 'select_0', 'sos1_0', 'select_1', 'sos1_1']
    >>> y, eqs = gp.formulations.maximum(x, over=j, big_m=100)
    >>> [d.name for d in y.domain]
    ['i']
    >>> y, eqs = gp.formulations.maximum(x[i, j], over=[i, j])
    >>> y.domain
    []

    """
    return _extremum("max", operands, over=over, big_m=big_m, native=native)


def absolute(
    operand: Operable,
    *,
    big_m: BigMType | None = None,
    native: bool = False,
) -> FormulationResult:
    """
    Creates a variable ``y`` that equals the absolute value of ``operand``,
    i.e. ``y == abs(operand)``. The operand can be a parameter, a variable or
    an expression, and ``y`` has the same domain.

    The absolute value is modeled as ``maximum(operand, -operand)``, so the
    generated symbols, the keys of the FormulationResult and the ``big_m`` and
    ``native`` arguments are the same as for :meth:`maximum
    <gamspy.formulations.maximum>`. ``big_m`` must be at least twice the
    largest absolute value that ``operand`` can take.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> x = gp.Variable(m, "x", domain=i)
    >>> y, eqs = gp.formulations.absolute(x)
    >>> [d.name for d in y.domain]
    ['i']
    >>> y, eqs = gp.formulations.absolute(x - 5, big_m=100)
    >>> len(eqs)
    5

    """
    _validate_operand(operand)
    return _extremum("max", (operand, -operand), over=None, big_m=big_m, native=native)


def _validate_binary_operands(operands: Sequence[BinaryType]) -> None:
    if len(operands) == 0:
        raise ValidationError("At least one operand is required")

    for operand in operands:
        if not _is_binary(operand):
            raise ValidationError("operands must be binary variables")


def _logical(
    operator: Literal["and", "or"],
    operands: tuple[BinaryType, ...],
    over: OverType | None,
) -> FormulationResult:
    _validate_binary_operands(operands)
    m, output_domain, reduced_domains, indices = _reduction(operands, over)
    result, output, output_declaration, output_equation_index = _add_output(
        m, output_domain, "binary"
    )

    terms = []
    for k, (operand, reduced_domain, (var_index, equation_index, _)) in enumerate(
        zip(operands, reduced_domains, indices, strict=True)
    ):
        if operator == "and":
            name, bound, term = f"upper_{k}", output <= operand, 1 - operand
        else:
            name, bound, term = f"lower_{k}", output >= operand, operand

        result.equations_created[name] = _add_equation(
            m, _declaration_domain(var_index), equation_index, bound, None
        )
        terms.append(
            gp.Sum(reduced_domain, term)  # ty: ignore[invalid-argument-type]
            if reduced_domain
            else term
        )

    total = sum(terms[1:], start=terms[0])
    if operator == "and":
        name, definition = "lower_0", output >= 1 - total
    else:
        name, definition = "upper_0", output <= total

    result.equations_created[name] = _add_equation(
        m, output_declaration, output_equation_index, definition, None
    )

    return result


def logical_and(
    *operands: BinaryType,
    over: OverType | None = None,
) -> FormulationResult:
    """
    Creates a binary variable ``y`` that equals the logical AND of the binary
    ``operands``, i.e. ``y == 1`` if and only if every operand is 1. If
    ``over`` is given, the AND is also taken over these sets, e.g.
    ``logical_and(b[i, j], over=j)`` creates ``y[i] == 1`` if and only if
    ``b[i, j] == 1`` for every ``j``.

    The domain of ``y`` is the union of the domains of the operands except
    ``over``. If an operand is indexed with a multidimensional subset, e.g.
    ``b[ij]`` with ``ij(i, j)``, ``y`` is declared over ``i`` and ``j`` but is
    only constrained for the elements of ``ij``.

    The formulation is linear and exact for binary operands :math:`b_k`:

    .. math::
        y \\leq b_k \\quad \\forall{k}

        y \\geq 1 - \\sum_{k}{(1 - b_k)}

    With ``over``, the sum also ranges over the elements of these sets.

    FormulationResult:
        - variables_created: ["output"]
        - equations_created: ["upper_k", "lower_0"]
        - ``k`` is the position of the operand in ``operands``.

    Parameters
    ----------
    *operands : Variable | ImplicitVariable
        Binary variables to combine.
    over : Set | Alias | Sequence[Set | Alias] | None, optional
        Sets to take the AND over.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> j = gp.Set(m, "j", records=range(2))
    >>> a = gp.Variable(m, "a", type="binary", domain=i)
    >>> b = gp.Variable(m, "b", type="binary", domain=[i, j])
    >>> y, eqs = gp.formulations.logical_and(a, b)
    >>> [d.name for d in y.domain]
    ['i', 'j']
    >>> res = gp.formulations.logical_and(b, over=j)
    >>> list(res.equations_created)
    ['upper_0', 'lower_0']

    """
    return _logical("and", operands, over)


def logical_or(
    *operands: BinaryType,
    over: OverType | None = None,
) -> FormulationResult:
    """
    Creates a binary variable ``y`` that equals the logical OR of the binary
    ``operands``, i.e. ``y == 1`` if and only if at least one operand is 1.
    If ``over`` is given, the OR is also taken over these sets, e.g.
    ``logical_or(b[i, j], over=j)`` creates ``y[i] == 1`` if and only if
    ``b[i, j] == 1`` for some ``j``.

    The domain of ``y`` is determined as for :meth:`logical_and
    <gamspy.formulations.logical_and>`. The formulation is linear and exact
    for binary operands :math:`b_k`:

    .. math::
        y \\geq b_k \\quad \\forall{k}

        y \\leq \\sum_{k}{b_k}

    With ``over``, the sum also ranges over the elements of these sets.

    FormulationResult:
        - variables_created: ["output"]
        - equations_created: ["lower_k", "upper_0"]
        - ``k`` is the position of the operand in ``operands``.

    Parameters
    ----------
    *operands : Variable | ImplicitVariable
        Binary variables to combine.
    over : Set | Alias | Sequence[Set | Alias] | None, optional
        Sets to take the OR over.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> j = gp.Set(m, "j", records=range(2))
    >>> a = gp.Variable(m, "a", type="binary", domain=i)
    >>> b = gp.Variable(m, "b", type="binary", domain=[i, j])
    >>> y, eqs = gp.formulations.logical_or(a, b[i, "0"])
    >>> [d.name for d in y.domain]
    ['i']
    >>> res = gp.formulations.logical_or(a, b, over=j)
    >>> list(res.equations_created)
    ['lower_0', 'lower_1', 'upper_0']

    """
    return _logical("or", operands, over)


def logical_not(operand: BinaryType) -> FormulationResult:
    """
    Creates a binary variable ``y`` that equals the logical NOT of the binary
    ``operand``, i.e. ``y == 1 - operand``. ``y`` has the domain of the
    operand and can be passed to :meth:`logical_and
    <gamspy.formulations.logical_and>`, :meth:`logical_or
    <gamspy.formulations.logical_or>` or :meth:`indicator
    <gamspy.formulations.indicator>`.

    FormulationResult:
        - variables_created: ["output"]
        - equations_created: ["complement"]

    Parameters
    ----------
    operand : Variable | ImplicitVariable
        Binary variable to negate.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> a = gp.Variable(m, "a", type="binary", domain=i)
    >>> b = gp.Variable(m, "b", type="binary", domain=i)
    >>> not_b, _ = gp.formulations.logical_not(b)
    >>> y, eqs = gp.formulations.logical_and(a, not_b)
    >>> [d.name for d in y.domain]
    ['i']

    """
    _validate_binary_operands([operand])
    m, output_domain, _, _ = _reduction([operand], None)
    result, output, output_declaration, output_equation_index = _add_output(
        m, output_domain, "binary"
    )
    result.equations_created["complement"] = _add_equation(
        m, output_declaration, output_equation_index, output == 1 - operand, None
    )

    return result


def _logical_pair(
    operator: Literal["xor", "imp", "eqv"],
    left: BinaryType,
    right: BinaryType,
) -> FormulationResult:
    _validate_binary_operands([left, right])
    m, output_domain, _, _ = _reduction([left, right], None)
    result, output, output_declaration, output_equation_index = _add_output(
        m, output_domain, "binary"
    )

    # each inequality forces the correct y for one or two combinations of left and right
    # e.g. for xor, y <= 2 - left - right forces y = 0 only when left = right = 1,
    # and for imp, y >= 1 - left forces y = 1 whenever left = 0
    if operator == "xor":
        lower = [left - right, right - left]
        upper = [left + right, 2 - left - right]
    elif operator == "imp":
        lower = [1 - left, right]
        upper = [1 - left + right]
    else:
        lower = [left + right - 1, 1 - left - right]
        upper = [1 - left + right, 1 + left - right]

    for k, bound in enumerate(lower):
        result.equations_created[f"lower_{k}"] = _add_equation(
            m, output_declaration, output_equation_index, output >= bound, None
        )
    for k, bound in enumerate(upper):
        result.equations_created[f"upper_{k}"] = _add_equation(
            m, output_declaration, output_equation_index, output <= bound, None
        )

    return result


def logical_xor(left: BinaryType, right: BinaryType) -> FormulationResult:
    """
    Creates a binary variable ``y`` that equals the exclusive disjunction of
    the binary operands, i.e. ``y == 1`` if and only if exactly one of
    ``left`` and ``right`` is 1.

    The domain of ``y`` is the union of the domains of the operands. The
    formulation is linear and exact for binary operands :math:`a` and
    :math:`b`:

    .. math::
        y \\geq a - b, \\quad y \\geq b - a

        y \\leq a + b, \\quad y \\leq 2 - a - b

    FormulationResult:
        - variables_created: ["output"]
        - equations_created: ["lower_0", "lower_1", "upper_0", "upper_1"]

    Parameters
    ----------
    left : Variable | ImplicitVariable
        First binary operand.
    right : Variable | ImplicitVariable
        Second binary operand.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> j = gp.Set(m, "j", records=range(2))
    >>> a = gp.Variable(m, "a", type="binary", domain=i)
    >>> b = gp.Variable(m, "b", type="binary", domain=j)
    >>> y, eqs = gp.formulations.logical_xor(a, b)
    >>> [d.name for d in y.domain]
    ['i', 'j']

    """
    return _logical_pair("xor", left, right)


def logical_imp(left: BinaryType, right: BinaryType) -> FormulationResult:
    """
    Creates a binary variable ``y`` that equals the logical implication
    ``left => right``, i.e. ``y == 0`` if and only if ``left`` is 1 and
    ``right`` is 0.

    The domain of ``y`` is the union of the domains of the operands. The
    formulation is linear and exact for binary operands :math:`a` and
    :math:`b`:

    .. math::
        y \\geq 1 - a, \\quad y \\geq b

        y \\leq 1 - a + b

    To enforce the implication instead of computing its value, use the
    constraint ``left <= right``.

    FormulationResult:
        - variables_created: ["output"]
        - equations_created: ["lower_0", "lower_1", "upper_0"]

    Parameters
    ----------
    left : Variable | ImplicitVariable
        Binary premise.
    right : Variable | ImplicitVariable
        Binary conclusion.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> a = gp.Variable(m, "a", type="binary", domain=i)
    >>> b = gp.Variable(m, "b", type="binary", domain=i)
    >>> res = gp.formulations.logical_imp(a, b)
    >>> list(res.equations_created)
    ['lower_0', 'lower_1', 'upper_0']

    """
    return _logical_pair("imp", left, right)


def logical_eqv(left: BinaryType, right: BinaryType) -> FormulationResult:
    """
    Creates a binary variable ``y`` that equals the logical equivalence of
    the binary operands, i.e. ``y == 1`` if and only if ``left`` and
    ``right`` are equal.

    The domain of ``y`` is the union of the domains of the operands. The
    formulation is linear and exact for binary operands :math:`a` and
    :math:`b`:

    .. math::
        y \\geq a + b - 1, \\quad y \\geq 1 - a - b

        y \\leq 1 - a + b, \\quad y \\leq 1 + a - b

    To enforce the equivalence instead of computing its value, use the
    constraint ``left == right``.

    FormulationResult:
        - variables_created: ["output"]
        - equations_created: ["lower_0", "lower_1", "upper_0", "upper_1"]

    Parameters
    ----------
    left : Variable | ImplicitVariable
        First binary operand.
    right : Variable | ImplicitVariable
        Second binary operand.

    Returns
    -------
    FormulationResult

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=range(3))
    >>> a = gp.Variable(m, "a", type="binary", domain=i)
    >>> b = gp.Variable(m, "b", type="binary", domain=i)
    >>> y, eqs = gp.formulations.logical_eqv(a, b)
    >>> len(eqs)
    4

    """
    return _logical_pair("eqv", left, right)
