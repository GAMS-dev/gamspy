from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeVar

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType
    from typing import TextIO

    import pandas as pd

    from gamspy import Container, Equation, Model, Options, Parameter, Variable
    from gamspy._algebra.number import Number
    from gamspy._backend.engine import EngineClient
    from gamspy._backend.neos import NeosClient
    from gamspy._types import ConditionType
    from gamspy.math import MathOp

_SQUARES_PROBLEMS = {"LP": "QCP", "RMIP": "RMIQCP", "MIP": "MIQCP", "CNS": "NLP"}
_COUNT_PROBLEMS = {
    "LP": "MIP",
    "QCP": "MIQCP",
    "NLP": "MINLP",
    "DNLP": "MINLP",
    "CNS": "MINLP",
}
_RELAXED_PROBLEMS = ("RMIP", "RMIQCP", "RMINLP")
_Value = TypeVar("_Value")


class Slack(NamedTuple):
    """
    Slack variables of a relaxed equation, declared over its domain.

    Attributes
    ----------
    positive : Variable
        Violation of an inequality, or the amount by which the left-hand side
        of an equality exceeds the right-hand side.
    negative : Variable | None
        Amount by which the left-hand side of an equality falls below the
        right-hand side. None for inequalities.
    """

    positive: Variable
    negative: Variable | None


@dataclass
class FeasibilityRelaxation:
    """
    Result of :meth:`gamspy.Model.feasibilityRelaxation`.

    Attributes
    ----------
    model : Model
        Minimizes the total weighted violation.
    optimality_model : Model | None
        Optimizes the objective of the original model subject to the relaxed
        equations. None if the original model has no objective.
    violation : Variable
        Total weighted violation.
    slacks : dict[Equation, Slack]
        Slack variables of each relaxed equation as a named tuple
        ``(positive, negative)`` over its domain. ``positive`` is the violation
        of an inequality or the excess of the left-hand side of an equality,
        ``negative`` the shortfall of the left-hand side of an equality and
        None for inequalities. With the ``count`` measure,
        a slack can exceed the violation; use :meth:`violations` to get the
        violations.
    """

    model: Model
    optimality_model: Model | None
    violation: Variable
    slacks: dict[Equation, Slack]
    _rows: dict[Equation, tuple[Equation, str]] = field(
        default_factory=dict, repr=False
    )

    def solve(
        self,
        tolerance: float = 0.0,
        solver: str | None = None,
        options: Options | None = None,
        solver_options: dict | None = None,
        output: TextIO | None = None,
        backend: Literal["local", "engine", "neos"] = "local",
        client: EngineClient | NeosClient | None = None,
    ) -> pd.DataFrame | None:
        """
        Solves ``model`` and, if it finds a solution, bounds ``violation`` by
        its level plus ``tolerance`` and solves ``optimality_model``. The bound
        is removed afterwards. The remaining arguments are passed to
        :meth:`gamspy.Model.solve`.

        Parameters
        ----------
        tolerance : float, optional
            Violation that ``optimality_model`` may add to the minimum.
            By default 0.0.

        Returns
        -------
        pd.DataFrame | None
            Summary of the last solve.
        """

        def run(model: Model) -> pd.DataFrame | None:
            return model.solve(
                solver=solver,
                options=options,
                solver_options=solver_options,
                output=output,
                backend=backend,
                client=client,
            )

        self.violation.up = gp.SpecialValues.POSINF
        summary = run(self.model)
        solved = (
            gp.ModelStatus.OptimalGlobal,
            gp.ModelStatus.OptimalLocal,
            gp.ModelStatus.Feasible,
            gp.ModelStatus.Integer,
        )
        if self.optimality_model is None or self.model.status not in solved:
            return summary

        self.violation.up = self.violation.l + tolerance
        try:
            return run(self.optimality_model)
        finally:
            self.violation.up = gp.SpecialValues.POSINF

    def violations(self, tolerance: float = 1e-6) -> dict[str, pd.DataFrame]:
        """
        Returns the violated rows of the relaxed equations after a solve.

        Parameters
        ----------
        tolerance : float, optional
            Violations up to this value are ignored. By default 1e-6.

        Returns
        -------
        dict[str, pd.DataFrame]
            Equation names mapped to their violated rows. The ``violation``
            column is the left-hand side minus the right-hand side, so it is
            negative if the left-hand side is too small. Equations without
            violations are omitted.

        Raises
        ------
        ValidationError
            If the relaxation has not been solved.
        """
        result = {}
        for equation, (row, operator) in self._rows.items():
            records = row.records
            if records is None:
                raise ValidationError(
                    "The relaxation must be solved before computing violations."
                )

            columns = list(records.columns[: row.dimension])
            if columns:
                records = records.set_index(columns)

            positive, negative = self.slacks[equation]
            level = _level(positive, records.index, columns)
            if operator == "=g=":
                difference = records["level"] - records["lower"] - level
            else:
                difference = records["level"] - records["upper"] + level

            if negative is not None:
                difference = difference - _level(negative, records.index, columns)

            if operator == "=l=":
                violated = difference > tolerance
            elif operator == "=g=":
                violated = difference < -tolerance
            else:
                violated = difference.abs() > tolerance

            if violated.any():
                frame = difference[violated].rename("violation").to_frame()
                result[equation.name] = (
                    frame.reset_index() if columns else frame.reset_index(drop=True)
                )

        return result


def _level(slack: Variable, index: pd.Index, columns: list[str]) -> pd.Series:
    records = slack.records
    if records is None:
        raise ValidationError(
            "The relaxation must be solved before computing violations."
        )

    if not columns:
        return records["level"]

    return records.set_index(columns)["level"].reindex(index).fillna(0.0)


def _name(container: Container, name: str, prefix: str) -> str:
    taken = name in container or name.casefold() in (
        model.casefold() for model in container.models
    )
    if taken or len(name) > 63:
        return utils._get_unique_name(prefix)

    return name


def _problem(problem: gp.Problem, measure: str) -> gp.Problem:
    if measure == "count":
        if problem in _RELAXED_PROBLEMS:
            raise ValidationError(
                f"`count` cannot be used with relaxed problem type {problem}."
            )
        return gp.Problem(_COUNT_PROBLEMS.get(problem, problem))

    if measure == "squares":
        return gp.Problem(_SQUARES_PROBLEMS.get(problem, problem))

    return gp.Problem.NLP if problem == gp.Problem.CNS else problem


def _targets(
    model: Model,
    equations: Sequence[Equation] | None,
    weights: dict[Equation, int | float | Parameter],
    big_m: dict[Equation, int | float | Parameter],
) -> tuple[list[Equation], list[Equation]]:
    if model._matches:
        raise ValidationError(
            "Feasibility relaxation is not supported for models with matches."
        )

    model_equations = [
        equation
        for equation in model.equations
        if equation.name not in gp.Model._autogen_symbols
    ]
    model_names = {equation.name for equation in model_equations}
    targets = model_equations if equations is None else list(equations)
    for equation in targets:
        if equation.name not in model_names:
            raise ValidationError(
                f"`{equation.name}` is not an equation of model `{model.name}`."
            )

    target_names = {equation.name for equation in targets}
    for label, given in (("Weight", weights), ("Big-M", big_m)):
        for equation in given:
            if equation.name not in target_names:
                raise ValidationError(
                    f"{label} given for `{equation.name}`, which is not relaxed."
                )

    return model_equations, targets


def _indexed(
    value: _Value, key: tuple | EllipsisType
) -> _Value | implicits.ImplicitParameter:
    # a weight or big-M Parameter over the domain of its equation
    if isinstance(value, gp.Parameter) and value.dimension > 0:
        return value[key]

    return value


def _penalty(
    term: expression.Expression | implicits.ImplicitVariable | MathOp,
    index: list,
    where: ConditionType | Number | bool | int | float | None,
) -> operable.Operable:
    controlled = [elem for elem in index if not isinstance(elem, str)]
    if not controlled:
        return term if where is None else term.where[where]

    domain = gp.Domain(*controlled)
    return gp.Sum(domain if where is None else domain.where[where], term)


def _slack(container: Container, equation: Equation, operator: str) -> Slack:
    def variable(suffix: str, description: str) -> Variable:
        return gp.Variable(
            container,
            name=_name(container, f"{equation.name}_{suffix}", "slack"),
            type="positive",
            domain=equation.domain,
            description=f"{description} {equation.name}",
        )

    if operator != "=e=":
        return Slack(variable("slack", "violation of"), None)

    return Slack(
        variable("slack", "excess of the left-hand side of"),
        variable("slack_neg", "shortfall of the left-hand side of"),
    )


def _amount(
    measure: str,
    slack: Slack,
    key: tuple | EllipsisType,
    where: ConditionType | Number | bool | int | float | None,
    equation: Equation,
    replaced: list[Equation],
    big_m: int | float | Parameter | implicits.ImplicitParameter | None,
) -> expression.Expression | implicits.ImplicitVariable | MathOp:
    positive, negative = slack
    if measure == "squares":
        squares = gp.math.sqr(positive[key])
        return squares if negative is None else squares + gp.math.sqr(negative[key])

    total = positive[key] if negative is None else positive[key] + negative[key]
    if measure == "sum":
        return total

    violated = gp.Variable(
        equation.container,
        name=_name(equation.container, f"{equation.name}_violated", "violated"),
        type="binary",
        domain=equation.domain,
        description=f"{equation.name} is violated",
    )
    # SOS1 needs no bounds but is not supported by every solver, e.g. BARON
    # and KNITRO. Big-M (total <= big_m * violated) works with any solver.
    zero = total <= 0
    linked = gp.formulations.indicator(
        violated[key], 0, zero if where is None else zero.where[where], big_m=big_m
    )
    replaced.extend(linked.equations_created.values())
    return violated[key]


def relax(
    model: Model,
    equations: Sequence[Equation] | None,
    weights: dict[Equation, int | float | Parameter] | None,
    measure: Literal["sum", "squares", "count"],
    big_m: int | float | dict[Equation, int | float | Parameter] | None = None,
) -> FeasibilityRelaxation:
    if measure not in ("sum", "squares", "count"):
        raise ValidationError(
            f"`measure` must be 'sum', 'squares' or 'count' but found `{measure}`."
        )

    if big_m is not None and measure != "count":
        raise ValidationError("`big_m` can only be used with the `count` measure.")

    weights = {} if weights is None else weights
    model_equations, targets = _targets(
        model, equations, weights, big_m if isinstance(big_m, dict) else {}
    )
    container = model.container
    problem = _problem(model.problem, measure)
    replaced: dict[str, list[Equation]] = {}
    rows_by_equation: dict[Equation, tuple[Equation, str]] = {}
    slacks: dict[Equation, Slack] = {}
    penalties: list[operable.Operable] = []
    for equation in targets:
        definition = equation._definition
        if (
            equation._indicator is not None
            or definition is None
            or not isinstance(definition.right, expression.Expression)
        ):
            raise ValidationError(f"Equation `{equation.name}` cannot be relaxed.")

        operator = definition.right.operator
        if operator == "=n=":
            continue

        if operator not in ("=e=", "=l=", "=g="):
            raise ValidationError(
                f"Equation `{equation.name}` of type `{equation.type}` cannot be relaxed."
            )

        left = definition.left
        where = None
        if isinstance(left, condition.Condition):
            where = left.condition
            left = left.conditioning_on

        assert isinstance(left, implicits.ImplicitEquation)
        index = left._reference_domain()
        key = tuple(index) if index else ...

        slack = _slack(container, equation, operator)
        slacks[equation] = slack
        positive, negative = slack

        lhs, rhs = definition.right.left, definition.right.right
        assert isinstance(lhs, operable.Operable)
        assert isinstance(rhs, operable.Operable | int | float)
        if negative is not None:
            row = lhs - positive[key] + negative[key] == rhs
        elif operator == "=l=":
            row = lhs - positive[key] <= rhs
        else:
            row = lhs + positive[key] >= rhs

        relaxed = gp.Equation(
            container,
            name=_name(container, f"{equation.name}_relaxed", "relaxed"),
            domain=equation.domain,
            description=f"relaxed {equation.name}",
        )
        if where is None:
            relaxed[key] = row
        else:
            relaxed[key].where[where] = row

        replaced[equation.name] = [relaxed]
        rows_by_equation[equation] = (relaxed, operator)

        weight = _indexed(weights.get(equation, 1), key)
        amount = _amount(
            measure,
            slack,
            key,
            where,
            equation,
            replaced[equation.name],
            _indexed(big_m.get(equation) if isinstance(big_m, dict) else big_m, key),
        )

        is_one = isinstance(weight, int | float) and weight == 1
        penalties.append(_penalty(amount if is_one else weight * amount, index, where))

    if not penalties:
        raise ValidationError("There are no equations to relax.")

    violation = gp.Variable(
        container,
        name=_name(container, f"{model.name}_violation", "violation"),
        description="total weighted violation",
    )
    violation_definition = gp.Equation(
        container,
        name=_name(container, f"{model.name}_violation_def", "violation"),
        definition=violation == reduce(lambda a, b: a + b, penalties),
    )

    relaxed_equations = [
        relaxed
        for equation in model_equations
        for relaxed in replaced.get(equation.name, [equation])
    ]
    relaxed_equations.append(violation_definition)

    relaxed_model = gp.Model(
        container,
        name=_name(container, f"{model.name}_feasrelax", "feasrelax"),
        problem=problem,
        equations=relaxed_equations,
        sense=gp.Sense.MIN,
        objective=violation,
        limited_variables=model._limited_variables,
    )

    optimality_model = None
    if model.sense != gp.Sense.FEASIBILITY:
        optimality_model = gp.Model(
            container,
            name=_name(container, f"{model.name}_feasopt", "feasopt"),
            problem=problem,
            equations=relaxed_equations,
            sense=model.sense,
            objective=model._objective,
            limited_variables=model._limited_variables,
        )

    return FeasibilityRelaxation(
        relaxed_model, optimality_model, violation, slacks, rows_by_equation
    )
