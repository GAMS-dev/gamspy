from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, cast

from gams.core.gdx import GMS_DT_EQU

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._algebra.sparse as sparse
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._internals import (
    EQU_TYPE,
    TRANSFER_TO_GAMS_EQUATION_SUBTYPES,
)
from gamspy._special_values import SpecialValues
from gamspy._symbols.base import VarEquSymbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from gamspy import Container, Variable
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._symbols.implicits import ImplicitEquation, ImplicitParameter
    from gamspy._types import DomainType, IndexType, VarEquRecordsType


EQ_TYPES = ["=e=", "=l=", "=g=", "=n=", "=x=", "=b="]
IRREGULAR_EQ_MAP = {"nonbinding": "=n=", "external": "=x=", "boolean": "=b="}
EQU_DEFAULT_VALUES = {
    "eq": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 0.0,
        "upper": 0.0,
        "scale": 1.0,
    },
    "geq": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 0.0,
        "upper": SpecialValues.POSINF,
        "scale": 1.0,
    },
    "leq": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": SpecialValues.NEGINF,
        "upper": 0.0,
        "scale": 1.0,
    },
    "nonbinding": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": SpecialValues.NEGINF,
        "upper": SpecialValues.POSINF,
        "scale": 1.0,
    },
    "external": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 0.0,
        "upper": 0.0,
        "scale": 1.0,
    },
    "boolean": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 0.0,
        "upper": 0.0,
        "scale": 1.0,
    },
}


class EquationType(Enum):
    """
    Enumeration of available equation types.
    """

    REGULAR = "regular"
    """Regular equations with =, >= and <= sign."""

    NONBINDING = "nonbinding"
    """No relationship implied between left-hand side and right-hand side. This equation type is ideally suited for use in MCP models and in variational inequalities."""

    EXTERNAL = "external"
    """Equation is defined by external programs."""

    BOOLEAN = "boolean"
    """Boolean equations."""

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Equation(VarEquSymbol):
    """
    Represents an Equation symbol in GAMS.

    Equations represent the constraints or relationships in a model. They can be
    defined using equality (==) or inequality (<=, >=) operators.
    See https://gamspy.readthedocs.io/en/latest/user/basics/equation.html

    Parameters
    ----------
    container : Container
        The Container object that this equation belongs to.
    name : str, optional
        Name of the equation. If not provided, a unique name is generated automatically.
    type : str, optional
        Type of the equation. Options: "regular", "nonbinding", "external", "boolean".
        Default is "regular".
    domain : DomainType, optional
        The domain of the equation. Can be a list of Sets/Aliases, a single Set/Alias,
        or strings representing set names. Use :data:`UNIVERSE <gamspy.UNIVERSE>` for the universe
        set (the bare string ``"*"`` is also accepted, but discouraged). Default is [] (scalar).
    definition : Variable | Operation | Expression, optional
        The mathematical definition of the equation. Can be set later via assignment.
    records : Sequence | np.ndarray | int | float | pd.DataFrame | pd.Series | dict, optional
        Initial records to populate the equation.
    domain_forwarding : bool | list[bool], optional
        If True, adding records to this equation will implicitly add new elements to the
        domain sets (if they are dynamic). Default is False.
    description : str, optional
        A human-readable description of the equation.
    uels_on_axes : bool, optional
        If True, implies that the Unique Element Labels (UELs) for the domain are
        contained in the axes (index/columns) of the provided `records` object.
    is_miro_output : bool, optional
        If True, flags this equation as an output symbol for GAMS MIRO. Default is False.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])
    >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
    >>> v = gp.Variable(m, "v", domain=[i])
    >>> e = gp.Equation(m, "e", domain=[i])
    >>> e[i] = a[i] <= v[i]

    """

    _attribute_slots: ClassVar[dict[str, str]] = {
        "l": "_l",
        "m": "_m",
        "lo": "_lo",
        "up": "_up",
        "scale": "_s",
        "stage": "_stage",
        "range": "_range",
        "slacklo": "_slacklo",
        "slackup": "_slackup",
        "slack": "_slack",
        "infeas": "_infeas",
    }
    _l: ImplicitParameter
    _m: ImplicitParameter
    _lo: ImplicitParameter
    _up: ImplicitParameter
    _s: ImplicitParameter
    _stage: ImplicitParameter
    _range: ImplicitParameter
    _slacklo: ImplicitParameter
    _slackup: ImplicitParameter
    _slack: ImplicitParameter
    _infeas: ImplicitParameter

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        type: str | EquationType = "regular",
        domain: DomainType | None = None,
        records: pd.DataFrame | None = None,
        description: str = "",
    ) -> Equation:
        obj = cast("Equation", object.__new__(cls))
        type = cast_type(type)
        obj._type = EQU_TYPE[type]
        obj._gams_type = cast("int", GMS_DT_EQU)
        obj._gams_subtype = cast("int", TRANSFER_TO_GAMS_EQUATION_SUBTYPES[type])
        obj._equation_listing = None
        obj._init_definition_state(None)
        obj._init_symbol(
            container,
            name,
            domain,
            records=records,
            description=description,
            validate=False,
        )
        obj._init_attributes()

        return obj

    def _redefine(
        self,
        container: Container | None = None,
        name: str | None = None,
        type: str | EquationType = "regular",
        domain: DomainType | None = None,
        definition: Variable | Operation | Expression | None = None,
        records: VarEquRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
        definition_domain: list | None = None,
    ):
        type = cast_type(type)
        if self.type != type.casefold():
            raise TypeError(
                "Cannot overwrite symbol in container unless equation"
                f" types are equal: `{self.type}` !="
                f" `{type.casefold()}`"
            )

        self._redefine_symbol(
            domain,
            domain_forwarding=domain_forwarding,
            description=description,
            is_miro_output=is_miro_output,
        )
        self._assignment = None
        self._equation_listing = None
        self._indicator = None
        self._definition_domain = definition_domain

        with self._miro_unprotected():
            self._init_definition(definition)

        self._redefine_records(records, uels_on_axes=uels_on_axes)

    def __init__(
        self,
        container: Container | None = None,
        name: str | None = None,
        type: str | EquationType = "regular",
        domain: DomainType | None = None,
        definition: Variable | Operation | Expression | None = None,
        records: VarEquRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
        definition_domain: list | None = None,
    ):
        container, name = self._resolve_container_and_name(
            container, name, prefix="e", is_miro=is_miro_output
        )
        self._type = cast_type(type)
        self._gams_type: int = GMS_DT_EQU
        self._gams_subtype: int = TRANSFER_TO_GAMS_EQUATION_SUBTYPES[self._type]
        self._equation_listing: list[str] | None = None
        self._init_definition_state(definition_domain)
        self._init_symbol(
            container,
            name,
            domain,
            description=description,
            domain_forwarding=domain_forwarding,
            is_miro_output=is_miro_output,
        )
        self._init_definition(definition)
        self._init_attributes()
        self._init_records(records, uels_on_axes=uels_on_axes)

    def _init_definition_state(self, definition_domain: list | None) -> None:
        self._definition: Expression | None = None
        # (binary variable, positions of its indices in the equation domain, value,
        # equation that makes GAMS generate the binary) of a native indicator
        self._indicator: tuple[Variable, tuple[int, ...], int, Equation] | None = None
        self._definition_domain = definition_domain

    def _serialize(self) -> dict:
        info = super()._serialize()

        if self._definition is not None:
            info["_definition"] = self._definition.getDeclaration()

        if self._indicator is not None:
            binary, positions, value, generation = self._indicator
            info["_indicator"] = [binary.name, list(positions), value, generation.name]

        return info

    def _deserialize_value(self, key: str, value: Any) -> Any:
        if key == "_definition":
            left, right = value.split(" .. ")
            return expression.Expression(left, "..", right[:-1])

        if key == "_indicator":
            binary_name, positions, indicator_value, generation_name = value
            return (
                self._container[binary_name],
                tuple(positions),
                indicator_value,
                self._container[generation_name],
            )

        return super()._deserialize_value(key, value)

    def __getitem__(self, indices: IndexType) -> ImplicitEquation:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitEquation(
            self,
            name=self.name,
            type=self.type,
            domain=domain,
        )

    def __setitem__(self, indices: IndexType, rhs: Expression):
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        self._set_definition(domain, rhs)

        self._container._synch_with_gams()

    def __repr__(self) -> str:
        return f"Equation(name='{self.name}', type='{self.type}', domain={self.domain})"

    def _init_definition(
        self,
        assignment: Variable | Operation | Expression | None = None,
    ):
        if assignment is None:
            return None

        domain = self.domain
        if self._definition_domain is not None:
            domain = validation.validate_domain(self, self._definition_domain)

        self._set_definition(domain, assignment)

    def _set_definition(self, domain, rhs):
        # self[domain] = rhs
        if isinstance(rhs, sparse.SparseAssignment):
            raise ValidationError(
                "`sparse` cannot be used in an equation definition because GAMS "
                "supports the sparse assignment operator ($=) in assignments only."
            )

        rhs_repr = rhs.gamsRepr()
        if self.type == "nonbinding" and not any(
            eq_type in rhs_repr for eq_type in EQ_TYPES
        ):
            # x - c -> x - c == 0
            rhs = rhs == gp.Number(0)

        rhs_repr = rhs.gamsRepr()
        if not any(eq_type in rhs_repr for eq_type in EQ_TYPES):
            raise ValidationError(
                "Equation definition must contain at least one equality sign such as ==, <= or >=."
            )

        if (
            self.type in IRREGULAR_EQ_MAP
            and "=e=" in rhs_repr
            and isinstance(rhs, expression.Expression)
        ):
            rhs = expression.Expression(
                rhs.left, IRREGULAR_EQ_MAP[self.type], rhs.right
            )

        if self.type == "external" and "=e=" not in rhs_repr:
            raise ValidationError("External equations must contain ==")

        statement = expression.Expression(
            implicits.ImplicitEquation(
                self,
                name=self.name,
                type=self.type,
                domain=domain,
            ),
            "..",
            rhs,
        )

        # Cannot validate definition if we are in a gp.Loop since the control indices can be provided by the gp.Loop
        if not self._container._in_loop:
            statement._validate_definition(utils._unpack(domain))

        self._container._add_statement(statement)
        self._definition = statement

    def _check_ambiguity(self) -> None:
        """Ambiguity check for MCP, EMP, MPEC models. See #610"""
        # Looks for =e=, =l= and =g= in an equation definition
        # with a stack based inorder traversal algorithm (O(N)).
        stack = []

        assert self._definition is not None
        node = self._definition.right
        while True:
            if node is not None:
                stack.append(node)
                node = getattr(node, "left", None)
            elif stack:
                node = stack.pop()
                if (
                    isinstance(node, expression.Expression)
                    and node.operator in {"=e=", "=l=", "=g=", "=x=", "=n="}
                    and not isinstance(node.right, operable.Operable)
                ):
                    raise ValidationError(
                        f"Definition of `{self.name}` is ambigiuous. Please "
                        "use gp.Number for numeric values or disable ambiguity "
                        "check via gp.set_options({'ALLOW_AMBIGUOUS_EQUATIONS': 'no'}). "
                        "Using numeric values in equations without gp.Number can result in "
                        f"different order than expected. Print `{self.name}.getDefinition()` to "
                        "make sure that the equation definition is as expected."
                    )
                node = getattr(node, "right", None)
            else:
                break  # pragma: no cover

    @property
    def l(self):
        """
        The Level of the equation (its current value).

        This corresponds to the `.l` suffix in GAMS. After a solve, this represents the
        value of the equation.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.l
        >>> repr.toValue()
        np.float64(10.0)

        """
        return self._l

    @l.setter
    def l(self, value: int | float | Expression):
        self._l[...] = value

    @property
    def m(self):
        """
        The Marginal (dual) value of the equation.

        This corresponds to the `.m` suffix in GAMS. It represents the shadow price
        or dual variable associated with the constraint.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.m
        >>> repr.toValue()
        np.float64(5.0)

        """
        return self._m

    @m.setter
    def m(self, value: int | float | Expression):
        self._m[...] = value

    @property
    def lo(self):
        """
        The Lower Bound of the equation.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.lo
        >>> repr.toValue()
        np.float64(-inf)

        """
        return self._lo

    @lo.setter
    def lo(self, value: int | float | Expression):
        self._lo[...] = value

    @property
    def up(self):
        """
        The Upper Bound of the equation.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.up
        >>> repr.toValue()
        np.float64(10.0)

        """
        return self._up

    @up.setter
    def up(self, value: int | float | Expression):
        self._up[...] = value

    @property
    def scale(self):
        """
        The Scale factor of the equation.

        This corresponds to the `.scale` suffix in GAMS, used for scaling the equation
        to improve numerical stability during solving.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.scale
        >>> repr.toValue()
        np.float64(1.0)

        """
        return self._s

    @scale.setter
    def scale(self, value: int | float | Expression):
        self._s[...] = value

    @property
    def stage(self):
        """
        The Stage of the equation.

        This corresponds to the `.stage` suffix in GAMS, often used in stochastic programming
        or model translation contexts.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.stage
        >>> repr.toValue()
        np.float64(1.0)

        """
        return self._stage

    @stage.setter
    def stage(self, value: int | float | Expression):
        self._stage[...] = value

    @property
    def range(self):
        """
        The Range of the equation.

        This corresponds to the `.range` suffix in GAMS. It is used to define the sensitivity
        range for range constraints.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.range
        >>> repr.toValue()
        np.float64(inf)

        """
        return self._range

    @property
    def slacklo(self):
        """
        The lower bound slack of the equation.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.slacklo
        >>> repr.toValue()
        np.float64(inf)

        """
        return self._slacklo

    @property
    def slackup(self):
        """
        The upper bound slack of the equation.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.slackup
        >>> repr.toValue()
        np.float64(0.0)

        """
        return self._slackup

    @property
    def slack(self):
        """
        The Slack of the equation.

        This corresponds to the `.slack` suffix. It represents the distance from the
        equation's bound (e.g., RHS - LHS for <= equations).

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.slack
        >>> repr.toValue()
        np.float64(0.0)

        """
        return self._slack

    @property
    def infeas(self):
        """
        The Infeasibility of the equation.

        Returns the amount by which the equation violates its bounds.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> repr = Parameter(m, "repr")
        >>> repr[...] = eq.infeas
        >>> repr.toValue()
        np.float64(0.0)

        """
        return self._infeas

    def computeInfeasibilities(self) -> pd.DataFrame | None:
        """
        Computes infeasibilities of the equation.

        Checks if the level value `.l` violates the bounds `.lo` and `.up`
        and returns a DataFrame containing the violations.

        Returns
        -------
        pd.DataFrame | None

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> e = gp.Equation(m, "e")
        >>> e.l[...] = -10
        >>> e.lo[...] = 5
        >>> e.computeInfeasibilities().values.tolist()
        [[-10.0, 0.0, 5.0, inf, 1.0, 15.0]]

        """
        return utils._calculate_infeasibilities(self)

    def getEquationListing(
        self,
        n: int | None = None,
        filters: list[list[str]] | None = None,
        infeasibility_threshold: float | None = None,
    ) -> str:
        """
        Returns the equation listing (log output) from the last solve.

        This requires the model to have been solved with the `equation_listing_limit` option enabled.

        Parameters
        ----------
        n : int, optional
            Maximum number of equations to return.
        filters : list[list[str]], optional
            Filters to select specific elements for the listing.
            The list size must match the equation's dimension.
        infeasibility_threshold: float, optional
            If set, only returns equations with infeasibility values greater than this threshold.

        Returns
        -------
        str
            The text listing of the generated equations.

        Raises
        ------
        ValidationError
            If the model was not solved with `equation_listing_limit`.
        ValidationError
            In case the length of the filters is different than the dimension of the equation.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, records=["item1", "item2"])
        >>> v = gp.Variable(m, domain=i)
        >>> z = gp.Variable(m)
        >>> e = gp.Equation(m, domain=i)
        >>> e[i] = v[i] * z >= 5
        >>> model = gp.Model(m, "test", equations=[e], problem="NLP", sense="MIN", objective=z)
        >>> summary = model.solve(options=gp.Options(equation_listing_limit=10))
        >>> print(e.getEquationListing())
        e(item1)..  (0)*v(item1) + (0)*z =G= 5 ; (LHS = 0, INFES = 5 ****)
        e(item2)..  (0)*v(item2) + (0)*z =G= 5 ; (LHS = 0, INFES = 5 ****)

        """
        if self._equation_listing is None:
            raise ValidationError(
                "The model must be solved with `equation_listing_limit` option for this functionality to work."
            )

        listings = self._equation_listing if filters is None else []

        if filters is not None:
            for listing in self._equation_listing:
                lhs, _ = listing.split("..")
                # symbol(elem1, elem2)
                _, domain = lhs[:-1].split("(")
                sets = domain.split(",")  # ["elem1", "elem2"]

                if len(filters) != len(sets):
                    raise ValidationError(
                        f"Filter size {len(filters)} must be equal to the domain size {len(sets)}"
                    )

                matches = 0
                for user_filter, set in zip(filters, sets, strict=False):
                    if set in user_filter or user_filter == []:
                        matches += 1

                if matches == len(sets):
                    listings.append(listing)

        def infeasibility_filter(listing):
            infeasibility = listing.split("INFES = ")

            if infeasibility_threshold is None:
                return True

            return (
                len(infeasibility) == 2
                and float(infeasibility[-1][:-6]) < infeasibility_threshold
            )

        return "\n".join(list(filter(infeasibility_filter, listings))[:n])

    @property
    def records(self) -> pd.DataFrame | None:
        """
        Returns the records (data) of the Equation as a DataFrame.

        Returns
        -------
        DataFrame | None

        Examples
        --------
        >>> from gamspy import Container, Parameter, Variable, Equation, Model
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq[...] = 2*x1 + 3*x2 <= 10
        >>> solved_model = Model(m, "my_model", equations=[eq], objective=10*x1 + 6*x2, sense="MAX").solve()
        >>> eq.toValue()
        np.float64(10.0)

        """
        return self._get_records()

    @records.setter
    def records(self, records: pd.DataFrame | None):
        self._assign_records(records)

    def __hash__(self):
        return id(self)

    def setRecords(
        self, records: VarEquRecordsType | None, uels_on_axes: bool = False
    ) -> None:
        """
        Sets the records (data) of the Equation.

        This allows manually setting the level, marginal, bounds, and scale for an equation.

        Parameters
        ----------
        records : Sequence | np.ndarray | int | float | pd.DataFrame | pd.Series | dict
            The data to load (e.g., list, numpy array, DataFrame). Tabular input
            must name its attribute columns (``level``, ``marginal``, ``lower``,
            ``upper``, ``scale``); every other column is read as a domain column,
            whatever order they appear in. Omitted attributes take their default.
        uels_on_axes : bool, optional
            If True, assumes domain elements are in the axes of the DataFrame. Default is False.

        Examples
        --------
        >>> from gamspy import Container, Variable, Equation
        >>> m = Container()
        >>> x1 = Variable(m, "x1", type="Positive")
        >>> x2 = Variable(m, "x2", type="Positive")
        >>> z = Variable(m, "z")
        >>> eq = Equation(m, "eq")
        >>> eq.setRecords(5)
        >>> eq.toValue()
        np.float64(5.0)

        """
        self._update_records(records, uels_on_axes=uels_on_axes)

    @property
    def _default_records(self) -> dict[str, float]:
        """Default records of an equation"""
        return EQU_DEFAULT_VALUES[self._type]

    @property
    def type(self):
        """
        The type of the equation.

        Common types include:


        - 'regular' (or 'eq', 'geq', 'leq'): Standard =e=, =g=, =l= constraints.
        - 'nonbinding' ('=n='): No relationship implied.
        - 'external' ('=x='): External equation.
        - 'boolean' ('=b='): Boolean equation.

        Returns
        -------
        str
            The type of equation

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> e = gp.Equation(m, "e", type="regular")
        >>> e.type
        'eq'

        """
        return self._type

    @type.setter
    def type(self, eq_type: str | EquationType):
        given_type = cast_type(eq_type)
        self._type = given_type

    def gamsRepr(self) -> str:
        """
        Returns the string representation of this Equation in the GAMS language.

        (e.g., 'e(i)').

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> e = gp.Equation(m, "e", domain=[i])
        >>> e.gamsRepr()
        'e(i)'

        """
        representation = self.name
        if self.domain:
            representation += self._get_domain_str(self._domain_forwarding)
        return representation

    def latexRepr(self) -> str:
        r"""
        Generates a LaTeX representation of the equation definition.

        Returns
        -------
        str
            LaTeX string defining the equation.

        Raises
        ------
        ValidationError
            If the equation has not been defined (assigned).

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1'])
        >>> v = gp.Variable(m, "v", domain=[i])
        >>> e = gp.Equation(m, "e", domain=[i])
        >>> e[i] = v[i] <= 10
        >>> print(e.latexRepr())
        $
        v_{i} \leq 10\hfill \forall i
        $

        """
        return self._latex_repr()

    def _latex_repr(self, matched_variables: Sequence[Variable] = ()) -> str:
        if self._definition is None:
            raise ValidationError(
                "Equation must be defined to get its latex representation."
            )

        # The LHS of an equation definition can either be an ImplicitEquation or a condition.
        # e.g. e[i] = ... or e[i].where[b[i]] = ...
        left = self._definition.left
        assert isinstance(left, (implicits.ImplicitEquation, condition.Condition))

        where = None
        if isinstance(left, condition.Condition):
            where = left.condition
            left = left.conditioning_on
            assert isinstance(left, implicits.ImplicitEquation)

        # A literal in the definition domain restricts the declared set at that position:
        # e[i, "j1"] = ... holds for all i where j = 'j1'
        conditions = []
        for position, literal in left._scalar_domains:
            declared = utils._get_domain_str([self.domain[position]], latex=True)[1:-1]
            literal = literal.replace("_", r"\_")
            conditions.append(f"{declared} = \\text{{`{literal}'}}")

        if where is not None:
            where_str, where_prec, _ = expression.latex_operand(where)
            if conditions and where_prec < expression.PRECEDENCE["and"]:
                where_str = f"({where_str})"
            conditions.append(where_str)

        condition_str = " \\wedge ".join(conditions)
        domain_str = utils._get_domain_str(left.domain, latex=True)[1:-1]
        right_side = ""
        if domain_str:
            right_side = f"\\hfill \\forall {domain_str}"
            if condition_str:
                right_side += f" ~ | ~ {condition_str}"
        elif condition_str:
            right_side = f"\\hfill \\text{{if }} {condition_str}"

        assert self._definition.right is not None
        definition_str = expression.get_operand_latex_repr(self._definition.right)

        if matched_variables:
            # Variables are matched to the equation position by position.
            positions = left._reference_domain()
            variable_strs = [
                variable[positions].latexRepr()
                if positions and variable.dimension == len(positions)
                else variable.latexRepr()
                for variable in matched_variables
            ]
            definition_str += " \\perp " + ",".join(variable_strs)

        return "$\n" + definition_str + right_side + "\n$"

    def getDeclaration(self) -> str:
        """
        Returns the GAMS declaration statement for this Equation.

        (e.g., 'Equation e(i);').

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
        >>> v = gp.Variable(m, "v", domain=[i])
        >>> e = gp.Equation(m, "e", domain=[i])
        >>> e.getDeclaration()
        'Equation e(i) / /;'

        """
        output = f"Equation {self.name}"

        if self.domain:
            output += self._get_domain_str(self._domain_forwarding)

        if self.description:
            output += ' "' + self.description + '"'

        if self._records is None:
            output += " / /"

        output += ";"
        return output

    def getAssignment(self) -> str:
        """
        Returns the latest GAMS assignment statement for this Equation (e.g. assigning .l).

        Returns
        -------
        str

        Raises
        ------
        ValidationError
            If the equation attributes have not been assigned a value.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> e = gp.Variable(m, "e", domain=[i])
        >>> e.l[i] = 0;
        >>> e.getAssignment()
        'e.l(i) = 0;'

        """
        if self._assignment is None:
            raise ValidationError("Equation was not assigned!")

        return self._assignment.getDeclaration()

    def getDefinition(self) -> str:
        """
        Returns the GAMS definition statement (algebra) for this Equation.

        (e.g., 'e(i) .. x(i) =l= 5;').

        Returns
        -------
        str

        Raises
        ------
        ValidationError
            If the equation algebra has not been defined.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
        >>> v = gp.Variable(m, "v", domain=[i])
        >>> e = gp.Equation(m, "e", domain=[i])
        >>> e[i] = a[i] <= v[i]
        >>> e.getDefinition()
        'e(i) .. a(i) =l= v(i);'

        """
        if self._definition is None:
            raise ValidationError("Equation is not defined!")

        return self._definition.getDeclaration()


def cast_type(type: str | EquationType) -> str:
    if isinstance(type, str):
        type = type.lower()
        if type not in (
            "eq",
            "geq",
            "leq",
            "regular",
            "nonbinding",
            "external",
            "boolean",
        ):
            raise ValueError(
                f"Allowed equation types: {EquationType.values()} but found {type}."
            )

        # assign eq by default
        if type == "regular":
            type = "eq"

    elif isinstance(type, EquationType):
        # assign eq by default
        type = "eq" if type == EquationType.REGULAR else str(type)

    return type
