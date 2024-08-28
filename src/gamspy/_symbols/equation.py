from __future__ import annotations

import builtins
import itertools
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any

import gams.transfer as gt
import pandas as pd
from gams.core.gdx import GMS_DT_EQU
from gams.transfer._internals import (
    EQU_TYPE,
    TRANSFER_TO_GAMS_EQUATION_SUBTYPES,
)

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Alias, Container, Set, Variable
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation


eq_types = ["=e=", "=l=", "=g="]

non_regular_map = {
    "nonbinding": "=n=",
    "external": "=x=",
    "boolean": "=b=",
}


class EquationType(Enum):
    REGULAR = "REGULAR"
    NONBINDING = "NONBINDING"
    EXTERNAL = "EXTERNAL"
    BOOLEAN = "BOOLEAN"

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Equation(gt.Equation, Symbol):
    """
    Represents an Equation symbol in GAMS.
    https://www.gams.com/latest/docs/UG_Equations.html

    Parameters
    ----------
    container : Container
        Container of the variable.
    name : str, optional
        Name of the equation. Name is autogenerated by default.
    type : str
        Type of the equation. "regular" by default.
    domain : list[Set | Alias | str] | Set | Alias | str, optional
        Domain of the variable.
    definition: Expression, optional
        Definition of the equation.
    records : Any, optional
        Records of the equation.
    domain_forwarding : bool, optional
        Whether the equation forwards the domain. See: https://gams.com/latest/docs/UG_SetDefinition.html#UG_SetDefinition_ImplicitSetDefinition
    description : str, optional
        Description of the equation.
    uels_on_axes: bool
        Assume that symbol domain information is contained in the axes of the given records.
    definition_domain: list, optional
        Definiton domain of the equation.
    is_miro_output : bool
        Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html

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

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        type: str | EquationType = "regular",
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        records: Any | None = None,
        description: str = "",
    ):
        # create new symbol object
        obj = Equation.__new__(
            cls,
            container,
            name,
            type,
            domain,
            records=records,
            description=description,
        )

        # set private properties directly
        type = cast_type(type)
        obj.type = EQU_TYPE[type]
        obj._gams_type = GMS_DT_EQU
        obj._gams_subtype = TRANSFER_TO_GAMS_EQUATION_SUBTYPES[type]
        obj._requires_state_check = False
        obj._container = container
        container._requires_state_check = True
        obj._name = name
        obj._domain = domain
        obj._domain_forwarding = False
        obj._description = description
        obj._records = None
        obj._modified = True

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj._definition = None
        obj.where = condition.Condition(obj)
        obj.container._add_statement(obj)
        obj._synchronize = True

        # create attributes
        obj._l, obj._m, obj._lo, obj._up, obj._s = obj._init_attributes()
        obj._stage = obj._create_attr("stage")
        obj._range = obj._create_attr("range")
        obj._slacklo = obj._create_attr("slacklo")
        obj._slackup = obj._create_attr("slackup")
        obj._slack = obj._create_attr("slack")
        obj._infeas = obj._create_attr("infeas")

        # miro support
        obj._is_miro_output = False

        return obj

    def __new__(
        cls,
        container: Container,
        name: str | None = None,
        type: str | EquationType = "regular",
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        definition: Variable | Operation | Expression | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
        definition_domain: list | None = None,
    ):
        if not isinstance(container, gp.Container):
            raise TypeError(
                f"Container must of type `Container` but found {container}"
            )

        if name is None:
            return object.__new__(cls)
        else:
            if not isinstance(name, str):
                raise TypeError(
                    f"Name must of type `str` but found {builtins.type(name)}"
                )
            try:
                symbol = container[name]
                if isinstance(symbol, cls):
                    return symbol

                raise TypeError(
                    f"Cannot overwrite symbol `{name}` in container"
                    " because it is not an Equation object)"
                )
            except KeyError:
                return object.__new__(cls)

    def __init__(
        self,
        container: Container,
        name: str | None = None,
        type: str | EquationType = "regular",
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        definition: Variable | Operation | Expression | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
        definition_domain: list | None = None,
    ):
        # miro support
        self._is_miro_output = is_miro_output

        self._synchronize = True

        # domain handling
        if domain is None:
            domain = []

        if isinstance(domain, (gp.Set, gp.Alias, str)):
            domain = [domain]

        if isinstance(domain, gp.math.Dim):
            domain = gp.math._generate_dims(container, domain.dims)

        # does symbol exist
        has_symbol = False
        if isinstance(getattr(self, "container", None), gp.Container):
            has_symbol = True

        if has_symbol:
            type = cast_type(type)
            if self.type != type.casefold():
                raise TypeError(
                    "Cannot overwrite symbol in container unless equation"
                    f" types are equal: `{self.type}` !="
                    f" `{type.casefold()}`"
                )

            if any(
                d1 != d2
                for d1, d2 in itertools.zip_longest(self.domain, domain)
            ):
                raise ValueError(
                    "Cannot overwrite symbol in container unless symbol"
                    " domains are equal"
                )

            if self.domain_forwarding != domain_forwarding:
                raise ValueError(
                    "Cannot overwrite symbol in container unless"
                    " 'domain_forwarding' is left unchanged"
                )

            # reset some properties
            self._requires_state_check = True
            self.container._requires_state_check = True
            if description != "":
                self.description = description

            previous_state = self.container.miro_protect
            self.container.miro_protect = False
            self.records = None
            self.modified = True

            # only set records if records are provided
            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            self.container.miro_protect = previous_state

        else:
            type = cast_type(type)

            if name is not None:
                name = validation.validate_name(name)

                if is_miro_output:
                    name = name.lower()  # type: ignore
            else:
                name = "e" + str(uuid.uuid4()).replace("-", "_")

            previous_state = container.miro_protect
            container.miro_protect = False
            super().__init__(
                container,
                name,
                type,
                domain,
                domain_forwarding=domain_forwarding,
                description=description,
                uels_on_axes=uels_on_axes,
            )

            if is_miro_output:
                container._miro_output_symbols.append(self.name)

            validation.validate_container(self, self.domain)

            self.where = condition.Condition(self)
            self.container._add_statement(self)
            self._definition_domain = definition_domain
            self._init_definition(definition)

            # create attributes
            (
                self._l,
                self._m,
                self._lo,
                self._up,
                self._s,
            ) = self._init_attributes()
            self._stage = self._create_attr("stage")
            self._range = self._create_attr("range")
            self._slacklo = self._create_attr("slacklo")
            self._slackup = self._create_attr("slackup")
            self._slack = self._create_attr("slack")
            self._infeas = self._create_attr("infeas")

            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            else:
                self.container._synch_with_gams()

            container.miro_protect = previous_state

    def __hash__(self):
        return id(self)

    def __getitem__(self, indices: tuple | str):
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitEquation(
            self,
            name=self.name,
            type=self.type,
            domain=domain,  # type: ignore  # noqa: E501
        )

    def __setitem__(
        self,
        indices: tuple | str | implicits.ImplicitSet,
        rhs: Expression,
    ):
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        self._set_definition(domain, rhs)

        self.container._synch_with_gams()
        self._winner = "gams"

    def __repr__(self) -> str:
        return f"Equation(name={self.name}, type={self.type}, domain={self.domain})"

    def _init_attributes(self) -> tuple:
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

    def _create_attr(self, attr_name):
        return implicits.ImplicitParameter(
            self,
            name=f"{self.name}.{attr_name}",
            records=self.records,
            domain=self.domain,
        )

    def _init_definition(
        self,
        assignment: Variable | Operation | Expression | None = None,
    ):
        if assignment is None:
            self._definition = None  # type: ignore
            return None

        domain = self.domain
        if self._definition_domain is not None:
            domain = validation.validate_domain(self, self._definition_domain)

        self._set_definition(domain, assignment)

    def _set_definition(self, domain, rhs):
        # self[domain] = rhs

        if not any(eq_type in rhs.gamsRepr() for eq_type in eq_types):
            raise ValidationError(
                "Equation definition must contain at least one equality sign such as ==, <= or >=."
            )

        if self.type == "external" and "=e=" not in rhs.gamsRepr():
            raise ValidationError("External equations must contain ==")

        if self.type in non_regular_map:
            rhs._replace_operator(non_regular_map[self.type])

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

        statement._validate_definition(utils._unpack(domain))

        self.container._add_statement(statement)
        self._definition = statement

    @property
    def l(self):  # noqa: E741, E743
        """
        Level

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
        Marginal

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
        Lower bound

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
        Upper bound

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
        Scale

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
        Stage

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
        Range

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
        Slack lower bound

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
        Slack upper bound

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
        Slack

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
        Infeasability

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

    def computeInfeasibilities(self) -> pd.DataFrame:
        """
        Computes infeasabilities of the equation

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> e = gp.Equation(m, "e")
        >>> e.l[...] = -10
        >>> e.lo[...] = 5
        >>> e.computeInfeasibilities().values.tolist()
        [[-10.0, 0.0, 5.0, 0.0, 1.0, 10.0]]

        """
        return utils._calculate_infeasibilities(self)

    def getEquationListing(
        self,
        n: int | None = None,
        filters: list[list[str]] | None = None,
        infeasibility_threshold: float | None = None,
    ) -> str:
        """
        Returns the generated equations.

        Parameters
        ----------
        n : int, optional
            Number of equations to be returned.
        filters : list[list[str]], optional
            Filters to be used.
        infeasibility_threshold: float, optional
            Filters out equations with infeasibilities that are above this value.

        Returns
        -------
        str

        Raises
        ------
        ValidationError
            In case the model is not solved yet with equation_listing_limit option.
        ValidationError
            In case the length of the filters is different than the dimension of the equation.
        """
        if not hasattr(self, "_equation_listing"):
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
                for user_filter, set in zip(filters, sets):
                    if set in user_filter or user_filter == []:
                        matches += 1

                # infeasibility = float(listing.split("INFES = ")[-1][:-6])
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
    def records(self):
        """
        Records of the Equation

        Returns
        -------
        DataFrame

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
        return self._records

    @records.setter
    def records(self, records):
        if records is not None and not isinstance(records, pd.DataFrame):
            raise TypeError("Symbol 'records' must be type DataFrame")

        # set records
        self._records = records

        self._requires_state_check = True
        self.modified = True

        self.container._requires_state_check = True
        self.container.modified = True

        if self._records is not None and self.domain_forwarding:
            self._domainForwarding()

            # reset state check flags for all symbols in the container
            for symbol in self.container.data.values():
                symbol._requires_state_check = True

    def setRecords(self, records: Any, uels_on_axes: bool = False) -> None:
        """
        Main convenience method to set standard pandas.DataFrame formatted
        records. If uels_on_axes=True setRecords will assume that all domain
        information is contained in the axes of the pandas object – data will be
        flattened (if necessary).

        Parameters
        ----------
        records : Any
        uels_on_axes : bool, optional

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
        super().setRecords(records, uels_on_axes)

        self.container._synch_with_gams()
        self._winner = "python"

    @property
    def type(self):
        """
        The type of equation;
        3. 'regular' -- equal, less than or greater than
        4. 'nonbinding', 'N', or '=N='  -- nonbinding relationship
        6. 'external', 'X', or '=X=' -- external equation
        7. 'boolean', 'B', or '=B=' -- boolean equation

        Returns
        -------
        str
            The type of equation
        """
        return self._type

    @type.setter
    def type(self, eq_type: str | EquationType):
        given_type = cast_type(eq_type)
        gt.Equation.type.fset(self, given_type)

    def gamsRepr(self) -> str:
        """
        Representation of this Equation in GAMS language.

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
        'e'

        """
        return self.name

    def latexRepr(self) -> str:
        if self._definition is None:
            raise ValidationError(
                "Equation must be defined to get its latex representation."
            )

        assert isinstance(
            self._definition.left,
            (implicits.ImplicitEquation, condition.Condition),
        )

        right_side = ""
        if isinstance(self._definition.left, implicits.ImplicitEquation):
            if len(self._definition.left.domain) > 0:
                domain_str = ",".join(
                    [symbol.name for symbol in self._definition.left.domain]
                )
                right_side = f"\\hfill \\forall {domain_str}"
        else:
            domain_str = ",".join(
                [
                    symbol.name
                    for symbol in self._definition.left.conditioning_on.domain
                ]
            )
            domain_str = f"\\forall {domain_str}"
            constraint_str = self._definition.left.condition.latexRepr()
            right_side = f"\\hfill {domain_str} ~ | ~ {constraint_str}"

        equation_str = (
            "$\n"
            + self._definition.right.latexRepr()
            + f"{right_side}"
            + "\n$"
        )

        return equation_str

    def getDeclaration(self) -> str:
        """
        Declaration of the Equation in GAMS

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
        'Equation e(i);'

        """
        output = f"Equation {self.name}"

        if self.domain:
            output += self._get_domain_str()

        if self.description:
            output += ' "' + self.description + '"'

        output += ";"
        return output

    def getDefinition(self) -> str:
        """
        Definition of the Equation in GAMS

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
        >>> e[i] = a[i] <= v[i]
        >>> e.getDefinition()
        'e(i) .. a(i) =l= v(i);'

        """
        if self._definition is None:
            raise ValidationError("Equation is not defined!")

        return self._definition.getDeclaration()


def cast_type(type: str | EquationType) -> str:
    if isinstance(type, str):
        if type.lower() not in [
            "eq",
            "geq",
            "leq",
            "regular",
            "nonbinding",
            "external",
            "boolean",
        ]:
            raise ValueError(
                "Allowed equation types:"
                f" {EquationType.values()} but found {type}."
            )

        # assign eq by default
        if type.upper() == "REGULAR":
            type = "eq"

    elif isinstance(type, EquationType):
        # assign eq by default
        type = "eq" if type == EquationType.REGULAR else str(type)

    return type
