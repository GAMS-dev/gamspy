from __future__ import annotations

import builtins
import itertools
import os
import threading
import weakref
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
from gams.core.gdx import GMS_DT_VAR

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._internals import TRANSFER_TO_GAMS_VARIABLE_SUBTYPES
from gamspy._special_values import SpecialValues
from gamspy._symbols.base import VarEquSymbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Container
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitVariable
    from gamspy._types import DomainType, IndexType, VarEquRecordsType

VAR_DEFAULT_VALUES = {
    "binary": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 0.0,
        "upper": 1.0,
        "scale": 1.0,
    },
    "integer": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 0.0,
        "upper": SpecialValues.POSINF,
        "scale": 1.0,
    },
    "positive": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 0.0,
        "upper": SpecialValues.POSINF,
        "scale": 1.0,
    },
    "negative": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": SpecialValues.NEGINF,
        "upper": 0.0,
        "scale": 1.0,
    },
    "free": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": SpecialValues.NEGINF,
        "upper": SpecialValues.POSINF,
        "scale": 1.0,
    },
    "sos1": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 0.0,
        "upper": SpecialValues.POSINF,
        "scale": 1.0,
    },
    "sos2": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 0.0,
        "upper": SpecialValues.POSINF,
        "scale": 1.0,
    },
    "semicont": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 1.0,
        "upper": SpecialValues.POSINF,
        "scale": 1.0,
    },
    "semiint": {
        "level": 0.0,
        "marginal": 0.0,
        "lower": 1.0,
        "upper": SpecialValues.POSINF,
        "scale": 1.0,
    },
}


class VariableType(Enum):
    """
    Enumeration of available variable types.
    """

    BINARY = "binary"
    """Discrete variable that can only take values of 0 or 1."""

    INTEGER = "integer"
    """Discrete variable that can only take integer values between the bounds."""

    POSITIVE = "positive"
    """No negative values are allowed for variable. The user may change both bounds from the default value."""

    NEGATIVE = "negative"
    """No positive values are allowed for variables. The user may change both bounds from the default value."""

    FREE = "free"
    """No bounds on variable. Both bounds may be changed from the default values by the user."""

    SOS1 = "sos1"
    """A set of variables, such that at most one variable within a group may have a non-zero value."""

    SOS2 = "sos2"
    """A set of variables, such that at most two variables within a group may have non-zero values and the two non-zero values are adjacent."""

    SEMICONT = "semicont"
    """Semi-continuous, must be zero or above a given minimum level."""

    SEMIINT = "semiint"
    """Semi-integer, must be zero or above a given minimum level and integer."""

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Variable(operable.Operable, VarEquSymbol):
    """
    Represents a Variable symbol in GAMS.

    Variables are the decision entities in a mathematical model. They can be
    free, positive, binary, integer, etc.
    See https://gamspy.readthedocs.io/en/latest/user/basics/variable.html

    Parameters
    ----------
    container : Container
        The Container object that this variable belongs to.
    name : str, optional
        Name of the variable. If not provided, a unique name is generated automatically.
    type : str, optional
        Type of the variable. Options: "free", "positive", "negative", "binary",
        "integer", "sos1", "sos2", "semicont", "semiint". Default is "free".
    domain : DomainType, optional
        The domain of the variable. Can be a list of Sets/Aliases, a single Set/Alias,
        or strings representing set names. Use "*" for the universe set. Default is [] (scalar).
    records : Sequence | pd.DataFrame | pd.Series | np.ndarray | int | float | dict, optional
        Initial records (level/marginal/bounds) to populate the variable.
    domain_forwarding : bool | list[bool], optional
        If True, adding records to this variable will implicitly add new elements to the
        domain sets (if they are dynamic). Default is False.
    description : str, optional
        A human-readable description of the variable.
    is_miro_output : bool, optional
        If True, flags this variable as an output symbol for GAMS MIRO. Default is False.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1', 'i2'])
    >>> v = gp.Variable(m, "v", domain=[i], type="positive", description="Production quantity")

    """

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        type: str = "free",
        domain: DomainType | None = None,
        records: VarEquRecordsType | None = None,
        description: str = "",
    ) -> Variable:
        # create new symbol object
        obj = object.__new__(cls)

        # legacy gtp attributes
        ## set private properties directly
        obj._type = type
        obj._gams_type = cast("int", GMS_DT_VAR)
        obj._gams_subtype = TRANSFER_TO_GAMS_VARIABLE_SUBTYPES[type]

        obj._container = cast(
            "Container",
            weakref.proxy(container)
            if not isinstance(container, weakref.ProxyType)
            else container,
        )
        obj.name = name
        obj._domain = obj._normalize_domain(obj._container, domain)
        obj._domain_forwarding = False
        obj._description = description
        obj._records = records
        obj._container._data.update({name: obj})

        # gamspy attributes
        obj._domain_violations = None
        obj._assignment = None
        obj.where = condition.Condition(obj)
        obj._latex_name = name.replace("_", r"\_")
        obj.container._add_statement(obj)
        obj._metadata = {}
        obj._should_load_from_gams = False
        obj._should_unload_to_gams = False
        obj._column_listing = None

        ## create attributes
        obj._l, obj._m, obj._lo, obj._up, obj._s = obj._init_attributes()
        obj._fx = obj._create_attr("fx")
        obj._prior = obj._create_attr("prior")
        obj._stage = obj._create_attr("stage")

        ## miro support
        obj._is_miro_output = False

        return obj

    def __new__(
        cls,
        container: Container | None = None,
        name: str | None = None,
        type: str = "free",
        domain: DomainType | None = None,
        records: VarEquRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
    ):
        if container is not None and not isinstance(container, gp.Container):
            invalid_type = builtins.type(container)
            raise TypeError(
                f"Container must of type `Container` but found {invalid_type}"
            )

        if name is None:
            return object.__new__(cls)
        else:
            if not isinstance(name, str):
                raise TypeError(
                    f"Name must of type `str` but found {builtins.type(name)}"
                )
            try:
                if not container:
                    container = gp._ctx_managers[
                        (os.getpid(), threading.get_native_id())
                    ]

                symbol = container._data[name]
            except KeyError:
                return object.__new__(cls)

        if isinstance(symbol, cls):
            return symbol

        raise TypeError(
            f"Cannot overwrite symbol `{symbol.name}` in container"
            " because it is not a Variable object)"
        )

    def __init__(
        self,
        container: Container | None = None,
        name: str | None = None,
        type: str = "free",
        domain: DomainType | None = None,
        records: VarEquRecordsType | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
    ):
        self._metadata: dict[str, Any] = {}
        self._assignment: Expression | None = None
        if is_miro_output and name is None:
            raise ValidationError("Please specify a name for miro symbols.")

        # miro support
        self._is_miro_output = is_miro_output
        self._domain_violations = None

        self._column_listing: list[str] | None = None

        # does symbol exist
        has_symbol = False
        if isinstance(getattr(self, "container", None), gp.Container):
            has_symbol = True

        if has_symbol:
            if self.type != type.casefold():
                raise TypeError(
                    "Cannot overwrite symbol in container unless variable"
                    f" types are equal: `{self.type}` !="
                    f" `{type.casefold()}`"
                )

            domain = self._normalize_domain(self.container, domain)
            if any(d1 != d2 for d1, d2 in itertools.zip_longest(self._domain, domain)):
                raise ValueError(
                    "Cannot overwrite symbol in container unless symbol"
                    " domains are equal"
                )

            if self._domain_forwarding != domain_forwarding:
                raise ValueError(
                    "Cannot overwrite symbol in container unless"
                    " 'domain_forwarding' is left unchanged"
                )

            # reset some properties
            if description != "":
                self._description = description

            previous_state = self._container._options.miro_protect
            self._container._options.miro_protect = False
            self._records: pd.DataFrame | None = None

            # only set records if records are provided
            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            self._container._options.miro_protect = previous_state

        else:
            if container is None:
                try:
                    container = gp._ctx_managers[
                        (os.getpid(), threading.get_native_id())
                    ]
                except KeyError as e:
                    raise ValidationError("Variable requires a container.") from e

            self._container = cast("Container", weakref.proxy(container))

            type = cast_type(type)

            if name is not None:
                name = validation.validate_name(name)

                if is_miro_output:
                    name = name.lower()
            else:
                name = self._container._get_symbol_name(prefix="v")

            self.name = name

            domain = self._normalize_domain(self._container, domain)
            self._domain = self._validate_domain(domain)
            self._domain_forwarding = domain_forwarding
            self.type = type
            self._description = description
            self._records = None
            self._gams_type = GMS_DT_VAR
            self._gams_subtype = TRANSFER_TO_GAMS_VARIABLE_SUBTYPES[self._type]
            self._latex_name = self.name.replace("_", r"\_")
            self._should_load_from_gams = False
            self._should_unload_to_gams = False
            self._container._data.update({name: self})

            if is_miro_output:
                self._container._miro_output_symbols.append(self.name)

            validation.validate_container(self, self._domain)
            self.where = condition.Condition(self)
            self._container._add_statement(self)

            # create attributes
            self._l, self._m, self._lo, self._up, self._s = self._init_attributes()
            self._fx = self._create_attr("fx")
            self._prior = self._create_attr("prior")
            self._stage = self._create_attr("stage")

            previous_state = self._container._options.miro_protect
            self._container._options.miro_protect = False
            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            else:
                if self._is_miro_output:
                    self._should_unload_to_gams = True

                self._container._synch_with_gams()

            self._container._options.miro_protect = True
            self._container._options.miro_protect = previous_state

    def _serialize(self) -> dict:
        info: dict[str, Any] = {
            "_domain_forwarding": self._domain_forwarding,
            "_is_miro_output": self._is_miro_output,
            "_metadata": self._metadata,
        }
        if self._assignment is not None:
            info["_assignment"] = self._assignment.getDeclaration()

        return info

    def _deserialize(self, info: dict) -> None:
        for key, value in info.items():
            if key == "_assignment":
                left, right = value.split(" = ")
                value = expression.Expression(left, "=", right[:-1])

            setattr(self, key, value)

        # Relink domain symbols
        new_domain = []
        for elem in self._domain:
            if elem == "*":
                new_domain.append(elem)
                continue
            new_domain.append(self._container[elem.name])

        self._domain = new_domain

        # Refresh the implicit parameters' domains
        self._update_attr_domains()

    def __getitem__(self, indices: IndexType) -> ImplicitVariable:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitVariable(self, name=self.name, domain=domain)

    def __eq__(self, other):
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def __repr__(self) -> str:
        return f"Variable(name='{self.name}', domain={self.domain}, type='{self.type}')"

    @property
    def T(self) -> ImplicitVariable:
        """
        Alias for the `.t()` method.

        Returns
        -------
        ImplicitVariable
        """
        return self.t()

    def t(self) -> ImplicitVariable:
        """
        Returns an ImplicitVariable derived from this
        variable by swapping its last two indices. This operation
        does not generate a new variable in GAMS.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> j = gp.Set(m, "j", records=['j1','j2'])
        >>> v = gp.Variable(m, "v", domain=[i, j])
        >>> v_t = v.t()
        >>> v_t.domain
        [Set(name='j', domain=['*']), Set(name='i', domain=['*'])]
        >>> v_t[i, j] # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        gamspy.exceptions.ValidationError:
        >>> v_t["j1", "i1"].gamsRepr()
        'v("i1","j1")'

        """
        from gamspy.math.matrix import permute

        dims = list(range(len(self.domain)))
        if len(dims) < 2:
            raise ValidationError(
                "Variable must contain at least 2 dimensions to transpose"
            )

        x = dims[-1]
        dims[-1] = dims[-2]
        dims[-2] = x
        return permute(self, dims)  # type: ignore

    def _init_attributes(
        self,
    ) -> tuple[
        ImplicitParameter,
        ImplicitParameter,
        ImplicitParameter,
        ImplicitParameter,
        ImplicitParameter,
    ]:
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

    def _create_attr(
        self, attr_name: Literal["l", "m", "lo", "up", "scale", "fx", "prior", "stage"]
    ) -> ImplicitParameter:
        return implicits.ImplicitParameter(
            self,
            name=f"{self.name}.{attr_name}",
            records=self.records,
            domain=self.domain,
        )

    def _update_attr_domains(self) -> None:
        self._l.__init__(
            self,
            name=f"{self.name}.l",
            records=self.records,
            domain=self.domain,
        )
        self._m.__init__(
            self,
            name=f"{self.name}.m",
            records=self.records,
            domain=self.domain,
        )
        self._lo.__init__(
            self,
            name=f"{self.name}.lo",
            records=self.records,
            domain=self.domain,
        )
        self._up.__init__(
            self,
            name=f"{self.name}.up",
            records=self.records,
            domain=self.domain,
        )
        self._s.__init__(
            self,
            name=f"{self.name}.scale",
            records=self.records,
            domain=self.domain,
        )
        self._fx.__init__(
            self,
            name=f"{self.name}.fx",
            records=self.records,
            domain=self.domain,
        )
        self._prior.__init__(
            self,
            name=f"{self.name}.prior",
            records=self.records,
            domain=self.domain,
        )
        self._stage.__init__(
            self,
            name=f"{self.name}.stage",
            records=self.records,
            domain=self.domain,
        )

    @property
    def l(self) -> ImplicitParameter:
        """
        The Level of the variable (its current value).

        This corresponds to the `.l` suffix in GAMS. After a solve, this holds the solution value.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=i, records=np.array([7, 18]))
        >>> x = gp.Variable(m, name="x", domain=i)
        >>> x.l[i] = d[i]
        >>> x.records.values.tolist()
        [['seattle', 7.0, 0.0, -inf, inf, 1.0], ['san-diego', 18.0, 0.0, -inf, inf, 1.0]]

        """
        return self._l

    @l.setter
    def l(self, value: int | float | Expression):
        self._l[...] = value

    @property
    def m(self) -> ImplicitParameter:
        """
        The Marginal (dual value) of the variable.

        This corresponds to the `.m` suffix in GAMS. Represents the reduced cost.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=i, records=np.array([7, 18]))
        >>> x = gp.Variable(m, name="x", domain=i)
        >>> x.m[i] = d[i]
        >>> x.records.values.tolist()
        [['seattle', 0.0, 7.0, -inf, inf, 1.0], ['san-diego', 0.0, 18.0, -inf, inf, 1.0]]

        """
        return self._m

    @m.setter
    def m(self, value: int | float | Expression):
        self._m[...] = value

    @property
    def lo(self) -> ImplicitParameter:
        """
        The Lower Bound of the variable.

        This corresponds to the `.lo` suffix in GAMS.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=i, records=np.array([7, 18]))
        >>> x = gp.Variable(m, name="x", domain=i)
        >>> x.lo[i] = d[i]
        >>> x.records.values.tolist()
        [['seattle', 0.0, 0.0, 7.0, inf, 1.0], ['san-diego', 0.0, 0.0, 18.0, inf, 1.0]]

        """
        return self._lo

    @lo.setter
    def lo(self, value: int | float | Expression):
        self._lo[...] = value

    @property
    def up(self) -> ImplicitParameter:
        """
        The Upper Bound of the variable.

        This corresponds to the `.up` suffix in GAMS.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=i, records=np.array([7, 18]))
        >>> x = gp.Variable(m, name="x", domain=i)
        >>> x.up[i] = d[i]
        >>> x.records.values.tolist()
        [['seattle', 0.0, 0.0, -inf, 7.0, 1.0], ['san-diego', 0.0, 0.0, -inf, 18.0, 1.0]]

        """
        return self._up

    @up.setter
    def up(self, value: int | float | Expression):
        self._up[...] = value

    @property
    def scale(self) -> ImplicitParameter:
        """
        The Scale factor of the variable.

        This corresponds to the `.scale` suffix in GAMS, used for scaling the variable
        to improve numerical stability.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=i, records=np.array([7, 18]))
        >>> x = gp.Variable(m, name="x", domain=i)
        >>> x.scale[i] = d[i]
        >>> x.records.values.tolist()
        [['seattle', 0.0, 0.0, -inf, inf, 7.0], ['san-diego', 0.0, 0.0, -inf, inf, 18.0]]

        """
        return self._s

    @scale.setter
    def scale(self, value: int | float | Expression):
        if self.type in ("integer", "binary"):
            raise ValidationError("Scales cannot be applied to discrete variables.")

        self._s[...] = value

    @property
    def fx(self) -> ImplicitParameter:
        """
        Fixed value of the variable.

        Setting `.fx` implies setting both `.lo` and `.up` to the same value.
        Reading `.fx` returns the current fixed level (if fixed).

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=i, records=np.array([7, 18]))
        >>> x = gp.Variable(m, name="x", domain=i)
        >>> x.fx[i] = d[i]
        >>> x.records.values.tolist()
        [['seattle', 7.0, 0.0, 7.0, 7.0, 1.0], ['san-diego', 18.0, 0.0, 18.0, 18.0, 1.0]]

        """
        return self._fx

    @fx.setter
    def fx(self, value: int | float | Expression):
        self._fx[...] = value

    @property
    def prior(self) -> ImplicitParameter:
        """
        Branching Priority.

        This corresponds to the `.prior` suffix in GAMS. Allows identifying a priority for
        branching on discrete variables. Valid only for discrete variable types (integer, binary).

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=i, records=np.array([7, 18]))
        >>> x = gp.Variable(m, name="x", domain=i, type="integer")
        >>> x.prior[i] = d[i]
        >>> x.records.values.tolist()
        [['seattle', 0.0, 0.0, 0.0, inf, 7.0], ['san-diego', 0.0, 0.0, 0.0, inf, 18.0]]

        """
        return self._prior

    @prior.setter
    def prior(self, value: int | float | Expression):
        if self.type not in ("integer", "binary"):
            raise ValidationError("Priorities can only be used on discrete variables.")

        self._prior[...] = value

    @property
    def stage(self) -> ImplicitParameter:
        """
        Branching Stage.

        This corresponds to the `.stage` suffix in GAMS. Used in stochastic programming or
        advanced branching strategies.

        Returns
        -------
        ImplicitParameter

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=i, records=np.array([7, 18]))
        >>> x = gp.Variable(m, name="x", domain=i, type="integer")
        >>> x.stage[i] = d[i]
        >>> x.records.values.tolist()
        [['seattle', 0.0, 0.0, 0.0, inf, 7.0], ['san-diego', 0.0, 0.0, 0.0, inf, 18.0]]

        """
        return self._stage

    @stage.setter
    def stage(self, value: int | float | Expression):
        self._stage[...] = value

    def computeInfeasibilities(self) -> pd.DataFrame | None:
        """
        Computes infeasibilities of the variable.

        Checks if the level value `.l` lies outside the bounds `.lo` and `.up`
        and returns a DataFrame containing the violations.

        Returns
        -------
        pd.DataFrame | None
            DataFrame showing the infeasible records.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> x = gp.Variable(m, name="x")
        >>> x.l[...] = -10
        >>> x.lo[...] = 5
        >>> x.computeInfeasibilities().values.tolist()
        [[-10.0, 0.0, 5.0, inf, 1.0, 15.0]]

        """
        return utils._calculate_infeasibilities(self)

    def getVariableListing(
        self,
        n: int | None = None,
        filters: list[list[str]] | None = None,
    ) -> str:
        """
        Returns the variable listing (log output) from the last solve.

        This requires the model to have been solved with the `variable_listing_limit` option enabled.

        Parameters
        ----------
        n : int, optional
            Maximum number of variables to return.
        filters : list[list[str]], optional
            Filters to select specific elements for the listing.
            The list size must match the variable's dimension.

        Returns
        -------
        str
            The text listing of the variable's status and values.

        Raises
        ------
        ValidationError
            If the model was not solved with `variable_listing_limit`.
        ValidationError
            If the filter size does not match the variable dimension.

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
        >>> summary = model.solve(options=gp.Options(variable_listing_limit=10))
        >>> print(v.getVariableListing())
        v(item1)
                        (.LO, .L, .UP, .M = -INF, 0, +INF, 0)
               (0)      e(item1)
        <BLANKLINE>
        v(item2)
                        (.LO, .L, .UP, .M = -INF, 0, +INF, 0)
               (0)      e(item2)
        <BLANKLINE>

        """
        if self._column_listing is None:
            raise ValidationError(
                "The model must be solved with `variable_listing_limit` option for this functionality to work."
            )

        listings = self._column_listing if filters is None else []

        if filters is not None:
            for listing in self._column_listing:
                lhs, *_ = listing.split("\n")
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

                # infeasibility = float(listing.split("INFES = ")[-1][:-6])
                if matches == len(sets):
                    listings.append(listing)

        return "\n".join(listings[:n])

    @property
    def records(self) -> pd.DataFrame | None:
        """
        Returns the records (data) of the Variable as a DataFrame.

        The DataFrame contains columns for the domain sets, and columns for
        level, marginal, lower, upper, and scale.

        Returns
        -------
        DataFrame | None

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=i, records=np.array([7, 18]))
        >>> x = gp.Variable(m, name="x", domain=i)
        >>> x.fx[i] = d[i]
        >>> x.records.values.tolist()
        [['seattle', 7.0, 0.0, 7.0, 7.0, 1.0], ['san-diego', 18.0, 0.0, 18.0, 18.0, 1.0]]

        """
        if self._should_load_from_gams:
            self._load_from_gams()

        return self._records

    @records.setter
    def records(self, records: pd.DataFrame | None):
        if records is not None and not isinstance(records, pd.DataFrame):
            raise TypeError("Symbol 'records' must be type DataFrame")

        self._records = records
        self._should_unload_to_gams = True
        self._handle_domain_forwarding()

    def __hash__(self):
        return id(self)

    def setRecords(
        self, records: VarEquRecordsType | None, uels_on_axes: bool = False
    ) -> None:
        """
        Sets the records (data) of the Variable.

        This is a convenience method to load data. It accepts various input formats.
        If `uels_on_axes=True`, it assumes domain information is in the pandas axes.

        Parameters
        ----------
        records : Sequence | np.ndarray | int | float | pd.DataFrame | pd.Series | dict
            The data to load (e.g., list, numpy array, DataFrame).
        uels_on_axes : bool, optional
            If True, assumes domain elements are in the axes of the DataFrame. Default is False.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> x = gp.Variable(m, name="x", domain=i)
        >>> x.setRecords(records=np.array([7, 18]))
        >>> x.records.values.tolist()
        [['seattle', 7.0, 0.0, -inf, inf, 1.0], ['san-diego', 18.0, 0.0, -inf, inf, 1.0]]

        """
        if records is None:
            self._container._add_statement(f"option clear={self.name};")
            self._container._synch_with_gams()
            self._records = None
            return

        self._setRecords(records, uels_on_axes=uels_on_axes)
        self._container._synch_with_gams()

    @property
    def _default_records(self):
        """Default records of a variable"""
        return VAR_DEFAULT_VALUES[self._type]

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, var_type: str | VariableType):
        """
        The type of variable; [binary, integer, positive, negative, free, sos1, sos2, semicont, semiint]

        Parameters
        ----------
        var_type : str
            The type of variable

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> x = gp.Variable(m, name="x", domain=i, type="positive")
        >>> x.type = "negative"
        >>> x.type
        'negative'

        """
        given_type = cast_type(var_type)
        self._type = given_type

    def gamsRepr(self) -> str:
        """
        Returns the string representation of this Variable in the GAMS language.

        (e.g., 'x(i)').

        Returns
        -------
        str
            The GAMS string representation.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> x = gp.Variable(m, name="x", domain=i, type="positive")
        >>> x.gamsRepr()
        'x(i)'

        """
        representation = self.name
        if self.domain:
            representation += self._get_domain_str(self._domain_forwarding)

        return representation

    def getDeclaration(self) -> str:
        """
        Returns the GAMS declaration statement for this Variable.

        (e.g., 'Positive Variable x(i);').

        Returns
        -------
        str
            The GAMS declaration string.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> v = gp.Variable(m, "v", domain=[i])
        >>> v.getDeclaration()
        'free Variable v(i) / /;'

        """
        output = self.type + " "
        output += f"Variable {self.gamsRepr()}"

        if self.description:
            output += ' "' + self.description + '"'

        if self._records is None:
            output += " / /"

        output += ";"

        return output

    def getAssignment(self) -> str:
        """
        Returns the latest GAMS assignment statement for this Variable.

        Returns
        -------
        str
            The GAMS assignment string.

        Raises
        ------
        ValidationError
            If the variable has not been assigned.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> v = gp.Variable(m, "v", domain=[i])
        >>> v.l[i] = 0;
        >>> v.getAssignment()
        'v.l(i) = 0;'

        """
        if self._assignment is None:
            raise ValidationError("Variable is not assigned!")

        return self._assignment.getDeclaration()


def cast_type(type: str | VariableType) -> str:
    if isinstance(type, str) and type.lower() not in VariableType.values():
        raise ValueError(
            f"Allowed variable types: {VariableType.values()} but found {type}."
        )

    if isinstance(type, VariableType):
        type = type.value

    return type.lower()
