from __future__ import annotations

import builtins
import itertools
import os
import threading
import uuid
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

import gams.transfer as gt
import pandas as pd
from gams.core.gdx import GMS_DT_VAR
from gams.transfer._internals import (
    TRANSFER_TO_GAMS_VARIABLE_SUBTYPES,
)

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Alias, Container, Set
    from gamspy._algebra.expression import Expression
    from gamspy.math.matrix import Dim


class VariableType(Enum):
    BINARY = "binary"
    INTEGER = "integer"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    FREE = "free"
    SOS1 = "sos1"
    SOS2 = "sos2"
    SEMICONT = "semicont"
    SEMIINT = "semiint"

    @classmethod
    def values(cls):
        """Convenience function to return all values of enum"""
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Variable(gt.Variable, operable.Operable, Symbol):
    """
    Represents a variable symbol in GAMS.
    https://gamspy.readthedocs.io/en/latest/user/basics/variable.html

    Parameters
    ----------
    container : Container
        Container of the variable.
    name : str, optional
        Name of the variable. Name is autogenerated by default.
    type : str, optional
        Type of the variable. "free" by default.
    domain : list[Set | Alias | str] | Set | Alias | str, optional
        Domain of the variable.
    records : Any, optional
        Records of the variable.
    domain_forwarding : bool, optional
        Whether the variable forwards the domain.
    description : str, optional
        Description of the variable.
    is_miro_output : bool
        Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])
    >>> v = gp.Variable(m, "v", domain=[i])

    """

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        type: str = "free",
        domain: list[Set | Alias | str]
        | Set
        | Alias
        | Dim
        | str
        | None = None,
        records: Any | None = None,
        description: str = "",
    ):
        # create new symbol object
        obj = Variable.__new__(
            cls,
            container,
            name,
            type,
            domain,
            records,
            description=description,
        )

        # set private properties directly
        obj._type = type
        obj._gams_type = GMS_DT_VAR
        obj._gams_subtype = TRANSFER_TO_GAMS_VARIABLE_SUBTYPES[type]

        obj._requires_state_check = False
        obj._container = container
        container._requires_state_check = True
        obj._name = name
        obj._domain = domain
        obj._domain_forwarding = False
        obj._description = description
        obj._records = records
        obj._modified = True

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj.where = condition.Condition(obj)
        obj.container._add_statement(obj)
        obj._synchronize = True

        # create attributes
        obj._l, obj._m, obj._lo, obj._up, obj._s = obj._init_attributes()
        obj._fx = obj._create_attr("fx")
        obj._prior = obj._create_attr("prior")
        obj._stage = obj._create_attr("stage")

        # miro support
        obj._is_miro_output = False

        return obj

    def __new__(
        cls,
        container: Container | None = None,
        name: str | None = None,
        type: str = "free",
        domain: list[Set | Alias | str]
        | Set
        | Alias
        | Dim
        | str
        | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
    ):
        ctx = None
        try:
            ctx = gp._ctx_managers[(os.getpid(), threading.get_native_id())]
        except KeyError:
            ...

        if ctx is None and not isinstance(container, gp.Container):
            invalid_type = builtins.type(container)
            raise TypeError(
                f"Container must of type `Container` but found {invalid_type}"
            )

        if name is None:
            obj = object.__new__(cls)

            if container is None:
                obj._ctx = ctx
            return obj
        else:
            if not isinstance(name, str):
                raise TypeError(
                    f"Name must of type `str` but found {builtins.type(name)}"
                )
            try:
                symbol = ctx[name] if ctx is not None else container[name]  # type: ignore
                if isinstance(symbol, cls):
                    return symbol

                raise TypeError(
                    f"Cannot overwrite symbol `{symbol.name}` in container"
                    " because it is not a Variable object)"
                )
            except KeyError:
                obj = object.__new__(cls)

                if container is None:
                    obj._ctx = ctx
                return obj

    def __init__(
        self,
        container: Container | None = None,
        name: str | None = None,
        type: str = "free",
        domain: list[Set | Alias | str]
        | Set
        | Alias
        | Dim
        | str
        | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_output: bool = False,
    ):
        if is_miro_output and name is None:
            raise ValidationError("Please specify a name for miro symbols.")

        # miro support
        self._is_miro_output = is_miro_output

        self._synchronize = True

        # domain handling
        if domain is None:
            domain = []

        if isinstance(domain, (gp.Set, gp.Alias, str)):
            domain = [domain]

        if isinstance(domain, gp.math.Dim):
            domain = gp.math._generate_dims(container, domain.dims)  # type: ignore

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

            previous_state = self.container._options.miro_protect
            self.container._options.miro_protect = False
            self.records = None
            self.modified = True

            # only set records if records are provided
            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            self.container._options.miro_protect = previous_state

        else:
            if hasattr(self, "_ctx"):
                container = self._ctx
            assert container is not None

            type = cast_type(type)

            if name is not None:
                name = validation.validate_name(name)

                if is_miro_output:
                    name = name.lower()  # type: ignore
            else:
                name = "v" + str(uuid.uuid4()).replace("-", "_")

            previous_state = container._options.miro_protect
            container._options.miro_protect = False
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

            # create attributes
            (
                self._l,
                self._m,
                self._lo,
                self._up,
                self._s,
            ) = self._init_attributes()
            self._fx = self._create_attr("fx")
            self._prior = self._create_attr("prior")
            self._stage = self._create_attr("stage")

            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            else:
                self.container._synch_with_gams()

            container._options.miro_protect = True

    def __getitem__(
        self, indices: Sequence | str
    ) -> implicits.ImplicitVariable:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitVariable(self, name=self.name, domain=domain)

    def __neg__(self):
        return expression.Expression(None, "-", self)

    def __eq__(self, other):
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def __repr__(self) -> str:
        return f"Variable(name='{self.name}', domain={self.domain}, type={self.type})"

    @property
    def T(self) -> implicits.ImplicitVariable:
        """See gamspy.Variable.t"""
        return self.t()

    def t(self) -> implicits.ImplicitVariable:
        """Returns an ImplicitVariable derived from this
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

        dims = [x for x in range(len(self.domain))]
        if len(dims) < 2:
            raise ValidationError(
                "Variable must contain at least 2 dimensions to transpose"
            )

        x = dims[-1]
        dims[-1] = dims[-2]
        dims[-2] = x
        return permute(self, dims)  # type: ignore

    def _init_attributes(self):
        level = self._create_attr("l")
        marginal = self._create_attr("m")
        lower = self._create_attr("lo")
        upper = self._create_attr("up")
        scale = self._create_attr("scale")
        return level, marginal, lower, upper, scale

    def _create_attr(self, attr_name):
        domain = self.domain
        return implicits.ImplicitParameter(
            self,
            name=f"{self.name}.{attr_name}",
            records=self.records,
            domain=domain,
        )

    def _update_attr_domains(self):
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
    def l(self):  # noqa: E741,E743
        """
        Level

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
    def m(self):
        """
        Marginal

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
    def lo(self):
        """
        Lower bound

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
    def up(self):
        """
        Upper bound

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
    def scale(self):
        """
        Scale

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
        if self.type in ["integer", "binary"]:
            raise ValidationError(
                "Scales cannot be applied to discrete variables."
            )

        self._s[...] = value

    @property
    def fx(self):
        """
        Fx

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
    def prior(self):
        """
        Prior

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
        if self.type not in ["integer", "binary"]:
            raise ValidationError(
                "Priorities can only be used on discrete variables."
            )

        self._prior[...] = value

    @property
    def stage(self):
        """
        Stage

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

    def computeInfeasibilities(self) -> pd.DataFrame:
        """
        Computes infeasabilities of the variable

        Returns
        -------
        pd.DataFrame

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
        Returns the generated variables.

        Parameters
        ----------
        n : int, optional
            Number of variables to be returned.
        filters : list[list[str]], optional
            Filters to be used.

        Returns
        -------
        str

        Raises
        ------
        ValidationError
            In case the model is not solved yet with variable_listing_limit option.
        ValidationError
            In case the length of the filters is different than the dimension of the variable.
        """
        if not hasattr(self, "_column_listing"):
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
                for user_filter, set in zip(filters, sets):
                    if set in user_filter or user_filter == []:
                        matches += 1

                # infeasibility = float(listing.split("INFES = ")[-1][:-6])
                if matches == len(sets):
                    listings.append(listing)

        return "\n".join(listings[:n])

    @property
    def records(self):
        """
        Records of the Variable

        Returns
        -------
        DataFrame

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
            for _, symbol in self.container.data.items():
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
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> x = gp.Variable(m, name="x", domain=i)
        >>> x.setRecords(records=np.array([7, 18]))
        >>> x.records.values.tolist()
        [['seattle', 7.0, 0.0, -inf, inf, 1.0], ['san-diego', 18.0, 0.0, -inf, inf, 1.0]]

        """
        super().setRecords(records, uels_on_axes)

        self.container._synch_with_gams()
        self._winner = "python"

    @property
    def type(self):
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
        gt.Variable.type.fset(self, given_type)

    def gamsRepr(self) -> str:
        """
        Representation of this Variable in GAMS language.

        Returns
        -------
        str

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
            representation += self._get_domain_str(self.domain_forwarding)

        return representation

    def getDeclaration(self) -> str:
        """
        Declaration of the Variable in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> v = gp.Variable(m, "v", domain=[i])
        >>> v.getDeclaration()
        'free Variable v(i);'

        """
        output = self.type + " "
        output += f"Variable {self.gamsRepr()}"

        if self.description:
            output += ' "' + self.description + '"'

        output += ";"

        return output

    def getAssignment(self) -> str:
        """
        Latest assignment to the Variable in GAMS

        Returns
        -------
        str

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
        if not hasattr(self, "_assignment"):
            raise ValidationError("Variable is not assigned!")

        return self._assignment.getDeclaration()


def cast_type(type: str | VariableType) -> str:
    if isinstance(type, str) and type.lower() not in VariableType.values():
        raise ValueError(
            f"Allowed variable types: {VariableType.values()} but"
            f" found {type}."
        )

    if isinstance(type, VariableType):
        type = type.value

    return type.lower()
