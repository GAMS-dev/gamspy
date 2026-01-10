from __future__ import annotations

import itertools
import os
import threading
from typing import TYPE_CHECKING, Any

import gams.transfer as gt
import numpy as np
import pandas as pd
from gams.core.gdx import GMS_DT_PAR

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    from gamspy import Alias, Container, Set
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._symbols.implicits import ImplicitParameter
    from gamspy.math.matrix import Dim


class Parameter(gt.Parameter, operable.Operable, Symbol):
    """
    Represents a Parameter symbol in GAMS.

    Parameters are used to hold data (scalars, vectors, or multi-dimensional arrays).
    See https://gamspy.readthedocs.io/en/latest/user/basics/parameter.html

    Parameters
    ----------
    container : Container
        The Container object that this parameter belongs to.
    name : str, optional
        Name of the parameter. If not provided, a unique name is generated automatically.
    domain : Sequence[Set | Alias | str] | Set | Alias | Dim | str, optional
        The domain of the parameter. Can be a list of Sets/Aliases, a single Set/Alias,
        or strings representing set names. Use "*" for the universe set. Default is [] (scalar).
    records : int | float | pd.DataFrame | np.ndarray | list, optional
        Initial values to populate the parameter. Can be a scalar, a list, a numpy array, or a pandas DataFrame.
    domain_forwarding : bool | list[bool], optional
        If True, adding records to this parameter will implicitly add new elements to the
        domain sets (if they are dynamic). Default is False.
    description : str, optional
        A human-readable description of the parameter.
    uels_on_axes : bool, optional
        If True, implies that the Unique Element Labels (UELs) for the domain are
        contained in the axes (index/columns) of the provided `records` object
        (e.g., pandas DataFrame). Default is False.
    is_miro_input : bool, optional
        If True, flags this parameter as an input symbol for GAMS MIRO. Default is False.
    is_miro_output : bool, optional
        If True, flags this parameter as an output symbol for GAMS MIRO. Default is False.
    is_miro_table : bool, optional
        If True, flags this parameter as a table symbol for GAMS MIRO. Default is False.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1', 'i2'])
    >>> a = gp.Parameter(m, "a", domain=[i], records=[['i1', 1], ['i2', 2]], description="Input data")

    """

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        domain: Sequence[Set | Alias | str] | Set | Alias | Dim | str | None = None,
        records: Any | None = None,
        description: str = "",
    ):
        if domain is None:
            domain = []

        if isinstance(domain, (gp.Set, gp.Alias, str)):
            domain = [domain]

        if isinstance(domain, gp.math.Dim):
            domain = gp.math._generate_dims(container, domain.dims)

        obj = Parameter.__new__(
            cls, container, name, domain, records, description=description
        )

        # set private properties directly
        obj._requires_state_check = False
        obj._container = container
        container._requires_state_check = True
        obj._name = name
        obj._domain = domain
        obj._domain_forwarding = False
        obj._description = description
        obj._records = records
        obj._modified = True
        obj._domain_violations = None

        # typing
        obj._gams_type = GMS_DT_PAR
        obj._gams_subtype = 0

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj._synchronize = True
        obj.where = condition.Condition(obj)
        obj._latex_name = name.replace("_", r"\_")
        obj.container._add_statement(obj)
        obj._metadata = {}
        obj._winner = "python"

        # miro support
        obj._is_miro_input = False
        obj._is_miro_output = False
        obj._is_miro_table = False

        return obj

    def __new__(
        cls,
        container: Container | None = None,
        name: str | None = None,
        domain: Sequence[Set | Alias | str] | Set | Alias | Dim | str | None = None,
        records: Any | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
        is_miro_table: bool = False,
    ):
        if container is not None and not isinstance(container, gp.Container):
            raise TypeError(
                f"Container must of type `Container` but found {type(container)}"
            )

        if name is None:
            return object.__new__(cls)
        else:
            if not isinstance(name, str):
                raise TypeError(f"Name must of type `str` but found {type(name)}")

            try:
                if not container:
                    container = gp._ctx_managers[
                        (os.getpid(), threading.get_native_id())
                    ]

                symbol = container[name]
                if isinstance(symbol, cls):
                    return symbol

                raise TypeError(
                    f"Cannot overwrite symbol `{name}` in container"
                    " because it is not a Parameter object"
                )
            except KeyError:
                return object.__new__(cls)

    def __init__(
        self,
        container: Container | None = None,
        name: str | None = None,
        domain: Sequence[Set | Alias | str] | Set | Alias | Dim | str | None = None,
        records: Any | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
        is_miro_table: bool = False,
    ):
        self._metadata: dict[str, Any] = {}
        if (is_miro_input or is_miro_output) and name is None:
            raise ValidationError("Please specify a name for miro symbols.")

        # miro support
        self._is_miro_input = is_miro_input
        self._is_miro_output = is_miro_output
        self._is_miro_table = is_miro_table
        self._is_miro_symbol = is_miro_input or is_miro_output or is_miro_table
        self._domain_violations = None

        self._synchronize = True
        self._winner = "python"

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
            self._requires_state_check = True
            self.container._requires_state_check = True
            if description != "":
                self.description = description

            previous_state = self.container._options.miro_protect
            self.container._options.miro_protect = False
            self._records = None
            self._modified = True

            # only set records if records are provided
            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            self.container._options.miro_protect = previous_state
        else:
            if container is None:
                try:
                    container = gp._ctx_managers[
                        (os.getpid(), threading.get_native_id())
                    ]
                except KeyError as e:
                    raise ValidationError("Parameter requires a container.") from e
            assert container is not None

            if name is not None:
                name = validation.validate_name(name)

                if is_miro_input or is_miro_output:
                    name = name.lower()  # type: ignore
            else:
                name = container._get_symbol_name(prefix="p")

            previous_state = container._options.miro_protect
            container._options.miro_protect = False
            super().__init__(
                container,
                name,
                domain,
                domain_forwarding=domain_forwarding,
                description=description,
                uels_on_axes=uels_on_axes,
            )
            self._latex_name = self.name.replace("_", r"\_")

            if is_miro_input:
                self._already_loaded = False
                container._miro_input_symbols.append(self.name)

            if is_miro_output:
                container._miro_output_symbols.append(self.name)

            validation.validate_container(self, self._domain)
            self.where = condition.Condition(self)
            self._assignment: Expression | None = None
            self.container._add_statement(self)

            if records is not None:
                super().setRecords(records, uels_on_axes=uels_on_axes)
                if self.dimension == 0 and not self._is_miro_symbol:
                    self._modified = False

                if gp.get_option("DROP_DOMAIN_VIOLATIONS"):
                    if self.hasDomainViolations():
                        self._domain_violations = self.getDomainViolations()
                        self.dropDomainViolations()
                    else:
                        self._domain_violations = None
            else:
                if not self._is_miro_symbol:
                    self._modified = False

            self.container._synch_with_gams(gams_to_gamspy=self._is_miro_input)

            container._options.miro_protect = previous_state

    def _serialize(self) -> dict:
        info = {
            "_domain_forwarding": self._domain_forwarding,
            "_is_miro_input": self._is_miro_input,
            "_is_miro_output": self._is_miro_output,
            "_is_miro_table": self._is_miro_table,
            "_metadata": self._metadata,
            "_synchronize": self._synchronize,
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
            new_domain.append(self.container[elem])

        self.domain = new_domain

    def __getitem__(
        self, indices: EllipsisType | slice | Sequence | str
    ) -> implicits.ImplicitParameter:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitParameter(self, name=self.name, domain=domain)

    def __setitem__(
        self,
        indices: EllipsisType | slice | Sequence | str | implicits.ImplicitSet,
        rhs: Operation | Expression | ImplicitParameter | float | int,
    ):
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        if self._is_miro_input and self.container._options.miro_protect:
            raise ValidationError(
                f"Cannot assign to protected miro input symbol {self.name}. `miro_protect`"
                " attribute of the container can be set to False to allow"
                " assigning to MIRO input symbols"
            )

        if isinstance(rhs, float):
            rhs = utils._map_special_values(rhs)  # type: ignore

        statement = expression.Expression(
            implicits.ImplicitParameter(self, name=self.name, domain=domain),
            "=",
            rhs,
        )

        statement._validate_definition(utils._unpack(domain))

        self.container._add_statement(statement)
        self._assignment = statement

        self.container._synch_with_gams(gams_to_gamspy=True)
        self._winner = "gams"

    def __eq__(self, other):
        op = "eq"
        if isinstance(
            other,
            (
                implicits.ImplicitVariable,
                expression.Expression,
                operation.Operation,
            ),
        ):
            op = "=e="
        return expression.Expression(self, op, other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def __repr__(self) -> str:
        return f"Parameter(name='{self.name}', domain={self.domain})"

    @property
    def T(self) -> implicits.ImplicitParameter:
        """
        Alias for the `.t()` method.

        Returns
        -------
        ImplicitParameter
        """
        return self.t()

    def t(self) -> implicits.ImplicitParameter:
        """
        Returns an ImplicitParameter derived from this
        parameter by swapping its last two indices. This operation
        does not generate a new parameter in GAMS.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> j = gp.Set(m, "j", records=['j1','j2'])
        >>> v = gp.Parameter(m, "v", domain=[i, j])
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
                "Parameter must contain at least 2 dimensions to transpose"
            )

        x = dims[-1]
        dims[-1] = dims[-2]
        dims[-2] = x
        return permute(self, dims)  # type: ignore

    @property
    def records(self):
        """
        Returns the records (data) of the Parameter as a DataFrame.

        Returns
        -------
        pd.DataFrame
            The dataframe containing the parameter's data.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> d = gp.Parameter(m, name="d", domain=[i])
        >>> d.setRecords(np.array([10, 25]))
        >>> d.toList()
        [('seattle', 10.0), ('san-diego', 25.0)]

        """
        return self._records

    @records.setter
    def records(self, records):
        if (
            hasattr(self, "_is_miro_input")
            and self._is_miro_input
            and self.container._options.miro_protect
        ):
            raise ValidationError(
                "Cannot assign to protected miro input symbols. `miro_protect`"
                " attribute of the container can be set to False to allow"
                " assigning to MIRO input symbols"
            )

        if records is not None and not isinstance(records, pd.DataFrame):
            raise TypeError("Symbol 'records' must be type DataFrame")

        # set records
        self._records = records

        self._requires_state_check = True
        self._modified = True

        self.container._requires_state_check = True
        self.container.modified = True

        if self._records is not None and self._domain_forwarding:
            self._domainForwarding()

            # reset state check flags for all symbols in the container
            for symbol in self.container.data.values():
                symbol._requires_state_check = True

    def __hash__(self):
        return id(self)

    def _setRecords(self, records: Any, *, uels_on_axes: bool = False) -> None:
        super().setRecords(records, uels_on_axes)

        if gp.get_option("DROP_DOMAIN_VIOLATIONS"):
            if self.hasDomainViolations():
                self._domain_violations = self.getDomainViolations()
                self.dropDomainViolations()
            else:
                self._domain_violations = None

    def setRecords(self, records: Any, uels_on_axes: bool = False) -> None:
        """
        Sets the records (data) of the Parameter.

        This is a convenience method to load data into the parameter. It handles various
        input formats like scalars, lists, numpy arrays, and pandas DataFrames.
        If `uels_on_axes=True`, it assumes that all domain information is contained
        in the axes (index/columns) of the pandas object, and the data will be flattened if necessary.

        Parameters
        ----------
        records : Any
            The data to load. Common formats:


            - Scalar: `10.5`
            - List of tuples/lists: `[['i1', 1], ['i2', 2]]`
            - numpy array: `np.array([1, 2])`
            - pandas DataFrame
        uels_on_axes : bool, optional
            If True, assumes domain elements are in the axes of the DataFrame. Default is False.

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i")
        >>> i.setRecords(["seattle", "san-diego"])
        >>> i.toList()
        ['seattle', 'san-diego']

        """
        self._setRecords(records, uels_on_axes=uels_on_axes)
        self.container._synch_with_gams(gams_to_gamspy=self._is_miro_input)
        self._winner = "python"

    def gamsRepr(self) -> str:
        """
        Returns the string representation of this Parameter in the GAMS language.

        The representation includes the parameter name and its domain (e.g., 'p(i, j)').

        Returns
        -------
        str
            The GAMS string representation.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> d = gp.Parameter(m, "d", domain=i)
        >>> d.gamsRepr()
        'd(i)'

        """
        representation = self.name
        if self.domain:
            representation += self._get_domain_str(self._domain_forwarding)

        return representation

    def getDeclaration(self) -> str:
        """
        Returns the GAMS declaration statement for this Parameter.

        This string is used internally to declare the parameter in the GAMS execution environment.

        Returns
        -------
        str
            The GAMS declaration statement (e.g., 'Parameter p(i);').

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
        >>> a.getDeclaration()
        'Parameter a(i);'

        """
        output = f"Parameter {self.gamsRepr()}"

        if self.description:
            output += ' "' + self.description + '"'

        if self.records is None:
            output += " / /"

        if self.dimension == 0 and self.records is not None:
            value = self.toValue()
            value = utils._map_special_values(value)
            if isinstance(value, float) and np.isnan(value):
                value = "Undf"

            output += f" / {value} /"

        output += ";"

        return output

    def getAssignment(self) -> str:
        """
        Returns the latest GAMS assignment statement for this Parameter.

        This string represents the last assignment operation performed on the parameter
        (e.g., 'p(i) = 5;').

        Returns
        -------
        str
            The GAMS assignment statement.

        Raises
        ------
        ValidationError
            If the parameter has not been assigned yet.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
        >>> a[i] = a[i] * 5
        >>> a.getAssignment()
        'a(i) = a(i) * 5;'

        """
        if self._assignment is None:
            raise ValidationError("Parameter was not assigned!")

        return self._assignment.getDeclaration()
