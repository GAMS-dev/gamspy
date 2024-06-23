from __future__ import annotations

import itertools
import uuid
from typing import TYPE_CHECKING, Any

import gams.transfer as gt
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
    from gamspy import Alias, Container, Set
    from gamspy._algebra.expression import Expression


class Parameter(gt.Parameter, operable.Operable, Symbol):
    """
    Represents a parameter symbol in GAMS.
    https://www.gams.com/latest/docs/UG_DataEntry.html#UG_DataEntry_Parameters

    Parameters
    ----------
    container : Container
        Container of the parameter.
    name : str, optional
        Name of the parameter. Name is autogenerated by default.
    domain : list[Set | Alias | str] | Set | Alias | str, optional
        Domain of the parameter.
    records : int | float | pd.DataFrame | np.ndarray | list, optional
        Records of the parameter.
    domain_forwarding : bool, optional
        Whether the parameter forwards the domain. See: https://gams.com/latest/docs/UG_SetDefinition.html#UG_SetDefinition_ImplicitSetDefinition
    description : str, optional
        Description of the parameter.
    uels_on_axes : bool
        Assume that symbol domain information is contained in the axes of the given records.
    is_miro_input : bool
        Whether the symbol is a GAMS MIRO input symbol. See: https://gams.com/miro/tutorial.html
    is_miro_output : bool
        Whether the symbol is a GAMS MIRO output symbol. See: https://gams.com/miro/tutorial.html

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=['i1','i2'])
    >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])

    """

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        domain: list[Set | Alias | str] | Set | Alias | str = [],
        records: Any | None = None,
        description: str = "",
    ):
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

        # typing
        obj._gams_type = GMS_DT_PAR
        obj._gams_subtype = 0

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj._is_dirty = False
        obj._synchronize = True

        obj.where = condition.Condition(obj)
        obj.container._add_statement(obj)

        # miro support
        obj._is_miro_input = False
        obj._is_miro_output = False
        obj._is_miro_table = False

        return obj

    def __new__(
        cls,
        container: Container,
        name: str | None = None,
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
        is_miro_table: bool = False,
    ):
        if not isinstance(container, gp.Container):
            raise TypeError(
                "Container must of type `Container` but found"
                f" {type(container)}"
            )

        if name is None:
            return object.__new__(cls)
        else:
            if not isinstance(name, str):
                raise TypeError(
                    f"Name must of type `str` but found {type(name)}"
                )

            try:
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
        container: Container,
        name: str | None = None,
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
        is_miro_table: bool = False,
    ):
        # miro support
        self._is_miro_input = is_miro_input
        self._is_miro_output = is_miro_output
        self._is_miro_table = is_miro_table

        self._synchronize = True

        # domain handling
        if domain is None:
            domain = []

        if isinstance(domain, (gp.Set, gp.Alias, str)):
            domain = [domain]

        # does symbol exist
        has_symbol = False
        if isinstance(getattr(self, "container", None), gp.Container):
            has_symbol = True

        if has_symbol:
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
            self._is_dirty = False

            if name is not None:
                name = validation.validate_name(name)

                if is_miro_input or is_miro_output:
                    name = name.lower()  # type: ignore
            else:
                name = "p" + str(uuid.uuid4()).replace("-", "_")

            previous_state = container.miro_protect
            container.miro_protect = False
            super().__init__(
                container,
                name,
                domain,
                domain_forwarding=domain_forwarding,
                description=description,
                uels_on_axes=uels_on_axes,
            )

            if is_miro_input:
                self._already_loaded = False
                container._miro_input_symbols.append(self.name)

            if is_miro_output:
                container._miro_output_symbols.append(self.name)

            validation.validate_container(self, self.domain)
            self.where = condition.Condition(self)
            self.container._add_statement(self)

            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            else:
                self.container._synch_with_gams()

            container.miro_protect = previous_state

    def __getitem__(self, indices: tuple | str) -> implicits.ImplicitParameter:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitParameter(self, name=self.name, domain=domain)

    def __setitem__(
        self,
        indices: tuple | str | implicits.ImplicitSet,
        rhs: Expression | float | int,
    ):
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        if self._is_miro_input and self.container.miro_protect:
            raise ValidationError(
                "Cannot assign to protected miro input symbols. `miro_protect`"
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

        self.container._add_statement(statement)
        self._assignment = statement

        self._is_dirty = True
        self.container._synch_with_gams()

    def __eq__(self, other):  # type: ignore
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

    def __neg__(self):
        return implicits.ImplicitParameter(
            self, name=f"-{self.name}", domain=self._domain
        )

    @property
    def records(self):
        """
        Records of the Parameter

        Returns
        -------
        DataFrame
        """
        if not self._is_dirty:
            return self._records

        self.container._synch_with_gams()

        return self._records

    @records.setter
    def records(self, records):
        if (
            hasattr(self, "_is_miro_input")
            and self._is_miro_input
            and self.container.miro_protect
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
        self.modified = True

        self.container._requires_state_check = True
        self.container.modified = True

        if self._records is not None and self.domain_forwarding:
            self._domainForwarding()
            self._mark_forwarded_domain_sets(self.domain_forwarding)

            # reset state check flags for all symbols in the container
            for symbol in self.container.data.values():
                symbol._requires_state_check = True

    def setRecords(self, records: Any, uels_on_axes: bool = False) -> None:
        super().setRecords(records, uels_on_axes)
        self.container._synch_with_gams()

    def gamsRepr(self) -> str:
        """
        Representation of this Parameter in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getDeclaration(self) -> str:
        """
        Declaration of the Parameter in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
        >>> a.getDeclaration()
        'Parameter a(i);'

        """
        statement_name = self.name
        if self.domain:
            statement_name += self._get_domain_str()

        output = f"Parameter {statement_name}"

        if self.description:
            output += ' "' + self.description + '"'

        output += ";"

        return output

    def getAssignment(self) -> str:
        """
        Latest assignment to the Parameter in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> a = gp.Parameter(m, "a", [i], records=[['i1',1],['i2',2]])
        >>> a[i] = a[i] * 5
        >>> a.getAssignment()
        'a(i) = (a(i) * 5);'

        """
        if not hasattr(self, "_assignment"):
            raise ValidationError("Parameter is not defined!")

        return self._assignment.getDeclaration()
