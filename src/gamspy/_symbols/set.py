from __future__ import annotations

import itertools
import uuid
from typing import TYPE_CHECKING, Any, Literal

import gams.transfer as gt
import pandas as pd
from gams.core.gdx import GMS_DT_SET

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Alias, Container
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits.implicit_set import ImplicitSet


class SetMixin:
    @property
    def pos(self):
        """
        Element position in the current set, starting with 1.

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.pos")

    @property
    def ord(self):
        """
        Same as .pos but for ordered sets only.

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.ord")

    @property
    def off(self):
        """
        Element position in the current set minus 1. So .off = .pos - 1

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.off")

    @property
    def rev(self):
        """
        Reverse element position in the current set, so the value for
        the last element is 0, the value for the penultimate is 1, etc.

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.rev")

    @property
    def uel(self):
        """
        Element position in the unique element list.

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.uel")

    @property
    def len(self):
        """
        Length of the set element name (a count of the number of characters).

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.len")

    @property
    def tlen(self):
        """
        Length of the set element text (a count of the number of characters).

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.tlen")

    @property
    def val(self):
        """
        If a set element is a number, this attribute gives the value of the number.
        For extended range arithmetic symbols, the symbols are reproduced.
        If a set element is a string that is not a number, then this attribute is
        not defined and trying to use it results in an error.

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.val")

    @property
    def tval(self):
        """
        If a set element text is a number, this attribute gives the value of the number.
        For extended range arithmetic symbols, the symbols are reproduced.
        If a set element text is a string that is not a number, then this attribute is
        not defined and trying to use it results in an error.

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.tval")

    @property
    def first(self):
        """
        Returns 1 for the first set element, otherwise 0.

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.first")

    @property
    def last(self):
        """
        Returns 1 for the last set element, otherwise 0.

        Returns
        -------
        ImplicitSet
        """
        return implicits.ImplicitSet(self, name=f"{self.name}.last")

    def lag(
        self: Alias | Set,
        n: int | Symbol | Expression,
        type: Literal["linear", "circular"] = "linear",
    ) -> ImplicitSet:
        """
        Lag operation shifts the values of a Set or Alias by one to the left

        Parameters
        ----------
        n : int | Symbol | Expression
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear

        Examples
        --------
        >>> import gamspy as gp
        >>>
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", description="time sequence", records=[f"y-{x}" for x in range(1987, 1992)])
        >>> a = gp.Parameter(m, name="a", domain=[t])
        >>> b = gp.Parameter(m, name="b", domain=[t])
        >>> c = gp.Parameter(m, name="c", domain=[t])
        >>> a[t] = 1986 + gp.Ord(t)
        >>> b[t] = -1
        >>> b[t] = a[t.lag(1, "linear")]
        >>> b.records.values.tolist()
        [['y-1988', 1987.0], ['y-1989', 1988.0], ['y-1990', 1989.0], ['y-1991', 1990.0]]
        >>> c[t] = a[t.lag(1, "circular")]
        >>> c.records.values.tolist()
        [['y-1987', 1991.0], ['y-1988', 1987.0], ['y-1989', 1988.0], ['y-1990', 1989.0], ['y-1991', 1990.0]]

        """
        jump = n if isinstance(n, int) else n.gamsRepr()  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(self, name=f"{self.name} -- {jump}")

        if type == "linear":
            return implicits.ImplicitSet(self, name=f"{self.name} - {jump}")

        raise ValueError("Lag type must be linear or circular")

    def lead(
        self: Set | Alias,
        n: int | Symbol | Expression,
        type: Literal["linear", "circular"] = "linear",
    ) -> ImplicitSet:
        """
        Lead shifts the values of a Set or Alias by one to the right

        Parameters
        ----------
        n : int | Symbol | Expression
        type : 'linear' or 'circular', optional

        Returns
        -------
        ImplicitSet

        Raises
        ------
        ValueError
            When type is not circular or linear

        Examples
        --------
        >>> import gamspy as gp
        >>>
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", description="time sequence", records=[f"y-{x}" for x in range(1987, 1992)])
        >>> a = gp.Parameter(m, name="a", domain=[t])
        >>> c = gp.Parameter(m, name="c", domain=[t])
        >>> d = gp.Parameter(m, name="d", domain=[t])
        >>> a[t] = 1986 + gp.Ord(t)
        >>> c[t] = -1
        >>> c[t.lead(2, "linear")] = a[t]
        >>> c.records.values.tolist()
        [['y-1987', -1.0], ['y-1988', -1.0], ['y-1989', 1987.0], ['y-1990', 1988.0], ['y-1991', 1989.0]]
        >>> d[t.lead(2, "circular")] = a[t]
        >>> d.records.values.tolist()
        [['y-1987', 1990.0], ['y-1988', 1991.0], ['y-1989', 1987.0], ['y-1990', 1988.0], ['y-1991', 1989.0]]

        """
        jump = n if isinstance(n, int) else n.gamsRepr()  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(self, name=f"{self.name} ++ {jump}")

        if type == "linear":
            return implicits.ImplicitSet(self, name=f"{self.name} + {jump}")

        raise ValueError("Lead type must be linear or circular")

    def sameAs(self: Set | Alias, other: Set | Alias | str) -> Expression:
        """
        Evaluates to true if this set is identical to the given set or alias, false otherwise.

        Parameters
        ----------
        other : Set | Alias

        Returns
        -------
        Expression
        """
        return gp.math.same_as(self, other)


class Set(gt.Set, operable.Operable, Symbol, SetMixin):
    """
    Represents a Set symbol in GAMS.
    https://www.gams.com/latest/docs/UG_SetDefinition.html

    Parameters
    ----------
    container : Container
        Container of the set.
    name : str, optional
        Name of the set. Name is autogenerated by default.
    domain : list[Set | Alias | str] | Set | Alias | str, optional
        Domain of the set.
    is_singleton : bool, optional
        Whether the set is a singleton set. Singleton sets cannot contain more than one element.
    records : pd.DataFrame | np.ndarray | list, optional
        Records of the set.
    domain_forwarding : bool, optional
        Whether the set forwards the domain. See: https://gams.com/latest/docs/UG_SetDefinition.html#UG_SetDefinition_ImplicitSetDefinition
    description : str, optional
        Description of the set.
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

    """

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        domain: list[Set | Alias | str] | Set | Alias | str = [],
        is_singleton: bool = False,
        records: Any | None = None,
        description: str = "",
    ):
        # create new symbol object
        obj = Set.__new__(
            cls,
            container,
            name,
            domain,
            is_singleton,
            records,
            description=description,
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
        obj._is_singleton = is_singleton

        # typing
        obj._gams_type = GMS_DT_SET
        obj._gams_subtype = 1 if obj.is_singleton else 0

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj._is_dirty = False
        obj.where = condition.Condition(obj)
        obj.container._add_statement(obj)
        obj._current_index = 0
        obj._synchronize = True

        # miro support
        obj._is_miro_input = False
        obj._is_miro_output = False

        return obj

    def __new__(
        cls,
        container: Container,
        name: str | None = None,
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        is_singleton: bool = False,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
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
                    " because it is not a Set object)"
                )
            except KeyError:
                return object.__new__(cls)

    def __init__(
        self,
        container: Container,
        name: str | None = None,
        domain: list[Set | Alias | str] | Set | Alias | str | None = None,
        is_singleton: bool = False,
        records: Any | None = None,
        domain_forwarding: bool = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
    ):
        self._is_miro_input = is_miro_input
        self._is_miro_output = is_miro_output

        self._synchronize = True

        # domain handling
        if domain is None:
            domain = ["*"]

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

            if self.is_singleton != is_singleton:
                raise ValueError(
                    "Cannot overwrite symbol in container unless"
                    " 'is_singleton' is left unchanged"
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
            self.where = condition.Condition(self)

            if name is not None:
                name = validation.validate_name(name)
                if is_miro_input or is_miro_output:
                    name = name.lower()  # type: ignore
            else:
                name = "s" + str(uuid.uuid4()).replace("-", "_")

            singleton_check(is_singleton, records)

            previous_state = container.miro_protect
            container.miro_protect = False

            super().__init__(
                container,
                name,
                domain,
                is_singleton,
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
            self.container._add_statement(self)

            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
            else:
                self.container._run()

            container.miro_protect = previous_state

    def __len__(self):
        if self.records is not None:
            return len(self.records.index)

        return 0

    def __getitem__(self, indices: tuple | str) -> implicits.ImplicitSet:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitSet(self, name=self.name, domain=domain)

    def __setitem__(self, indices: tuple | str, rhs):
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        if isinstance(rhs, bool):
            rhs = "yes" if rhs is True else "no"  # type: ignore

        statement = expression.Expression(
            implicits.ImplicitSet(self, name=self.name, domain=domain),
            "=",
            rhs,
        )

        self.container._add_statement(statement)
        self._assignment = statement

        self._is_dirty = True
        self.container._run()

    @property
    def records(self):
        """
        Records of the Set

        Returns
        -------
        DataFrame
        """
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
            self._mark_forwarded_domain_sets()

            # reset state check flags for all symbols in the container
            for symbol in self.container.data.values():
                symbol._requires_state_check = True

    def setRecords(self, records: Any, uels_on_axes: bool = False) -> None:
        super().setRecords(records, uels_on_axes)
        self.container._run()

    def gamsRepr(self) -> str:
        """
        Representation of this Set in GAMS language.

        Returns
        -------
        str
        """
        return self.name

    def getDeclaration(self) -> str:
        """
        Declaration of the Set in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> i.getDeclaration()
        'Set i(*);'

        """
        output = f"Set {self.name}"

        if self._is_singleton:
            output = f"Singleton {output}"

        output += self._get_domain_str()

        if self.description:
            output += f' "{self.description}"'

        output += ";"

        return output

    def getAssignment(self) -> str:
        """
        Latest assignment to the Set in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> i['i1'] = False
        >>> i.getAssignment()
        'i("i1") = no;'

        """
        if not hasattr(self, "_assignment"):
            raise ValidationError("Set is not assigned!")

        return self._assignment.getDeclaration()


def singleton_check(is_singleton: bool, records: Any | None):
    if is_singleton and records is not None and len(records) > 1:
        raise ValidationError(
            "Singleton set records size cannot be more than one."
        )
