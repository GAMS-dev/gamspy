from __future__ import annotations

import itertools
import os
import threading
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
import gamspy.utils as utils
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    from gamspy import Alias, Container
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitSet
    from gamspy._types import OperableType
    from gamspy.math import Dim
    from gamspy.math.misc import MathOp


class SetMixin:
    @property
    def pos(self: Set | Alias) -> ImplicitParameter:
        """
        Returns the element position in the current set, starting with 1.

        This attribute corresponds to the `.pos` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            The position of the element.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego", "new-york"], description="canning plants")
        >>> i.pos.records.values.tolist()
        [['seattle', 'position', 1.0], ['san-diego', 'position', 2.0], ['new-york', 'position', 3.0]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.pos")

    @property
    def ord(self: Set | Alias) -> ImplicitParameter:
        """
        Same as .pos but for ordered sets only.

        Returns
        -------
        ImplicitParameter
            The ordinal position of the element.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego", "new-york"], description="canning plants")
        >>> i.ord.records.values.tolist()
        [['seattle', 'order', 1.0], ['san-diego', 'order', 2.0], ['new-york', 'order', 3.0]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.ord")

    @property
    def off(self: Set | Alias) -> ImplicitParameter:
        """
        Returns the element position in the current set minus 1.

        Mathematically: `.off` = `.pos` - 1.
        This attribute corresponds to the `.off` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            The offset position of the element.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego", "new-york"], description="canning plants")
        >>> i.off.records.values.tolist()
        [['san-diego', 'off', 1.0], ['new-york', 'off', 2.0]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.off")

    @property
    def rev(self: Set | Alias) -> ImplicitParameter:
        """
        Returns the reverse element position in the current set.

        The value for the last element is 0, the penultimate is 1, and so on.
        This attribute corresponds to the `.rev` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            The reverse position value.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego", "new-york"], description="canning plants")
        >>> i.rev.records.values.tolist()
        [['seattle', 'reverse', 2.0], ['san-diego', 'reverse', 1.0]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.rev")

    @property
    def uel(self: Set | Alias) -> ImplicitParameter:
        """
        Returns the element position in the global Unique Element List (UEL).

        This attribute corresponds to the `.uel` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            The position index in the UEL table.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego", "new-york"], description="canning plants")
        >>> i.uel.records.values.tolist()
        [['seattle', 'uel_position', 1.0], ['san-diego', 'uel_position', 2.0], ['new-york', 'uel_position', 3.0]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.uel")

    @property
    def len(self: Set | Alias) -> ImplicitParameter:
        """
        Returns the length of the set element name (count of characters).

        This attribute corresponds to the `.len` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            The character count of the element name.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego", "new-york"], description="canning plants")
        >>> i.len.records.values.tolist()
        [['seattle', 'length', 7.0], ['san-diego', 'length', 9.0], ['new-york', 'length', 8.0]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.len")

    @property
    def tlen(self: Set | Alias) -> ImplicitParameter:
        """
        Returns the length of the set element explanatory text (count of characters).

        This attribute corresponds to the `.tlen` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            The character count of the element text.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=[("seattle", "Wisconsin"), ("san-diego", ""), ("new-york", " ")], description="canning plants")
        >>> i.tlen.records.values.tolist()
        [['seattle', 'text_length', 9.0], ['new-york', 'text_length', 1.0]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.tlen")

    @property
    def val(self: Set | Alias) -> ImplicitParameter:
        """
        Returns the numerical value if the set element name is a number.

        If the element is not a number, this attribute is undefined and may result in an error or ignored record.
        This attribute corresponds to the `.val` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            The numerical value of the element name.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["12", "20", "-13.4"], description="canning plants")
        >>> i.val.records.values.tolist()
        [['12', 'value', 12.0], ['20', 'value', 20.0], ['-13.4', 'value', -13.4]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.val")

    @property
    def tval(self: Set | Alias) -> ImplicitParameter:
        """
        Returns the numerical value if the set element text is a number.

        If the element text is not a number, this attribute is undefined.
        This attribute corresponds to the `.tval` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            The numerical value of the element text.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=[("seattle", "12"), ("san-diego", ""), ("new-york", "-13.4")], description="canning plants")
        >>> i.tval.records.values.tolist()
        [['seattle', 'text_value', 12.0], ['new-york', 'text_value', -13.4]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.tval")

    @property
    def first(self: Set | Alias) -> ImplicitParameter:
        """
        Returns 1 if the element is the first in the set, otherwise 0.

        This attribute corresponds to the `.first` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            An implicit parameter with value 1 for the first element.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego", "new-york"], description="canning plants")
        >>> i.first.records.values.tolist()
        [['seattle', 'is_first', 1.0]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.first")

    @property
    def last(self: Set | Alias) -> ImplicitParameter:
        """
        Returns 1 if the element is the last in the set, otherwise 0.

        This attribute corresponds to the `.last` attribute in GAMS.

        Returns
        -------
        ImplicitParameter
            An implicit parameter with value 1 for the last element.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego", "new-york"], description="canning plants")
        >>> i.last.records.values.tolist()
        [['new-york', 'is_last', 1.0]]

        """
        return implicits.ImplicitParameter(self, f"{self.name}.last")

    def lag(
        self,
        n: OperableType,
        type: Literal["linear", "circular"] = "linear",
    ) -> ImplicitSet:
        """
        Shifts the values of a Set or Alias by `n` positions to the left (lag).

        Parameters
        ----------
        n : OperableType
            The number of positions to shift. Can be an integer or a GAMS symbol.
        type : 'linear' or 'circular', optional
            The type of lag to perform:
            - 'linear' (default): Elements shifted out of bounds are dropped.
            - 'circular': Elements shifted out of bounds wrap around to the end.

        Returns
        -------
        ImplicitSet
            The shifted set expression.

        Raises
        ------
        ValueError
            If `type` is not 'linear' or 'circular'.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", description="time sequence", records=[f"y-{x}" for x in range(1987, 1992)])
        >>> a = gp.Parameter(m, name="a", domain=[t])
        >>> b = gp.Parameter(m, name="b", domain=[t])
        >>> a[t] = 1986 + gp.Ord(t)
        >>> b[t] = a[t.lag(1, "linear")]
        >>> b.records.values.tolist()
        [['y-1988', 1987.0], ['y-1989', 1988.0], ['y-1990', 1989.0], ['y-1991', 1990.0]]

        """
        assert isinstance(self, (gp.Set, gp.Alias))
        jump = n if isinstance(n, int) else n.gamsRepr()  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(self, name=self.name, extension=f" -- {jump}")

        if type == "linear":
            return implicits.ImplicitSet(self, name=self.name, extension=f" - {jump}")

        raise ValueError("Lag type must be linear or circular")

    def lead(
        self,
        n: OperableType,
        type: Literal["linear", "circular"] = "linear",
    ) -> ImplicitSet:
        """
        Shifts the values of a Set or Alias by `n` positions to the right (lead).

        Parameters
        ----------
        n : OperableType
            The number of positions to shift. Can be an integer or a GAMS symbol.
        type : 'linear' or 'circular', optional
            The type of lead to perform:
            - 'linear' (default): Elements shifted out of bounds are dropped.
            - 'circular': Elements shifted out of bounds wrap around to the beginning.

        Returns
        -------
        ImplicitSet
            The shifted set expression.

        Raises
        ------
        ValueError
            If `type` is not 'linear' or 'circular'.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", description="time sequence", records=[f"y-{x}" for x in range(1987, 1992)])
        >>> a = gp.Parameter(m, name="a", domain=[t])
        >>> c = gp.Parameter(m, name="c", domain=[t])
        >>> a[t] = 1986 + gp.Ord(t)
        >>> c[t.lead(2, "linear")] = a[t]
        >>> c.records.values.tolist()
        [['y-1989', 1987.0], ['y-1990', 1988.0], ['y-1991', 1989.0]]

        """
        assert isinstance(self, (gp.Set, gp.Alias))
        jump = n if isinstance(n, int) else f"({n.gamsRepr()})"  # type: ignore

        if type == "circular":
            return implicits.ImplicitSet(self, name=self.name, extension=f" ++ {jump}")

        if type == "linear":
            return implicits.ImplicitSet(self, name=self.name, extension=f" + {jump}")

        raise ValueError("Lead type must be linear or circular")

    def sameAs(self, other: Set | Alias | str) -> MathOp:
        """
        Evaluates to True if the current set element is identical to the given symbol or string.

        This corresponds to the `sameAs` operator in GAMS.

        Parameters
        ----------
        other : Set | Alias | str
            The other set, alias, or string label to compare against.

        Returns
        -------
        MathOp
            A boolean expression that evaluates to True (1) if they match, False (0) otherwise.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
        >>> j = gp.Set(m, name="j", records=["new-york", "seattle"])
        >>> attr = gp.Parameter(m, "attr", domain=[i, j])
        >>> attr[i,j] = i.sameAs(j)
        >>> attr.records.values.tolist()
        [['seattle', 'seattle', 1.0]]

        """
        assert isinstance(self, (gt.Set, gt.Alias))
        assert isinstance(other, (gt.Set, gt.Alias, str))
        return gp.math.same_as(self, other)


class Set(gt.Set, operable.Operable, Symbol, SetMixin):
    """
    Represents a Set symbol in GAMS.

    See https://gamspy.readthedocs.io/en/latest/user/basics/set.html for more details.

    Parameters
    ----------
    container : Container
        The Container object that this set belongs to.
    name : str, optional
        Name of the set. If not provided, a unique name is generated automatically.
    domain : Sequence[Set | Alias | str] | Set | Alias | str, optional
        The domain of the set. Can be a list of other Sets/Aliases, a single Set/Alias,
        or strings representing set names. Use "*" for the universe set. Default is ["*"].
    is_singleton : bool, optional
        If True, restricts the set to contain at most one element. Default is False.
    records : pd.DataFrame | np.ndarray | list, optional
        Initial elements to populate the set.
    domain_forwarding : bool | list[bool], optional
        If True, adding records to this set will implicitly add new elements to the
        domain sets (if they are dynamic). Default is False.
    description : str, optional
        A human-readable description of the set.
    uels_on_axes : bool, optional
        If True, implies that the Unique Element Labels (UELs) for the domain are
        contained in the axes (index/columns) of the provided `records` object
        (e.g., pandas DataFrame). Default is False.
    is_miro_input : bool, optional
        If True, flags this set as an input symbol for GAMS MIRO. Default is False.
    is_miro_output : bool, optional
        If True, flags this set as an output symbol for GAMS MIRO. Default is False.

    Examples
    --------
    Simple set:


    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = m.addSet("i", records=["a", "b"])


    Indexed set:


    >>> j = m.addSet("j", domain=i)


    Singleton set:


    >>> s = m.addSet("s", is_singleton=True, records=["s1"])

    """

    @classmethod
    def _constructor_bypass(
        cls,
        container: Container,
        name: str,
        domain: Sequence[Set | Alias | str] | Set | Alias | str | None = None,
        is_singleton: bool = False,
        records: Any | None = None,
        description: str = "",
    ):
        if domain is None:
            domain = ["*"]

        if isinstance(domain, (gp.Set, gp.Alias, str)):
            domain = [domain]

        if isinstance(domain, gp.math.Dim):
            domain = gp.math._generate_dims(container, domain.dims)

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
        obj._domain_violations = None

        # typing
        obj._gams_type = GMS_DT_SET
        obj._gams_subtype = 1 if obj.is_singleton else 0

        # add to container
        container.data.update({name: obj})

        # gamspy attributes
        obj.where = condition.Condition(obj)
        obj._latex_name = name.replace("_", r"\_")
        obj.container._add_statement(obj)
        obj._synchronize = True
        obj._metadata = {}
        obj._winner = "python"

        # miro support
        obj._is_miro_input = False
        obj._is_miro_output = False

        return obj

    def __new__(
        cls,
        container: Container | None = None,
        name: str | None = None,
        domain: Sequence[Set | Alias | str] | Set | Alias | str | Dim | None = None,
        is_singleton: bool = False,
        records: Any | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
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
                    " because it is not a Set object)"
                )
            except KeyError:
                return object.__new__(cls)

    def __init__(
        self,
        container: Container | None = None,
        name: str | None = None,
        domain: Sequence[Set | Alias | str] | Set | Alias | str | Dim | None = None,
        is_singleton: bool = False,
        records: Any | None = None,
        domain_forwarding: bool | list[bool] = False,
        description: str = "",
        uels_on_axes: bool = False,
        is_miro_input: bool = False,
        is_miro_output: bool = False,
    ):
        self._metadata: dict[str, Any] = {}
        if (is_miro_input or is_miro_output) and name is None:
            raise ValidationError("Please specify a name for miro symbols.")

        self._is_miro_input = is_miro_input
        self._is_miro_output = is_miro_output
        self._is_miro_symbol = is_miro_input or is_miro_output
        self._domain_violations = None

        self._synchronize = True
        self._winner = "python"

        # domain handling
        if domain is None:
            domain = ["*"]

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

            if self.is_singleton != is_singleton:
                raise ValueError(
                    "Cannot overwrite symbol in container unless"
                    " 'is_singleton' is left unchanged"
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
                    raise ValidationError("Set requires a container.") from e
            assert container is not None

            self.where = condition.Condition(self)
            self._assignment: Expression | None = None

            if name is not None:
                name = validation.validate_name(name)
                if is_miro_input or is_miro_output:
                    name = name.lower()  # type: ignore
            else:
                name = container._get_symbol_name(prefix="s")

            self._singleton_check(is_singleton, records, domain)
            previous_state = container._options.miro_protect
            container._options.miro_protect = False

            super().__init__(
                container,
                name,
                domain,
                is_singleton,
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
            self.container._add_statement(self)

            if records is not None:
                self.setRecords(records, uels_on_axes=uels_on_axes)
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
            "_metadata": self._metadata,
            "_synchronize": self._synchronize,
        }
        if self._assignment is not None:
            info["_assignment"] = self._assignment.getDeclaration()

        return info

    def _deserialize(self, info: dict) -> None:
        # Set attributes
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

    def __len__(self):
        if self.records is not None:
            return len(self.records.index)

        return 0

    def __getitem__(
        self, indices: Sequence | str | int | EllipsisType | slice
    ) -> implicits.ImplicitSet:
        domain = validation.validate_domain(self, indices)

        return implicits.ImplicitSet(self, name=self.name, domain=domain)

    def __setitem__(
        self,
        indices: Sequence | str | int | implicits.ImplicitSet | EllipsisType | slice,
        rhs,
    ):
        # self[domain] = rhs
        domain = validation.validate_domain(self, indices)

        if isinstance(rhs, bool):
            rhs = "yes" if rhs is True else "no"  # type: ignore

        statement = expression.Expression(
            implicits.ImplicitSet(self, name=self.name, domain=domain),
            "=",
            rhs,
        )

        statement._validate_definition(utils._unpack(domain))

        self.container._add_statement(statement)
        self._assignment = statement

        self.container._synch_with_gams(gams_to_gamspy=True)
        self._winner = "gams"

    def __repr__(self) -> str:
        return f"Set(name='{self.name}', domain={self.domain})"

    def _singleton_check(
        self,
        is_singleton: bool,
        records: Any | None,
        domain: Sequence[Set | Alias | str],
    ):
        if is_singleton:
            if records is not None and len(records) != 1:
                raise ValidationError("Singleton set records size must be one.")

            if len(domain) != 1:
                raise ValidationError(
                    f"Length of the domain of the singleton set must be 1 but found {len(domain)}"
                )

    @property
    def records(self):
        """
        Records of the Set

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import gamspy as gp
        >>> import numpy as np
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i")
        >>> i.setRecords(["seattle", "san-diego"])
        >>> i.records.values.tolist()
        [['seattle', ''], ['san-diego', '']]

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
        Sets the records (elements) of the Set.

        This is a convenience method to load data into the set. It handles various
        input formats like lists and pandas DataFrames.

        Parameters
        ----------
        records : Any
            The data to load. Common formats:


            - List of strings: `['i1', 'i2']`
            - List of tuples (for multi-dimensional sets): `[('a', '1'), ('b', '2')]`
            - pandas DataFrame.
        uels_on_axes : bool, optional
            If True, assumes that the domain information is located in the axes
            (index/columns) of the `records` object rather than the data values.
            Use this when passing a DataFrame where the indices represent the set elements.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, name="i")
        >>> i.setRecords(["seattle", "san-diego"])
        >>> i.records.values.tolist()
        [['seattle', ''], ['san-diego', '']]

        """
        self._setRecords(records, uels_on_axes=uels_on_axes)
        self.container._synch_with_gams(gams_to_gamspy=self._is_miro_input)
        self._winner = "python"

    def gamsRepr(self) -> str:
        """
        Representation of this Set in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", domain=["*"], records=['i1','i2'])
        >>> i.gamsRepr()
        'i'

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

        if self.is_singleton:
            output = f"Singleton {output}"

        output += self._get_domain_str(self._domain_forwarding)

        if self.description:
            output += f' "{self.description}"'

        if self.records is None:
            output += " / /"

        output += ";"

        return output

    def getAssignment(self) -> str:
        """
        Latest assignment to the Set in GAMS

        Returns
        -------
        str

        Raises
        ------
        ValueError
            When type is not circular or linear

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=['i1','i2'])
        >>> i['i1'] = False
        >>> i.getAssignment()
        'i("i1") = no;'

        """
        if self._assignment is None:
            raise ValidationError("Set was not assigned!")

        return self._assignment.getDeclaration()
