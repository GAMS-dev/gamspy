from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.domain as domain
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._symbols.symbol import Symbol

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Alias, Parameter, Set, Variable
    from gamspy._algebra.domain import Domain
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.number import Number
    from gamspy._algebra.operation import Card, Operation, Ord
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
    )
    from gamspy.math import MathOp


class Condition(operable.Operable):
    """
    Condition class allows conditioning on GAMSPy constructs such as Symbols and Expressions.

    Parameters
    ----------
    symbol: ImplicitSymbol | Expression
        Reference to the symbol to be conditioned.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=["i1", "i2", "i3"])
    >>> a = gp.Parameter(m, "a", domain=i, records=[("i1", 10), ("i2", 20), ("i3", 30)])
    >>> # Conditional assignment: Set values to 1 where they exceed 15
    >>> a[i].where[a[i] > 15] = 1
    >>> a.records.values.tolist()
    [['i1', 10.0], ['i2', 1.0], ['i3', 1.0]]

    """

    def __init__(
        self,
        conditioning_on: ImplicitSymbol
        | Set
        | Alias
        | Parameter
        | Variable
        | Expression
        | Operation
        | Domain
        | Number
        | Card
        | Ord
        | MathOp,
        condition: Operation
        | Expression
        | ImplicitParameter
        | ImplicitSet
        | int
        | None = None,
    ):
        self.conditioning_on = conditioning_on
        self.condition = condition
        self._where = None
        self.container = None
        if hasattr(conditioning_on, "container"):
            self.container = conditioning_on.container

        self.domain = None
        if hasattr(conditioning_on, "domain"):
            self.domain = conditioning_on.domain

    @property
    def where(self):
        return Condition(self)

    def __getitem__(
        self,
        condition: Operation | Expression | ImplicitParameter | ImplicitSet,
    ) -> Condition:
        if isinstance(condition, expression.Expression):
            condition._fix_equalities()

        return Condition(self.conditioning_on, condition)

    def __setitem__(self, condition, rhs):
        # conditioning_on.where[condition] = rhs
        eq_types = (syms.Equation, implicits.ImplicitEquation)
        if isinstance(rhs, bool):
            rhs = "yes" if rhs is True else "no"

        op_type = ".." if isinstance(self.conditioning_on, eq_types) else "="

        if isinstance(condition, expression.Expression):
            condition._fix_equalities()

        lhs = Condition(self.conditioning_on, condition)
        statement = expression.Expression(lhs, op_type, rhs)

        if isinstance(self.conditioning_on, ImplicitSymbol):
            statement._validate_definition(utils._unpack(self.conditioning_on.domain))

        self.conditioning_on.container._add_statement(statement)

        if isinstance(self.conditioning_on, ImplicitSymbol):
            self.conditioning_on.parent._assignment = statement
            self.conditioning_on.parent._winner = "gams"
        elif isinstance(self.conditioning_on, Symbol):
            self.conditioning_on._assignment = statement
            self.conditioning_on._winner = "gams"

        if isinstance(self.conditioning_on, implicits.ImplicitEquation):
            self.conditioning_on.parent._definition = statement

        self.conditioning_on.container._synch_with_gams(gams_to_gamspy=True)

    def __repr__(self) -> str:
        return f"Condition(conditioning_on={self.conditioning_on}, condition={self.condition})"

    @property
    def dimension(self) -> int:
        """
        The dimension of the records of the condition.

        Returns
        -------
        int
            Dimensionality

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2", "i3"])
        >>> a = gp.Parameter(m, "a", domain=i, records=[("i1", 10), ("i2", 20), ("i3", 5)])
        >>> condition = a[i].where[a[i] > 9]
        >>> condition.dimension
        1

        """
        if self.domain is None:
            return 0

        return len(self.domain)

    @property
    def records(self) -> pd.DataFrame | None:
        """
        The records of the condition.

        Returns
        -------
        pd.DataFrame | None

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2", "i3"])
        >>> a = gp.Parameter(m, "a", domain=i, records=[("i1", 10), ("i2", 20), ("i3", 5)])
        >>> condition = a[i].where[a[i] > 9]
        >>> condition.records.values.tolist()
        [['i1', 10.0], ['i2', 20.0]]

        """
        assert self.container is not None
        assert self.domain is not None
        if isinstance(
            self.conditioning_on,
            (syms.Set, syms.Alias, implicits.ImplicitSet),
        ):
            temp_name = "c" + utils._get_unique_name()
            temp_sym = syms.Set._constructor_bypass(
                self.container,
                temp_name,
                self.domain,  # type: ignore
            )
            temp_sym[...] = self
            del self.container.data[temp_name]
        elif isinstance(self.conditioning_on, domain.Domain):
            temp_name = "c" + utils._get_unique_name()
            temp_sym = syms.Set._constructor_bypass(
                self.container,
                temp_name,
                self.domain,  # type: ignore
            )
            temp_sym[...].where[self.condition] = True
            del self.container.data[temp_name]
        else:
            temp_name = "c" + utils._get_unique_name()
            temp_sym = syms.Parameter._constructor_bypass(
                self.container,
                temp_name,
                self.domain,  # type: ignore
            )
            temp_sym[...] = self
            del self.container.data[temp_name]

        return temp_sym.records

    def gamsRepr(self) -> str:
        """
        Representation of this condition in GAMS.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> a = gp.Parameter(m, "a", domain=i)
        >>> condition = a[i].where[a[i] > 5]
        >>> condition.gamsRepr()
        'a(i) $ (a(i) > 5)'

        """
        condition_str = (
            self.condition.gamsRepr()  # type: ignore
            if hasattr(self.condition, "gamsRepr")
            else str(self.condition)
        )
        conditioning_on_str = self.conditioning_on.gamsRepr()

        if isinstance(self.condition, bool):
            condition_str = str(int(self.condition))

        if isinstance(self.conditioning_on, expression.Expression):
            conditioning_on_str = f"({conditioning_on_str})"

        return f"{conditioning_on_str} $ ({condition_str})"  # type: ignore

    def getDeclaration(self) -> str:
        """
        Declaration of this condition in GAMS.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> a = gp.Parameter(m, "a", domain=i)
        >>> condition = a[i].where[a[i] > 5]
        >>> condition.getDeclaration()
        'a(i) $ (a(i) > 5)'

        """
        return self.gamsRepr()

    def latexRepr(self) -> str:
        """
        Representation of this condition in Latex.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> a = gp.Parameter(m, "a", domain=i)
        >>> condition = a[i].where[a[i] > 2]
        >>> condition.latexRepr()
        'a_{i} ~ | ~ a_{i} > 2'

        """
        condition_str = (
            self.condition.latexRepr()  # type: ignore
            if hasattr(self.condition, "latexRepr")
            else str(self.condition)
        )
        return f"{self.conditioning_on.latexRepr()} ~ | ~ {condition_str}"  # type: ignore
