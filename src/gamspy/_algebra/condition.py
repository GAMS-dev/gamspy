from __future__ import annotations

from typing import TYPE_CHECKING, cast

import gamspy._algebra.domain as domain
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._algebra.sparse as sparse
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._container import Container
from gamspy._internals import DataSource
from gamspy._notation import LATEX, UNICODE
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy.exceptions import FatalError, ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Parameter
    from gamspy._algebra.domain import Domain
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.number import Number
    from gamspy._algebra.operation import Card, Operation, Ord
    from gamspy._algebra.sparse import SparseAssignment
    from gamspy._notation import Notation
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitSet
    from gamspy._types import ConditionType, ImplicitSymbolType, SymbolType
    from gamspy.math import MathOp


class Condition(operable.Operable):
    """
    Condition class allows conditioning on GAMSPy constructs such as symbols and expressions.

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
        conditioning_on: SymbolType
        | ImplicitSymbolType
        | Expression
        | Operation
        | Domain
        | Number
        | Card
        | Ord
        | MathOp,
        condition: ConditionType | Number | bool | int | float | None = None,
    ):
        self.conditioning_on = conditioning_on
        self.condition = condition
        self._where = None
        self.container: Container | None = None
        if (
            hasattr(conditioning_on, "container")
            and conditioning_on.container is not None
        ):
            self.container = cast("Container | None", conditioning_on.container)
        elif hasattr(condition, "container") and condition.container is not None:
            self.container = cast("Container | None", condition.container)

        self.domain = None
        if hasattr(conditioning_on, "domain"):
            self.domain = conditioning_on.domain

    def __eq__(self, other):
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def __hash__(self):
        return id(self)

    def __getitem__(
        self, condition: ConditionType | Number | bool | int | float
    ) -> Condition:
        return Condition(self.conditioning_on, condition)

    def __setitem__(
        self,
        condition: ConditionType | Number | bool | int | float,
        rhs: Expression
        | Operation
        | Condition
        | MathOp
        | Number
        | int
        | float
        | bool
        | Parameter
        | ImplicitParameter
        | ImplicitSet
        | SparseAssignment,
    ):
        if not isinstance(self.container, Container):
            raise FatalError("Cannot determine the container of the expression!")

        # conditioning_on.where[condition] = rhs
        eq_types = (syms.Equation, implicits.ImplicitEquation)
        right_operand, op_type = sparse._unwrap(rhs)
        if isinstance(right_operand, bool):
            right_operand = "yes" if right_operand is True else "no"

        if isinstance(self.conditioning_on, eq_types):
            if op_type == sparse.SPARSE:
                raise ValidationError(
                    "`sparse` cannot be used in an equation definition because GAMS "
                    "supports the sparse assignment operator ($=) in assignments only."
                )

            op_type = ".."

        lhs = Condition(self.conditioning_on, condition)
        statement = expression.Expression(lhs, op_type, right_operand)
        self.container._add_statement(statement)

        if isinstance(self.conditioning_on, ImplicitSymbol):
            # Cannot validate definition if we are in a gp.Loop since the control indices can be provided by the gp.Loop
            if not self.container._in_loop:
                statement._validate_definition(
                    utils._unpack(self.conditioning_on.domain)
                )

            if isinstance(self.conditioning_on, implicits.ImplicitEquation):
                self.conditioning_on.parent._definition = statement
            else:
                self.conditioning_on.parent._assignment = statement  # ty: ignore[unresolved-attribute]

            self.conditioning_on.container._synch_with_gams()
            self.conditioning_on.parent._should_load_from = DataSource.GAMS  # ty: ignore[unresolved-attribute]

        elif isinstance(self.conditioning_on, syms.Alias):
            self.conditioning_on._assignment = statement
            self.conditioning_on.container._synch_with_gams()
            self.conditioning_on.alias_with._should_load_from = DataSource.GAMS
        elif isinstance(
            self.conditioning_on,
            (syms.Set, syms.Parameter, syms.Variable, syms.Equation),
        ):
            self.conditioning_on._assignment = statement
            self.conditioning_on.container._synch_with_gams()
            self.conditioning_on._should_load_from = DataSource.GAMS

        else:
            self.container._synch_with_gams()

    def __repr__(self) -> str:
        return f"Condition(conditioning_on={self.conditioning_on}, condition={self.condition})"

    def __len__(self):
        if self.records is not None:
            return len(self.records.index)

        return 0

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
        if not isinstance(self.container, Container):
            raise FatalError("Cannot determine the container of the expression!")

        if self.domain is None:
            raise FatalError("Cannot determine the domain of the expression!")

        if isinstance(
            self.conditioning_on,
            (syms.Set, syms.Alias, implicits.ImplicitSet),
        ):
            with self.container._temp_symbol(
                syms.Set,
                self.domain,  # ty: ignore[invalid-argument-type]
            ) as temp_set:
                temp_set[...] = self
            return temp_set.records
        elif isinstance(self.conditioning_on, domain.Domain):
            with self.container._temp_symbol(
                syms.Set,
                self.domain,  # ty: ignore[invalid-argument-type]
            ) as temp_set:
                temp_set[...].where[self.condition] = True  # ty: ignore[invalid-assignment]
            return temp_set.records

        with self.container._temp_symbol(
            syms.Parameter,
            self.domain,  # ty: ignore[invalid-argument-type]
        ) as temp_param:
            temp_param[...] = self
        return temp_param.records

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
        return self._create_gams_repr()

    def _create_gams_repr(self, *, logical: bool = False) -> str:
        condition_str = expression.get_logical_gams_repr(self.condition)
        conditioning_on_str = (
            expression.get_logical_gams_repr(self.conditioning_on)
            if logical
            else self.conditioning_on.gamsRepr()
        )

        if isinstance(self.condition, bool):
            condition_str = str(int(self.condition))

        if isinstance(self.conditioning_on, expression.Expression):
            conditioning_on_str = f"({conditioning_on_str})"

        return f"{conditioning_on_str} $ ({condition_str})"

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
        return self._math_repr(LATEX)

    def unicodeRepr(self) -> str:
        """
        Representation of this condition in mathematical notation with Unicode
        characters.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, "i", records=["i1", "i2"])
        >>> a = gp.Parameter(m, "a", domain=i)
        >>> condition = a[i].where[a[i] >= 2]
        >>> print(condition.unicodeRepr())
        a[i] | a[i] ≥ 2

        """
        return self._math_repr(UNICODE)

    def _math_repr(self, notation: Notation) -> str:
        conditioning_on_str, conditioning_on_prec, conditioning_on_open = (
            expression.math_operand(self.conditioning_on, notation)
        )
        if conditioning_on_prec < expression.LEAF_PRECEDENCE or conditioning_on_open:
            conditioning_on_str = f"({conditioning_on_str})"

        condition_str = expression.get_operand_math_repr(self.condition, notation)
        return f"{conditioning_on_str}{notation.such_that}{condition_str}"
