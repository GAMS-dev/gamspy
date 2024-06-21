from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._symbols.implicits.implicit_variable import ImplicitVariable

if TYPE_CHECKING:
    from gamspy import Equation, Parameter, Set, Variable
    from gamspy._algebra.expression import Expression


class ImplicitParameter(ImplicitSymbol, operable.Operable):
    def __init__(
        self,
        parent: Parameter | Variable | Equation,
        name: str,
        domain: list[Set | str] = [],
        records: Any | None = None,
    ) -> None:
        """Implicit Parameter

        Parameters
        ----------
        parent : Parameter | Variable | Equation
        name : str
        domain : List[Set | str], optional
        records : Any, optional
        """
        super().__init__(parent, name, domain)
        self._records = records
        self._assignment = None

    def __neg__(self) -> ImplicitParameter:
        return ImplicitParameter(
            parent=self.parent,
            name=f"-{self.name}",
            domain=self.domain,
        )

    def __invert__(self):
        return expression.Expression("", "not", self)

    def __getitem__(self, indices: list | str) -> ImplicitParameter:
        domain = validation.validate_domain(self, indices)

        return ImplicitParameter(
            parent=self.parent, name=self.name, domain=domain
        )

    def __setitem__(self, indices: list | str, rhs: Expression) -> None:
        # self[domain] = rhs
        domain = validation._transform_given_indices(self.domain, indices)

        if isinstance(rhs, float):
            rhs = utils._map_special_values(rhs)  # type: ignore

        statement = expression.Expression(
            ImplicitParameter(
                parent=self.parent, name=self.name, domain=domain
            ),
            "=",
            rhs,
        )

        self.container._add_statement(statement)
        self.parent._assignment = statement

        self.parent._is_dirty = True
        self.container._run()

    @property
    def dimension(self):
        return self.parent.dimension

    def __eq__(self, other):  # type: ignore
        op = "eq"
        if isinstance(
            other,
            (ImplicitVariable, expression.Expression, operation.Operation),
        ):
            op = "=e="
        return expression.Expression(self, op, other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    def gamsRepr(self) -> str:
        """Representation of the parameter in GAMS syntax.

        Returns:
            str: String representation of the parameter in GAMS syntax.
        """
        representation = self.name
        if self.domain:
            representation += utils._get_domain_str(self.domain)

        return representation
