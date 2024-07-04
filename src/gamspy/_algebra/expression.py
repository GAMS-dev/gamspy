from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import gamspy._algebra.condition as condition
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._extrinsic import ExtrinsicFunction
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    import gamspy._algebra.expression as expression
    from gamspy._algebra.operation import Operation

    OperandType = Optional[
        Union[
            int,
            float,
            str,
            Operation,
            expression.Expression,
            Symbol,
            ImplicitSymbol,
            MathOp,
        ]
    ]

GMS_MAX_LINE_LENGTH = 80000
LINE_LENGTH_OFFSET = 79000


class Expression(operable.Operable):
    """
    Expression of two operands and an operation.

    Parameters
    ----------
    left: str | int | float | Parameter | Variable | None
        Left operand
    data: str
        Operation
    right: str | int | float | Parameter | Variable | None
        Right operand

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> a = gp.Parameter(m, name="a")
    >>> b = gp.Parameter(m, name="b")
    >>> expression = a * b
    >>> expression.gamsRepr()
    '(a * b)'

    """

    def __init__(
        self,
        left: OperandType,
        data: str | MathOp | ExtrinsicFunction,
        right: OperandType,
    ):
        self.left = (
            utils._map_special_values(left)
            if isinstance(left, float)
            else left
        )
        self.data = data
        self.right = (
            utils._map_special_values(right)
            if isinstance(right, float)
            else right
        )

        if data == "=" and isinstance(right, Expression):
            right._fix_equalities()
        self.representation = self._create_representation()
        self.where = condition.Condition(self)

    def _create_representation(self) -> str:
        left_str, right_str = self._get_operand_representations()
        out_str = self._create_output_str(left_str, right_str)

        # Adapt to GAMS quirks
        if self.data in ["=", ".."] and out_str[0] == "(":
            # (voycap(j,k)$vc(j,k)).. sum(.) -> not valid
            # voycap(j,k)$vc(j,k).. sum(.)   -> valid
            match_index = utils._get_matching_paranthesis_indices(out_str)
            out_str = out_str[1:match_index] + out_str[match_index + 1 :]

        return out_str

    def _get_operand_representations(self) -> tuple[str, str]:
        if self.left is None:
            left_str = ""
        else:
            left_str = (
                str(self.left)
                if isinstance(self.left, (int, float, str))
                else self.left.gamsRepr()
            )

        if self.right is None:
            right_str = ""
        else:
            right_str = (
                str(self.right)
                if isinstance(self.right, (int, float, str))
                else self.right.gamsRepr()
            )

        # ((((ord(n) - 1) / 10) * -1) + ((ord(n) / 10) * 0)); -> not valid
        # ((((ord(n) - 1) / 10) * (-1)) + ((ord(n) / 10) * 0)); -> valid
        if isinstance(self.left, (int, float)) and self.left < 0:
            left_str = f"({left_str})"

        if isinstance(self.right, (int, float)) and self.right < 0:
            right_str = f"({right_str})"

        return left_str, right_str

    def _create_output_str(self, left_str: str, right_str: str) -> str:
        # get around 80000 line length limitation in GAMS
        length = len(left_str) + len(self.data) + len(right_str)
        if length >= GMS_MAX_LINE_LENGTH - LINE_LENGTH_OFFSET:
            out_str = f"{left_str} {self.data}\n {right_str}"
        else:
            out_str = f"{left_str} {self.data} {right_str}"

        if self.data in ["..", "="]:
            return f"{out_str};"

        if self.data in ["=g=", "=l=", "=e=", "=n=", "=x=", "=c=", "=b="]:
            return out_str

        return f"({out_str})"

    def __eq__(self, other):  # type: ignore
        return Expression(self, "=e=", other)

    def __ne__(self, other):  # type: ignore
        return Expression(self, "ne", other)

    def __neg__(self):
        return Expression(None, "-", self)

    def __bool__(self):
        raise ValidationError(
            "An expression cannot be used as a truth value. If you are "
            "trying to generate an expression, use binary operators "
            "instead (e.g. &, |, ^). For more details, see: "
            "https://gamspy.readthedocs.io/en/latest/user/gamspy_for_gams_users.html#logical-operations"
        )

    def _replace_operator(self, operator: str):
        self.data = operator
        self.representation = self._create_representation()

    def gamsRepr(self) -> str:
        """
        Representation of this Expression in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = gp.Parameter(m, name="a")
        >>> b = gp.Parameter(m, name="b")
        >>> expression = a * b
        >>> expression.gamsRepr()
        '(a * b)'

        """
        return self.representation

    def getDeclaration(self) -> str:
        """
        Declaration of the Expression in GAMS

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = gp.Parameter(m, name="a")
        >>> b = gp.Parameter(m, name="b")
        >>> expression = a * b
        >>> expression.getDeclaration()
        '(a * b)'

        """
        return self.gamsRepr()

    def _fix_equalities(self) -> None:
        # Equality operations on Parameter and Variable objects generate
        # GAMS equality signs: =g=, =e=, =l=. If these signs appear on
        # assignments, replace them with regular equality ops.
        EQ_MAP: dict[Any, str] = {"=g=": ">=", "=e=": "eq", "=l=": "<="}
        stack = []

        node: Expression | None = self
        while True:
            if node is not None:
                stack.append(node)
                node = getattr(node, "left", None)
            elif stack:
                node = stack.pop()

                if isinstance(node, Expression) and node.data in EQ_MAP:
                    node._replace_operator(EQ_MAP[node.data])

                if isinstance(node, operation.Operation) and isinstance(
                    node.expression, Expression
                ):
                    node.expression._fix_equalities()

                node = getattr(node, "right", None)
            else:
                break  # pragma: no cover

        self.representation = self._create_representation()

    def _find_all_symbols(self) -> list[str]:
        symbols: list[str] = []
        stack = []

        node = self
        while True:
            if node is not None:
                stack.append(node)
                node = getattr(node, "left", None)  # type: ignore
            elif stack:
                node = stack.pop()

                if isinstance(node, (Symbol, ImplicitSymbol)):
                    for index, elem in enumerate(node.domain):
                        if isinstance(elem, (Symbol, ImplicitSymbol)):
                            path = validation.get_domain_path(elem)
                            for name in path:
                                if name not in symbols and " " not in name:
                                    symbols.append(name)

                        if (
                            isinstance(node, ImplicitSymbol)
                            and isinstance(elem, str)
                            and elem != "*"
                        ):
                            symbol = node.parent.domain[index]
                            if (
                                not isinstance(symbol, str)
                                and symbol.name not in symbols
                            ):
                                symbols.append(symbol.name)

                    if node.name not in symbols:
                        symbols.append(node.name)
                elif isinstance(node, Expression) and isinstance(
                    node.data, MathOp
                ):
                    stack += list(node.data.elements)
                if isinstance(node, operation.Operation):
                    stack += node.domain
                    node = node.expression
                else:
                    node = getattr(node, "right", None)
            else:
                break  # pragma: no cover

        return symbols

    def _find_symbols_in_conditions(self) -> list[str]:
        symbols: list[str] = []
        stack = []

        node = self
        while True:
            if node is not None:
                stack.append(node)
                node = getattr(node, "left", None)  # type: ignore
            elif stack:
                node = stack.pop()

                if isinstance(node, Expression) and node.data == "$":
                    condition = node.right

                    if isinstance(condition, Expression):
                        symbols += condition._find_all_symbols()
                    elif isinstance(condition, ImplicitSymbol):
                        symbols.append(condition.parent.name)

                if isinstance(node, operation.Operation):
                    stack += node.domain
                    node = node.expression
                else:
                    node = getattr(node, "right", None)
            else:
                break  # pragma: no cover

        return symbols
