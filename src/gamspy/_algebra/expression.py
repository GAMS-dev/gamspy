from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import gamspy as gp
import gamspy._algebra.condition as condition
import gamspy._algebra.domain as domain
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    import gamspy._algebra.expression as expression
    from gamspy import Alias, Set, Variable
    from gamspy._algebra.operation import Operation
    from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
    from gamspy._symbols.symbol import Symbol

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


@dataclass
class DomainPlaceHolder:
    indices: list[tuple[str, int]]


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
        self, left: OperandType, data: str | MathOp, right: OperandType
    ):
        self.left = left
        self.data = data
        self.right = right
        if data == "=" and isinstance(right, Expression):
            right._fix_equalities()
        self.representation = self._create_representation()
        self.where = condition.Condition(self)
        self._create_domain()
        left_control = getattr(left, "controlled_domain", [])
        right_control = getattr(right, "controlled_domain", [])
        self.controlled_domain: list[Set | Alias] = list(
            set([*left_control, *right_control])
        )
        self.container = None
        if left is not None and hasattr(left, "container"):
            self.container = left.container
        elif right is not None and hasattr(right, "container"):
            self.container = right.container

    def _create_domain(self):
        if self.left is None or isinstance(self.left, (int, float, str)):
            left_domain = []  # left is a scalar
        elif isinstance(self.left, domain.Domain):
            left_domain = self.left.sets
        else:
            left_domain = self.left.domain

        self._left_domain = left_domain

        if self.right is None or isinstance(self.right, (int, float, str)):
            right_domain = []  # right is a scalar
        elif isinstance(self.right, domain.Domain):
            right_domain = self.right.sets
        else:
            right_domain = self.right.domain

        self._right_domain = right_domain
        set_to_index = {}
        for i, d in enumerate(left_domain):
            if isinstance(d, str):
                continue  # string domains are fixed and they do not count

            if d not in set_to_index:
                set_to_index[d] = []

            set_to_index[d].append(("l", i))

        for i, d in enumerate(right_domain):
            if isinstance(d, str):
                continue  # string domains are fixed and they do not count

            if d not in set_to_index:
                set_to_index[d] = []

            set_to_index[d].append(("r", i))

        shadow_domain = []
        result_domain = []
        for d in [*left_domain, *right_domain]:
            if isinstance(d, str):
                continue  # string domains are fixed and they do not count

            if d not in result_domain:
                result_domain.append(d)
                indices = set_to_index[d]
                shadow_domain.append(DomainPlaceHolder(indices=indices))

        self._shadow_domain = shadow_domain
        self.domain = result_domain
        self.dimension = validation.get_dimension(self.domain)

    def __getitem__(self, indices):
        indices = validation.validate_domain(self, indices)
        left_domain = [d for d in self._left_domain]
        right_domain = [d for d in self._right_domain]
        for i, s in enumerate(indices):
            for lr, pos in self._shadow_domain[i].indices:
                if lr == "l":
                    left_domain[pos] = s
                else:
                    right_domain[pos] = s

        left = self.left[left_domain] if left_domain else self.left
        right = self.right[right_domain] if right_domain else self.right

        return Expression(left, self.data, right)

    def _create_representation(self):
        left_str, right_str = self._get_operand_representations()
        out_str = self._create_output_str(left_str, right_str)

        # Adapt to GAMS quirks
        if isinstance(self.left, (domain.Domain, syms.Set, syms.Alias)):
            return out_str[1:-1]

        if self.data in ["=", ".."] and out_str[0] == "(":
            # (voycap(j,k)$vc(j,k)).. sum(.) -> not valid
            # voycap(j,k)$vc(j,k).. sum(.)   -> valid
            match_index = utils._get_matching_paranthesis_indices(out_str)
            out_str = out_str[1:match_index] + out_str[match_index + 1 :]

        return out_str

    def _get_operand_representations(self) -> tuple[str, str]:
        if isinstance(self.left, float):
            self.left = utils._map_special_values(self.left)

        if isinstance(self.right, float):
            self.right = utils._map_special_values(self.right)

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

    def _find_variables(self) -> list[Variable]:
        stack = []
        variables: list[Variable] = []

        node: OperandType = self
        while True:
            if node is not None:
                stack.append(node)
                node = getattr(node, "left", None)
            elif stack:
                node = stack.pop()

                if hasattr(node, "data") and isinstance(node.data, MathOp):
                    variables += node.data._find_variables()

                if isinstance(node, gp.Variable):
                    variables.append(node.name)
                elif isinstance(node, implicits.ImplicitVariable):
                    variables.append(node.parent.name)
                elif isinstance(node, operation.Operation):
                    operation_variables = node._extract_variables()
                    variables += operation_variables

                node = getattr(node, "right", None)
            else:
                break  # pragma: no cover

        return list(set(variables))

    def _fix_equalities(self):
        # Equality operations on Parameter and Variable objects generate
        # GAMS equality signs: =g=, =e=, =l=. If these signs appear on
        # assignments, replace them with regular equality ops.
        EQ_MAP = {"=g=": ">=", "=e=": "eq", "=l=": "<="}
        stack = []

        node = self
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
