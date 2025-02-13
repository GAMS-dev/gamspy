from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import gamspy._algebra.condition as condition
import gamspy._algebra.domain as domain
import gamspy._algebra.number as number
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._extrinsic import ExtrinsicFunction
from gamspy._symbols.implicits import ImplicitSet
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy import Alias, Set
    from gamspy._symbols.implicits import ImplicitEquation
    from gamspy._types import OperableType

GMS_MAX_LINE_LENGTH = 80000
LINE_LENGTH_OFFSET = 79000

LATEX_DATA_MAP = {
    "=g=": "\\geq",
    "=l=": "\\leq",
    "=e=": "=",
    "*": "\\cdot",
    "and": "\\wedge",
    "or": "\\vee",
    "xor": "\\oplus",
    "$": "|",
}


@dataclass
class DomainPlaceHolder:
    indices: list[tuple[str, int]]


def peek(stack):
    if len(stack) > 0:
        return stack[-1]
    return None


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
        left: OperableType | ImplicitEquation | None,
        data: str | MathOp | ExtrinsicFunction,
        right: OperableType | None,
    ):
        self.left = (
            utils._map_special_values(left) if type(left) is float else left
        )
        self.data = data
        self.right = (
            utils._map_special_values(right) if type(right) is float else right
        )

        if data == "=" and isinstance(right, Expression):
            right._fix_equalities()

        self.representation = self._create_output_str()
        self.where = condition.Condition(self)
        self._create_domain()
        left_control = getattr(left, "controlled_domain", [])
        right_control = getattr(right, "controlled_domain", [])
        self.controlled_domain: list[Set | Alias] = list(
            {*left_control, *right_control}
        )
        self.container = None
        if hasattr(left, "container"):
            self.container = left.container  # type: ignore
        elif hasattr(right, "container"):
            self.container = right.container  # type: ignore

    def _create_domain(self):
        for loc, result in [
            (self.left, "_left_domain"),
            (self.right, "_right_domain"),
        ]:
            if isinstance(loc, condition.Condition):
                loc = loc.conditioning_on

            if loc is None or type(loc) in (int, float, str):
                result_domain = []  # left is a scalar
            elif type(loc) is domain.Domain:
                result_domain = loc.sets
            else:
                result_domain = loc.domain

            setattr(self, result, result_domain)

        left_domain = self._left_domain
        right_domain = self._right_domain

        set_to_index = {}

        for domain_char, domain_ptr in (
            ("l", left_domain),
            ("r", right_domain),
        ):
            for i, d in enumerate(domain_ptr):
                if isinstance(d, str):
                    continue  # string domains are fixed and they do not count

                if d not in set_to_index:
                    set_to_index[d] = []

                set_to_index[d].append((domain_char, i))

        shadow_domain = []
        result_domain = []
        for d in [*left_domain, *right_domain]:
            if isinstance(d, str):
                continue

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

    def _get_operand_representations(self) -> tuple[str, str]:
        left_str, right_str = "", ""
        if self.left is not None:
            left_str = (
                str(self.left)
                if type(self.left) in (int, float, str)
                else self.left.gamsRepr()  # type: ignore
            )

        if self.right is not None:
            right_str = (
                str(self.right)
                if type(self.right) in (int, float, str)
                else self.right.gamsRepr()  # type: ignore
            )

        # ((((ord(n) - 1) / 10) * -1) + ((ord(n) / 10) * 0));   -> not valid
        # ((((ord(n) - 1) / 10) * (-1)) + ((ord(n) / 10) * 0)); -> valid
        if type(self.left) in (int, float) and self.left < 0:  # type: ignore
            left_str = f"({left_str})"

        if type(self.right) in (int, float) and self.right < 0:  # type: ignore
            right_str = f"({right_str})"

        # (voycap(j,k)$vc(j,k)) .. sum(.) -> not valid
        #  voycap(j,k)$vc(j,k)  .. sum(.) -> valid
        if self.data in ["..", "="] and type(self.left) is condition.Condition:
            left_str = left_str[1:-1]

        return left_str, right_str

    def _create_output_str(self) -> str:
        left_str, right_str = self._get_operand_representations()

        # get around 80000 line length limitation in GAMS
        length = len(left_str) + len(self.data) + len(right_str)
        if length >= GMS_MAX_LINE_LENGTH - LINE_LENGTH_OFFSET:
            out_str = f"{left_str} {self.data}\n {right_str}"
        else:
            out_str = f"{left_str} {self.data} {right_str}"

        if self.data == ".":
            return out_str.replace(" ", "")

        if self.data in ["..", "="]:
            return f"{out_str};"

        if self.data in ["=g=", "=l=", "=e=", "=n=", "=x=", "=c=", "=b="]:
            return out_str

        return f"({out_str})"

    def __eq__(self, other):
        return Expression(self, "=e=", other)

    def __ne__(self, other):
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

    def __repr__(self) -> str:
        return f"Expression(left={self.left}, data={self.data}, right={self.right})"

    def _replace_operator(self, operator: str):
        self.data = operator
        self.representation = self._create_output_str()

    def latexRepr(self) -> str:
        """
        Representation of this Expression in Latex.

        Returns
        -------
        str
        """
        if type(self.left) is float:
            self.left = utils._map_special_values(self.left)

        if type(self.right) is float:
            self.right = utils._map_special_values(self.right)

        if self.left is None:
            left_str = ""
        else:
            left_str = (
                str(self.left)
                if type(self.left) in (int, float, str)
                else self.left.latexRepr()  # type: ignore
            )

        if self.right is None:
            right_str = ""
        else:
            right_str = (
                str(self.right)
                if type(self.right) in (int, float, str)
                else self.right.latexRepr()  # type: ignore
            )

        data = self.data
        if type(self.data) is str:
            data = LATEX_DATA_MAP.get(self.data, self.data)

        data_str = (
            str(data) if type(data) in (int, float, str) else data.latexRepr()  # type: ignore
        )

        if self.data == "/":
            return f"\\frac{{{left_str}}}{{{right_str}}}"

        return f"{left_str} {data_str} {right_str}"

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
        # Uses a stack based post-order traversal algorithm.
        EQ_MAP: dict[str, str] = {"=g=": ">=", "=e=": "eq", "=l=": "<="}
        stack = []
        root = self

        while True:
            while root is not None:
                if hasattr(root, "right"):
                    stack.append(root.right)

                stack.append(root)
                root = root.left if hasattr(root, "left") else None  # type: ignore

            if len(stack) == 0:
                break

            root = stack.pop()

            if isinstance(root, Expression):
                if root.data in EQ_MAP:
                    root._replace_operator(EQ_MAP[root.data])
                else:
                    root.representation = root._create_output_str()

            last_item = peek(stack)
            if (
                hasattr(root, "right")
                and last_item is not None
                and last_item is root.right
            ):
                stack.pop()
                stack.append(root)
                root = root.right
            else:
                root = None

    def _find_all_symbols(self) -> list[str]:
        # Finds all symbols in an expression with a stack based inorder
        # traversal algorithm (O(N)).
        symbols: list[str] = []
        stack = []

        node = self
        while True:
            if node is not None:
                stack.append(node)
                node = getattr(node, "left", None)  # type: ignore
            elif stack:
                node = stack.pop()

                if isinstance(node, Symbol):
                    if node.name not in symbols:
                        symbols.append(node.name)
                    stack += node.domain
                    node = None
                elif isinstance(node, ImplicitSymbol):
                    if node.parent.name not in symbols:
                        symbols.append(node.parent.name)
                    stack += node.domain
                    stack += node.container[node.parent.name].domain
                    node = None
                elif isinstance(node, operation.Operation):
                    stack += node.op_domain
                    node = node.rhs
                elif isinstance(node, condition.Condition):
                    stack.append(node.conditioning_on)

                    if isinstance(node.condition, Expression):
                        node = node.condition
                    else:
                        stack.append(node.condition)
                        node = None
                elif isinstance(node, (operation.Ord, operation.Card)):
                    stack.append(node._symbol)
                    node = None
                elif isinstance(node, MathOp):
                    if isinstance(node.elements[0], Expression):
                        node = node.elements[0]
                    else:
                        stack += node.elements
                        node = None
                elif isinstance(node, ExtrinsicFunction):
                    stack += list(node.args)
                    node = None
                elif isinstance(node, Expression) and isinstance(
                    node.data, MathOp
                ):
                    node = node.data
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

                if isinstance(node, condition.Condition):
                    given_condition = node.condition

                    if isinstance(given_condition, Expression):
                        symbols += given_condition._find_all_symbols()
                    elif isinstance(given_condition, ImplicitSymbol):
                        symbols.append(given_condition.parent.name)

                if isinstance(node, operation.Operation):
                    stack += node.op_domain
                    node = node.rhs
                else:
                    node = getattr(node, "right", None)
            else:
                break  # pragma: no cover

        return symbols

    def _validate_definition(
        self, control_stack: list[Set | Alias | ImplicitSet]
    ) -> None:
        stack = []

        node = self.right
        while True:
            if node is not None:
                stack.append(node)
                node = getattr(node, "left", None)  # type: ignore
            elif stack:
                node = stack.pop()

                if isinstance(node, operation.Operation):
                    node._validate_operation(control_stack.copy())
                elif isinstance(node, ImplicitSymbol):
                    for elem in node.domain:
                        if hasattr(elem, "is_singleton") and elem.is_singleton:
                            continue

                        if (
                            isinstance(elem, Symbol)
                            and elem not in control_stack
                        ):
                            raise ValidationError(
                                f"Uncontrolled set `{elem}` entered as constant!"
                            )
                        elif (
                            isinstance(elem, ImplicitSymbol)
                            and elem.parent not in control_stack
                        ):
                            raise ValidationError(
                                f"Uncontrolled set `{elem.parent}` entered as constant!"
                            )

                node = getattr(node, "right", None)
            else:
                break  # pragma: no cover


class SetExpression(Expression):
    def __init__(
        self,
        left: OperableType,
        data: Literal["+", "-", "*", "not"],
        right: OperableType,
    ):
        super().__init__(left, data, right)
        self._adjust_left_right()

    def _adjust_left_right(self) -> None:
        if isinstance(self.left, (ImplicitSet, SetExpression)):
            if type(self.right) in (int, float):
                if self.right == 0:
                    self.right = "no"
                elif self.right == 1:
                    self.right = "yes"
                else:
                    raise ValidationError(
                        f"Incompatible operand `{self.right}` for the set operation `{self.data}`."
                    )
            elif (
                type(self.right) is condition.Condition
                and type(self.right.conditioning_on) is number.Number
            ):
                if self.right.conditioning_on._value == 0:
                    self.right.conditioning_on._value = "no"
                elif self.right.conditioning_on._value == 1:
                    self.right.conditioning_on._value = "yes"
                raise ValidationError(
                    f"Incompatible operand `{self.right}` for the set operation `{self.data}`."
                )

        if isinstance(self.right, (ImplicitSet, SetExpression)):
            if type(self.left) in (int, float):
                if self.left == 0:
                    self.left = "no"
                elif self.left == 1:
                    self.left = "yes"
                else:
                    raise ValidationError(
                        f"Incompatible operand `{self.left}` for the set operation `{self.data}`."
                    )
            elif (
                type(self.left) is condition.Condition
                and type(self.left.conditioning_on) is number.Number
            ):
                if self.left.conditioning_on._value == 0:
                    self.left.conditioning_on._value = "no"
                elif self.left.conditioning_on._value == 1:
                    self.left.conditioning_on._value = "yes"
                else:
                    raise ValidationError(
                        f"Incompatible operand `{self.left}` for the set operation `{self.data}`."
                    )

        self.representation = self._create_output_str()
