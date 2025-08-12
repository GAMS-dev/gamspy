from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import gamspy._algebra.condition as condition
import gamspy._algebra.domain as domain
import gamspy._algebra.number as number
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._symbols as gp_syms
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._config import get_option
from gamspy._extrinsic import ExtrinsicFunction
from gamspy._symbols.implicits import ImplicitSet
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._symbols.symbol import Symbol
from gamspy.exceptions import ValidationError
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from numbers import Real

    import pandas as pd

    from gamspy import Alias, Set
    from gamspy._symbols.implicits import ImplicitEquation
    from gamspy._types import OperableType

GMS_MAX_LINE_LENGTH = 80000
LINE_LENGTH_OFFSET = 79000


@dataclass
class DomainPlaceHolder:
    indices: list[tuple[str, int]]


def peek(stack):
    if len(stack) > 0:
        return stack[-1]
    return None


# Higher number means higher precedence.
PRECEDENCE = {
    "..": 0,
    "=": 0,
    "or": 1,
    "xor": 2,
    "and": 3,
    "=e=": 4,
    "=n=": 4,
    "=b=": 4,
    "eq": 4,
    "ne": 4,
    ">=": 4,
    "<=": 4,
    ">": 4,
    "<": 4,
    "=g=": 4,
    "=l=": 4,
    "=x=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "not": 7,
    "u-": 7,
}

# Defines how operators of the same precedence are grouped.
ASSOCIATIVITY = {
    "or": "left",
    "xor": "left",
    "and": "left",
    "=e=": "left",
    "=n=": "left",
    "=x=": "left",
    "=b=": "left",
    "eq": "left",
    "ne": "left",
    "=": "left",
    ">=": "left",
    "<=": "left",
    ">": "left",
    "<": "left",
    "=g=": "left",
    "=l=": "left",
    "..": "non",
    "+": "left",
    "-": "left",
    "*": "left",
    "/": "left",
    "not": "right",
    "u-": "right",
}

# Precedence for a leaf node is considered infinite.
LEAF_PRECEDENCE = float("inf")


def get_operand_repr(operand) -> str:
    if hasattr(operand, "gamsRepr"):
        return operand.gamsRepr()

    representation = str(operand)

    # b[i] * -1   -> not valid
    # b[i] * (-1) -> valid
    if isinstance(operand, (int, float)) and operand < 0:
        representation = f"({representation})"

    return representation


def create_expression_string(root_node: Expression) -> str:
    """
    Creates a string representation of a binary expression tree without recursion.
    It uses an iterative post-order traversal to build the expression,
    adding parentheses only when necessary based on operator precedence and
    associativity rules.
    """
    if not isinstance(root_node, Expression):
        return get_operand_repr(root_node)

    # 1. Get nodes in post-order using an iterative approach (2 stacks)
    s1: list[OperableType | ImplicitEquation | str] = [root_node]
    post_order_nodes = []
    while s1:
        node = s1.pop()
        post_order_nodes.append(node)
        if isinstance(node, Expression):
            if node.left is not None:
                s1.append(node.left)
            if node.right is not None:
                s1.append(node.right)

    # 2. Build the expression string using the post-order list
    eval_stack: list[tuple[str, Real]] = []
    for node in reversed(post_order_nodes):
        if not isinstance(node, Expression):
            eval_stack.append((get_operand_repr(node), LEAF_PRECEDENCE))
            continue

        op = node.operator
        op_prec = PRECEDENCE[op]
        op_assoc = ASSOCIATIVITY[op]

        # Handle unary ops
        if op in ("u-", "not"):
            operand_str, operand_prec = eval_stack.pop()

            # Add parentheses if the operand's operator has lower precedence
            if operand_prec < op_prec:
                operand_str = f"({operand_str})"

            if op == "u-":
                new_str = f"(-{operand_str})"
                # A parenthesized expression has the highest precedence
                eval_stack.append((new_str, LEAF_PRECEDENCE))
            else:  # Standard handling for 'not'
                new_str = f"not {operand_str}"
                eval_stack.append((new_str, op_prec))

        # Handle binary ops
        else:
            right_str, right_prec = eval_stack.pop()
            left_str, left_prec = eval_stack.pop()

            if left_prec < op_prec or (
                left_prec == op_prec and op_assoc == "right"
            ):
                left_str = f"({left_str})"

            if right_prec < op_prec or (
                right_prec == op_prec and op_assoc == "left"
            ):
                right_str = f"({right_str})"

            # get around 80000 line length limitation in GAMS
            length = len(left_str) + len(op) + len(right_str)
            if length >= GMS_MAX_LINE_LENGTH - LINE_LENGTH_OFFSET:
                new_str = f"{left_str} {op}\n {right_str}"
            else:
                new_str = f"{left_str} {op} {right_str}"
            eval_stack.append((new_str, op_prec))

    final_string = eval_stack[0][0]

    if root_node.operator in ("=", ".."):
        return f"{final_string};"

    return final_string


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
    'a * b'

    """

    def __init__(
        self,
        left: OperableType | ImplicitEquation | None,
        operator: str,
        right: OperableType | str | None,
    ):
        self.left = (
            utils._map_special_values(left)
            if isinstance(left, float)
            else left
        )
        self.operator = operator
        self.right = (
            utils._map_special_values(right)
            if isinstance(right, float)
            else right
        )

        if operator == "=" and isinstance(right, Expression):
            right._fix_equalities()

        self._representation: str | None = None
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

    @property
    def representation(self) -> str:
        if self._representation is None:
            self._representation = create_expression_string(self)

        return self._representation

    def _create_domain(self):
        for loc, result in (
            (self.left, "_left_domain"),
            (self.right, "_right_domain"),
        ):
            if isinstance(loc, condition.Condition):
                loc = loc.conditioning_on

            if loc is None or isinstance(loc, (int, float, str)):
                result_domain = []  # left is a scalar
            elif isinstance(loc, domain.Domain):
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
        for d in (*left_domain, *right_domain):
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

        return Expression(left, self.operator, right)

    @property
    def records(self) -> pd.DataFrame | None:
        """
        Evaluates the expression and returns the resulting records.

        Returns
        -------
        pd.DataFrame | None

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = gp.Parameter(m, records=5)
        >>> b = gp.Parameter(m, records=6)
        >>> (a + b).records
           value
        0   11.0

        """
        assert self.container is not None
        temp_name = "a" + utils._get_unique_name()
        temp_param = gp_syms.Parameter._constructor_bypass(
            self.container, temp_name, self.domain
        )
        temp_param[...] = self
        del self.container.data[temp_name]
        return temp_param.records

    def toValue(self) -> float | None:
        """
        Convenience method to return expression records as a Python float. Only possible if there is a single record as a result of the expression evaluation.

        Returns
        -------
        float | None

        Raises
        ------
        TypeError
            In case the dimension of the expression is not zero.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = gp.Parameter(m, records=5)
        >>> b = gp.Parameter(m, records=6)
        >>> (a + b).toValue()
        np.float64(11.0)

        """
        if self.dimension != 0:
            raise TypeError(
                f"Cannot extract value data for non-scalar expressions (expression dimension is {self.dimension})"
            )

        records = self.records
        if records is not None:
            return records["value"][0]

        return records

    def toList(self) -> list | None:
        """
        Convenience method to return the records of the expression as a list.

        Returns
        -------
        list | None

        Examples
        --------
        >>> import numpy as np
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, records=range(3))
        >>> a = gp.Parameter(m, domain=i, records=np.array([1,2,3]))
        >>> b = gp.Parameter(m, domain=i, records=np.array([4,5,6]))
        >>> (a + b).toList()
        [['0', 5.0], ['1', 7.0], ['2', 9.0]]

        """
        records = self.records
        if records is not None:
            return records.values.tolist()

        return None

    def __eq__(self, other):
        return Expression(self, "=e=", other)

    def __ne__(self, other):
        return Expression(self, "ne", other)

    def __bool__(self):
        raise ValidationError(
            "An expression cannot be used as a truth value. If you are "
            "trying to generate an expression, use binary operators "
            "instead (e.g. &, |, ^). For more details, see: "
            "https://gamspy.readthedocs.io/en/latest/user/gamspy_for_gams_users.html#logical-operations"
        )

    def __repr__(self) -> str:
        return f"Expression(left={self.left}, data={self.operator}, right={self.right})"

    def _replace_operator(self, operator: str):
        self.operator = operator

    def latexRepr(self) -> str:
        """
        Representation of this Expression in Latex.

        Returns
        -------
        str
        """
        data_map = {
            "=g=": "\\geq",
            "=l=": "\\leq",
            "=e=": "=",
            "*": "\\cdot",
            "and": "\\wedge",
            "or": "\\vee",
            "xor": "\\oplus",
            "$": "|",
        }

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
                else self.left.latexRepr()  # type: ignore
            )

        if self.right is None:
            right_str = ""
        else:
            right_str = (
                str(self.right)
                if isinstance(self.right, (int, float, str))
                else self.right.latexRepr()  # type: ignore
            )

        operator = self.operator
        if isinstance(self.operator, str):
            operator = data_map.get(self.operator, self.operator)

        operator_str = (
            str(operator)
            if isinstance(operator, (int, float, str))
            else operator.latexRepr()
        )

        if self.operator == "/":
            return f"\\frac{{{left_str}}}{{{right_str}}}"

        return f"({left_str} {operator_str} {right_str})"

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
        'a * b'

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
        'a * b'

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

            if isinstance(root, Expression) and root.operator in EQ_MAP:
                root._replace_operator(EQ_MAP[root.operator])

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
        if not get_option("DOMAIN_VALIDATION") or not get_option(
            "DOMAIN_VALIDATION"
        ):
            return

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
            if isinstance(self.right, (int, float)):
                if self.right == 0:
                    self.right = "no"
                elif self.right == 1:
                    self.right = "yes"
                else:
                    raise ValidationError(
                        f"Incompatible operand `{self.right}` for the set operation `{self.operator}`."
                    )
            elif isinstance(self.right, condition.Condition) and isinstance(
                self.right.conditioning_on, number.Number
            ):
                if self.right.conditioning_on._value == 0:
                    self.right.conditioning_on._value = "no"
                elif self.right.conditioning_on._value == 1:
                    self.right.conditioning_on._value = "yes"
                raise ValidationError(
                    f"Incompatible operand `{self.right}` for the set operation `{self.operator}`."
                )

        if isinstance(self.right, (ImplicitSet, SetExpression)):
            if isinstance(self.left, (int, float)):
                if self.left == 0:
                    self.left = "no"
                elif self.left == 1:
                    self.left = "yes"
                else:
                    raise ValidationError(
                        f"Incompatible operand `{self.left}` for the set operation `{self.operator}`."
                    )
            elif isinstance(self.left, condition.Condition) and isinstance(
                self.left.conditioning_on, number.Number
            ):
                if self.left.conditioning_on._value == 0:
                    self.left.conditioning_on._value = "no"
                elif self.left.conditioning_on._value == 1:
                    self.left.conditioning_on._value = "yes"
                else:
                    raise ValidationError(
                        f"Incompatible operand `{self.left}` for the set operation `{self.operator}`."
                    )
