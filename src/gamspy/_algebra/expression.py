from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import gamspy._algebra.condition as condition
import gamspy._algebra.domain as domain
import gamspy._algebra.number as number
import gamspy._algebra.operable as operable
import gamspy._algebra.operation as operation
import gamspy._algebra.sparse as sparse
import gamspy._symbols as gp_syms
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy._config import get_option
from gamspy._extrinsic import ExtrinsicFunction
from gamspy._special_values import SpecialValues
from gamspy._symbols.base import BaseSymbol
from gamspy._symbols.implicits import ImplicitSet
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
from gamspy._universe import is_universe
from gamspy.exceptions import GamspyException, ValidationError
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np
    import pandas as pd

    from gamspy import Alias, Container, Set
    from gamspy._symbols.implicits import ImplicitEquation
    from gamspy._types import OperableType

GMS_MAX_LINE_LENGTH = 80000
LINE_LENGTH_OFFSET = 79000


@dataclass
class DomainPlaceHolder:
    indices: list[tuple[str, int]]


# Higher number means higher precedence.
PRECEDENCE = {
    "..": 0,
    "=": 0,
    "$=": 0,
    "or": 1,
    "xor": 1,
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
    "$=": "left",
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

ASSIGNMENT_OPERATORS = (sparse.SPARSE, "=")
STATEMENT_OPERATORS = (*ASSIGNMENT_OPERATORS, "..")

# Equality operations generate GAMS equation signs (=g=, =e=, =l=). Outside
# of an equation definition, they have to be written as relational operators.
LOGICAL_OPERATORS = {"=g=": ">=", "=e=": "eq", "=l=": "<="}


def split_assignment(declaration: str) -> tuple[str, str, str]:
    for operator in ASSIGNMENT_OPERATORS:
        left, separator, right = declaration.partition(f" {operator} ")
        if separator:
            return left, operator, right

    raise ValidationError(f"`{declaration}` is not a valid GAMS assignment.")


def get_operand_gams_repr(operand) -> str:
    if hasattr(operand, "gamsRepr"):
        return operand.gamsRepr()

    representation = str(operand)

    # b[i] * -1   -> not valid
    # b[i] * (-1) -> valid
    # The same goes for -INF, which is a string after special values are mapped.
    if representation.startswith("-"):
        representation = f"({representation})"

    return representation


# In LaTeX, a leading minus sign is not wrapped in parentheses, so a negated
# operand binds like a binary minus: a + (-b) and a \cdot (-b) keep their
# parentheses while -a + b does not get any.
LATEX_NEGATION_PRECEDENCE = PRECEDENCE["-"]

# A condition (x ~ | ~ cond) is always parenthesized when it is an operand.
LATEX_CONDITION_PRECEDENCE = 0.5

LATEX_SPECIAL_VALUES = {
    "EPS": "\\text{EPS}",
    "NA": "\\text{NA}",
    "UNDF": "\\text{UNDF}",
    "INF": "\\infty",
    "-INF": "-\\infty",
}


def _latex_number(value: int | float) -> tuple[str, float]:
    if isinstance(value, float):
        if SpecialValues.isEps(value):
            return LATEX_SPECIAL_VALUES["EPS"], LEAF_PRECEDENCE
        if SpecialValues.isNA(value):
            return LATEX_SPECIAL_VALUES["NA"], LEAF_PRECEDENCE
        if SpecialValues.isUndef(value):
            return LATEX_SPECIAL_VALUES["UNDF"], LEAF_PRECEDENCE
        if SpecialValues.isPosInf(value):
            return LATEX_SPECIAL_VALUES["INF"], LEAF_PRECEDENCE
        if SpecialValues.isNegInf(value):
            return LATEX_SPECIAL_VALUES["-INF"], LATEX_NEGATION_PRECEDENCE

    representation = str(abs(value))
    precedence = LEAF_PRECEDENCE
    if "e" in representation:
        # 1.5e-07 -> 1.5 \cdot 10^{-7}
        mantissa, exponent = representation.split("e")
        representation = f"10^{{{int(exponent)}}}"
        if float(mantissa) != 1:
            representation = f"{mantissa} \\cdot {representation}"
            precedence = PRECEDENCE["*"]

    if value < 0:
        return f"-{representation}", LATEX_NEGATION_PRECEDENCE

    return representation, precedence


def latex_operand(operand) -> tuple[str, float, bool]:
    if isinstance(operand, Expression):
        return _create_latex_expression(operand)

    if isinstance(operand, number.Number):
        operand = operand._value

    if isinstance(operand, (int, float)):
        return *_latex_number(operand), False

    if isinstance(operand, str) and operand in LATEX_SPECIAL_VALUES:
        precedence = (
            LATEX_NEGATION_PRECEDENCE if operand.startswith("-") else LEAF_PRECEDENCE
        )
        return LATEX_SPECIAL_VALUES[operand], precedence, False

    if isinstance(operand, operation.Operation):
        return operand.latexRepr(), LEAF_PRECEDENCE, True

    if isinstance(operand, condition.Condition):
        return operand.latexRepr(), LATEX_CONDITION_PRECEDENCE, False

    if hasattr(operand, "latexRepr"):
        return operand.latexRepr(), LEAF_PRECEDENCE, False

    return str(operand), LEAF_PRECEDENCE, False


def get_operand_latex_repr(operand) -> str:
    return latex_operand(operand)[0]


def get_logical_gams_repr(operand) -> str:
    if isinstance(operand, Expression):
        return create_gams_expression(operand, logical=True)

    if isinstance(operand, condition.Condition):
        return operand._create_gams_repr(logical=True)

    return get_operand_gams_repr(operand)


def create_gams_expression(root_node: Expression, *, logical: bool = False) -> str:
    """
    Creates GAMS representation of a binary expression tree without recursion.
    It uses an iterative post-order traversal to build the expression,
    adding parentheses only when necessary based on operator precedence and
    associativity rules. If logical is True, equation signs are written as
    relational operators. The right-hand side of an assignment is always
    written in logical form.
    """
    # 1. Get nodes in post-order (left - right - parent).
    s1: list[tuple[OperableType | ImplicitEquation | str, bool]] = [
        (root_node, logical)
    ]
    post_order_nodes = []
    while s1:
        node, is_logical = s1.pop()
        post_order_nodes.append((node, is_logical))
        if isinstance(node, Expression):
            if node.left is not None:
                s1.append((node.left, is_logical))
            if node.right is not None:
                s1.append(
                    (node.right, is_logical or node.operator in ASSIGNMENT_OPERATORS)
                )

    # 2. Build the GAMS expression
    eval_stack: list[tuple[str, float]] = []
    for node, is_logical in reversed(post_order_nodes):
        if not isinstance(node, Expression):
            representation = (
                get_logical_gams_repr(node)
                if is_logical
                else get_operand_gams_repr(node)
            )
            eval_stack.append((representation, LEAF_PRECEDENCE))
            continue

        op = node.operator
        if is_logical:
            op = LOGICAL_OPERATORS.get(op, op)
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
                new_str = f"(not {operand_str})"
                eval_stack.append((new_str, op_prec))

        # Handle binary ops
        else:
            right_str, right_prec = eval_stack.pop()
            left_str, left_prec = eval_stack.pop()

            if left_prec < op_prec or (left_prec == op_prec and op_assoc == "right"):
                left_str = f"({left_str})"

            if right_prec < op_prec or (right_prec == op_prec and op_assoc == "left"):
                right_str = f"({right_str})"

            # get around 80000 line length limitation in GAMS
            length = len(left_str) + len(op) + len(right_str)
            if length >= GMS_MAX_LINE_LENGTH - LINE_LENGTH_OFFSET:
                new_str = f"{left_str} {op}\n {right_str}"
            else:
                new_str = f"{left_str} {op} {right_str}"
            eval_stack.append((new_str, op_prec))

    final_string = eval_stack[0][0]

    if root_node.operator in STATEMENT_OPERATORS:
        return f"{final_string};"

    return final_string


def create_latex_expression(root_node: Expression) -> str:
    """
    Creates LaTeX representation of a binary expression tree without recursion.
    It uses an iterative post-order traversal to build the expression,
    adding parentheses only when necessary based on operator precedence and
    associativity rules.
    """
    return _create_latex_expression(root_node)[0]


def _create_latex_expression(root_node: Expression) -> tuple[str, float, bool]:
    op_map = {
        "=g=": "\\geq",
        "=l=": "\\leq",
        "=e=": "=",
        "=n=": "\\text{=n=}",
        "=x=": "\\text{=x=}",
        "=b=": "\\text{=b=}",
        "eq": "=",
        "ne": "\\neq",
        ">=": "\\geq",
        "<=": "\\leq",
        "*": "\\cdot",
        "and": "\\wedge",
        "or": "\\vee",
        "xor": "\\oplus",
        "$=": "\\stackrel{\\$}{=}",
    }

    # 1. Get nodes in post-order (left - right - parent).
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

    # 2. Build the LaTeX expression. Each entry also tracks whether the string
    # ends with a big operator (e.g. a sum) whose scope extends to the right.
    eval_stack: list[tuple[str, float, bool]] = []
    for node in reversed(post_order_nodes):
        if not isinstance(node, Expression):
            eval_stack.append(latex_operand(node))
            continue

        op = node.operator
        op_prec = PRECEDENCE[op]
        op_assoc = ASSOCIATIVITY[op]

        # Handle unary ops
        if op in ("u-", "not"):
            operand_str, operand_prec, operand_open = eval_stack.pop()

            # Add parentheses if the operand's operator has lower precedence
            if operand_prec < op_prec:
                operand_str = f"({operand_str})"
                operand_open = False

            if op == "u-":
                eval_stack.append(
                    (f"-{operand_str}", LATEX_NEGATION_PRECEDENCE, operand_open)
                )
            else:
                eval_stack.append((f"\\neg {operand_str}", op_prec, operand_open))

        # Handle binary ops
        else:
            right_str, right_prec, right_open = eval_stack.pop()
            left_str, left_prec, left_open = eval_stack.pop()

            # The fraction bar already groups both operands.
            if op == "/":
                eval_stack.append(
                    (f"\\frac{{{left_str}}}{{{right_str}}}", LEAF_PRECEDENCE, False)
                )
                continue

            # (sum_i x_i) * y would otherwise read as sum_i (x_i * y).
            if (
                left_prec < op_prec
                or (left_prec == op_prec and op_assoc == "right")
                or (left_open and op == "*")
            ):
                left_str = f"({left_str})"

            if right_prec < op_prec or (right_prec == op_prec and op_assoc == "left"):
                right_str = f"({right_str})"
                right_open = False

            op = op_map.get(op, op)

            # get around 80000 line length limitation in GAMS
            length = len(left_str) + len(op) + len(right_str)
            if length >= GMS_MAX_LINE_LENGTH - LINE_LENGTH_OFFSET:
                new_str = f"{left_str} {op}\n {right_str}"
            else:
                new_str = f"{left_str} {op} {right_str}"
            eval_stack.append((new_str, op_prec, right_open))

    return eval_stack[0]


def _describe_graph_node(node):
    """
    Return (label, shape, children) for a single node of an expression
    tree. children is a list of (child, edge_label) tuples; leaves
    return an empty list.
    """
    OP_SHAPE = "box"
    LEAF_SHAPE = "ellipse"

    if isinstance(node, Expression):
        children = []
        if node.left is not None:
            children.append((node.left, None))
        if node.right is not None:
            children.append((node.right, None))
        return node.operator, OP_SHAPE, children

    if isinstance(node, operation.Operation):
        children: list[tuple] = [(index, "index") for index in node.op_domain]
        children.append((node.rhs, None))
        return node._op_name, OP_SHAPE, children

    if isinstance(node, condition.Condition):
        children = [(node.conditioning_on, None)]
        if node.condition is not None:
            children.append((node.condition, "cond"))
        return "$", OP_SHAPE, children

    if isinstance(node, operation.Ord):
        return "ord", OP_SHAPE, [(node._symbol, None)]

    if isinstance(node, operation.Card):
        return "card", OP_SHAPE, [(node._symbol, None)]

    if isinstance(node, MathOp):
        return node.op_name, OP_SHAPE, [(el, None) for el in node.elements]

    if isinstance(node, ExtrinsicFunction):
        return node.name, OP_SHAPE, [(arg, None) for arg in node.args]

    if isinstance(node, domain.Domain):
        return "domain", OP_SHAPE, [(s, None) for s in node.sets]

    # Leaf: bare symbol, implicit symbol, Number, or raw scalar.
    return get_operand_gams_repr(node), LEAF_SHAPE, []


def find_symbols(root_node) -> Iterator[str]:
    """
    Yields the names of all symbols that root_node refers to, including the
    sets in their domains, with a stack-based traversal (O(N)).
    """
    seen: set[str] = set()
    stack = [root_node]
    while stack:
        node = stack.pop()

        if isinstance(node, BaseSymbol):
            if node.name in seen:
                continue

            seen.add(node.name)
            yield node.name

            if type(node) is gp_syms.Alias:
                stack.append(node.alias_with)

            stack.extend(
                elem
                for elem in node.domain
                if not isinstance(elem, str) and elem.name != node.name
            )
        elif isinstance(node, ImplicitSymbol):
            stack.append(node.parent)
            stack.extend(node.domain)
        else:
            stack.extend(child for child, _ in _describe_graph_node(node)[2])


def find_symbols_in_conditions(root_node) -> Iterator[str]:
    """
    Yields the names of all symbols that are used in a condition anywhere in
    root_node.
    """
    seen: set[str] = set()
    stack = [root_node]
    while stack:
        node = stack.pop()

        # Symbols are leaves. Their domains are not conditions.
        if isinstance(node, (BaseSymbol, ImplicitSymbol)):
            continue

        if isinstance(node, condition.Condition):
            for name in find_symbols(node.condition):
                if name not in seen:
                    seen.add(name)
                    yield name

        stack.extend(child for child, _ in _describe_graph_node(node)[2])


def create_graph(root_node):
    """
    Build a graphviz.Digraph representing the tree structure of ``root_node``
    (an ``Expression``, ``Operation``, or ``Condition``). Every node in the tree
    becomes a graph node and every parent -> child relationship becomes a
    directed edge. Symbols and numbers are leaves.
    """
    try:
        import graphviz
    except ModuleNotFoundError as e:
        raise ValidationError(
            "graphviz is required for `toGraph()`. Install it with "
            "`pip install gamspy[graph]`. Rendering the graph additionally "
            "requires the Graphviz system binaries (e.g. `apt install graphviz`)."
        ) from e

    graph = graphviz.Digraph()
    counter = 0

    # Each stack item is (node, parent_id, edge_label). A running counter keeps
    # node ids unique so a symbol appearing multiple times gets a distinct node
    # per occurrence rather than merging into one.
    stack: list[tuple] = [(root_node, None, None)]
    while stack:
        node, parent_id, edge_label = stack.pop()

        counter += 1
        node_id = f"n{counter}"

        label, shape, children = _describe_graph_node(node)
        graph.node(node_id, label=label, shape=shape)

        if parent_id is not None:
            graph.edge(parent_id, node_id, label=edge_label)

        # Push children reversed so they are processed (and drawn) left to right.
        for child, child_label in reversed(children):
            stack.append((child, node_id, child_label))

    return graph


class Expression(operable.Operable):
    """
    Represents an expression involving two operands and an operator.

    This class constructs a binary expression tree that can be evaluated or
    translated into GAMS syntax.

    Parameters
    ----------
    left : OperableType | ImplicitEquation | None
        Left operand.
    operator : str
        The operator symbol (e.g., '+', '-', '*', '=', 'eq').
    right : OperableType | str | None
        Right operand.

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
        left: OperableType | ImplicitEquation | str | None,
        operator: str,
        right: OperableType | str | None,
    ):
        self.left = utils._map_special_values(left) if isinstance(left, float) else left
        self.operator = operator
        self.right = (
            utils._map_special_values(right) if isinstance(right, float) else right
        )

        self._representation: str | None = None
        self._left_domain: Sequence = []
        self._right_domain: Sequence = []
        self._create_domain()
        left_control = getattr(left, "controlled_domain", [])
        right_control = getattr(right, "controlled_domain", [])
        self.controlled_domain: list[Set | Alias] = list(
            {*left_control, *right_control}
        )
        self.container: Container | None = None
        if hasattr(left, "container") and left.container is not None:
            self.container = left.container  # ty: ignore[invalid-assignment]
        elif hasattr(right, "container") and right.container is not None:
            self.container = right.container  # ty: ignore[invalid-assignment]

        self.where = condition.Condition(self)

    @property
    def representation(self) -> str:
        if self._representation is None:
            self._representation = create_gams_expression(self)

        return self._representation

    def _create_domain(self):
        left = self.left
        if isinstance(left, condition.Condition):
            left = left.conditioning_on

        if left is None or isinstance(left, (int, float, str)):
            left_domain = []  # left is a scalar
        elif isinstance(left, domain.Domain):
            left_domain = left.sets
        else:
            left_domain = left.domain  # ty: ignore[unresolved-attribute]

        right = self.right
        if isinstance(right, condition.Condition):
            right = right.conditioning_on

        if right is None or isinstance(right, (int, float, str)):
            right_domain = []  # right is a scalar
        elif isinstance(right, domain.Domain):
            right_domain = right.sets
        else:
            right_domain = right.domain  # ty: ignore[unresolved-attribute]

        self._left_domain = left_domain
        self._right_domain = right_domain

        if not left_domain and not right_domain:
            self._shadow_domain = []
            self.domain = []
            self.dimension = 0
            return

        set_to_index: dict = {}
        for domain_char, domain_ptr in (
            ("l", left_domain),
            ("r", right_domain),
        ):
            for i, d in enumerate(domain_ptr):
                if isinstance(d, str) or is_universe(d):
                    continue

                indices = set_to_index.get(d)
                if indices is None:
                    set_to_index[d] = [(domain_char, i)]
                else:
                    indices.append((domain_char, i))

        self._shadow_domain = [
            DomainPlaceHolder(indices=indices) for indices in set_to_index.values()
        ]
        self.domain = list(set_to_index)
        self.dimension = validation.get_dimension(self.domain)

    def __getitem__(self, indices):
        indices = validation.validate_domain(self, indices)
        left_domain = list(self._left_domain)
        right_domain = list(self._right_domain)
        for i, s in enumerate(indices):
            for lr, pos in self._shadow_domain[i].indices:
                if lr == "l":
                    left_domain[pos] = s
                else:
                    right_domain[pos] = s

        left = self.left[left_domain] if left_domain else self.left  # ty: ignore[not-subscriptable, invalid-argument-type]
        right = self.right[right_domain] if right_domain else self.right  # ty: ignore[invalid-argument-type, not-subscriptable]

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
        if self.container is None:
            raise GamspyException(
                "Could not discover the container from the expression. Therefore, cannot call .records on this expression."
            )

        temp_name = "autotemp" + utils._get_unique_name()
        temp_param = gp_syms.Parameter._constructor_bypass(
            self.container, temp_name, self.domain
        )
        temp_param[...] = self
        del self.container._data[temp_name]
        return temp_param.records

    def toValue(self) -> float:
        """
        Convenience method to return expression records as a Python float.
        Only possible if there is a single record as a result of the expression evaluation.

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
        if records is None:  # pragma: no cover
            raise ValidationError(
                "Could not get the value of the expression. Please report to support@gams.com."
            )

        return records["value"][0]

    def toList(self) -> list:
        """
        Convenience method to return the records of the expression as a list.

        Returns
        -------
        list

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

        return []

    def __eq__(self, other):
        return Expression(self, "=e=", other)

    def __ne__(self, other):
        return Expression(self, "ne", other)

    def __hash__(self):
        return id(self)

    def __bool__(self):
        raise ValidationError(
            "An expression cannot be used as a truth value. If you are "
            "trying to generate an expression, use binary operators "
            "instead (e.g. &, |, ^). For more details, see: "
            "https://gamspy.readthedocs.io/en/latest/user/gamspy_for_gams_users.html#logical-operations"
        )

    def __repr__(self) -> str:
        return f"Expression(left={self.left}, data={self.operator}, right={self.right})"

    def latexRepr(self) -> str:
        """
        Returns the LaTeX representation of this Expression.

        Returns
        -------
        str
            The LaTeX string.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = gp.Parameter(m, name="a")
        >>> b = gp.Parameter(m, name="b")
        >>> (a + b).latexRepr()
        'a + b'

        """
        return create_latex_expression(self)

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

    def toGraph(self):
        """
        Return a ``graphviz.Digraph`` of this expression's tree.

        Each operator/operation is drawn as a box and each symbol or number as
        a leaf, so the structure of the expression can be inspected visually or
        rendered to a file. Requires the optional ``graphviz`` dependency
        (``pip install gamspy[graph]``); a :class:`~gamspy.exceptions.ValidationError`
        is raised if it is not installed.

        Returns
        -------
        graphviz.Digraph

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> a = gp.Parameter(m, name="a")
        >>> b = gp.Parameter(m, name="b")
        >>> c = gp.Parameter(m, name="c")
        >>> d = gp.Parameter(m, name="d")
        >>> graph = (a * b + c / d).toGraph()  # doctest: +SKIP
        >>> graph.render("expr", format="svg")  # doctest: +SKIP

        """
        return create_graph(self)

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

    def _validate_definition(
        self, control_stack: list[Set | Alias | ImplicitSet]
    ) -> None:
        if not get_option("VALIDATION") or not get_option("DOMAIN_VALIDATION"):
            return

        # LHS carries the controlled indices only the right-hand side must be checked.
        _validate_controlled(self.right, control_stack)


def _is_acting_like_set(operand) -> bool:
    """True if the operand acts as a set in GAMS set algebra."""
    if isinstance(operand, ImplicitSet):
        return True

    if isinstance(operand, str):
        return operand in ("yes", "no")

    if isinstance(operand, SetExpression):
        return operand._acting_like_set

    if isinstance(operand, condition.Condition):
        return _is_acting_like_set(operand.conditioning_on)

    if isinstance(operand, number.Number):
        return operand._value in ("yes", "no")

    return False


class SetExpression(Expression):
    """
    Represents an expression involving set operations.

    This class handles operations specifically for Sets and Aliases, such as
    union, intersection, difference, and complement.
    """

    def __init__(
        self,
        left: OperableType | None,
        data: Literal["+", "-", "*", "not"],
        right: OperableType,
    ):
        super().__init__(left, data, right)
        self._acting_like_set = True
        self._adjust_left_right()

    def _adjust_left_right(self) -> None:
        if isinstance(self.left, bool):
            self.left = int(self.left)
        if isinstance(self.right, bool):
            self.right = int(self.right)

        if self.operator == "not":
            self._acting_like_set = _is_acting_like_set(self.right)
            return

        left_is_set = _is_acting_like_set(self.left)
        right_is_set = _is_acting_like_set(self.right)

        if left_is_set == right_is_set:
            # Either a pure set operation or a purely numeric expression.
            self._acting_like_set = left_is_set
            return

        # One operand is a set, the other is not. 0 and 1 are interchangeable
        # with no and yes, so the operation stays a pure set operation.
        num_attr = "right" if left_is_set else "left"
        num_operand = getattr(self, num_attr)

        if isinstance(num_operand, (int, float)):
            if num_operand in (0, 1):
                setattr(self, num_attr, "yes" if num_operand else "no")
                return
        elif isinstance(num_operand, condition.Condition) and isinstance(
            num_operand.conditioning_on, number.Number
        ):
            if num_operand.conditioning_on._value in (0, 1):
                num_operand.conditioning_on._value = (
                    "yes" if num_operand.conditioning_on._value else "no"
                )
                return
        elif not isinstance(num_operand, SetExpression):
            # Operands such as parameters, lag/lead operations or expressions
            # over them are left untouched. e.g. s(i) + t.lag(1) or s(i) + yearval[ll]
            return

        # A numeric operand other than 0 and 1 makes GAMS evaluate the whole
        # expression numerically. GAMS does not allow mixing numbers and sets
        # in additions, so a set operand of + or - must be coerced to its 0/1
        # value by multiplying it with 1: e.g. `2 - s(i)` is invalid GAMS while
        # `2 - 1*s(i)` is valid.
        self._acting_like_set = False
        if self.operator != "*":
            set_attr = "left" if left_is_set else "right"
            setattr(self, set_attr, Expression(1, "*", getattr(self, set_attr)))


LAG_LEAD_OPERATORS = ("+", "-", "++", "--")
INVERSE_LAG_LEAD_OPERATORS = {"+": "-", "-": "+", "++": "--", "--": "++"}


def _lag_lead_operator(direction: Literal["+", "-"], type: str) -> str:
    if type == "circular":
        return direction * 2

    if type == "linear":
        return direction

    raise ValueError(
        f"{'Lead' if direction == '+' else 'Lag'} type must be linear or circular"
    )


class ShiftExpression(Expression):
    """
    Represents a lag or lead operation on an ordered set, e.g. ``t - 1``.

    A shift denotes the element that is ``jump`` positions before (lag) or
    after (lead) the current element of ``parent``. It is an index expression
    and not a set reference of its own: wherever a shift is used as an index,
    the shifted set is the one that is put under control.

    Parameters
    ----------
    parent : Set | Alias
        The ordered set that is shifted.
    operator : '+' | '-' | '++' | '--'
        ``-``/``+`` are linear lag/lead, ``--``/``++`` are circular lag/lead.
    jump : OperableType
        Number of positions to shift by. Anything that evaluates to an integer
        in GAMS, e.g. an int, a Parameter or Card.

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> t = gp.Set(m, name="t", records=[f"y{idx}" for idx in range(3)])
    >>> t.lag(1).gamsRepr()
    't - 1'
    >>> t.lead(gp.Card(t), "circular").gamsRepr()
    't ++ (card(t))'

    """

    def __init__(
        self,
        parent: Set | Alias,
        operator: str,
        jump: OperableType,
    ) -> None:
        if operator not in LAG_LEAD_OPERATORS:
            raise ValidationError(
                f"`{operator}` is not a valid lag/lead operator. Valid lag/lead"
                f" operators are {LAG_LEAD_OPERATORS}"
            )

        self.parent = parent
        super().__init__(parent, operator, jump)

    @property
    def jump(self) -> OperableType:
        """Number of positions the parent set is shifted by."""
        return self.right  # ty: ignore[invalid-return-type]

    @property
    def is_circular(self) -> bool:
        """True for circular (``++``, ``--``) shifts."""
        return len(self.operator) == 2

    def _create_domain(self) -> None:
        # A shift occupies a single index position but carries no domain of its own.
        self._left_domain = []
        self._right_domain = []
        self._shadow_domain = []
        self.domain = ["*"]
        self.dimension = 1

    def _jump_repr(self, *, latex: bool = False) -> str:
        if latex:
            representation = get_operand_latex_repr(self.jump)
        else:
            representation = get_operand_gams_repr(self.jump)

        if isinstance(self.jump, (int, float)):
            return representation

        return f"({representation})"

    def __eq__(self, other):
        # A shift is an index and not a value. Identity check instead of building expression.
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)

    def __add__(self, other: OperableType) -> Expression:
        """
        Shifts this lag/lead operation by `n` more positions to the right.

        Parameters
        ----------
        other

        Returns
        -------
        ShiftExpression

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", records=[f"y{idx}" for idx in range(3)])
        >>> (t + 1 + 2).gamsRepr()
        't + 3'

        """
        if isinstance(
            other, (gp_syms.Set, gp_syms.Alias, ImplicitSet, ShiftExpression)
        ):
            return super().__add__(other)

        return self.lead(other)

    def __sub__(self, other: OperableType) -> Expression:
        """
        Shifts this lag/lead operation by `n` more positions to the left.

        Parameters
        ----------
        other

        Returns
        -------
        ShiftExpression

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", records=[f"y{idx}" for idx in range(3)])
        >>> (t - 1 - 2).gamsRepr()
        't - 3'

        """
        if isinstance(
            other, (gp_syms.Set, gp_syms.Alias, ImplicitSet, ShiftExpression)
        ):
            return super().__sub__(other)

        return self.lag(other)

    def __ge__(self, other: OperableType) -> Expression:
        return Expression(self, ">=", other)

    def __le__(self, other: OperableType) -> Expression:
        return Expression(self, "<=", other)

    def __getitem__(self, indices) -> None:
        raise ValidationError(
            f"`{self.gamsRepr()}` is a lag/lead operation and cannot be indexed."
        )

    def __repr__(self) -> str:
        return (
            f"ShiftExpression(parent={self.parent}, operator='{self.operator}',"
            f" jump={self.jump})"
        )

    def lag(
        self, n: OperableType, type: Literal["linear", "circular"] = "linear"
    ) -> ShiftExpression:
        """
        Shifts this lag/lead operation by `n` more positions to the left.

        Parameters
        ----------
        n : OperableType
            The number of positions to shift.
        type : 'linear' or 'circular', optional
            The type of lag to perform.

        Returns
        -------
        ShiftExpression

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", records=[f"y{idx}" for idx in range(3)])
        >>> t.lag(1).lag(2).gamsRepr()
        't - 3'

        """
        return self._combine(_lag_lead_operator("-", type), n)

    def lead(
        self, n: OperableType, type: Literal["linear", "circular"] = "linear"
    ) -> ShiftExpression:
        """
        Shifts this lag/lead operation by `n` more positions to the right.

        Parameters
        ----------
        n : OperableType
            The number of positions to shift.
        type : 'linear' or 'circular', optional
            The type of lead to perform.

        Returns
        -------
        ShiftExpression

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", records=[f"y{idx}" for idx in range(3)])
        >>> t.lag(1).lead(3).gamsRepr()
        't + 2'

        """
        return self._combine(_lag_lead_operator("+", type), n)

    def _combine(self, operator: str, jump: OperableType) -> ShiftExpression:
        """
        Folds another shift into this one.

        GAMS allows one lag/lead operator per index position only (`t - 1 - 1`
        does not compile), therefore the jumps are combined instead of the
        operators being chained.
        """
        if (len(operator) == 2) != self.is_circular:
            raise ValidationError(
                f"`{self.gamsRepr()}` is a"
                f" {'circular' if self.is_circular else 'linear'} lag/lead"
                " operation and cannot be combined with a"
                f" {'linear' if self.is_circular else 'circular'} one. Combine"
                " the jumps and apply a single lag/lead operation instead."
            )

        same_direction = operator[0] == self.operator[0]
        new_operator = self.operator

        if isinstance(self.jump, (int, float)) and isinstance(jump, (int, float)):
            new_jump = self.jump + jump if same_direction else self.jump - jump
            if new_jump < 0:
                new_jump = -new_jump
                new_operator = INVERSE_LAG_LEAD_OPERATORS[new_operator]
        else:
            new_jump = self.jump + jump if same_direction else self.jump - jump

        return ShiftExpression(self.parent, new_operator, new_jump)

    @property
    def records(self) -> pd.DataFrame | None:
        raise ValidationError(".records is not allowed for lag/lead operations.")

    def toDense(self) -> np.ndarray:
        raise ValidationError(".toDense is not allowed for lag/lead operations.")

    def toValue(self) -> float:
        raise ValidationError(".toValue is not allowed for lag/lead operations.")

    def toList(self) -> list:
        raise ValidationError(".toList is not allowed for lag/lead operations.")

    def latexRepr(self) -> str:
        """
        Returns the LaTeX representation of this lag/lead operation.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", records=[f"y{idx}" for idx in range(3)])
        >>> t.lag(1).latexRepr()
        't - 1'

        """
        return (
            f"{self.parent.latexRepr()} {self.operator} {self._jump_repr(latex=True)}"
        )

    def gamsRepr(self) -> str:
        """
        Representation of this lag/lead operation in GAMS language.

        Returns
        -------
        str

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> t = gp.Set(m, name="t", records=[f"y{idx}" for idx in range(3)])
        >>> t.lead(2, "circular").gamsRepr()
        't ++ 2'

        """
        return f"{self.parent.gamsRepr()} {self.operator} {self._jump_repr()}"


def _check_uncontrolled_indices(
    node: ImplicitSymbol, control_stack: list[Set | Alias | ImplicitSet]
) -> None:
    """Raise if any set used as an index of node is not controlled.

    A set is controlled when it is an index of the symbol being defined or
    of an enclosing operation (Sum etc.). Singleton sets are always allowed.
    """
    for elem in node.domain:
        if hasattr(elem, "is_singleton") and elem.is_singleton:
            continue

        if isinstance(elem, BaseSymbol) and not utils.isin(elem, control_stack):
            raise ValidationError(f"Uncontrolled set `{elem}` entered as constant!")

        if isinstance(elem, (ImplicitSymbol, ShiftExpression)) and not utils.isin(
            elem.parent, control_stack
        ):
            raise ValidationError(
                f"Uncontrolled set `{elem.parent}` entered as constant!"
            )


def _validate_controlled(node, control_stack: list[Set | Alias | ImplicitSet]) -> None:
    """
    Walk a right-hand-side value expression and ensure every set used as an index is
    controlled by control_stack.
    """
    stack = [node]
    while stack:
        current = stack.pop()

        if isinstance(current, operation.Operation):
            current._validate_operation(list(control_stack))
        elif isinstance(current, ImplicitSymbol):
            _check_uncontrolled_indices(current, control_stack)
        elif isinstance(current, condition.Condition):
            stack.append(current.conditioning_on)
            stack.append(current.condition)
        elif isinstance(current, (operation.Ord, operation.Card)):
            stack.append(current._symbol)
        elif isinstance(current, MathOp):
            stack.extend(current.elements)
        elif isinstance(current, ExtrinsicFunction):
            stack.extend(current.args)
        elif isinstance(current, Expression):
            stack.append(current.left)
            stack.append(current.right)
        # Other leaves (numbers, literals, bare sets, ...) carry no index to validate.
