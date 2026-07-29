from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gamspy._algebra.condition import Condition
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.number import Number
    from gamspy._algebra.operation import Card, Operation, Ord
    from gamspy._symbols import Parameter
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitSet
    from gamspy.math import MathOp

# GAMS operator that performs a sparse assignment.
SPARSE = "$="


class SparseAssignment:
    """
    Wrapper that marks the right-hand side of an assignment as sparse.

    Instances are created by :meth:`gamspy.sparse <gamspy.sparse>` and are only
    meaningful as the right-hand side of an assignment.
    """

    __slots__ = ("rhs",)

    def __init__(self, rhs: Any) -> None:
        self.rhs = rhs

    def __repr__(self) -> str:
        return f"SparseAssignment(rhs={self.rhs})"


def sparse(
    rhs: Expression
    | Operation
    | Condition
    | MathOp
    | Parameter
    | ImplicitParameter
    | ImplicitSet
    | Card
    | Ord
    | Number
    | int
    | float
    | bool
    | str,
) -> SparseAssignment:
    """
    Marks the right-hand side of an assignment as a sparse assignment.

    A sparse assignment (``$=`` in GAMS) assigns to a tuple of the left-hand
    side only if the right-hand side evaluates to a non-zero value. Tuples for
    which the right-hand side is zero are left untouched. Therefore,
    ``p[i] = sparse(rhs)`` is equivalent to ``p[i].where[rhs] = rhs`` but the
    right-hand side is evaluated only once. This can be significantly faster if
    the right-hand side is expensive to evaluate, e.g. a
    :meth:`Sum <gamspy.Sum>`.

    Note that the right-hand side is always evaluated, so unlike a condition on
    the left-hand side, a sparse assignment does not protect against undefined
    arithmetic such as a division by zero.

    Parameters
    ----------
    rhs : Expression | Operation | Condition | MathOp | Parameter | ImplicitParameter | ImplicitSet | Card | Ord | Number | int | float | bool | str
        Right-hand side of the assignment.

    Returns
    -------
    SparseAssignment

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, "i", records=["i1", "i2", "i3"])
    >>> d1 = gp.Parameter(m, "d1", domain=i, records=[("i1", 1), ("i2", 1), ("i3", 1)])
    >>> d2 = gp.Parameter(m, "d2", domain=i, records=[("i2", -2)])
    >>> d3 = gp.Parameter(m, "d3", domain=i)
    >>> d3[i] = d1[i]
    >>> d3[i] = gp.sparse(d2[i])
    >>> d3.records.values.tolist()
    [['i1', 1.0], ['i2', -2.0], ['i3', 1.0]]
    >>> d3.getAssignment()
    'd3(i) $= d2(i);'

    """
    return SparseAssignment(rhs)


def _unwrap(rhs: Any, operator: str = "=") -> tuple[Any, str]:
    """
    Unwraps a sparse assignment right-hand side.

    Returns the right-hand side and the assignment operator to be used, which is
    ``$=`` if the right-hand side was marked with ``sparse`` and ``operator``
    otherwise.
    """
    if isinstance(rhs, SparseAssignment):
        return rhs.rhs, SPARSE

    return rhs, operator
