from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import gamspy as gp
from gamspy._algebra.condition import Condition
from gamspy._algebra.domain import Domain
from gamspy._symbols.implicits import ImplicitSet
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Alias, Container, Set


class Loop:
    """
    A context manager to execute a group of statements iteratively for each member of a set or domain.

    The Loop class maps to the GAMS `loop` statement. It is particularly useful for
    cases where parallel assignments are not sufficient, such as iterative calculations,
    nested loops, or modifying models and solving them repeatedly.

    Parameters
    ----------
    indices : Set | Alias | ImplicitSet | Condition | Domain | Sequence[Set | Alias]
        The controlling domain of the loop. This can be a single Set, a sequence of Sets,
        or a domain restricted by a logical condition (using `.where`).

    Examples
    --------
    **1. Simple iteration over a single Set:**


    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> t = gp.Set(m, records=["1985", "1986", "1987"])
    >>> pop = gp.Parameter(m, domain=t, records=[("1985", 3456)])
    >>> growth = gp.Parameter(m, domain=t, records=[("1985", 25.3), ("1986", 27.3)])
    >>> with gp.Loop(t):
    ...     pop[t + 1] = pop[t] + growth[t]

    **2. Iteration with a logical condition (dollar condition):**
    You can restrict the loop domain using the `.where` attribute on Sets or Domains.


    >>> i = gp.Set(m, records=["i1", "i2", "i3"])
    >>> j = gp.Set(m, records=["j1", "j2", "j3"])
    >>> q = gp.Parameter(m, domain=[i, j], records=[("i1", "j1", 1), ("i1", "j2", 3)])
    >>> x = gp.Parameter(m, records=1)
    >>> with gp.Loop(gp.Domain(i, j).where[q[i, j] > 0]):
    ...     x[...] = x[...] + q[i, j]

    **3. Nested Loops:**
    Loops can be nested using standard Python indentation.


    >>> a = gp.Parameter(m, domain=[i, j])
    >>> b = gp.Parameter(m)
    >>> a.generateRecords()
    >>> with gp.Loop(i):
    ...     with gp.Loop(j):
    ...         b[...] = a[i, j]

    """

    def __init__(
        self,
        indices: Set | Alias | ImplicitSet | Condition | Domain | Sequence[Set | Alias],
    ):
        self.indices = indices
        self.container = self._find_container()

    def _find_container(self) -> Container:
        if isinstance(self.indices, (gp.Set, gp.Alias, Condition, Domain, ImplicitSet)):
            return self.indices.container
        elif isinstance(self.indices, Sequence):
            for elem in self.indices:
                if hasattr(elem, "container"):
                    return elem.container

        raise ValidationError(
            f"`{type(self.indices)}` is not an allowed type for a loop index. "
        )

    def _index_repr(self) -> str:
        if isinstance(self.indices, (gp.Set, gp.Alias, Condition, Domain, ImplicitSet)):
            return self.indices.gamsRepr()
        elif isinstance(self.indices, Sequence):
            representations = [index.gamsRepr() for index in self.indices]
            return f"({','.join(representations)})"

        raise ValidationError(
            f"`{type(self.indices)}` is not an allowed type for a loop index. "
        )

    def __enter__(self):
        self.container._in_loop += 1

        self.container._add_statement(f"loop({self._index_repr()},")

    def __exit__(self, exc_type, exc, tb):
        self.container._in_loop -= 1

        self.container._add_statement(");")
        if self.container._in_loop == 0:  # Run only in the most outer loop
            self.container._synch_with_gams(gams_to_gamspy=True)
