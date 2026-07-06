from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import gamspy as gp
import gamspy._gdx as gdxio
from gamspy._algebra.condition import Condition
from gamspy._algebra.domain import Domain
from gamspy._symbols.implicits import ImplicitSet
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Alias, Container, Parameter, Set
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Card, Operation
    from gamspy._symbols.implicits import ImplicitParameter
    from gamspy.math import MathOp

# Dictionary to track the container used in the most recent If/ElseIf block
_last_containers: dict[tuple[int, int], Container] = {}


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
        indices: Set
        | Alias
        | ImplicitSet
        | Condition
        | Domain
        | Sequence[Set | Alias]
        | MathOp,
    ):
        self.indices = indices
        self._loop_number = -1
        self.container = self._find_container()

    def _find_container(self) -> Container:
        if isinstance(
            self.indices,
            (gp.Set, gp.Alias, Condition, Domain, ImplicitSet, gp.math.MathOp),
        ):
            return self.indices.container  # type: ignore
        elif isinstance(self.indices, Sequence):
            for elem in self.indices:
                if hasattr(elem, "container"):
                    return elem.container

        raise ValidationError(
            f"`{type(self.indices)}` is not an allowed type for a loop index. "
        )

    def _index_repr(self) -> str:
        if isinstance(
            self.indices,
            (gp.Set, gp.Alias, Condition, Domain, ImplicitSet, gp.math.MathOp),
        ):
            return self.indices.gamsRepr()
        elif isinstance(self.indices, Sequence):
            representations = [index.gamsRepr() for index in self.indices]
            return f"({','.join(representations)})"

        raise ValidationError(
            f"`{type(self.indices)}` is not an allowed type for a loop index. "
        )

    @property
    def Break(self) -> None:
        """
        Breaks the execution of the current loop prematurely.

        This property maps to the GAMS `break` statement. Note that you can only
        break out of the innermost loop currently executing. Attempting to break an
        outer loop from within an inner loop will raise a ValidationError.

        Raises
        ------
        ValidationError
            If attempting to break an outer loop without breaking the inner loop first.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, records=["i1", "i2", "i3"])
        >>> j = gp.Set(m, records=["j1", "j2", "j3"])
        >>> cnt = gp.Parameter(m, records=0)
        >>> with gp.Loop(i) as loop:
        ...     with gp.Loop(j) as loop2:
        ...         cnt[...] += 1
        ...         loop2.Break  # Successfully breaks the inner loop
        ...     loop.Break       # Successfully breaks the outer loop

        """
        if self._loop_number < self.container._in_loop:
            raise ValidationError(
                "You cannot break this loop. You should break the inner loop first."
            )

        self.container._add_statement("break;")

    @property
    def Continue(self) -> None:
        """
        Skips the remaining statements in the current iteration and proceeds to the next one.

        This property maps to the GAMS `continue` statement. It gives additional
        control over the execution of loop structures by allowing you to bypass
        the rest of the loop block for the current domain element.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Set(m, records=["i1", "i2", "i3", "i4"])
        >>> cnt = gp.Parameter(m, records=0)
        >>> with gp.Loop(i) as loop:
        ...     with gp.If(gp.Ord(i) == 2):
        ...         loop.Continue  # Skips incrementing for "i2"
        ...     cnt[...] += 1

        """
        self.container._add_statement("continue;")

    def __enter__(self) -> Loop:
        self.container._in_loop += 1
        self._loop_number = self.container._in_loop

        self.container._add_statement(f"loop({self._index_repr()},")

        return self

    def __exit__(self, exc_type, exc, tb):
        self.container._in_loop -= 1

        self.container._add_statement(");")

        # An exception occurred inside the with block.
        # Don't do synchronization that may raise another exception.
        if exc_type is not None:
            return False

        # Run only in the most outer loop
        if self.container._in_loop == 0:
            self.container._add_statement(
                f"execute_unload '{self.container._gdx_out}';"
            )
            self.container._last_control_flow = "loop"
            self.container._synch_with_gams()
            symbol_names = gdxio._get_symbol_names_from_gdx(
                self.container.system_directory, self.container._gdx_out
            )
            for name in symbol_names:
                self.container._data[name]._should_load_from_gams = True


class For:
    """
    A context manager to execute a group of statements iteratively over a numerical range.

    The For class maps to the GAMS `for` statement. It allows you to iterate over a
    range of numerical values, incrementing or decrementing a scalar parameter at each step.
    It is useful for iterative algorithmic calculations that require a numerical counter,
    rather than iterating over elements of a set.

    Parameters
    ----------
    index : Parameter
        A scalar Parameter used as the numerical loop counter.
    start : int | float | Parameter | Expression | Card | Operation | MathOp
        The starting value of the loop counter.
    end : int | float | Parameter | Expression | Card | Operation | MathOp
        The final value of the loop counter.
    step : int | float | Parameter | Expression | Card | Operation | MathOp, optional
        The increment or decrement step size. Defaults to 1.
    direction : Litera['to', 'downto']
        The direction of the step. 'to' steps upwards, 'downto' steps downwards. Defaults to 'to'.

    Examples
    --------
    **1. Simple iteration over a numerical range:**


    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Parameter(m)
    >>> cnt = gp.Parameter(m, records=0)
    >>> with gp.For(i, 1, 10):
    ...     cnt[...] += i

    **2. Iterating backwards**
    When a negative step is provided, the loop iterate downwards.


    >>> x = gp.Parameter(m, records=10)
    >>> with gp.For(i, 10, 1, 2, direction="downto"):
    ...     x[...] = x[...] - 2

    **3. Using Parameters as loop bounds:**
    You can use other parameters or expressions to define the boundaries of the loop.


    >>> start_val = gp.Parameter(m, records=5)
    >>> end_val = gp.Parameter(m, records=15)
    >>> with gp.For(i, start_val, end_val):
    ...     cnt[...] += 1

    """

    def __init__(
        self,
        index: Parameter,
        start: int
        | float
        | Parameter
        | ImplicitParameter
        | Expression
        | Card
        | Operation
        | MathOp,
        end: int
        | float
        | Parameter
        | ImplicitParameter
        | Expression
        | Card
        | Operation
        | MathOp,
        step: int
        | float
        | Parameter
        | ImplicitParameter
        | Expression
        | Card
        | Operation
        | MathOp = 1,
        direction: Literal["to", "downto"] = "to",
    ):
        if not isinstance(index, gp.Parameter):
            raise TypeError(
                f"`index` must be a scalar Parameter but given {type(index)}"
            )

        if index.dimension != 0:
            raise ValidationError(
                f"`index` parameter must be a scalar but given index dimension is {index.dimension}"
            )

        self.index = index
        self.start = start
        self.end = end
        self.step = step
        self.direction = direction
        self._loop_number = -1
        self.container = index.container

    @property
    def Break(self) -> None:
        """
        Breaks the execution of the current loop prematurely.

        This property maps to the GAMS `break` statement. Note that you can only
        break out of the innermost loop currently executing. Attempting to break an
        outer loop from within an inner loop will raise a ValidationError.

        Raises
        ------
        ValidationError
            If attempting to break an outer loop without breaking the inner loop first.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Parameter(m)
        >>> cnt = gp.Parameter(m, records=0)
        >>> with gp.For(i, 1, 10) as my_for:
        ...     cnt[...] += 1
        ...     with gp.If(i == 5):
        ...         my_for.Break  # Exits the loop when `i` reaches 5

        """
        if self._loop_number < self.container._in_loop:
            raise ValidationError(
                "You cannot break this for loop. You should break the inner loop first."
            )

        self.container._add_statement("break;")

    @property
    def Continue(self) -> None:
        """
        Skips the remaining statements in the current iteration and proceeds to the next one.

        This property maps to the GAMS `continue` statement. It gives additional
        control over the execution of loop structures by allowing you to bypass
        the rest of the loop block for the current counter value.

        Examples
        --------
        >>> import gamspy as gp
        >>> m = gp.Container()
        >>> i = gp.Parameter(m)
        >>> cnt = gp.Parameter(m, records=0)
        >>> with gp.For(i, 1, 10) as my_for:
        ...     with gp.If(i == 5):
        ...         my_for.Continue  # Skips incrementing `cnt` when `i` is 5
        ...     cnt[...] += 1

        """
        self.container._add_statement("continue;")

    def __enter__(self) -> For:
        self.container._in_loop += 1
        self._loop_number = self.container._in_loop

        index_str = self.index.gamsRepr()
        start_str = (
            str(self.start)
            if isinstance(self.start, (int, float))
            else self.start.gamsRepr()
        )
        end_str = (
            str(self.end) if isinstance(self.end, (int, float)) else self.end.gamsRepr()
        )
        step_str = (
            str(self.step)
            if isinstance(self.step, (int, float))
            else self.step.gamsRepr()
        )
        self.container._add_statement(
            f"for({index_str} = {start_str} {self.direction} {end_str} by {step_str}, "
        )

        return self

    def __exit__(self, exc_type, exc, tb):
        self.container._in_loop -= 1

        self.container._add_statement(");")

        # An exception occurred inside the with block.
        # Don't do synchronization that may raise another exception.
        if exc_type is not None:
            return False

        if self.container._in_loop == 0:  # Run only in the most outer loop
            self.container._add_statement(
                f"execute_unload '{self.container._gdx_out}';"
            )
            self.container._last_control_flow = "for"
            self.container._synch_with_gams()
            symbol_names = gdxio._get_symbol_names_from_gdx(
                self.container.system_directory, self.container._gdx_out
            )
            for name in symbol_names:
                self.container._data[name]._should_load_from_gams = True


class While:
    """
    A context manager to execute a group of statements repeatedly as long as a
    condition evaluates to True.

    The While class maps to the GAMS `while` statement. It is useful for
    processes that must repeat an unknown number of times until a specific
    logical condition is met.

    Parameters
    ----------
    condition : Expression | Condition | Operation | MathOp | Parameter
        The logical condition that must remain true to continue executing the nested statements.

    Examples
    --------
    **1. Iteratively dividing a number:**

    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> x = gp.Parameter(m, records=100)
    >>> cnt = gp.Parameter(m, records=0)
    >>> with gp.While(x > 1):
    ...     x[...] = x / 2
    ...     cnt[...] += 1

    """

    def __init__(
        self, condition: Expression | Condition | Operation | MathOp | Parameter
    ):
        self.condition = condition

        if not isinstance(condition.container, gp.Container):
            raise ValidationError(
                f"Could not find the container in the given condition `{condition}`. Hence, gp.While operation is not possible."
            )

        self.container = condition.container
        self._loop_number = -1

    @property
    def Break(self) -> None:
        """
        Breaks the execution of the current while loop prematurely.

        This property maps to the GAMS `break` statement. Note that you can only
        break out of the innermost loop currently executing.

        Raises
        ------
        ValidationError
            If attempting to break an outer loop without breaking the inner loop first.
        """
        if self._loop_number < self.container._in_loop:
            raise ValidationError(
                "You cannot break this while loop. You should break the inner loop first."
            )

        self.container._add_statement("break;")

    @property
    def Continue(self) -> None:
        """
        Skips the remaining statements in the current iteration and proceeds to the next one.
        """
        self.container._add_statement("continue;")

    def __enter__(self) -> While:
        self.container._in_loop += 1
        self._loop_number = self.container._in_loop

        representation = self.condition.gamsRepr()
        representation = gp.utils._replace_equality_signs(representation)
        self.container._add_statement(f"while({representation},")

        return self

    def __exit__(self, exc_type, exc, tb):
        self.container._in_loop -= 1

        self.container._add_statement(");")

        # An exception occurred inside the with block.
        # Don't do synchronization that may raise another exception.
        if exc_type is not None:
            return False

        if self.container._in_loop == 0:  # Run only in the most outer loop
            self.container._add_statement(
                f"execute_unload '{self.container._gdx_out}';"
            )
            self.container._last_control_flow = "while"
            self.container._synch_with_gams()
            symbol_names = gdxio._get_symbol_names_from_gdx(
                self.container.system_directory, self.container._gdx_out
            )
            for name in symbol_names:
                self.container._data[name]._should_load_from_gams = True


class If:
    """
    A context manager to conditionally execute a group of statements.

    The If class maps to the GAMS `if` statement. It allows you to branch
    conditionally around a group of execution statements within a loop.

    Parameters
    ----------
    condition : Expression
        The logical condition that must be satisfied to execute the nested statements.

    Examples
    --------
    **1. Skipping iterations conditionally:**


    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, records=[f"i{idx}" for idx in range(1, 11)])
    >>> cnt = gp.Parameter(m, records=0)
    >>> with gp.Loop(i) as loop:
    ...     with gp.If(gp.math.mod(gp.Ord(i), 2) == 0):
    ...         loop.Continue
    ...     cnt[...] += 1

    **2. Breaking a loop based on a condition:**


    >>> with gp.Loop(i) as loop:
    ...     with gp.If(i.sameAs("i6")):
    ...         loop.Break
    ...     cnt[...] += 1

    """

    def __init__(
        self, condition: Expression | Condition | Operation | MathOp | Parameter
    ):
        self.condition = condition

        if not isinstance(condition.container, gp.Container):
            raise ValidationError(
                f"Could not find the container in the given condition `{condition}`. Hence, gp.If operation is not possible."
            )

        self.container = condition.container

        # Track the container for potential succeeding ElseIf/Else statements
        pid = os.getpid()
        tid = threading.get_native_id()
        _last_containers[(pid, tid)] = self.container

    def __enter__(self):
        if not self.container._in_loop:
            raise ValidationError(
                "`gp.If` context manager can only be used in `gp.Loop` context managers. Use regular Python if statements instead."
            )

        representation = self.condition.gamsRepr()
        representation = gp.utils._replace_equality_signs(representation)
        self.container._add_statement(f"if ({representation},")

    def __exit__(self, exc_type, exc, tb):
        self.container._add_statement(");")
        self.container._last_control_flow = "if"


class ElseIf:
    """
    A context manager to conditionally execute a group of statements if the preceding
    `If` or `ElseIf` condition was False and the current condition is True.

    Parameters
    ----------
    condition : Expression
        The logical condition that must be satisfied to execute the nested statements.
    """

    def __init__(
        self, condition: Expression | Condition | Operation | MathOp | Parameter
    ):
        self.condition = condition

        if not isinstance(condition.container, gp.Container):
            raise ValidationError(
                f"Could not find the container in the given condition `{condition}`. Hence, gp.ElseIf operation is not possible."
            )

        self.container = condition.container

        # Track the container for potential succeeding ElseIf/Else statements
        pid = os.getpid()
        tid = threading.get_native_id()
        _last_containers[(pid, tid)] = self.container

    def __enter__(self):
        if not self.container._in_loop:
            raise ValidationError(
                "`gp.ElseIf` context manager can only be used in `gp.Loop` context managers."
            )

        if getattr(self.container, "_last_control_flow", None) not in ("if", "elseif"):
            raise ValidationError(
                "`gp.ElseIf` must follow a `gp.If` or `gp.ElseIf` block."
            )

        last_statement = self.container._unsaved_statements[-1]
        if (
            not self.container._unsaved_statements
            or not isinstance(last_statement, str)
            or self.container._unsaved_statements[-1] != ");"
        ):
            raise ValidationError(
                "`gp.ElseIf` must immediately follow a `gp.If` or `gp.ElseIf` block without any intervening statements."
            )

        # Remove the closing parenthesis of the previous block to continue the chain
        self.container._unsaved_statements.pop()

        representation = self.condition.gamsRepr()
        representation = gp.utils._replace_equality_signs(representation)
        self.container._add_statement(f"elseif {representation},")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.container._add_statement(");")
        self.container._last_control_flow = "elseif"


class Else:
    """
    A context manager to execute a group of statements if all preceding `If` and `ElseIf`
    conditions were False.
    """

    def __init__(self):
        pid = os.getpid()
        tid = threading.get_native_id()
        container = _last_containers.get((pid, tid))

        if not isinstance(container, gp.Container):
            raise ValidationError(
                "Could not find the container. Hence, gp.Else operation is not possible. "
                "Ensure gp.Else follows a gp.If or gp.ElseIf statement."
            )

        self.container = container

    def __enter__(self):
        if not getattr(self.container, "_in_loop", 0):
            raise ValidationError(
                "`gp.Else` context manager can only be used in `gp.Loop` context managers."
            )

        if getattr(self.container, "_last_control_flow", None) not in ("if", "elseif"):
            raise ValidationError(
                "`gp.Else` must follow a `gp.If` or `gp.ElseIf` block."
            )

        last_statement = self.container._unsaved_statements[-1]
        if (
            not self.container._unsaved_statements
            or not isinstance(last_statement, str)
            or self.container._unsaved_statements[-1] != ");"
        ):
            raise ValidationError(
                "`gp.Else` must immediately follow a `gp.If` or `gp.ElseIf` block without any intervening statements."
            )

        # Remove the closing parenthesis of the previous block to continue the chain
        self.container._unsaved_statements.pop()

        self.container._add_statement("else")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.container._add_statement(");")
        self.container._last_control_flow = "else"
