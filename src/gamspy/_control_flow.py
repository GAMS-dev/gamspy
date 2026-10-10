from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import gamspy as gp
import gamspy._gdx as gdxio
from gamspy._algebra.condition import Condition
from gamspy._algebra.domain import Domain
from gamspy._algebra.expression import _validate_controlled, get_logical_gams_repr
from gamspy._config import get_option
from gamspy._internals import ATTR_PREFIX, DataSource
from gamspy._symbols.implicits import ImplicitSet
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Container, Parameter
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Card, Operation
    from gamspy._symbols.implicits import ImplicitParameter
    from gamspy._types import ConditionType, OperationIndexType
    from gamspy.math import MathOp

# Dictionary to track the container used in the most recent If/ElseIf block
_last_containers: dict[tuple[int, int], Container] = {}


def _synch_loop_with_gams(container: Container) -> None:
    """Runs the statements of the outermost loop and loads the results back."""
    models = list(container._models_solved_in_loop.values())
    container._models_solved_in_loop.clear()

    gdx_out = container._gdx_out
    container._options._set_extra_options(
        {"gdx": gdx_out, "gdxSymbols": "newOrChangedNoData"}
    )
    container._synch_with_gams()
    container._options._set_extra_options({})
    symbol_names = gdxio._get_symbol_names_from_gdx(container.system_directory, gdx_out)
    container._should_load_from(symbol_names, source=DataSource.GAMS)

    for model in models:
        # A solve under a condition that never held writes no attribute file.
        if os.path.exists(model._attr_gdx_file):
            model._update_model_attributes()


class Loop:
    """
    A context manager to execute a group of statements iteratively for each member of a set or domain.

    The Loop class maps to the GAMS `loop` statement. It is particularly useful for
    cases where parallel assignments are not sufficient, such as iterative calculations,
    nested loops, or modifying models and solving them repeatedly.

    Parameters
    ----------
    indices : Set | Alias | ImplicitSet | Condition | Domain | MathOp | Sequence[Set | Alias | ImplicitSet | Condition | Domain | MathOp]
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

    def __init__(self, indices: OperationIndexType):
        self.indices = indices
        self._loop_number = -1
        self.container = self._find_container()

    def _find_container(self) -> Container:
        if isinstance(
            self.indices,
            (gp.Set, gp.Alias, Condition, Domain, ImplicitSet, gp.math.MathOp),
        ):
            return self.indices.container  # ty: ignore[invalid-return-type]
        elif isinstance(self.indices, Sequence):
            for elem in self.indices:
                container = getattr(elem, "container", None)
                if isinstance(container, gp.Container):
                    return container

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
            if self.container._in_loop == 0:
                self.container._models_solved_in_loop.clear()
            return False

        # Run only in the most outer loop
        if self.container._in_loop == 0:
            self.container._last_control_flow = "loop"
            _synch_loop_with_gams(self.container)


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
            if self.container._in_loop == 0:
                self.container._models_solved_in_loop.clear()
            return False

        if self.container._in_loop == 0:  # Run only in the most outer loop
            self.container._last_control_flow = "for"
            _synch_loop_with_gams(self.container)


class While:
    """
    A context manager to execute a group of statements repeatedly as long as a
    condition evaluates to True.

    The While class maps to the GAMS `while` statement. It is useful for
    processes that must repeat an unknown number of times until a specific
    logical condition is met.

    Parameters
    ----------
    condition : ConditionType
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

    def __init__(self, condition: ConditionType):
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

        representation = get_logical_gams_repr(self.condition)
        self.container._add_statement(f"while({representation},")

        return self

    def __exit__(self, exc_type, exc, tb):
        self.container._in_loop -= 1

        self.container._add_statement(");")

        # An exception occurred inside the with block.
        # Don't do synchronization that may raise another exception.
        if exc_type is not None:
            if self.container._in_loop == 0:
                self.container._models_solved_in_loop.clear()
            return False

        if self.container._in_loop == 0:  # Run only in the most outer loop
            self.container._last_control_flow = "while"
            _synch_loop_with_gams(self.container)


# This flag remembers whether a block of the outermost if chain has already been taken.
_IF_CHAIN_FLAG = ATTR_PREFIX + "if_flag"


def _resolve_if_container(
    condition: ConditionType, container: Container | None, construct: str
) -> Container:
    deduced = getattr(condition, "container", None)

    if container is None:
        if not isinstance(deduced, gp.Container):
            raise ValidationError(
                f"Could not find the container in the given condition `{condition}`. "
                f"Hence, {construct} operation is not possible. Provide it with the "
                "`container` argument."
            )

        container = deduced
    elif not isinstance(container, gp.Container):
        raise TypeError(f"`container` must be a Container but given {type(container)}")
    elif isinstance(deduced, gp.Container) and deduced is not container:
        raise ValidationError(
            f"The container of the given condition `{condition}` is different "
            f"than the `container` argument of {construct}."
        )

    # Track the container for potential succeeding ElseIf/Else statements
    _last_containers[(os.getpid(), threading.get_native_id())] = container

    return container


def _condition_repr(condition: ConditionType, container: Container) -> str:
    # On the outermost level, no index of the condition is controlled.
    if (
        not container._in_loop
        and get_option("VALIDATION")
        and get_option("DOMAIN_VALIDATION")
    ):
        _validate_controlled(condition, [])

    return get_logical_gams_repr(condition)


def _continue_if_chain(container: Container, construct: str) -> bool:
    """
    Reopens the preceding If/ElseIf block. Returns True if the chain is at the
    outermost level, where the preceding block has already been executed.
    """
    if container._last_control_flow not in ("if", "elseif"):
        raise ValidationError(
            f"`{construct}` must follow a `gp.If` or `gp.ElseIf` block."
        )

    if container._in_loop:
        statements = container._unsaved_statements
        is_chained = (
            bool(statements)
            and isinstance(statements[-1], str)
            and statements[-1] == ");"
        )
    else:
        is_chained = container._open_if_chain

    if not is_chained:
        raise ValidationError(
            f"`{construct}` must immediately follow a `gp.If` or `gp.ElseIf` block without any intervening statements."
        )

    if container._in_loop:
        # Remove the closing parenthesis of the previous block to continue the chain
        container._unsaved_statements.pop()
        return False

    container._in_loop += 1
    return True


def _close_if_block(
    container: Container,
    is_outermost: bool,
    kind: Literal["if", "elseif", "else"],
    exc_type: type[BaseException] | None,
) -> None:
    if is_outermost:
        container._in_loop -= 1

    container._add_statement(");")
    if is_outermost and kind == "elseif":
        container._add_statement(");")

    container._last_control_flow = kind

    if not is_outermost:
        return

    # An exception occurred inside the with block.
    # Don't do synchronization that may raise another exception.
    if exc_type is not None:
        container._models_solved_in_loop.clear()
        return

    _synch_loop_with_gams(container)
    container._open_if_chain = kind != "else"


class If:
    """
    A context manager to conditionally execute a group of statements.

    The If class maps to the GAMS `if` statement. It allows you to branch
    conditionally around a group of execution statements. It can be used within
    loops as well as on the outermost level. On the outermost level, the statements
    of the block are executed when the block is closed.

    Parameters
    ----------
    condition : ConditionType
        The logical condition that must be satisfied to execute the nested statements.
    container : Container, optional
        The container to add the statements to. Required only if the container
        cannot be deduced from the condition.

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

    **3. Branching on the outermost level:**


    >>> x = gp.Parameter(m, records=15)
    >>> with gp.If(x > 10):
    ...     x[...] = 10
    >>> with gp.Else():
    ...     x[...] = 0
    >>> x.toValue()
    np.float64(10.0)

    **4. Providing the container explicitly:**


    >>> with gp.If(gp.Number(1) == 1, container=m):
    ...     cnt[...] = 1

    """

    def __init__(self, condition: ConditionType, container: Container | None = None):
        self.condition = condition
        self.container = _resolve_if_container(condition, container, "gp.If")
        self._is_outermost = False

    def __enter__(self) -> None:
        representation = _condition_repr(self.condition, self.container)

        if self.container._in_loop:
            self.container._add_statement(f"if ({representation},")
            return

        self._is_outermost = True
        self.container._in_loop += 1
        self.container._add_statement(f"Scalar {_IF_CHAIN_FLAG};")
        self.container._add_statement(f"{_IF_CHAIN_FLAG} = 0;")
        self.container._add_statement(f"if ({representation}, {_IF_CHAIN_FLAG} = 1;")

    def __exit__(self, exc_type, exc, tb):
        _close_if_block(self.container, self._is_outermost, "if", exc_type)


class ElseIf:
    """
    A context manager to conditionally execute a group of statements if the preceding
    `If` or `ElseIf` condition was False and the current condition is True.

    Parameters
    ----------
    condition : ConditionType
        The logical condition that must be satisfied to execute the nested statements.
    container : Container, optional
        The container to add the statements to. Required only if the container
        cannot be deduced from the condition.
    """

    def __init__(self, condition: ConditionType, container: Container | None = None):
        self.condition = condition
        self.container = _resolve_if_container(condition, container, "gp.ElseIf")
        self._is_outermost = False

    def __enter__(self) -> ElseIf:
        representation = _condition_repr(self.condition, self.container)
        self._is_outermost = _continue_if_chain(self.container, "gp.ElseIf")

        if self._is_outermost:
            # The condition is nested to avoid evaluating it once a preceding block is taken.
            self.container._add_statement(f"if (not {_IF_CHAIN_FLAG},")
            self.container._add_statement(
                f"if ({representation}, {_IF_CHAIN_FLAG} = 1;"
            )
        else:
            self.container._add_statement(f"elseif {representation},")

        return self

    def __exit__(self, exc_type, exc, tb):
        _close_if_block(self.container, self._is_outermost, "elseif", exc_type)


class Else:
    """
    A context manager to execute a group of statements if all preceding `If` and `ElseIf`
    conditions were False.
    """

    def __init__(self) -> None:
        pid = os.getpid()
        tid = threading.get_native_id()
        container = _last_containers.get((pid, tid))

        if not isinstance(container, gp.Container):
            raise ValidationError(
                "Could not find the container. Hence, gp.Else operation is not possible. "
                "Ensure gp.Else follows a gp.If or gp.ElseIf statement."
            )

        self.container = container
        self._is_outermost = False

    def __enter__(self) -> Else:
        self._is_outermost = _continue_if_chain(self.container, "gp.Else")

        if self._is_outermost:
            self.container._add_statement(f"if (not {_IF_CHAIN_FLAG},")
        else:
            self.container._add_statement("else")

        return self

    def __exit__(self, exc_type, exc, tb):
        _close_if_block(self.container, self._is_outermost, "else", exc_type)
