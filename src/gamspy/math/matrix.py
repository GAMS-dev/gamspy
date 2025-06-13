from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gamspy._symbols.implicits as implicits
import gamspy.math
import gamspy.utils as utils
from gamspy._symbols.parameter import Parameter
from gamspy._symbols.set import Set
from gamspy._symbols.variable import Variable
from gamspy.exceptions import GamspyException, ValidationError

if TYPE_CHECKING:
    from gamspy import Container
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation
    from gamspy._symbols.alias import Alias
    from gamspy.math.misc import MathOp


def vector_norm(
    x: (
        Parameter
        | Variable
        | implicits.ImplicitParameter
        | implicits.ImplicitVariable
        | Expression
        | Operation
    ),
    ord: float | int = 2,
    dim: list[int] | list[Set | Alias] | None = None,
) -> Operation | Expression | MathOp:
    """
    Returns the vector norm of the provided vector x. If ord is not an even integer, absolute value is used which
    requires DNLP.

    Parameters
    ----------
    x : Parameter | Variable | implicits.ImplicitParameter | implicits.ImplicitVariable | Expression | Operation
    ord: int | float
    dim: list[int] | list[Set | Alias], optional

    Returns
    -------
    Expression | Operation | MathOp

    Examples
    --------
    >>> import gamspy as gp
    >>> import math
    >>> m = gp.Container()
    >>> i = gp.Set(m, name="i", records=["i1", "i2"])
    >>> vec = gp.Parameter(m, "vec", domain=[i], records=[("i1", 3), ("i2", 4)])
    >>> vlen = gp.Parameter(m, "vlen", domain=[])
    >>> vlen[...] = gp.math.vector_norm(vec)
    >>> math.isclose(vlen.toValue(), 5, rel_tol=1e-4)
    True

    """
    import gamspy._algebra.operation as operation
    from gamspy._symbols.alias import Alias

    if isinstance(ord, float):
        if ord.is_integer():
            ord = int(ord)
        elif ord in (float("inf"), float("-inf")):
            raise ValidationError("Infinity norms are not supported")

    if ord == 0:
        raise ValidationError("0 norm is not supported")

    even = isinstance(ord, int) and ord % 2 == 0
    domain = x.domain

    if len(domain) == 0:
        raise ValidationError("Provided argument is a scalar")

    sum_domain = domain
    if dim is not None:
        if not isinstance(dim, Iterable):
            raise ValidationError("dim must be an Iterable")

        if len(dim) == 0:
            raise ValidationError("If dim is provided, it must contain items")

        if isinstance(dim[0], int):
            for item in dim:
                if not isinstance(item, int):
                    raise ValidationError(
                        "If dim is provided, either all items must be integers"
                        " or all items must be Sets"
                    )

            sum_domain = []
            for d in dim:
                sum_domain.append(domain[d])
        else:
            for item in dim:
                if not isinstance(item, (Set, Alias)):
                    raise ValidationError(
                        "If dim is provided, either all items must be integers"
                        " or all items must be Sets"
                    )

            sum_domain = dim

    if ord == 2:
        return gamspy.math.sqrt(
            operation.Sum(sum_domain, gamspy.math.sqr(x[domain])),
            safe_cancel=True,
        )
    elif even:
        return gamspy.math.rpower(
            operation.Sum(sum_domain, x[domain] ** ord), (1 / ord)
        )
    elif ord == 1:
        return operation.Sum(sum_domain, gamspy.math.abs(x[domain]))

    return gamspy.math.rpower(
        operation.Sum(sum_domain, gamspy.math.abs(x[domain]) ** ord),
        (1 / ord),
    )


def next_alias(symbol: Alias | Set) -> Alias:
    """Provided the set or alias, it returns the next alias.
    If it is not found, it creates the alias. This function is
    mainly for matrix multiplication conflict resolution but
    it might be helpful in the cases where you need to generate
    many aliases from a set.

    Parameters
    ----------
    symbol : Set | Alias

    Returns
    -------
    Alias

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, name="i", records=["i1", "i2", "i3"])
    >>> j = gp.math.next_alias(i)
    >>> j.name
    'AliasOfi_2'
    >>> k = gp.math._generate_dims(m, [10])[0]
    >>> k.name
    'DenseDim10_1'
    >>> k2 = gp.math.next_alias(k)
    >>> k2.name
    'DenseDim10_2'

    """

    current = symbol
    if symbol.name.startswith("DenseDim") or symbol.name.startswith("AliasOf"):
        prefix, num = symbol.name.split("_")
    else:
        prefix, num = f"AliasOf{symbol.name}", 1

    num = int(num) + 1
    expected_name = f"{prefix}_{num}"
    find_x = symbol.container.data.get(expected_name, None)
    if find_x is None:
        find_x = symbol.container.addAlias(expected_name, alias_with=current)

    return find_x


@dataclass
class Dim:
    dims: list[int]


def dim(dims: list[int] | tuple[int, ...]) -> Dim:
    """Returns an array where each element
    corresponds to a set where the dimension of the
    set is equal to the element in dims. If same dimension
    size used, then next free alias will be returned.
    Symbols are generated once the Dim object is passed to
    a constructor that supports it.

    Parameters
    ----------
    dims: list[int] | tuple[int, ...]

    Returns
    -------
    Dim

    Examples
    --------
    >>> import gamspy as gp
    >>> import math
    >>> m = gp.Container()
    >>> a = gp.math.dim([10, 20]) # nothing generated yet
    >>> a
    Dim(dims=[10, 20])
    >>> par = gp.Parameter(m, name="par", domain=a) # now two sets are generated
    >>> par.domain
    [Set(name='DenseDim10_1', domain=['*']), Set(name='DenseDim20_1', domain=['*'])]
    >>> par2 = gp.Parameter(m, name="par2", domain=a) # same 2 sets are used
    >>> par2.domain
    [Set(name='DenseDim10_1', domain=['*']), Set(name='DenseDim20_1', domain=['*'])]

    """
    for x in dims:
        if not isinstance(x, int):
            raise ValidationError("Dimensions must be integers")

    return Dim(dims=dims)  # type: ignore


def _generate_dims(
    m: Container, dims: list[int] | tuple[int, ...]
) -> list[Alias | Set]:
    sets_so_far = []
    for x in dims:
        expected_name = f"DenseDim{x}_1"
        find_x = m.data.get(expected_name, None)
        if find_x is None:
            find_x = m.addSet(name=expected_name, records=range(x))

        while find_x in sets_so_far:
            find_x = next_alias(find_x)

        sets_so_far.append(find_x)

    return sets_so_far


def trace(
    x: (
        Parameter
        | implicits.ImplicitParameter
        | Variable
        | implicits.ImplicitVariable
    ),
    axis1: int = 0,
    axis2: int = 1,
) -> Operation:
    """Returns trace of the given input x.
    By default trace of zeroth and first axis used. `axis1` and `axis2` parameters
    control on which axes to get trace. Domains at the axis1 and axis2 must be same
    or aliases.


    Parameters
    ----------
    x: (
        Parameter
        | implicits.ImplicitParameter
        | Variable
        | implicits.ImplicitVariable
    )
    axis1=0
    axis2=1

    Returns
    -------
    Operation

    Examples
    --------
    >>> import gamspy as gp
    >>> import numpy as np
    >>> m = gp.Container()
    >>> identity = np.eye(3, 3)
    >>> mat = gp.Parameter(m, name="mat", domain=gp.math.dim([3, 3]), records=identity, uels_on_axes=True)
    >>> sc = gp.Parameter(m, name="sc", domain=[])
    >>> sc[...] = gp.math.trace(mat)
    >>> int(sc.toDense())
    3

    """
    import gamspy._algebra.operation as operation

    if len(x.domain) < 2:
        raise ValidationError("Trace requires at least 2 dimensions")

    if not utils.setBaseEqual(x.domain[axis1], x.domain[axis2]):
        raise ValidationError("Matrix dimensions are not equal")

    domain = [i for i in x.domain]
    domain[axis1] = domain[axis2]

    return operation.Sum(domain[axis2], x[domain])


def permute(
    x: (
        Parameter
        | implicits.ImplicitParameter
        | Variable
        | implicits.ImplicitVariable
    ),
    dims: list[int],
) -> implicits.ImplicitVariable | implicits.ImplicitParameter:
    """Permutes the dimensions provided input `x` using `dim`.
    Similar to PyTorch permute.

    Parameters
    ----------
    x: (
        Parameter
        | implicits.ImplicitParameter
        | Variable
        | implicits.ImplicitVariable
    )
    dims: list[int]

    Returns
    -------
    implicits.ImplicitVariable | implicits.ImplicitParameter

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> i = gp.Set(m, name="i")
    >>> j = gp.Set(m, name="j")
    >>> k = gp.Set(m, name="k")
    >>> p = gp.Parameter(m, name="p", domain=[i, j, k])
    >>> p2 = gp.math.permute(p, [2, 0, 1])
    >>> p2.domain
    [Set(name='k', domain=['*']), Set(name='i', domain=['*']), Set(name='j', domain=['*'])]

    """
    # TODO Accept permuting expressions!
    # Might be needed in some context
    for i in dims:
        if not isinstance(i, int):
            raise ValidationError("Permute dimensions must be integers")

    dims_len = len(dims)
    if min(dims) != 0 or max(dims) != dims_len - 1:
        raise ValidationError(
            "Permute requires the order of indices from 0 to n-1"
        )

    if len(set(dims)) != dims_len:
        raise ValidationError("Permute dimensions must be unique")

    permuted_domain = utils._permute_domain(x.domain, dims)
    if isinstance(x, Parameter):
        return implicits.ImplicitParameter(
            x,
            name=x.name,
            records=x.records,
            domain=permuted_domain,
            permutation=dims,
        )
    elif isinstance(x, implicits.ImplicitParameter):
        if x.permutation is not None:
            dims = utils._permute_domain(x.permutation, dims)

        return implicits.ImplicitParameter(
            x.parent,
            name=x.name,
            records=x._records,
            domain=permuted_domain,
            permutation=dims,
            scalar_domains=x._scalar_domains,
        )
    elif isinstance(x, Variable):
        return implicits.ImplicitVariable(
            x, name=x.name, domain=permuted_domain, permutation=dims
        )
    elif isinstance(x, implicits.ImplicitVariable):
        if x.permutation is not None:
            dims = utils._permute_domain(x.permutation, dims)

        return implicits.ImplicitVariable(
            x.parent,
            name=x.name,
            domain=permuted_domain,
            permutation=dims,
            scalar_domains=x._scalar_domains,
        )

    raise GamspyException(f"permute not implemented for {type(x)}")


def _validate_matrix_mult_dims(left, right):
    """Validates the dimensions for the matrix multiplication"""
    left_len = len(left.domain)
    right_len = len(right.domain)

    dim_no_match_err = "Matrix multiplication dimensions do not match"

    if left_len == 0:
        raise ValidationError(
            "Matrix multiplication requires at least 1 domain, left side"
            " is a scalar"
        )

    if right_len == 0:
        raise ValidationError(
            "Matrix multiplication requires at least 1 domain, right side"
            " is a scalar"
        )

    lr = (left_len, right_len)

    left_controlled = getattr(left, "controlled_domain", [])
    right_controlled = getattr(right, "controlled_domain", [])
    controlled_domain = [*left_controlled, *right_controlled]

    if lr == (1, 1):
        # Dot product
        if not utils.setBaseEqual(left.domain[0], right.domain[0]):
            raise ValidationError("Dot product requires same domain")

        sum_domain = left.domain[0]
        while sum_domain in controlled_domain:
            sum_domain = next_alias(sum_domain)

        return [sum_domain], [sum_domain], sum_domain
    elif lr == (2, 2):
        # Matrix multiplication
        if not utils.setBaseEqual(left.domain[1], right.domain[0]):
            raise ValidationError(dim_no_match_err)

        left_domain = left.domain[0]
        right_domain = right.domain[1]
        if left_domain == right_domain:
            left_domain = next_alias(left_domain)

        sum_domain = left.domain[1]
        while (
            sum_domain in (left_domain, right_domain)
            or sum_domain in controlled_domain
        ):
            sum_domain = next_alias(sum_domain)

        return (
            [left_domain, sum_domain],
            [sum_domain, right_domain],
            sum_domain,
        )
    elif lr == (1, 2):
        # Vector matrix, vector 1-prepended
        if not utils.setBaseEqual(left.domain[0], right.domain[0]):
            raise ValidationError(dim_no_match_err)

        sum_domain = left.domain[0]
        right_domain = right.domain[1]

        while right_domain in controlled_domain:
            right_domain = next_alias(right_domain)

        while sum_domain == right_domain or sum_domain in controlled_domain:
            sum_domain = next_alias(sum_domain)

        return [sum_domain], [sum_domain, right_domain], sum_domain
    elif lr == (2, 1):
        # Matrix vector, ordinary
        if not utils.setBaseEqual(left.domain[1], right.domain[0]):
            raise ValidationError(dim_no_match_err)

        sum_domain = left.domain[1]
        while left.domain[0] == sum_domain or sum_domain in controlled_domain:
            sum_domain = next_alias(sum_domain)

        return [left.domain[0], sum_domain], [sum_domain], sum_domain
    elif left_len == 1 and right_len > 2:
        # Vector batched-matrix, vector 1-prepended
        if not utils.setBaseEqual(left.domain[0], right.domain[-2]):
            raise ValidationError(dim_no_match_err)

        sum_domain = left.domain[0]
        while (
            sum_domain in right.domain[:-2]
            or sum_domain == right.domain[-1]
            or sum_domain in controlled_domain
        ):
            sum_domain = next_alias(sum_domain)

        return (
            [sum_domain],
            [*right.domain[:-2], sum_domain, right.domain[-1]],
            sum_domain,
        )
    elif left_len > 2 and right_len == 1:
        # batched-matrix vector, ordinary
        if not utils.setBaseEqual(left.domain[-1], right.domain[0]):
            raise ValidationError(dim_no_match_err)

        sum_domain = left.domain[-1]
        while (
            sum_domain in left.domain[:-1] or sum_domain in controlled_domain
        ):
            sum_domain = next_alias(sum_domain)

        return (
            [*left.domain[:-1], sum_domain],
            [sum_domain],
            sum_domain,
        )
    elif left_len >= 2 and right_len >= 2:
        # batched-matrix batched-matrix
        if not utils.setBaseEqual(left.domain[-1], right.domain[-2]):
            raise ValidationError(dim_no_match_err)

        batch_dim_1 = left.domain[:-2]
        batch_dim_2 = right.domain[:-2]

        if len(batch_dim_1) > 0 and len(batch_dim_2) > 0:
            if len(batch_dim_1) != len(batch_dim_2):
                raise ValidationError("Batch dimensions do not match")

            if any([x != y for x, y in zip(batch_dim_1, batch_dim_2)]):
                raise ValidationError("Batch dimensions do not match")

        left_domain = left.domain[-2]
        right_domain = right.domain[-1]
        while right_domain == left_domain or right_domain in batch_dim_1:
            right_domain = next_alias(right_domain)

        while left_domain == right_domain or left_domain in batch_dim_2:
            left_domain = next_alias(left_domain)

        sum_domain = left.domain[-1]
        while (
            sum_domain in left.domain[:-1]
            or sum_domain in right.domain[:-2]
            or sum_domain in (right_domain, left_domain)
            or sum_domain in controlled_domain
        ):
            sum_domain = next_alias(sum_domain)

        return (
            [*left.domain[:-2], left_domain, sum_domain],
            [*right.domain[:-2], sum_domain, right_domain],
            sum_domain,
        )

    raise ValidationError(
        f"Matrix multiplication for left dim: {left_len},"
        f" right dim: {right_len} not implemented"
    )
