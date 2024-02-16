#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

from typing import List, Union

import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._symbols.parameter import Parameter
from gamspy._symbols.set import Set
from gamspy._symbols.variable import Variable
from gamspy.exceptions import ValidationError


def next_alias(symbol):
    from gamspy._symbols.alias import Alias

    current = symbol
    if symbol.name.startswith("DenseDim") or symbol.name.startswith("AliasOf"):
        prefix, num = symbol.name.split("_")
    else:
        prefix, num = f"AliasOf{symbol.name}", 1

    num = int(num) + 1
    expected_name = f"{prefix}_{num}"
    find_x = symbol.container.data.get(expected_name, None)
    if not find_x:
        find_x = Alias(
            symbol.container, name=expected_name, alias_with=current
        )

    return find_x


def dim(m, dims: List[int]):
    """Returns an array where each element
    corresponds to a set where the dimension of the
    set is equal to the element in dims"""
    for x in dims:
        if not isinstance(x, int):
            raise ValidationError("Dimensions must be integers")

    sets_so_far = []
    for x in dims:
        expected_name = f"DenseDim{x}_1"
        find_x = m.data.get(expected_name, None)
        if not find_x:
            find_x = Set(m, name=expected_name, records=range(x))

        while find_x in sets_so_far:
            find_x = next_alias(find_x)

        sets_so_far.append(find_x)

    return sets_so_far


# TODO add documentation for these!
def trace(
    x: Union[
        Parameter,
        implicits.ImplicitParameter,
        Variable,
        implicits.ImplicitVariable,
    ]
):
    raise NotImplementedError()


def permute(
    x: Union[
        Parameter,
        implicits.ImplicitParameter,
        Variable,
        implicits.ImplicitVariable,
    ],
    dims: List[int],
):
    # TODO Accept permuting expressions!
    # Might be needed in some context
    dims_len = len(dims)
    if min(dims) != 0 or max(dims) != dims_len - 1:
        raise ValidationError(
            "Permute requires the order of indices from 0 to n-1"
        )

    if len(set(dims)) != dims_len:
        raise ValidationError("Permute dimensions must be unique")

    for i in dims:
        if not isinstance(i, int):
            raise ValidationError("Permute dimensions must be integers")

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
        )
    elif isinstance(x, Variable):
        return implicits.ImplicitVariable(
            x, name=x.name, domain=permuted_domain, permutation=dims
        )
    elif isinstance(x, implicits.ImplicitVariable):
        if x.permutation is not None:
            dims = utils._permute_domain(x.permutation, dims)

        return implicits.ImplicitVariable(
            x.parent, name=x.name, domain=permuted_domain, permutation=dims
        )


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
        if not utils.set_base_eq(left.domain[0], right.domain[0]):
            raise ValidationError("Dot product requires same domain")

        sum_domain = left.domain[0]
        while sum_domain in controlled_domain:
            sum_domain = next_alias(sum_domain)

        return [sum_domain], [sum_domain], sum_domain
    elif lr == (2, 2):
        # Matrix multiplication
        if not utils.set_base_eq(left.domain[1], right.domain[0]):
            raise ValidationError(dim_no_match_err)

        left_domain = left.domain[0]
        right_domain = right.domain[1]
        if left_domain == right_domain:
            left_domain = next_alias(left_domain)

        sum_domain = left.domain[1]
        while (
            sum_domain in [left_domain, right_domain]
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
        if not utils.set_base_eq(left.domain[0], right.domain[0]):
            raise ValidationError(dim_no_match_err)

        sum_domain = left.domain[0]
        while sum_domain == right.domain[1] or sum_domain in controlled_domain:
            sum_domain = next_alias(sum_domain)

        return [sum_domain], [sum_domain, right.domain[1]], sum_domain
    elif lr == (2, 1):
        # Matrix vector, ordinary
        if not utils.set_base_eq(left.domain[1], right.domain[0]):
            raise ValidationError(dim_no_match_err)

        sum_domain = left.domain[1]
        while left.domain[0] == sum_domain or sum_domain in controlled_domain:
            sum_domain = next_alias(sum_domain)

        return [left.domain[0], sum_domain], [sum_domain], sum_domain
    elif left_len == 1 and right_len > 2:
        # Vector batched-matrix, vector 1-prepended
        if not utils.set_base_eq(left.domain[0], right.domain[-2]):
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
        if not utils.set_base_eq(left.domain[-1], right.domain[0]):
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
        if not utils.set_base_eq(left.domain[-1], right.domain[-2]):
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
        if left_domain == right_domain:
            left_domain = next_alias(left_domain)

        sum_domain = left.domain[-1]
        while (
            sum_domain in left.domain[:-1]
            or sum_domain in right.domain[:-2]
            or sum_domain in [right_domain, left_domain]
            or sum_domain in controlled_domain
        ):
            sum_domain = next_alias(sum_domain)

        return (
            [*left.domain[:-2], left_domain, sum_domain],
            [*right.domain[:-2], sum_domain, right_domain],
            sum_domain,
        )
    else:
        raise ValidationError(
            f"Matrix multiplication for left dim: {left_len},"
            f" right dim: {right_len} not implemented"
        )
