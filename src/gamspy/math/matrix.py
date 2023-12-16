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

from typing import Tuple, List
from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.expression as expression
import gamspy.utils as utils

import gamspy._symbols.implicits as implicits
from gamspy.exceptions import GamspyException

from gamspy._symbols.parameter import Parameter
from gamspy._symbols.variable import Variable


def permute(
    x: Union[Parameter, implicits.ImplicitParameter, Variable, implicits.ImplicitVariable],
    dims: List[int]
):
    dims_len = len(dims)
    if min(dims) != 0 or max(dims) != dims_len - 1:
        raise GamspyException("Permute requires the order of indices from 0 to n-1")

    if len(set(dims)) != dims_len:
        raise GamspyException("Permute dimensions must be unique")

    for i in dims:
        if not isinstance(i, int):
            raise GamspyException("Permute dimensions must be integers")

    permuted_domain = utils._permute_domain(x.domain, dims)
    if isinstance(x, Parameter):
        return implicits.ImplicitParameter(
            x,
            name=x.name,
            records=x.records,
            domain=permuted_domain,
            permutation=dims
        )
    elif isinstance(x, implicits.ImplicitParameter):
        if x.permutation is not None:
            dims = utils._permute_domain(x.permutation, dims)

        return implicits.ImplicitParameter(
            x.parent,
            name=x.name,
            records=x._records,
            domain=permuted_domain,
            permutation=dims
        )
    elif isinstance(Variable):
        raise NotImplemented()
    elif isinstance(implicits.ImplicitVariable):
        raise NotImplemented()





