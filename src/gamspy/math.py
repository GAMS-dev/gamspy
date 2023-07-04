#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2017-2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2017-2023 GAMS Software GmbH <support@gams.com>
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

import math
import gamspy._algebra._expression as expression
from typing import Union


def abs(value) -> Union[expression.Expression, float]:
    if isinstance(value, (int, float)):
        return math.fabs(value)
    return expression.Expression("abs(", value.gamsRepr(), ")")


def sqrt(value) -> Union[expression.Expression, float]:
    if isinstance(value, (int, float)):
        return math.sqrt(value)
    return expression.Expression("sqrt(", value.gamsRepr(), ")")


def exp(value) -> Union[expression.Expression, float]:
    if isinstance(value, (int, float)):
        return math.exp(value)
    return expression.Expression("exp(", value.gamsRepr(), ")")


def power(base, exponent):
    if isinstance(base, (int, float)) and isinstance(exponent, (int, float)):
        return base**exponent

    base_str = (
        str(base) if isinstance(base, (int, float, str)) else base.gamsRepr()
    )
    exponent_str = (
        str(exponent)
        if isinstance(exponent, (int, float, str))
        else exponent.gamsRepr()
    )
    return expression.Expression("power(", f"{base_str},{exponent_str}", ")")


def mod(dividend, divider) -> Union[expression.Expression, int, float]:
    if isinstance(dividend, (int, float)) and isinstance(
        divider, (int, float)
    ):
        return dividend % divider

    dividend_str = (
        str(dividend)
        if isinstance(dividend, (int, float))
        else dividend.gamsRepr()
    )
    divider_str = (
        str(divider)
        if isinstance(divider, (int, float))
        else divider.gamsRepr()
    )
    return expression.Expression("mod(" + dividend_str, ",", divider_str + ")")


def min(values) -> expression.Expression:
    values_str = ",".join([value.gamsRepr() for value in values])
    return expression.Expression("min(", values_str, ")")


def max(values) -> expression.Expression:
    values_str = ",".join([value.gamsRepr() for value in values])
    return expression.Expression("max(", values_str, ")")


def log(value) -> Union[expression.Expression, float]:
    if isinstance(value, (int, float)):
        return math.log(value)
    return expression.Expression("log(", value.gamsRepr(), ")")


def log2(value) -> Union[expression.Expression, float]:
    if isinstance(value, (int, float)):
        return math.log2(value)
    return expression.Expression("log2(", value.gamsRepr(), ")")


def log10(value) -> Union[expression.Expression, float]:
    if isinstance(value, (int, float)):
        return math.log10(value)
    return expression.Expression("log10(", value.gamsRepr(), ")")


def Round(value, decimal: int = 0) -> expression.Expression:
    return expression.Expression(
        "round(", value.gamsRepr() + f", {decimal}", ")"
    )


def sin(value) -> Union[expression.Expression, float]:
    if isinstance(value, (int, float)):
        return math.sin(value)
    return expression.Expression("sin(", value.gamsRepr(), ")")


def cos(value) -> Union[expression.Expression, float]:
    if isinstance(value, (int, float)):
        return math.cos(value)
    return expression.Expression("cos(", value.gamsRepr(), ")")


def uniform(lower_bound, upper_bound) -> expression.Expression:
    return expression.Expression(
        "uniform(", f"{lower_bound},{upper_bound}", ")"
    )


def uniformInt(lower_bound, upper_bound) -> expression.Expression:
    return expression.Expression(
        "uniformInt(", f"{lower_bound},{upper_bound}", ")"
    )


def normal(
    mean: Union[int, float], dev: Union[int, float]
) -> expression.Expression:
    return expression.Expression("normal(", f"{mean},{dev}", ")")
