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
import gamspy._algebra.expression as expression


class Operable:
    """
    A mixin class that overloads the magic operations of a class
    to be used in Expressions
    """

    # +, -, /, *, **
    def __add__(self, other):
        return expression.Expression(self, "+", other)

    def __radd__(self, other):
        return expression.Expression(other, "+", self)

    def __sub__(self, other):
        return expression.Expression(self, "-", other)

    def __rsub__(self, other):
        return expression.Expression(other, "-", self)

    def __truediv__(self, other):
        return expression.Expression(self, "/", other)

    def __rtruediv__(self, other):
        return expression.Expression(other, "/", self)

    def __mul__(self, other):
        return expression.Expression(self, "*", other)

    def __rmul__(self, other):
        return expression.Expression(other, "*", self)

    def __pow__(self, other):
        if isinstance(other, int) and other == 2:
            return expression.Expression("sqr(", self.gamsRepr(), ")")
        return expression.Expression(self, "**", other)

    # not, and, or, xor
    def __and__(self, other):
        return expression.Expression(self, "and", other)

    def __rand__(self, other):
        return expression.Expression(other, "and", self)

    def __or__(self, other):
        return expression.Expression(self, "or", other)

    def __ror__(self, other):
        return expression.Expression(other, "or", self)

    def __xor__(self, other):
        return expression.Expression(self, "xor", other)

    def __rxor__(self, other):
        return expression.Expression(other, "xor", self)

    # <, <=, >, >=, ==, !=
    def __lt__(self, other):
        return expression.Expression(self, "<", other)

    def __le__(self, other):
        return expression.Expression(self, "=l=", other)

    def __gt__(self, other):
        return expression.Expression(self, ">", other)

    def __ge__(self, other):
        return expression.Expression(self, "=g=", other)

    def __ne__(self, other):  # type: ignore
        return expression.Expression(self, "ne", other)

    # ~ -> not
    def __invert__(self):
        return expression.Expression("", "not", self)
