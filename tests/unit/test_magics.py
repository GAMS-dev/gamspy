from __future__ import annotations

import pytest

import gamspy as gp
import gamspy._algebra.expression as expression
from gamspy import Container, Parameter, Set, Variable
from gamspy.math import sqr

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    markets = ["new-york", "chicago", "topeka"]
    demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

    yield m, markets, demands


def test_magics(data):
    m, markets, demands = data
    i = Set(m, name="i", records=markets)

    b = Parameter(m, name="b", domain=[i], records=demands)

    x = Variable(m, name="x", domain=[i], type="Positive")

    # ADD
    # Parameter + Variable, Variable + Parameter,
    # Parameter + builtin, builtin + Parameter
    op1 = b[i] + x[i]
    assert op1.gamsRepr() == "b(i) + x(i)"
    op2 = x[i] + b[i]
    assert op2.gamsRepr() == "x(i) + b(i)"
    op3 = b[i] + 5
    assert op3.gamsRepr() == "b(i) + 5"
    op4 = 5 + b[i]
    assert op4.gamsRepr() == "5 + b(i)"

    # SUB
    # Parameter - Variable, Variable - Parameter,
    # Parameter - builtin, builtin - Parameter
    op1 = b[i] - x[i]
    assert op1.gamsRepr() == "b(i) - x(i)"
    op2 = x[i] - b[i]
    assert op2.gamsRepr() == "x(i) - b(i)"
    op3 = b[i] - 5
    assert op3.gamsRepr() == "b(i) - 5"
    op4 = 5 - b[i]
    assert op4.gamsRepr() == "5 - b(i)"

    # MUL
    # Parameter * Variable, Variable * Parameter,
    # Parameter * builtin, builtin * Parameter
    op1 = b[i] * x[i]
    assert op1.gamsRepr() == "b(i) * x(i)"
    op2 = x[i] * b[i]
    assert op2.gamsRepr() == "x(i) * b(i)"
    op3 = b[i] * -5
    assert op3.gamsRepr() == "b(i) * (-5)"
    op4 = -5 * b[i]
    assert op4.gamsRepr() == "(-5) * b(i)"

    # DIV
    # Parameter / Variable, Variable / Parameter,
    # Parameter / builtin, builtin / Parameter
    op1 = b[i] / x[i]
    assert op1.gamsRepr() == "b(i) / x(i)"
    op2 = x[i] / b[i]
    assert op2.gamsRepr() == "x(i) / b(i)"
    op3 = b[i] / 5
    assert op3.gamsRepr() == "b(i) / 5"
    op4 = 5 / b[i]
    assert op4.gamsRepr() == "5 / b(i)"

    # POW
    # Parameter ** Variable, Variable ** Parameter
    op1 = b[i] ** x[i]
    assert op1.gamsRepr() == "rPower(b(i),x(i))"
    op2 = x[i] ** b[i]
    assert op2.gamsRepr() == "rPower(x(i),b(i))"

    # Set/Parameter/Variable ** 2
    op1 = i**2
    assert op1.gamsRepr() == "power(i,2)"
    op2 = b[i] ** 2
    assert op2.gamsRepr() == "power(b(i),2)"
    op3 = x[i] ** 2
    assert op3.gamsRepr() == "power(x(i),2)"

    # Set/Parameter/Variable ** 0.5
    op1 = i**0.5
    assert op1.gamsRepr() == "sqrt(i)"
    op2 = b[i] ** 0.5
    assert op2.gamsRepr() == "sqrt(b(i))"
    op3 = x[i] ** 0.5
    assert op3.gamsRepr() == "sqrt(x(i))"

    m = Container()
    j = Parameter(m, "j", records=5)
    k = Parameter(m, "k", records=2)
    l = Parameter(m, "l")

    l[...] = (j**k) ** 0.5
    assert l.toValue() == 5.0

    # AND
    # Parameter and Variable, Variable and Parameter
    op1 = b[i] & x[i]
    assert op1.gamsRepr() == "b(i) and x(i)"
    op2 = x[i] & b[i]
    assert op2.gamsRepr() == "x(i) and b(i)"

    # RAND
    op1 = 5 & b[i]
    assert op1.gamsRepr() == "5 and b(i)"

    # OR
    # Parameter or Variable, Variable or Parameter
    op1 = b[i] | x[i]
    assert op1.gamsRepr() == "b(i) or x(i)"
    op2 = x[i] | b[i]
    assert op2.gamsRepr() == "x(i) or b(i)"

    # ROR
    op1 = 5 | b[i]
    assert op1.gamsRepr() == "5 or b(i)"

    # XOR
    # Parameter xor Variable, Variable xor Parameter
    op1 = b[i] ^ x[i]
    assert op1.gamsRepr() == "b(i) xor x(i)"
    op2 = x[i] ^ b[i]
    assert op2.gamsRepr() == "x(i) xor b(i)"

    # RXOR
    op1 = 5 ^ x[i]
    assert op1.gamsRepr() == "5 xor x(i)"

    # LT
    # Parameter < Variable, Variable < Parameter
    op1 = b[i] < x[i]
    assert op1.gamsRepr() == "b(i) < x(i)"
    op2 = x[i] < b[i]
    assert op2.gamsRepr() == "x(i) < b(i)"

    # LE
    # Parameter <= Variable, Variable <= Parameter
    op1 = b[i] <= x[i]
    assert op1.gamsRepr() == "b(i) =l= x(i)"
    op2 = x[i] <= b[i]
    assert op2.gamsRepr() == "x(i) =l= b(i)"

    # GT
    # Parameter > Variable, Variable > Parameter
    op1 = b[i] > x[i]
    assert op1.gamsRepr() == "b(i) > x(i)"
    op2 = x[i] > b[i]
    assert op2.gamsRepr() == "x(i) > b(i)"

    # GE
    # Parameter >= Variable, Variable >= Parameter
    op1 = b[i] >= x[i]
    assert op1.gamsRepr() == "b(i) =g= x(i)"
    op2 = x[i] >= b[i]
    assert op2.gamsRepr() == "x(i) =g= b(i)"

    # NE
    # Parameter != Variable, Variable != Parameter
    op1 = b[i] != x[i]
    assert op1.gamsRepr() == "b(i) ne x(i)"
    op2 = x[i] != b[i]
    assert op2.gamsRepr() == "x(i) ne b(i)"

    # E
    # Parameter == Variable, Variable == Parameter
    op1 = b[i] == x[i]
    assert op1.gamsRepr() == "b(i) =e= x(i)"
    op2 = x[i] == b[i]
    assert op2.gamsRepr() == "x(i) =e= b(i)"
    op3 = b[i] == b[i]
    assert op3.gamsRepr() == "b(i) eq b(i)"

    # not
    # not Parameter/Variable
    op1 = ~b[i]
    assert op1.gamsRepr() == "not b(i)"
    op2 = ~x[i]
    assert op2.gamsRepr() == "not x(i)"

    # unary
    op1 = -(b[i] + x[i])
    assert op1.gamsRepr() == "(-(b(i) + x(i)))"
    op2 = -(x[i] + b[i])
    assert op2.gamsRepr() == "(-(x(i) + b(i)))"
    assert isinstance(-sqr(k), expression.Expression)
    assert -sqr(k).toValue() == -4

    with pytest.raises(TypeError):
        gp.Number("bla")

    m = gp.Container()
    a = gp.Parameter(m)
    a[...] = gp.Number(2) != 0
    assert a.toValue() == 1
