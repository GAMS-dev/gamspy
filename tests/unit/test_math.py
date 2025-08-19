from __future__ import annotations

import sys

import numpy as np
import pytest

import gamspy.math as gams_math
from gamspy import Container, Equation, Model, Parameter, Set, Sum, Variable
from gamspy._symbols.implicits import ImplicitVariable
from gamspy.exceptions import ValidationError
from gamspy.math.misc import MathOp

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    markets = ["new-york", "chicago", "topeka"]
    demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

    yield m, markets, demands


def test_math(data):
    m, markets, demands = data
    i = Set(m, name="i", records=markets)

    b = Parameter(m, name="b", domain=[i], records=demands)
    s1 = Parameter(m, name="s1", records=5)
    s2 = Parameter(m, name="s2", records=3)
    s3 = Parameter(m, name="s3", records=6)

    v = Variable(m, name="v", domain=[i])

    # mod
    op = s1 % s2
    assert isinstance(op, MathOp)
    assert op.gamsRepr() == "mod(s1,s2)"

    # abs
    op2 = gams_math.abs(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "abs(b(i))"

    # ceil
    op2 = gams_math.ceil(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "ceil(b(i))"

    # centropy
    op2 = gams_math.centropy(v[i], b[i])
    assert op2.gamsRepr() == "centropy(v(i),b(i),1e-20)"
    op2 = gams_math.centropy(v[i], b[i], 1e-15)
    assert op2.gamsRepr() == "centropy(v(i),b(i),1e-15)"
    pytest.raises(ValueError, gams_math.centropy, v[i], b[i], -1)
    pytest.raises(TypeError, gams_math.centropy, v[i], b[i], s1)

    # cvPower
    op2 = gams_math.cv_power(3, b[i])
    assert op2.gamsRepr() == "cvPower(3,b(i))"
    pytest.raises(ValueError, gams_math.cv_power, s1, b[i])
    pytest.raises(ValueError, gams_math.cv_power, -1, b[i])

    # rPower
    op2 = gams_math.rpower(b[i], 3)
    assert op2.gamsRepr() == "rPower(b(i),3)"

    # signPower
    op2 = gams_math.sign_power(b[i], 3)
    assert op2.gamsRepr() == "signPower(b(i),3)"
    pytest.raises(ValueError, gams_math.sign_power, b[i], s1)
    pytest.raises(ValueError, gams_math.sign_power, b[i], -5)

    # vcPower
    op2 = gams_math.vc_power(b[i], 3)
    assert op2.gamsRepr() == "vcPower(b(i),3)"

    # sllog10
    op1 = gams_math.sllog10(5)
    assert op1.gamsRepr() == "sllog10(5,1e-150)"

    # sqlog10
    op1 = gams_math.sqlog10(5)
    assert op1.gamsRepr() == "sqlog10(5,1e-150)"

    # sqrt
    op2 = gams_math.sqrt(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "sqrt(b(i))"

    # exp
    op2 = gams_math.exp(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "exp(b(i))"

    # power
    op2 = gams_math.power(b[i], 3)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "power(b(i),3)"

    # sqr
    op2 = gams_math.sqr(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "sqr(b(i))"

    # mod
    op2 = gams_math.mod(b[i], 3)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "mod(b(i),3)"

    # min
    op2 = gams_math.Min(s1, s2, s3)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "min(s1,s2,s3)"

    # max
    op2 = gams_math.Max(s1, s2, s3)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "max(s1,s2,s3)"

    # log
    op2 = gams_math.log(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "log(b(i))"

    # logit
    op1 = gams_math.logit(5)
    assert op1.gamsRepr() == "logit(5)"

    # logBeta
    op1 = gams_math.log_beta(3, 5)
    assert op1.gamsRepr() == "logBeta(3,5)"

    # logGamma
    op1 = gams_math.log_gamma(3)
    assert op1.gamsRepr() == "logGamma(3)"

    # log2
    op2 = gams_math.log2(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "log2(b(i))"

    # log10
    op2 = gams_math.log10(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "log10(b(i))"

    # round
    op2 = gams_math.Round(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "round(b(i),0)"

    # sin
    op2 = gams_math.sin(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "sin(b(i))"

    # sinh
    op2 = gams_math.sinh(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "sinh(b(i))"

    # asin
    op2 = gams_math.asin(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "arcsin(b(i))"

    # cos
    op2 = gams_math.cos(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "cos(b(i))"

    # cosh
    op2 = gams_math.cosh(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "cosh(b(i))"

    # arccos
    op2 = gams_math.acos(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "arccos(b(i))"

    # cos
    op2 = gams_math.cos(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "cos(b(i))"

    # tan
    op2 = gams_math.tan(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "tan(b(i))"

    # tanh
    op2 = gams_math.tanh(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "tanh(b(i))"

    # arctan
    op2 = gams_math.atan(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "arctan(b(i))"

    # arctan2
    op2 = gams_math.atan2(b[i], 3)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "arctan2(b(i),3)"

    # floor
    op2 = gams_math.floor(b[i])
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "floor(b(i))"

    # div
    op2 = gams_math.div(b[i], 3)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "div(b(i),3)"

    # div0
    op2 = gams_math.div0(b[i], 3)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "div0(b(i),3)"

    # factorial
    op2 = gams_math.factorial(5)
    assert op2.gamsRepr() == "fact(5)"

    # fractional
    op2 = gams_math.fractional(b[i])
    assert op2.gamsRepr() == "frac(b(i))"

    # truncate
    op2 = gams_math.truncate(b[i])
    assert op2.gamsRepr() == "trunc(b(i))"

    # slexp
    op1 = gams_math.slexp(5)
    assert op1.gamsRepr() == "slexp(5,150)"

    # sqexp
    op1 = gams_math.sqexp(5)
    assert op1.gamsRepr() == "sqexp(5,150)"

    # dist
    op2 = gams_math.dist(b[i], 3)
    assert op2.gamsRepr() == "eDist(b(i),3)"

    # uniform
    op2 = gams_math.uniform(0, 1)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "uniform(0,1)"

    # uniformInt
    op2 = gams_math.uniformInt(0, 1)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "uniformInt(0,1)"

    # normal
    op2 = gams_math.normal(mean=0, dev=1)
    assert isinstance(op2, MathOp)
    assert op2.gamsRepr() == "normal(0,1)"

    # sign
    p = Parameter(m, "p", domain=[i])
    op2 = gams_math.sign(p[i])
    assert op2.gamsRepr() == "sign(p(i))"

    # binomial
    op1 = gams_math.binomial(3, 5)
    assert isinstance(op1, MathOp)
    assert op1.gamsRepr() == "binomial(3,5)"
    op2 = gams_math.binomial(b[i], 3)
    assert op2.gamsRepr() == "binomial(b(i),3)"


def test_math_2(data):
    m, _, _ = data
    m = Container()
    i = Set(m, "i", records=["1", "2"])
    a = Parameter(m, "a", domain=[i], records=[("1", 1), ("2", 2)])

    op1 = gams_math.entropy(1)
    assert op1.gamsRepr() == "entropy(1)"

    op1 = gams_math.beta(1, 2)
    assert op1.gamsRepr() == "beta(1,2)"

    op1 = gams_math.regularized_beta(1, 2, 3)
    assert op1.gamsRepr() == "betaReg(1,2,3)"

    op1 = gams_math.gamma(1)
    assert op1.gamsRepr() == "gamma(1)"

    op1 = gams_math.regularized_gamma(1, 2)
    assert op1.gamsRepr() == "gammaReg(1,2)"

    op1 = gams_math.lse_max(a[i])
    assert op1.gamsRepr() == "lseMax(a(i))"

    pytest.raises(ValidationError, gams_math.lse_max)

    op1 = gams_math.lse_max_sc(a[i], a[i])
    assert op1.gamsRepr() == "lseMaxSc(a(i),a(i))"

    pytest.raises(ValidationError, gams_math.lse_max_sc, 5)

    op1 = gams_math.lse_min(a[i])
    assert op1.gamsRepr() == "lseMin(a(i))"

    pytest.raises(ValidationError, gams_math.lse_min)

    op1 = gams_math.lse_min_sc(a[i], a[i])
    assert op1.gamsRepr() == "lseMinSc(a(i),a(i))"

    pytest.raises(ValidationError, gams_math.lse_min_sc, 5)

    op1 = gams_math.ncp_cm(a[i], a[i], 3)
    assert op1.gamsRepr() == "ncpCM(a(i),a(i),3)"

    op1 = gams_math.ncp_f(a[i], a[i])
    assert op1.gamsRepr() == "ncpF(a(i),a(i),0)"

    op1 = gams_math.ncpVUpow(a[i], a[i])
    assert op1.gamsRepr() == "ncpVUpow(a(i),a(i),0)"

    op1 = gams_math.ncpVUsin(a[i], a[i])
    assert op1.gamsRepr() == "ncpVUsin(a(i),a(i),0)"

    op1 = gams_math.poly(a[i], 3, 5, 7)
    assert op1.gamsRepr() == "poly(a(i),3,5,7)"
    pytest.raises(ValidationError, gams_math.poly, a[i], 3)

    op1 = gams_math.rand_binomial(1, 2)
    assert op1.gamsRepr() == "randBinomial(1,2)"

    op1 = gams_math.rand_linear(1, 2, 3)
    assert op1.gamsRepr() == "randLinear(1,2,3)"

    op1 = gams_math.rand_triangle(1, 2, 3)
    assert op1.gamsRepr() == "randTriangle(1,2,3)"

    op1 = gams_math.slrec(a[i])
    assert op1.gamsRepr() == "slrec(a(i),1e-10)"

    op1 = gams_math.sqrec(a[i])
    assert op1.gamsRepr() == "sqrec(a(i),1e-10)"

    op1 = gams_math.errorf(a[i])
    assert op1.gamsRepr() == "errorf(a(i))"

    # sigmoid
    op1 = gams_math.sigmoid(2.3)
    assert op1.gamsRepr() == "sigmoid(2.3)"

    op2 = gams_math.sigmoid(a[i])
    assert op2.gamsRepr() == "sigmoid(a(i))"

    # power
    op1 = a[i] ** 3
    assert op1.gamsRepr() == "power(a(i),3)"

    op2 = a[i] ** 2.999999
    assert op2.gamsRepr() == "power(a(i),2.999999)"

    op3 = a[i] ** 2.5
    assert op3.gamsRepr() == "rPower(a(i),2.5)"

    # rpower
    op1 = 3 ** a[i]
    assert op1.gamsRepr() == "rPower(3,a(i))"


def test_logical(data):
    m, _, _ = data
    m = Container()

    o = Set(m, "o", records=[f"pos{idx}" for idx in range(1, 11)])
    p = Set(m, "p", records=[f"opt{idx}" for idx in range(1, 6)])
    sumc = Variable(m, "sumc", domain=[o, p])
    op = Variable(m, "op", domain=[o, p])
    defopLS = Equation(m, "defopLS", domain=[o, p])
    defopLS[o, p] = op[o, p] == gams_math.ifthen(sumc[o, p] >= 0.5, 1, 0)
    assert (
        defopLS.getDefinition()
        == "defopLS(o,p) .. op(o,p) =e= ifthen(sumc(o,p) >= 0.5,1,0);"
    )

    # bool_and
    op1 = gams_math.bool_and(2, 3)
    assert op1.gamsRepr() == "bool_and(2,3)"

    op2 = gams_math.bool_and(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "bool_and(sumc(o,p),op(o,p))"

    # bool_eqv
    op1 = gams_math.bool_eqv(2, 3)
    assert op1.gamsRepr() == "bool_eqv(2,3)"

    op2 = gams_math.bool_eqv(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "bool_eqv(sumc(o,p),op(o,p))"

    # bool_imp
    op1 = gams_math.bool_imp(2, 3)
    assert op1.gamsRepr() == "bool_imp(2,3)"

    op2 = gams_math.bool_imp(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "bool_imp(sumc(o,p),op(o,p))"

    # bool_not
    op1 = gams_math.bool_not(2)
    assert op1.gamsRepr() == "bool_not(2)"

    op2 = gams_math.bool_not(sumc[o, p])
    assert op2.gamsRepr() == "bool_not(sumc(o,p))"

    # bool_or
    op1 = gams_math.bool_or(2, 3)
    assert op1.gamsRepr() == "bool_or(2,3)"

    op2 = gams_math.bool_or(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "bool_or(sumc(o,p),op(o,p))"

    # bool_xor
    op1 = gams_math.bool_xor(2, 3)
    assert op1.gamsRepr() == "bool_xor(2,3)"

    op2 = gams_math.bool_xor(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "bool_xor(sumc(o,p),op(o,p))"

    # rel_eq
    op1 = gams_math.rel_eq(2, 3)
    assert op1.gamsRepr() == "rel_eq(2,3)"

    op2 = gams_math.rel_eq(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "rel_eq(sumc(o,p),op(o,p))"

    # rel_ge
    op1 = gams_math.rel_ge(2, 3)
    assert op1.gamsRepr() == "rel_ge(2,3)"

    op2 = gams_math.rel_ge(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "rel_ge(sumc(o,p),op(o,p))"

    # rel_gt
    op1 = gams_math.rel_gt(2, 3)
    assert op1.gamsRepr() == "rel_gt(2,3)"

    op2 = gams_math.rel_gt(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "rel_gt(sumc(o,p),op(o,p))"

    # rel_le
    op1 = gams_math.rel_le(2, 3)
    assert op1.gamsRepr() == "rel_le(2,3)"

    op2 = gams_math.rel_le(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "rel_le(sumc(o,p),op(o,p))"

    # rel_lt
    op1 = gams_math.rel_lt(2, 3)
    assert op1.gamsRepr() == "rel_lt(2,3)"

    op2 = gams_math.rel_lt(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "rel_lt(sumc(o,p),op(o,p))"

    # rel_ne
    op1 = gams_math.rel_ne(2, 3)
    assert op1.gamsRepr() == "rel_ne(2,3)"

    op2 = gams_math.rel_ne(sumc[o, p], op[o, p])
    assert op2.gamsRepr() == "rel_ne(sumc(o,p),op(o,p))"


def test_relu(data, relu_type=gams_math.relu_with_binary_var):
    m, _, _ = data

    i = Set(m, name="i", records=["i1", "i2", "i3"], description="plants")

    budget = 200

    c = Parameter(
        m,
        name="c",
        records=[("i1", 100), ("i2", 50), ("i3", 90)],
        description="min investment before operation",
        domain=[i],
    )

    g = Parameter(
        m,
        name="g",
        records=[("i1", 2), ("i2", 1.2), ("i3", 1.5)],
        description="gain per unit after min investment",
        domain=[i],
    )

    x = Variable(
        m,
        name="x",
        description="investment in plant i",
        domain=[i],
        type="Positive",
    )

    if relu_type == gams_math.relu_with_binary_var:
        y, eqs = relu_type(x - c, default_lb=-100, default_ub=200)

        y, b, eqs = relu_type(
            x - c, default_lb=-100, default_ub=200, return_binary_var=True
        )
        assert b.type == "binary"
        assert len(eqs) == 3  # adds three equations
    else:
        y, eqs = relu_type(x - c)
        y, s, eqs = relu_type(x - c, return_slack_var=True)
        assert isinstance(y, ImplicitVariable)
        assert isinstance(s, ImplicitVariable)
        assert id(y.parent) == id(s.parent)
        assert len(y.parent.domain) == len((x - c).domain) + 1
        assert len(eqs) == 1  # adds one equation

    total_budget = Equation(m, name="check_budget")
    total_budget[...] = Sum(i, x[i]) <= budget

    budget = Model(
        m,
        name="budget",
        equations=m.getEquations(),
        problem="MIP",
        sense="max",
        objective=Sum(i, y[i] * g[i]),
    )
    budget.solve(output=sys.stdout)
    assert np.isclose(budget.objective_value, 200.0)


def test_leaky_relu(data):
    m, _, _ = data
    leaky_relu = gams_math.leaky_relu_with_binary_var

    i = Set(m, name="i", records=["i1", "i2", "i3"], description="plants")

    x = Variable(m, name="x", domain=i, type="free")
    x.lo[...] = -100
    x.up[...] = 100

    x2 = Parameter(m, name="x2", domain=i)
    x2[...] = 43

    y, eqs = leaky_relu(x2, 0.1)
    assert len(eqs) == 4

    y, eqs = leaky_relu(x, 0.1)
    assert len(eqs) == 4

    y, b, eqs = leaky_relu(x, 0.1, return_binary_var=True)
    assert len(eqs) == 4
    assert b.type == "binary"

    # must be in (0, 1)
    pytest.raises(ValidationError, leaky_relu, x, -1)
    pytest.raises(ValidationError, leaky_relu, x, 0)
    pytest.raises(ValidationError, leaky_relu, x, 2)
    pytest.raises(ValidationError, leaky_relu, x, 1)

    x_vals = [-100, -50, 0, 50, 100]
    y_vals = [-10, -5, 0, 50, 100]
    b_vals = [[0], [0], [0, 1], [1], [1]]

    model = Model(
        m, name="leaky_relu", equations=m.getEquations(), problem="MIP"
    )

    for x_val, y_val, b_val in zip(x_vals, y_vals, b_vals):
        x.fx[...] = x_val
        model.solve()
        assert y.toDense()[0] == y_val
        assert b.toDense()[0] in b_val


def test_relu_2(data):
    m, markets, demands = data
    m = Container()

    i = Set(m, name="i", records=["i1", "i2", "i3"], description="plants")

    budget = 200

    c = Parameter(
        m,
        name="c",
        records=[("i1", 100), ("i2", 50), ("i3", 90)],
        description="min investment before operation",
        domain=[i],
    )

    g = Parameter(
        m,
        name="g",
        records=[("i1", 2), ("i2", 1.2), ("i3", 1.5)],
        description="gain per unit after min investment",
        domain=[i],
    )

    x = Variable(
        m,
        name="x",
        description="investment in plant i",
        domain=[i],
        type="Positive",
    )

    y, eqs = gams_math.relu_with_complementarity_var(x - c)
    assert len(eqs) == 2

    total_budget = Equation(m, name="check_budget")
    total_budget[...] = Sum(i, x[i]) <= budget

    budget = Model(
        m,
        name="budget",
        equations=m.getEquations(),
        problem="NLP",
        sense="max",
        objective=Sum(i, y[i] * g[i]),
    )
    # give solver a different starting point, since solver is local
    # it will stop at a local optimum
    y.l["i1"] = 0
    y.l["i2"] = 200
    y.l["i3"] = 0
    budget.solve(output=sys.stdout)

    assert np.isclose(budget.objective_value, 180.0)


def test_relu_3(data):
    test_relu(data, gams_math.relu_with_sos1_var)


def test_log_softmax(data):
    m, _, _ = data
    m = Container()

    labels = Set(m, name="labels", domain=gams_math.dim([30, 3]))
    for i in range(30):
        labels[str(i), str(i % 3)] = 1

    x = Variable(m, name="x", domain=gams_math.dim([30, 3]))

    x.lo[...] = -5
    x.up[...] = 5

    p = Parameter(m, name="p", domain=gams_math.dim([30, 3]))

    # log_softmax requires bare value
    pytest.raises(ValidationError, gams_math.log_softmax, x - p)
    pytest.raises(ValidationError, gams_math.log_softmax, x[...])
    # dim out of bounds
    pytest.raises(IndexError, gams_math.log_softmax, x, 2)

    # this uses LSE in background
    y, eqs = gams_math.log_softmax(x)
    assert "lseMax" in eqs[0].getDefinition()

    # this cannot use LSE in background
    y, eqs = gams_math.log_softmax(x, 0)
    assert "lseMax" not in eqs[0].getDefinition()

    # this won't use LSE in background because of skip_intrinsic
    y, eqs = gams_math.log_softmax(x, skip_intrinsic=True)
    assert "lseMax" not in eqs[0].getDefinition()

    nll = Variable(m, name="nll")

    set_loss = Equation(m, name="set_loss")
    set_loss[...] = nll == Sum(labels[y.domain], -y)


def test_softmax(data):
    m, markets, demands = data
    m = Container()

    labels = Set(m, name="labels", domain=gams_math.dim([30, 3]))
    for i in range(30):
        labels[str(i), str(i % 3)] = 1

    x = Variable(m, name="x", domain=gams_math.dim([30, 3]))

    x.lo[...] = -5
    x.up[...] = 5

    p = Parameter(m, name="p", domain=gams_math.dim([30, 3]))

    # softmax requires bare value
    pytest.raises(ValidationError, gams_math.softmax, x - p)
    pytest.raises(ValidationError, gams_math.softmax, x[...])
    # dim out of bounds
    pytest.raises(IndexError, gams_math.softmax, x, 2)

    y, equations = gams_math.softmax(x)
    assert "exp" in equations[0].getDefinition()


def test_tanh_activation(data):
    m, *_ = data
    m = Container()

    x = Variable(m, name="x", domain=gams_math.dim([30, 3]))

    x.lo[...] = -5
    x.up[...] = "inf"

    y, eqs = gams_math.activation.tanh(x)

    assert len(y.domain) == 2
    assert len(y.domain[0]) == 30
    assert len(y.domain[1]) == 3
    assert len(eqs) == 1

    assert np.isclose(y.records["lower"].iloc[0], np.tanh(-5))
    assert np.isclose(y.records["upper"].iloc[0], 1)
