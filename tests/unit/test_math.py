from __future__ import annotations

import os
import unittest

import pandas as pd

import gamspy._algebra.expression as expression
import gamspy.math as gams_math
from gamspy import Container
from gamspy import Equation
from gamspy import Parameter
from gamspy import Set
from gamspy import Variable
from gamspy.exceptions import ValidationError


class MathSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )

    def test_math(self):
        # Prepare data
        demands = pd.DataFrame(
            [["new-york", 325], ["chicago", 300], ["topeka", 275]]
        )

        # Set
        i = Set(self.m, name="i", records=["new-york", "chicago", "topeka"])

        # Parameter
        b = Parameter(self.m, name="b", domain=[i], records=demands)
        s1 = Parameter(self.m, name="s1", records=5)
        s2 = Parameter(self.m, name="s2", records=3)
        s3 = Parameter(self.m, name="s3", records=6)

        # Variable
        v = Variable(self.m, name="v", domain=[i])

        # abs
        op1 = gams_math.abs(-5)
        self.assertEqual(op1, 5)
        op2 = gams_math.abs(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( abs(b(i)) )")

        # ceil
        op2 = gams_math.ceil(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( ceil(b(i)) )")

        # centropy
        op2 = gams_math.centropy(v[i], b[i])
        self.assertEqual(op2.gamsRepr(), "( centropy(v(i),b(i),1e-20) )")
        op2 = gams_math.centropy(v[i], b[i], 1e-15)
        self.assertEqual(op2.gamsRepr(), "( centropy(v(i),b(i),1e-15) )")
        self.assertRaises(ValueError, gams_math.centropy, v[i], b[i], -1)

        # cvPower
        op2 = gams_math.cv_power(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "( cvPower(b(i),3) )")

        # rPower
        op2 = gams_math.rpower(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "( rPower(b(i),3) )")

        # signPower
        op2 = gams_math.sign_power(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "( signPower(b(i),3) )")

        # vcPower
        op2 = gams_math.vc_power(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "( vcPower(b(i),3) )")

        # sllog10
        op1 = gams_math.sllog10(5)
        self.assertEqual(op1.gamsRepr(), "( sllog10(5,1e-150) )")

        # sqlog10
        op1 = gams_math.sqlog10(5)
        self.assertEqual(op1.gamsRepr(), "( sqlog10(5,1e-150) )")

        # sqrt
        op2 = gams_math.sqrt(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( sqrt(b(i)) )")

        # exp
        op2 = gams_math.exp(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( exp(b(i)) )")

        # power
        op2 = gams_math.power(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( power(b(i),3) )")

        # sqr
        op2 = gams_math.sqr(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( sqr(b(i)) )")

        # mod
        op2 = gams_math.mod(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( mod(b(i),3) )")

        # min
        op2 = gams_math.Min(s1, s2, s3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( min(s1,s2,s3) )")

        # max
        op2 = gams_math.Max(s1, s2, s3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( max(s1,s2,s3) )")

        # log
        op2 = gams_math.log(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( log(b(i)) )")

        # logit
        op1 = gams_math.logit(5)
        self.assertEqual(op1.gamsRepr(), "( logit(5) )")

        # logBeta
        op1 = gams_math.log_beta(3, 5)
        self.assertEqual(op1.gamsRepr(), "( logBeta(3,5) )")

        # logGamma
        op1 = gams_math.log_gamma(3, 5)
        self.assertEqual(op1.gamsRepr(), "( logGamma(3,5) )")

        # log2
        op2 = gams_math.log2(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( log2(b(i)) )")

        # log10
        op2 = gams_math.log10(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( log10(b(i)) )")

        # round
        op1 = gams_math.Round(5.3)
        self.assertEqual(op1, 5)

        op2 = gams_math.Round(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( round(b(i),0) )")

        # sin
        op2 = gams_math.sin(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( sin(b(i)) )")

        # sinh
        op2 = gams_math.sinh(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( sinh(b(i)) )")

        # asin
        op2 = gams_math.asin(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( arcsin(b(i)) )")

        # cos
        op2 = gams_math.cos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( cos(b(i)) )")

        # cosh
        op2 = gams_math.cosh(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( cosh(b(i)) )")

        # arccos
        op2 = gams_math.acos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( arccos(b(i)) )")

        # cos
        op2 = gams_math.cos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( cos(b(i)) )")

        # tan
        op2 = gams_math.tan(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( tan(b(i)) )")

        # tanh
        op2 = gams_math.tanh(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( tanh(b(i)) )")

        # arctan
        op2 = gams_math.atan(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( arctan(b(i)) )")

        # arctan2
        op2 = gams_math.atan2(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( arctan2(b(i),3) )")

        # floor
        op2 = gams_math.floor(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( floor(b(i)) )")

        # div
        op2 = gams_math.div(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( div(b(i),3) )")

        # div0
        op2 = gams_math.div0(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( div0(b(i),3) )")

        # factorial
        op2 = gams_math.factorial(b[i])
        self.assertEqual(op2.gamsRepr(), "( fact(b(i)) )")

        # fractional
        op2 = gams_math.fractional(b[i])
        self.assertEqual(op2.gamsRepr(), "( frac(b(i)) )")

        # truncate
        op2 = gams_math.truncate(b[i])
        self.assertEqual(op2.gamsRepr(), "( trunc(b(i)) )")

        # slexp
        op1 = gams_math.slexp(5)
        self.assertEqual(op1.gamsRepr(), "( slexp(5,150) )")

        # sqexp
        op1 = gams_math.sqexp(5)
        self.assertEqual(op1.gamsRepr(), "( sqexp(5,150) )")

        # dist
        self.assertRaises(Exception, gams_math.dist, (1, 2), 5)

        op2 = gams_math.dist(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "( eDist(b(i),3) )")

        # uniform
        op2 = gams_math.uniform(0, 1)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( uniform(0,1) )")

        # uniformInt
        op2 = gams_math.uniformInt(0, 1)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "( uniformInt(0,1) )")

        # normal
        op2 = gams_math.normal(mean=0, dev=1)
        self.assertTrue(op2, expression.Expression)
        self.assertEqual(op2.gamsRepr(), "( normal(0,1) )")

        # sign
        p = Parameter(self.m, "p", domain=[i])
        op2 = gams_math.sign(p[i])
        self.assertEqual(op2.gamsRepr(), "( sign(p(i)) )")

        # binomial
        op1 = gams_math.binomial(3, 5)
        self.assertTrue(isinstance(op1, expression.Expression))
        self.assertEqual(op1.gamsRepr(), "( binomial(3,5) )")
        op2 = gams_math.binomial(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "( binomial(b(i),3) )")

    def test_math_2(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )
        i = Set(m, "i", records=["1", "2"])
        a = Parameter(m, "a", domain=[i], records=[("1", 1), ("2", 2)])

        op1 = gams_math.entropy(1)
        self.assertEqual(op1.gamsRepr(), "( entropy(1) )")

        op1 = gams_math.beta(1, 2)
        self.assertEqual(op1.gamsRepr(), "( beta(1,2) )")

        op1 = gams_math.regularized_beta(1, 2, 3)
        self.assertEqual(op1.gamsRepr(), "( betaReg(1,2,3) )")

        op1 = gams_math.gamma(1)
        self.assertEqual(op1.gamsRepr(), "( gamma(1) )")

        op1 = gams_math.regularized_gamma(1, 2)
        self.assertEqual(op1.gamsRepr(), "( gammaReg(1,2) )")

        op1 = gams_math.lse_max(a[i])
        self.assertEqual(op1.gamsRepr(), "( lseMax(a(i)) )")

        self.assertRaises(ValidationError, gams_math.lse_max)

        op1 = gams_math.lse_max_sc(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "( lseMaxSc(a(i),a(i)) )")

        self.assertRaises(ValidationError, gams_math.lse_max_sc, 5)

        op1 = gams_math.lse_min(a[i])
        self.assertEqual(op1.gamsRepr(), "( lseMin(a(i)) )")

        self.assertRaises(ValidationError, gams_math.lse_min)

        op1 = gams_math.lse_min_sc(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "( lseMinSc(a(i),a(i)) )")

        self.assertRaises(ValidationError, gams_math.lse_min_sc, 5)

        op1 = gams_math.ncp_cm(a[i], a[i], 3)
        self.assertEqual(op1.gamsRepr(), "( ncpCM(a(i),a(i),3) )")

        op1 = gams_math.ncp_f(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "( ncpF(a(i),a(i),0) )")

        op1 = gams_math.ncpVUpow(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "( ncpVUpow(a(i),a(i),0) )")

        op1 = gams_math.ncpVUsin(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "( ncpVUsin(a(i),a(i),0) )")

        op1 = gams_math.poly(a[i], 3, 5)
        self.assertEqual(op1.gamsRepr(), "( poly(a(i),3,5) )")

        op1 = gams_math.rand_binomial(1, 2)
        self.assertEqual(op1.gamsRepr(), "( randBinomial(1,2) )")

        op1 = gams_math.rand_linear(1, 2, 3)
        self.assertEqual(op1.gamsRepr(), "( randLinear(1,2,3) )")

        op1 = gams_math.rand_triangle(1, 2, 3)
        self.assertEqual(op1.gamsRepr(), "( randTriangle(1,2,3) )")

        op1 = gams_math.slrec(a[i])
        self.assertEqual(op1.gamsRepr(), "( slrec(a(i),1e-10) )")

        op1 = gams_math.sqrec(a[i])
        self.assertEqual(op1.gamsRepr(), "( sqrec(a(i),1e-10) )")

        op1 = gams_math.errorf(a[i])
        self.assertEqual(op1.gamsRepr(), "( errorf(a(i)) )")

        # sigmoid
        op1 = gams_math.sigmoid(2.3)
        self.assertEqual(op1.gamsRepr(), "( sigmoid(2.3) )")

        op2 = gams_math.sigmoid(a[i])
        self.assertEqual(op2.gamsRepr(), "( sigmoid(a(i)) )")

        # power
        op1 = a[i] ** 3
        self.assertEqual(op1.gamsRepr(), "( power(a(i),3) )")

        op2 = a[i] ** 2.999999
        self.assertEqual(op2.gamsRepr(), "( power(a(i),2.999999) )")

        op3 = a[i] ** 2.5
        self.assertEqual(op3.gamsRepr(), "( rPower(a(i),2.5) )")

    def test_logical(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )

        o = Set(m, "o", records=[f"pos{idx}" for idx in range(1, 11)])
        p = Set(m, "p", records=[f"opt{idx}" for idx in range(1, 6)])
        sumc = Variable(m, "sumc", domain=[o, p])
        op = Variable(m, "op", domain=[o, p])
        defopLS = Equation(m, "defopLS", domain=[o, p])
        defopLS[o, p] = op[o, p] == gams_math.ifthen(sumc[o, p] >= 0.5, 1, 0)
        self.assertEqual(
            defopLS._definition.gamsRepr(),
            "defopLS(o,p) .. op(o,p) =e= ( ifthen(sumc(o,p) >= 0.5,1,0) );",
        )

        # bool_and
        op1 = gams_math.bool_and(2, 3)
        self.assertEqual(op1.gamsRepr(), "( bool_and(2,3) )")

        op2 = gams_math.bool_and(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( bool_and(sumc(o,p),op(o,p)) )")

        # bool_eqv
        op1 = gams_math.bool_eqv(2, 3)
        self.assertEqual(op1.gamsRepr(), "( bool_eqv(2,3) )")

        op2 = gams_math.bool_eqv(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( bool_eqv(sumc(o,p),op(o,p)) )")

        # bool_imp
        op1 = gams_math.bool_imp(2, 3)
        self.assertEqual(op1.gamsRepr(), "( bool_imp(2,3) )")

        op2 = gams_math.bool_imp(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( bool_imp(sumc(o,p),op(o,p)) )")

        # bool_not
        op1 = gams_math.bool_not(2)
        self.assertEqual(op1.gamsRepr(), "( bool_not(2) )")

        op2 = gams_math.bool_not(sumc[o, p])
        self.assertEqual(op2.gamsRepr(), "( bool_not(sumc(o,p)) )")

        # bool_or
        op1 = gams_math.bool_or(2, 3)
        self.assertEqual(op1.gamsRepr(), "( bool_or(2,3) )")

        op2 = gams_math.bool_or(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( bool_or(sumc(o,p),op(o,p)) )")

        # bool_xor
        op1 = gams_math.bool_xor(2, 3)
        self.assertEqual(op1.gamsRepr(), "( bool_xor(2,3) )")

        op2 = gams_math.bool_xor(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( bool_xor(sumc(o,p),op(o,p)) )")

        # rel_eq
        op1 = gams_math.rel_eq(2, 3)
        self.assertEqual(op1.gamsRepr(), "( rel_eq(2,3) )")

        op2 = gams_math.rel_eq(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( rel_eq(sumc(o,p),op(o,p)) )")

        # rel_ge
        op1 = gams_math.rel_ge(2, 3)
        self.assertEqual(op1.gamsRepr(), "( rel_ge(2,3) )")

        op2 = gams_math.rel_ge(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( rel_ge(sumc(o,p),op(o,p)) )")

        # rel_gt
        op1 = gams_math.rel_gt(2, 3)
        self.assertEqual(op1.gamsRepr(), "( rel_gt(2,3) )")

        op2 = gams_math.rel_gt(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( rel_gt(sumc(o,p),op(o,p)) )")

        # rel_le
        op1 = gams_math.rel_le(2, 3)
        self.assertEqual(op1.gamsRepr(), "( rel_le(2,3) )")

        op2 = gams_math.rel_le(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( rel_le(sumc(o,p),op(o,p)) )")

        # rel_lt
        op1 = gams_math.rel_lt(2, 3)
        self.assertEqual(op1.gamsRepr(), "( rel_lt(2,3) )")

        op2 = gams_math.rel_lt(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( rel_lt(sumc(o,p),op(o,p)) )")

        # rel_ne
        op1 = gams_math.rel_ne(2, 3)
        self.assertEqual(op1.gamsRepr(), "( rel_ne(2,3) )")

        op2 = gams_math.rel_ne(sumc[o, p], op[o, p])
        self.assertEqual(op2.gamsRepr(), "( rel_ne(sumc(o,p),op(o,p)) )")


def math_suite():
    suite = unittest.TestSuite()
    tests = [
        MathSuite(name) for name in dir(MathSuite) if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(math_suite())
