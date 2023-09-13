import math
import unittest

import pandas as pd

import gamspy._algebra.expression as expression
import gamspy.math as gams_math
from gamspy import Container
from gamspy import Parameter
from gamspy import Set
from gamspy import Variable


class MathSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_math(self):
        # Prepare data
        demands = pd.DataFrame(
            [["new-york", 325], ["chicago", 300], ["topeka", 275]]
        )

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        # Parameter
        b = Parameter(self.m, name="b", domain=[j], records=demands)
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
        self.assertEqual(op2.gamsRepr(), "(abs( b(i) ))")

        # ceil
        op1 = gams_math.ceil(7.5)
        self.assertTrue(isinstance(op1, int) and op1 == 8)
        op2 = gams_math.ceil(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(ceil( b(i) ))")

        # centropy
        op2 = gams_math.centropy(v[i], b[i])
        self.assertEqual(op2.gamsRepr(), "(centropy( v(i),b(i),1e-20 ))")
        op2 = gams_math.centropy(v[i], b[i], 1e-15)
        self.assertEqual(op2.gamsRepr(), "(centropy( v(i),b(i),1e-15 ))")
        self.assertRaises(ValueError, gams_math.centropy, v[i], b[i], -1)

        # cvPower
        op1 = gams_math.cv_power(5, 3)
        self.assertEqual(op1, 125)
        op2 = gams_math.cv_power(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "(cvPower( b(i),3 ))")

        # rPower
        op1 = gams_math.rpower(5, 3)
        self.assertEqual(op1, 125)
        op2 = gams_math.rpower(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "(rPower( b(i),3 ))")

        # signPower
        op1 = gams_math.sign_power(5, 3)
        self.assertEqual(op1, 125)
        op2 = gams_math.sign_power(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "(signPower( b(i),3 ))")

        # vcPower
        op1 = gams_math.vc_power(5, 3)
        self.assertEqual(op1, 125)
        op2 = gams_math.vc_power(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "(vcPower( b(i),3 ))")

        # sllog10
        op1 = gams_math.sllog10(5)
        self.assertEqual(op1.gamsRepr(), "(sllog10( 5,1e-150 ))")

        # sqlog10
        op1 = gams_math.sqlog10(5)
        self.assertEqual(op1.gamsRepr(), "(sqlog10( 5,1e-150 ))")

        # sqrt
        op1 = gams_math.sqrt(9)
        self.assertEqual(op1, 3)
        op2 = gams_math.sqrt(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(sqrt( b(i) ))")

        # exp
        op1 = gams_math.exp(3)
        self.assertEqual(op1, math.exp(3))
        op2 = gams_math.exp(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(exp( b(i) ))")

        # power
        op1 = gams_math.power(2, 3)
        self.assertEqual(op1, math.pow(2, 3))
        op2 = gams_math.power(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(power( b(i),3 ))")

        # sqr
        op1 = gams_math.sqr(4)
        self.assertEqual(op1, 4**2)
        op2 = gams_math.sqr(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(power( b(i),2 ))")

        # mod
        op1 = gams_math.mod(5, 2)
        self.assertEqual(op1, 1)
        op2 = gams_math.mod(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(mod(b(i) , 3))")

        # min
        op2 = gams_math.min(s1, s2, s3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(min( s1,s2,s3 ))")

        # max
        op2 = gams_math.max(s1, s2, s3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(max( s1,s2,s3 ))")

        # log
        op1 = gams_math.log(3)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.log(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(log( b(i) ))")

        # logit
        op1 = gams_math.logit(5)
        self.assertEqual(op1.gamsRepr(), "(logit( 5 ))")

        # logBeta
        op1 = gams_math.log_beta(3, 5)
        self.assertEqual(op1.gamsRepr(), "(logBeta( 3,5 ))")

        # logGamma
        op1 = gams_math.log_gamma(3, 5)
        self.assertEqual(op1.gamsRepr(), "(logGamma( 3,5 ))")

        # log2
        op1 = gams_math.log2(8)
        self.assertEqual(op1, 3)
        op2 = gams_math.log2(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(log2( b(i) ))")

        # log10
        op1 = gams_math.log10(100)
        self.assertEqual(op1, 2)
        op2 = gams_math.log10(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(log10( b(i) ))")

        # round
        op2 = gams_math.Round(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(round( b(i), 0 ))")

        # sin
        op1 = gams_math.sin(8)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.sin(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(sin( b(i) ))")

        # sinh
        op1 = gams_math.sinh(5)
        self.assertAlmostEqual(op1, 74.203, 3)
        op2 = gams_math.sinh(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(sinh( b(i) ))")

        # asin
        op1 = gams_math.asin(0.5)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.asin(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(arcsin( b(i) ))")

        # cos
        op1 = gams_math.cos(8)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.cos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(cos( b(i) ))")

        # cosh
        op1 = gams_math.cosh(5)
        self.assertAlmostEqual(op1, 74.209, 2)
        op2 = gams_math.cosh(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(cosh( b(i) ))")

        # arccos
        op1 = gams_math.acos(0.5)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.acos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(arccos( b(i) ))")

        # cos
        op1 = gams_math.cos(8)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.cos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(cos( b(i) ))")

        # tan
        op1 = gams_math.tan(0.5)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.tan(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(tan( b(i) ))")

        # tanh
        op1 = gams_math.tanh(5)
        self.assertAlmostEqual(op1, 0.999, 2)
        op2 = gams_math.tanh(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(tanh( b(i) ))")

        # arctan
        op1 = gams_math.atan(7.5)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.atan(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(arctan( b(i) ))")

        # arctan2
        op1 = gams_math.atan2(5, 3)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.atan2(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(arctan2( b(i), 3 ))")

        # floor
        op1 = gams_math.floor(7.5)
        self.assertTrue(isinstance(op1, int) and op1 == 7)
        op2 = gams_math.floor(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(floor( b(i) ))")

        # div
        op1 = gams_math.div(6, 3)
        self.assertTrue(op1 == 2)
        op2 = gams_math.div(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(div( b(i), 3 ))")

        # div0
        op1 = gams_math.div0(6, 0)
        self.assertTrue(op1 == 1e299)
        op1 = gams_math.div0(6, 2)
        self.assertTrue(op1 == 3)
        op2 = gams_math.div0(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(div0( b(i), 3 ))")

        # factorial
        op1 = gams_math.factorial(5)
        self.assertEqual(op1, 120)
        op2 = gams_math.factorial(b[i])
        self.assertEqual(op2.gamsRepr(), "(fact( b(i) ))")

        # fractional
        op1 = gams_math.fractional(5.3)
        self.assertAlmostEqual(op1, 0.299, 2)
        op2 = gams_math.fractional(b[i])
        self.assertEqual(op2.gamsRepr(), "(frac( b(i) ))")

        # truncate
        op1 = gams_math.truncate(5.3)
        self.assertEqual(op1, 5)
        op2 = gams_math.truncate(b[i])
        self.assertEqual(op2.gamsRepr(), "(trunc( b(i) ))")

        # slexp
        op1 = gams_math.slexp(5)
        self.assertEqual(op1.gamsRepr(), "(slexp( 5,150 ))")

        # sqexp
        op1 = gams_math.sqexp(5)
        self.assertEqual(op1.gamsRepr(), "(sqexp( 5,150 ))")

        # dist
        op1 = gams_math.dist((1, 2), (2, 3))
        self.assertTrue(isinstance(op1, float))
        self.assertRaises(Exception, gams_math.dist, (1, 2), 5)

        op2 = gams_math.dist(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "(eDist( b(i), 3 ))")

        # uniform
        op2 = gams_math.uniform(0, 1)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(uniform( 0,1 ))")

        # uniformInt
        op2 = gams_math.uniformInt(0, 1)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(uniformInt( 0,1 ))")

        # normal
        op2 = gams_math.normal(mean=0, dev=1)
        self.assertTrue(op2, expression.Expression)
        self.assertEqual(op2.gamsRepr(), "(normal( 0,1 ))")

        # sign
        p = Parameter(self.m, "p", domain=[i])
        op2 = gams_math.sign(p[i])
        self.assertEqual(op2.gamsRepr(), "(sign( p(i) ))")

        # binomial
        op1 = gams_math.binomial(3, 5)
        self.assertTrue(isinstance(op1, expression.Expression))
        self.assertEqual(op1.gamsRepr(), "(binomial( 3,5 ))")
        op2 = gams_math.binomial(b[i], 3)
        self.assertEqual(op2.gamsRepr(), "(binomial( b(i),3 ))")


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
