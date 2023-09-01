import math
import unittest

import pandas as pd

import gamspy._algebra.expression as expression
import gamspy.math as gams_math
from gamspy import (
    Container,
    Parameter,
    Set,
    Variable,
)


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

        # arctan
        op1 = gams_math.atan(7.5)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.atan(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(arctan( b(i) ))")

        # floor
        op1 = gams_math.floor(7.5)
        self.assertTrue(isinstance(op1, int) and op1 == 7)
        op2 = gams_math.floor(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(floor( b(i) ))")

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

        p = Parameter(self.m, "p", domain=[i])
        op2 = gams_math.sign(p[i])
        self.assertEqual(op2.gamsRepr(), "(sign( p(i) ))")


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
