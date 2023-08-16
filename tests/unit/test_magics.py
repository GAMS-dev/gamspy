import unittest

import pandas as pd
from gamspy import (
    Container,
    Set,
    Parameter,
    Variable,
)


class MagicsSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_magics(self):
        # Prepare data
        demands = pd.DataFrame(
            [["new-york", 325], ["chicago", 300], ["topeka", 275]]
        )

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        # Parameter
        b = Parameter(self.m, name="b", domain=[j], records=demands)

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")

        # ADD
        # Parameter + Variable, Variable + Parameter,
        # Parameter + builtin, builtin + Parameter
        op1 = b[i] + x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) + x(i))")
        op2 = x[i] + b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) + b(i))")
        op3 = b[i] + 5
        self.assertEqual(op3.gamsRepr(), "(b(i) + 5)")
        op4 = 5 + b[i]
        self.assertEqual(op4.gamsRepr(), "(5 + b(i))")

        # SUB
        # Parameter - Variable, Variable - Parameter,
        # Parameter - builtin, builtin - Parameter
        op1 = b[i] - x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) - x(i))")
        op2 = x[i] - b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) - b(i))")
        op3 = b[i] - 5
        self.assertEqual(op3.gamsRepr(), "(b(i) - 5)")
        op4 = 5 - b[i]
        self.assertEqual(op4.gamsRepr(), "(5 - b(i))")

        # MUL
        # Parameter * Variable, Variable * Parameter,
        # Parameter * builtin, builtin * Parameter
        op1 = b[i] * x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) * x(i))")
        op2 = x[i] * b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) * b(i))")
        op3 = b[i] * -5
        self.assertEqual(op3.gamsRepr(), "(b(i) * (-5))")
        op4 = -5 * b[i]
        self.assertEqual(op4.gamsRepr(), "((-5) * b(i))")

        # DIV
        # Parameter / Variable, Variable / Parameter,
        # Parameter / builtin, builtin / Parameter
        op1 = b[i] / x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) / x(i))")
        op2 = x[i] / b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) / b(i))")
        op3 = b[i] / 5
        self.assertEqual(op3.gamsRepr(), "(b(i) / 5)")
        op4 = 5 / b[i]
        self.assertEqual(op4.gamsRepr(), "(5 / b(i))")

        # POW
        # Parameter ** Variable, Variable ** Parameter
        op1 = b[i] ** x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) ** x(i))")
        op2 = x[i] ** b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) ** b(i))")

        # Set/Parameter/Variable ** 2
        op1 = i**2
        self.assertEqual(op1.gamsRepr(), "(sqr( i ))")
        op2 = b[i] ** 2
        self.assertEqual(op2.gamsRepr(), "(sqr( b(i) ))")
        op3 = x[i] ** 2
        self.assertEqual(op3.gamsRepr(), "(sqr( x(i) ))")

        # AND
        # Parameter and Variable, Variable and Parameter
        op1 = b[i] & x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) and x(i))")
        op2 = x[i] & b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) and b(i))")

        # RAND
        op1 = 5 & b[i]
        self.assertEqual(op1.gamsRepr(), "(5 and b(i))")

        # OR
        # Parameter or Variable, Variable or Parameter
        op1 = b[i] | x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) or x(i))")
        op2 = x[i] | b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) or b(i))")

        # ROR
        op1 = 5 | b[i]
        self.assertEqual(op1.gamsRepr(), "(5 or b(i))")

        # XOR
        # Parameter xor Variable, Variable xor Parameter
        op1 = b[i] ^ x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) xor x(i))")
        op2 = x[i] ^ b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) xor b(i))")

        # RXOR
        op1 = 5 ^ x[i]
        self.assertEqual(op1.gamsRepr(), "(5 xor x(i))")

        # LT
        # Parameter < Variable, Variable < Parameter
        op1 = b[i] < x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) < x(i))")
        op2 = x[i] < b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) < b(i))")

        # LE
        # Parameter <= Variable, Variable <= Parameter
        op1 = b[i] <= x[i]
        self.assertEqual(op1.gamsRepr(), "b(i) =l= x(i)")
        op2 = x[i] <= b[i]
        self.assertEqual(op2.gamsRepr(), "x(i) =l= b(i)")

        # GT
        # Parameter > Variable, Variable > Parameter
        op1 = b[i] > x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) > x(i))")
        op2 = x[i] > b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) > b(i))")

        # GE
        # Parameter >= Variable, Variable >= Parameter
        op1 = b[i] >= x[i]
        self.assertEqual(op1.gamsRepr(), "b(i) =g= x(i)")
        op2 = x[i] >= b[i]
        self.assertEqual(op2.gamsRepr(), "x(i) =g= b(i)")

        # NE
        # Parameter != Variable, Variable != Parameter
        op1 = b[i] != x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) ne x(i))")
        op2 = x[i] != b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) ne b(i))")

        # E
        # Parameter == Variable, Variable == Parameter
        op1 = b[i] == x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) = x(i))")
        op2 = x[i] == b[i]
        self.assertEqual(op2.gamsRepr(), "x(i) =e= b(i)")
        op3 = b[i] == b[i]
        self.assertEqual(op3.gamsRepr(), "(b(i) = b(i))")

        # not
        # not Parameter/Variable
        op1 = ~b[i]
        self.assertEqual(op1.gamsRepr(), "( not b(i))")
        op2 = ~x[i]
        self.assertEqual(op2.gamsRepr(), "( not x(i))")


def magics_suite():
    suite = unittest.TestSuite()
    tests = [
        MagicsSuite(name)
        for name in dir(MagicsSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(magics_suite())
