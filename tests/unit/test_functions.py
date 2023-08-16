import unittest

from gamspy.functions import ifthen
from gamspy import (
    Container,
    Equation,
    Set,
    Variable,
)


class FunctionsSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_functions(self):
        m = Container()

        o = Set(m, "o", records=[f"pos{idx}" for idx in range(1, 11)])
        p = Set(m, "p", records=[f"opt{idx}" for idx in range(1, 6)])
        sumc = Variable(m, "sumc", domain=[o, p])
        op = Variable(m, "op", domain=[o, p])
        defopLS = Equation(m, "defopLS", domain=[o, p])
        defopLS[o, p] = op[o, p] == ifthen(sumc[o, p] >= 0.5, 1, 0)
        self.assertEqual(
            list(m._statements_dict.values())[-1].gamsRepr(),
            "defopLS(o,p) .. op(o,p) =e= (ifthen(sumc(o,p) >= 0.5, 1, 0)  );",
        )


def functions_suite():
    suite = unittest.TestSuite()
    tests = [
        FunctionsSuite(name)
        for name in dir(FunctionsSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(functions_suite())
