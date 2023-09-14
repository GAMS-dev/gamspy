import os
import unittest

import numpy as np
import pandas as pd

from gamspy import Alias
from gamspy import Container
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


class ParameterSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_parameter_string(self):
        canning_plants = pd.DataFrame(["seattle", "san-diego", "topeka"])

        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        a = Parameter(
            self.m,
            name="a",
            domain=[i],
            records=pd.DataFrame(
                [["seattle", 350], ["san-diego", 600], ["topeka", 500]]
            ),
            description="distances",
        )

        self.assertEqual(
            a.getStatement(),
            'Parameter a(i) "distances";',
        )

        b = Parameter(self.m, "b")
        self.assertEqual(b.getStatement(), "Parameter b;")
        self.assertEqual((b == 5).gamsRepr(), "(b = 5)")
        self.assertEqual((-b).name, "-b")

    def test_implicit_parameter_string(self):
        canning_plants = pd.DataFrame(["seattle", "san-diego", "topeka"])

        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        a = Parameter(
            self.m,
            name="a",
            domain=[i],
            records=pd.DataFrame(
                [["seattle", 350], ["san-diego", 600], ["topeka", 500]]
            ),
        )

        self.assertEqual(a[i].gamsRepr(), "a(i)")
        a[i].assign = a[i] * 5
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "a(i) = (a(i) * 5);",
        )

        a[i] = -a[i] * 5
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "a(i) = (-a(i) * 5);",
        )

    def test_implicit_parameter_assignment(self):
        canning_plants = pd.DataFrame(["seattle", "san-diego", "topeka"])

        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        a = Parameter(
            self.m,
            name="a",
            domain=[i],
            records=pd.DataFrame(
                [["seattle", 350], ["san-diego", 600], ["topeka", 500]]
            ),
        )

        b = Parameter(
            self.m,
            name="b",
            domain=[i],
            records=pd.DataFrame(
                [["seattle", 350], ["san-diego", 600], ["topeka", 500]]
            ),
        )

        a[i] = Sum(i, b[i])
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "a(i) = sum(i,b(i));",
        )

        v = Variable(self.m, "v", domain=[i])
        v.l[i] = v.l[i] * 5
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "v.l(i) = (v.l(i) * 5);",
        )

    def test_equality(self):
        j = Set(self.m, "j")
        h = Set(self.m, "h")
        hp = Alias(self.m, "hp", h)
        lamb = Parameter(self.m, "lambda", domain=[j, h])
        gamma = Parameter(self.m, "gamma", domain=[j, h])
        gamma[j, h] = Sum(hp.where[Ord(hp) >= Ord(h)], lamb[j, hp])
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "gamma(j,h) = sum(hp $ (ord(hp) >= ord(h)),lambda(j,hp));",
        )

    def test_override(self):
        # Parameter record override
        s = Set(self.m, name="s", records=[str(i) for i in range(1, 4)])
        c = Parameter(self.m, name="c", domain=[s])
        c = self.m.addParameter(
            name="c",
            domain=[s],
            records=[("1", 1), ("2", 2), ("3", 3)],
            description="new description",
        )
        self.assertEqual(c.description, "new description")

        # Try to add the same parameter
        self.assertRaises(ValueError, self.m.addParameter, "c", [s, s])

    def test_undef(self):
        m = Container()
        _ = Parameter(
            m, name="rho", records=[np.nan]
        )  # Instead of using numpy there might be a NA from the math package

        self.assertEqual(
            m.generateGamsString(),
            "$onUNDF\nParameter rho;\n$gdxLoad"
            f" {m.workspace.working_directory}{os.sep}default.gdx"
            " rho\n$offUNDF\n",
        )


def parameter_suite():
    suite = unittest.TestSuite()
    tests = [
        ParameterSuite(name)
        for name in dir(ParameterSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(parameter_suite())
