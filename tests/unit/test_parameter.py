from __future__ import annotations

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
from gamspy.exceptions import ValidationError


class ParameterSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            delayed_execution=os.getenv("DELAYED_EXECUTION", False)
        )

    def test_parameter_creation(self):
        # no name
        self.assertRaises(TypeError, Parameter, self.m)

        # non-str type name
        self.assertRaises(TypeError, Parameter, self.m, 5)

        # no container
        self.assertRaises(TypeError, Parameter)

        # non-container type container
        self.assertRaises(TypeError, Parameter, 5, "j")

        # try to create a symbol with same name but different type
        _ = Set(self.m, "i")
        self.assertRaises(TypeError, Parameter, self.m, "i")

        # get already created symbol
        j1 = Parameter(self.m, "j")
        j2 = Parameter(self.m, "j")
        self.assertEqual(id(j1), id(j2))

        # Parameter and domain containers are different
        m = Container(
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
        )
        set1 = Set(self.m, "set1")
        with self.assertRaises(ValidationError):
            _ = Parameter(m, "param1", domain=[set1])

    def test_parameter_string(self):
        canning_plants = pd.DataFrame(["seattle", "san-diego", "topeka"])

        # Check if the name is reserved
        self.assertRaises(ValidationError, Parameter, self.m, "set")

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
        self.assertEqual((b == 5).gamsRepr(), "(b eq 5)")
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

        a[i] = -a[i] * 5
        self.assertEqual(
            self.m._unsaved_statements[-1].gamsRepr(),
            "a(i) = (-a(i) * 5);",
        )

        cont = Container(delayed_execution=False, working_directory=".")

        s = Set(cont, "s")
        m = Set(cont, "m")
        A = Parameter(cont, "A", domain=[s, m])

        A.domain = ["s", "m"]
        self.assertEqual(A.getStatement(), "Parameter A(*,*);")

    def test_parameter_assignment(self):
        m = Container(
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
        )

        i = Set(self.m, "i")
        j = Set(m, "j")
        a = Parameter(self.m, "a", domain=[i])

        with self.assertRaises(ValidationError):
            a[j] = 5

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
            self.m._unsaved_statements[-1].getStatement(),
            "a(i) = sum(i,b(i));",
        )

        v = Variable(self.m, "v", domain=[i])
        v.l[i] = v.l[i] * 5
        self.assertEqual(
            self.m._unsaved_statements[-1].getStatement(),
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
            self.m._unsaved_statements[-1].gamsRepr(),
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
        m = Container(
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
        )
        _ = Parameter(
            m, name="rho", records=[np.nan]
        )  # Instead of using numpy there might be a NA from the math package

        self.assertEqual(
            m.generateGamsString(),
            "$onMultiR\n$onUNDF\n$gdxIn"
            f" {m._gdx_in}\nParameter"
            " rho;\n$load rho\n"
            "$offUNDF\n$gdxIn\n"
            f"execute_unload '{m._gdx_out}' \n",
        )

    def test_assignment_dimensionality(self):
        j1 = Set(self.m, "j1")
        j2 = Set(self.m, "j2")
        j3 = Parameter(self.m, "j3", domain=[j1, j2])
        with self.assertRaises(ValidationError):
            j3["bla"] = 5

        j4 = Set(self.m, "j4")

        with self.assertRaises(ValidationError):
            j3[j1, j2, j4] = 5

        with self.assertRaises(ValidationError):
            j3[j1, j2] = j3[j1, j2, j4] * 5

    def test_domain_verification(self):
        m = Container()
        i1 = Set(m, "i1", records=["i1", "i2"])
        a1 = Parameter(m, "a1", domain=i1, records=[("i1", 1), ("i2", 2)])
        a1["i1"] = 5

        with self.assertRaises(ValidationError):
            a1["i3"] = 5

        with self.assertRaises(ValidationError):
            a1["i3"] = a1["i3"] * 5


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
