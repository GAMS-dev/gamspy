from __future__ import annotations

import os
import unittest

import numpy as np
import pandas as pd
from gamspy import Alias, Container, Ord, Parameter, Set, Sum, Variable
from gamspy.exceptions import GamspyException, ValidationError


class ParameterSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None)
        )
        self.canning_plants = ["seattle", "san-diego"]
        self.markets = ["new-york", "chicago", "topeka"]
        self.distances = [
            ["seattle", "new-york", 2.5],
            ["seattle", "chicago", 1.7],
            ["seattle", "topeka", 1.8],
            ["san-diego", "new-york", 2.5],
            ["san-diego", "chicago", 1.8],
            ["san-diego", "topeka", 1.4],
        ]
        self.capacities = [["seattle", 350], ["san-diego", 600]]
        self.demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

    def test_parameter_creation(self):
        # no name is fine
        a = Parameter(self.m)
        with self.assertRaises(ValidationError):
            _ = a.getAssignment()

        # non-str type name
        self.assertRaises(TypeError, Parameter, self.m, 5)

        # no container
        self.assertRaises((TypeError, ValidationError), Parameter)

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
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        set1 = Set(self.m, "set1")
        with self.assertRaises(ValidationError):
            _ = Parameter(m, "param1", domain=[set1])

    def test_parameter_string(self):
        # Check if the name is reserved
        self.assertRaises(ValidationError, Parameter, self.m, "set")

        i = Set(
            self.m,
            name="i",
            records=self.canning_plants,
            description="Canning Plants",
        )
        a = Parameter(
            self.m,
            name="a",
            domain=[i],
            records=self.capacities,
            description="capacities",
        )

        self.assertEqual(
            a.getDeclaration(),
            'Parameter a(i) "capacities";',
        )

        b = Parameter(self.m, "b")
        self.assertEqual(b.getDeclaration(), "Parameter b;")
        self.assertEqual((b == 5).gamsRepr(), "(b eq 5)")
        self.assertEqual((-b).getDeclaration(), "( - b)")

    def test_implicit_parameter_string(self):
        m = Container()

        i = Set(
            m,
            name="i",
            records=self.canning_plants,
            description="Canning Plants",
        )
        a = Parameter(
            m,
            name="a",
            domain=[i],
            records=self.capacities,
        )

        self.assertEqual(a[i].gamsRepr(), "a(i)")

        a[i] = -a[i] * 5

        self.assertEqual(
            a.getAssignment(),
            "a(i) = (( - a(i)) * 5);",
        )

        cont = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )

        s = Set(cont, "s")
        m = Set(cont, "m")
        A = Parameter(cont, "A", domain=[s, m])

        A.domain = ["s", "m"]
        self.assertEqual(A.getDeclaration(), "Parameter A(*,*);")

    def test_parameter_assignment(self):
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )

        i = Set(self.m, "i")
        j = Set(m, "j")
        a = Parameter(self.m, "a", domain=[i])

        with self.assertRaises(ValidationError):
            a[j] = 5

    def test_implicit_parameter_assignment(self):
        m = Container()
        i = Set(
            m,
            name="i",
            records=self.canning_plants,
            description="Canning Plants",
        )
        a = Parameter(
            m,
            name="a",
            domain=[i],
            records=self.capacities,
        )

        b = Parameter(
            m,
            name="b",
            domain=[i],
            records=self.capacities,
        )

        a[i] = b[i]
        self.assertEqual(
            a.getAssignment(),
            "a(i) = b(i);",
        )

        v = Variable(m, "v", domain=[i])
        v.l[i] = v.l[i] * 5

        self.assertEqual(
            v.getAssignment(),
            "v.l(i) = (v.l(i) * 5);",
        )

    def test_equality(self):
        m = Container()
        j = Set(m, "j")
        h = Set(m, "h")
        hp = Alias(m, "hp", h)
        lamb = Parameter(m, "lambda", domain=[j, h])
        gamma = Parameter(m, "gamma", domain=[j, h])
        gamma[j, h] = Sum(hp.where[Ord(hp) >= Ord(h)], lamb[j, hp])
        self.assertEqual(
            gamma.getAssignment(),
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
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        _ = Parameter(
            m, name="rho", records=[np.nan]
        )  # Instead of using numpy there might be a NA from the math package

        self.assertEqual(
            m.generateGamsString(),
            "$onMultiR\n$onUNDF\nParameter"
            f" rho;\n$gdxIn {m._gdx_in}\n$loadDC rho\n$gdxIn\n$offUNDF\n",
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
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        i1 = Set(m, "i1", records=["i1", "i2"])
        a1 = Parameter(m, "a1", domain=i1, records=[("i1", 1), ("i2", 2)])
        a1["i1"] = 5

        with self.assertRaises(ValidationError):
            a1["i3"] = 5

        with self.assertRaises(ValidationError):
            a1["i3"] = a1["i3"] * 5

    def test_uels_on_axes(self):
        s = pd.Series(index=["a", "b", "c"], data=[i + 1 for i in range(3)])
        i = Parameter(self.m, "i", ["*"], records=s, uels_on_axes=True)
        self.assertEqual(i.records.value.tolist(), [1, 2, 3])

    def test_domain_violation(self):
        col = Set(
            self.m, "col", records=[("col" + str(i), i) for i in range(1, 10)]
        )
        row = Set(
            self.m, "row", records=[("row" + str(i), i) for i in range(1, 10)]
        )

        initial_state_data = pd.DataFrame(
            [
                [0, 0, 0, 0, 8, 6, 0, 0, 0],
                [0, 7, 0, 9, 0, 2, 0, 0, 0],
                [6, 9, 0, 0, 0, 0, 2, 0, 8],
                [8, 0, 0, 0, 9, 0, 7, 0, 2],
                [4, 0, 0, 0, 0, 0, 0, 0, 3],
                [2, 0, 9, 0, 1, 0, 0, 0, 4],
                [5, 0, 3, 0, 0, 0, 0, 7, 6],
                [0, 0, 0, 5, 0, 8, 0, 2, 0],
                [0, 0, 0, 3, 7, 0, 0, 0, 0],
            ],
            index=["roj" + str(i) for i in range(1, 10)],
            columns=["col" + str(i) for i in range(1, 10)],
        )

        with self.assertRaises(GamspyException):
            _ = Parameter(
                self.m,
                "initial_state",
                domain=[row, col],
                records=initial_state_data,
                uels_on_axes=True,
            )

    def test_expert_sync(self):
        i_list = [str(i) for i in range(4)]
        m = Container()
        i = Set(m, "i", records=i_list)
        f = Parameter(m, "f", domain=i)
        f["0"] = 0
        f["1"] = 1

        f.synchronize = False
        for n in range(2, 4):
            f[str(n)] = f[str(n - 2)] + f[str(n - 1)]

        self.assertEqual(f.records.value.tolist(), [1.0])
        f.synchronize = True
        self.assertEqual(f.records.value.tolist(), [1.0, 1.0, 2.0])

    def test_control_domain(self):
        i = Set(self.m, "i", records=["i1", "i2"])
        j = Set(self.m, "j", records=["j1", "j2"])

        a = Parameter(self.m, "a", domain=i)
        b = Parameter(self.m, "b", domain=j, records=[("j1", 1), ("j2", 2)])

        with self.assertRaises(ValidationError):
            a[i] = b[j]

        with self.assertRaises(ValidationError):
            a[i.lead(1)] = b[j]

        with self.assertRaises(ValidationError):
            a[i] = b[j.lead(1)]

        with self.assertRaises(ValidationError):
            a[i.lead(1)] = b[j.lead(1)]


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
