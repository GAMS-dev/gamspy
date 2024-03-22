from __future__ import annotations

import gc
import os
import unittest

import gamspy.utils as utils
import pandas as pd
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    UniverseAlias,
    Variable,
)
from gamspy.exceptions import GamspyException, ValidationError


class ContainerSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None)
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

    def test_container(self):
        import gams.transfer as gt

        with self.assertRaises(TypeError):
            m = Container(options={"bla": "bla"})

        i = gt.Set(self.m, "i")
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["i"], Set))

        j = gt.Alias(self.m, "j", i)
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["j"], Alias))

        a = gt.Parameter(self.m, "a")
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["a"], Parameter))

        v = gt.Variable(self.m, "v")
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["v"], Variable))

        e = gt.Equation(self.m, "e", type="eq")
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["e"], Equation))

        # Test getters
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

        i = Set(m, "i")
        self.assertTrue(isinstance(m["i"], Set))

        j = Alias(m, "j", i)
        self.assertTrue(isinstance(m["j"], Alias))

        a = Parameter(m, "a")
        self.assertTrue(isinstance(m["a"], Parameter))

        v = Variable(m, "v")
        self.assertTrue(isinstance(m["v"], Variable))

        e = Equation(m, "e")
        self.assertTrue(isinstance(m["e"], Equation))

        self.assertEqual(m.getSets(), [i])
        self.assertEqual(m.getAliases(), [j])
        self.assertEqual(m.getParameters(), [a])
        self.assertEqual(m.getVariables(), [v])
        self.assertEqual(m.getEquations(), [e])

        # test addX syntax
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        i1 = m.addSet("i")
        self.assertRaises(ValueError, m.addSet, "i", i1)
        self.assertTrue(isinstance(i1, Set))
        i2 = m.addSet("i")
        self.assertTrue(id(i1) == id(i2))
        i3 = m.addSet("i", records=["new_record"], description="new desc")
        self.assertTrue(id(i1) == id(i3))
        self.assertRaises(ValueError, m.addSet, "i", [j])
        self.assertRaises(ValueError, m.addSet, "i", None, 5)

        j1 = m.addAlias("j", i1)
        self.assertTrue(isinstance(j1, Alias))
        j2 = m.addAlias("j", i1)
        self.assertTrue(id(j1) == id(j2))
        j3 = m.addAlias("j", j2)
        self.assertTrue(id(j3) == id(j2))

        a1 = m.addParameter("a")
        self.assertRaises(ValueError, m.addParameter, "a", i1)
        self.assertTrue(isinstance(a1, Parameter))
        a2 = m.addParameter("a")
        self.assertTrue(id(a1) == id(a2))
        self.assertRaises(ValueError, m.addParameter, "a", ["*"])
        self.assertRaises(ValueError, m.addParameter, "a", None, None, 5)

        v1 = m.addVariable("v")
        self.assertRaises(ValueError, m.addVariable, "v", "free", domain=i1)
        self.assertTrue(isinstance(v1, Variable))
        v2 = m.addVariable("v", description="blabla", records=pd.DataFrame())
        self.assertTrue(id(v1) == id(v2))
        self.assertRaises(ValueError, m.addVariable, "v", "free", ["*"])
        self.assertRaises(TypeError, m.addVariable, "v", "dayum")

        e1 = m.addEquation("e")
        self.assertRaises(ValueError, m.addEquation, "e", "regular", i1)
        self.assertTrue(isinstance(e1, Equation))
        e2 = m.addEquation("e")
        self.assertTrue(id(e1) == id(e2))
        self.assertRaises(ValueError, m.addEquation, "e", "bla")
        self.assertRaises(TypeError, m.addEquation, "e", "leq")
        e3 = m.addEquation("e", records=pd.DataFrame())
        self.assertTrue(id(e3) == id(e1))

    def test_read_write(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        _ = Set(m, "i", records=["i1", "i2"])
        _ = Set(m, "j", records=["j1", "j2"])
        m.write("test.gdx")

        _ = Set(self.m, name="k", records=["k1", "k2"])
        self.m.read("test.gdx", ["i"])
        self.assertEqual(list(self.m.data.keys()), ["k", "i"])

    def test_loadRecordsFromGdx(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        a = Parameter(
            self.m, name="a", domain=[i], records=[("i1", 1), ("i2", 2)]
        )
        self.m.write("test.gdx")

        # Load all
        new_container = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        i = Set(new_container, name="i")
        a = Parameter(new_container, name="a", domain=[i])
        new_container.loadRecordsFromGdx("test.gdx")

        self.assertEqual(i.records.values.tolist(), [["i1", ""], ["i2", ""]])

        self.assertEqual(a.records.values.tolist(), [["i1", 1.0], ["i2", 2.0]])

        # Load specific symbols
        new_container2 = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        i = Set(new_container2, name="i")
        a = Parameter(new_container2, name="a", domain=[i])
        new_container2.loadRecordsFromGdx("test.gdx", ["i"])

        self.assertEqual(i.records.values.tolist(), [["i1", ""], ["i2", ""]])
        self.assertIsNone(a.records)

    def test_enums(self):
        self.assertEqual(str(Problem.LP), "LP")
        self.assertEqual(str(Sense.MAX), "MAX")

        self.assertEqual(
            Problem.values(),
            [
                "LP",
                "NLP",
                "QCP",
                "DNLP",
                "MIP",
                "RMIP",
                "MINLP",
                "RMINLP",
                "MIQCP",
                "RMIQCP",
                "MCP",
                "CNS",
                "MPEC",
                "RMPEC",
                "EMP",
                "MPSGE",
            ],
        )

        self.assertEqual(Sense.values(), ["MIN", "MAX", "FEASIBILITY"])

    def test_arbitrary_gams_code(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        i = Set(m, "i", records=["i1", "i2"])
        i["i1"] = False
        m._addGamsCode("scalar piHalf / [pi/2] /;", import_symbols=["piHalf"])
        self.assertTrue("piHalf" in m.data)
        self.assertEqual(m["piHalf"].records.values[0][0], 1.5707963267948966)

        pi = Parameter(m, "pi")
        with self.assertRaises(ValidationError):
            m._addGamsCode("scalar pi / pi /;", import_symbols=[pi])

    def test_system_directory(self):
        import gamspy_base

        expected_path = gamspy_base.__path__[0]

        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

        if os.getenv("SYSTEM_DIRECTORY", None) is None:
            self.assertEqual(m.system_directory.lower(), expected_path.lower())

            self.assertEqual(
                utils._get_gamspy_base_directory().lower(),
                expected_path.lower(),
            )

    def test_non_empty_working_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            # (1) run model once to fill the working directory
            m = Container(
                system_directory=os.getenv("SYSTEM_DIRECTORY", None),
                working_directory=td,
            )
            i = Set(m, name="i", records=["i1"])
            x = Variable(m, name="x", type="Positive")

            dummy = Model(
                m,
                name="dummy",
                equations=[],
                problem="LP",
                sense=Sense.MIN,
                objective=Sum((i,), x),
            )
            dummy.solve()

            # (2) make sure to not use default_restart.g00 generated by (1)
            m = Container(
                system_directory=os.getenv("SYSTEM_DIRECTORY", None),
                working_directory=td,
            )
            i = Set(m, name="i", records=["i1"])
            x = Variable(m, name="x", type="Positive")

            dummy = Model(
                m,
                name="dummy",
                equations=[],
                problem="LP",
                sense=Sense.MIN,
                objective=Sum((i,), x),
            )
            dummy.solve()

    def test_write_load_on_demand(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        i = Set(m, name="i", records=["i1"])
        p1 = Parameter(m, name="p1", domain=[i], records=[["i1", 1]])
        p2 = Parameter(m, name="p2", domain=[i])
        p2[i] = p1[i]
        m.write("data.gdx")
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            load_from="data.gdx",
        )
        self.assertEqual(m["p2"].toList(), [("i1", 1.0)])

    def test_copy(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            working_directory=".",
        )

        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])
        _ = Alias(m, "k", alias_with=j)
        _ = UniverseAlias(m)

        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(m, name="x", domain=[i, j], type="Positive")

        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        self.assertRaises(ValidationError, m.copy, ".")
        new_cont = m.copy(working_directory="test")
        self.assertEqual(m.data.keys(), new_cont.data.keys())

        transport = Model(
            new_cont,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        transport.solve()

        self.assertIsNotNone(new_cont.gamsJobName())
        self.assertAlmostEqual(transport.objective_value, 153.675, 3)

    def test_generate_gams_string(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

        i = Set(m, "i")
        _ = Alias(m, "a", i)
        _ = Parameter(m, "p")
        _ = Variable(m, "v")
        _ = Equation(m, "e")

        self.assertEqual(
            m.generateGamsString(),
            f"$onMultiR\n$onUNDF\n$gdxIn {m._gdx_in}\nSet i(*);\n$loadDC i\n$offUNDF\n$gdxIn\n$onMultiR\n$onUNDF\n$gdxIn {m._gdx_in}\nAlias(i,a);\n$loadDC i\n$offUNDF\n$gdxIn\n$onMultiR\n$onUNDF\n$gdxIn {m._gdx_in}\nParameter p;\n$loadDC p\n$offUNDF\n$gdxIn\n$onMultiR\n$onUNDF\n$gdxIn {m._gdx_in}\nfree Variable v;\n$loadDC v\n$offUNDF\n$gdxIn\n$onMultiR\n$onUNDF\n$gdxIn {m._gdx_in}\nEquation e;\n$loadDC e\n$offUNDF\n$gdxIn\n",
        )

    def test_removal_of_autogenerated_symbols(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(m, name="x", domain=[i, j], type="Positive")

        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            m,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        transport.solve()
        self.assertEqual(
            list(m.data.keys()),
            ["i", "j", "a", "b", "d", "c", "x", "supply", "demand"],
        )

    def test_write(self):
        from gamspy import SpecialValues

        _ = Parameter(self.m, "a", records=SpecialValues.EPS)
        self.m.write("test.gdx", eps_to_zero=True)

        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            load_from="test.gdx",
        )
        self.assertEqual(int(m["a"].toValue()), 0)

    def test_read(self):
        _ = Parameter(self.m, "a", records=5)
        self.m.write("test.gdx")

        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        m.read("test.gdx", load_records=False)
        self.assertIsNone(m["a"].records, None)

    def test_debugging_level(self):
        from gamspy.math import sqrt

        with self.assertRaises(ValidationError):
            _ = Container(debugging_level="wrong_level")

        global working_directory

        def test_delete_success():
            global working_directory
            m = Container(debugging_level="delete")
            working_directory = m.working_directory
            _ = Equation(m, "e")

        test_delete_success()
        gc.collect()
        self.assertFalse(os.path.exists(working_directory))

        def test_delete_err():
            global working_directory
            m = Container(debugging_level="delete")
            working_directory = m.working_directory
            e = Equation(m, "e")
            with self.assertRaises(GamspyException):
                e[:] = sqrt(e) == 5

        test_delete_err()
        gc.collect()
        self.assertFalse(os.path.exists(working_directory))

        def test_keep_success():
            m = Container(debugging_level="keep")
            global working_directory
            working_directory = m.working_directory
            _ = Equation(m, "e")

        test_keep_success()
        gc.collect()
        self.assertTrue(os.path.exists(working_directory))

        def test_keep_err():
            m = Container(debugging_level="keep")
            global working_directory
            working_directory = m.working_directory
            e = Equation(m, "e")
            with self.assertRaises(GamspyException):
                e[:] = sqrt(e) == 5

        test_keep_err()
        gc.collect()
        self.assertTrue(os.path.exists(working_directory))

        def test_keep_on_error_success():
            m = Container(debugging_level="keep_on_error")
            global working_directory
            working_directory = m.working_directory
            _ = Equation(m, "e")

        test_keep_on_error_success()
        gc.collect()
        self.assertFalse(os.path.exists(working_directory))

        def test_keep_on_error_err():
            m = Container(debugging_level="keep_on_error")
            global working_directory
            working_directory = m.working_directory
            e = Equation(m, "e")
            with self.assertRaises(GamspyException):
                e[:] = sqrt(e) == 5

        test_keep_on_error_err()
        gc.collect()
        self.assertTrue(os.path.exists(working_directory))


def container_suite():
    suite = unittest.TestSuite()
    tests = [
        ContainerSuite(name)
        for name in dir(ContainerSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(container_suite())
