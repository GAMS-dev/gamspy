from __future__ import annotations

import gc
import os
import unittest

import gamspy.utils as utils
import pandas as pd
import urllib3
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
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
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
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
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

    def test_str(self):
        self.assertEqual(str(self.m), f"<Empty Container ({hex(id(self.m))})>")

        _ = Set(self.m, "i")
        self.assertEqual(
            str(self.m),
            f"<Container ({hex(id(self.m))}) with {len(self.m)} symbols: {self.m.data.keys()}>",
        )

    def test_read_write(self):
        gdx_path = os.path.join("tmp", "test.gdx")

        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        _ = Set(m, "i", records=["i1", "i2"])
        _ = Set(m, "j", records=["j1", "j2"])
        m.write(gdx_path)

        _ = Set(self.m, name="k", records=["k1", "k2"])
        self.m.read(gdx_path, ["i"])
        self.assertEqual(list(self.m.data.keys()), ["k", "i"])

    def test_loadRecordsFromGdx(self):
        gdx_path = os.path.join("tmp", "test.gdx")

        i = Set(self.m, name="i", records=["i1", "i2"])
        a = Parameter(
            self.m, name="a", domain=[i], records=[("i1", 1), ("i2", 2)]
        )
        self.m.write(gdx_path)

        # Load all
        new_container = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        i = Set(new_container, name="i")
        a = Parameter(new_container, name="a", domain=[i])
        new_container.loadRecordsFromGdx(gdx_path)

        self.assertEqual(i.records.values.tolist(), [["i1", ""], ["i2", ""]])

        self.assertEqual(a.records.values.tolist(), [["i1", 1.0], ["i2", 2.0]])

        # Load specific symbols
        new_container2 = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        i = Set(new_container2, name="i")
        a = Parameter(new_container2, name="a", domain=[i])
        new_container2.loadRecordsFromGdx(gdx_path, ["i"])

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
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        i = Set(m, "i", records=["i1", "i2"])
        i["i1"] = False
        m.addGamsCode("scalar piHalf / [pi/2] /;")
        self.assertTrue("piHalf" in m.data)
        self.assertEqual(m["piHalf"].records.values[0][0], 1.5707963267948966)

    def test_add_gams_code_on_actual_models(self):
        links = {
            "LP": "https://gams.com/latest/gamslib_ml/trnsport.1",
            "MIP": "https://gams.com/latest/gamslib_ml/prodsch.9",
            "NLP": "https://gams.com/latest/gamslib_ml/weapons.18",
            "MCP": "https://gams.com/latest/gamslib_ml/wallmcp.127",
            "CNS": "https://gams.com/latest/gamslib_ml/camcns.209",
            "DNLP": "https://gams.com/latest/gamslib_ml/linear.23",
            "MINLP": "https://gams.com/latest/gamslib_ml/meanvarx.113",
            "QCP": "https://gams.com/latest/gamslib_ml/himmel11.95",
            "MIQCP": "https://gams.com/latest/gamslib_ml/qalan.282",
            "MPSGE": "https://gams.com/latest/gamslib_ml/hansmge.147",
        }

        for link in links.values():
            data = urllib3.request("GET", link).data.decode("utf-8")
            m = Container()
            m.addGamsCode(data)

    def test_system_directory(self):
        import gamspy_base

        expected_path = gamspy_base.__path__[0]

        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )

        if os.getenv("GAMSPY_GAMS_SYSDIR", None) is None:
            self.assertEqual(m.system_directory.lower(), expected_path.lower())

            self.assertEqual(
                utils._get_gamspy_base_directory().lower(),
                expected_path.lower(),
            )
        else:
            self.assertEqual(
                m.system_directory, os.environ["GAMSPY_GAMS_SYSDIR"]
            )

    def test_write_load_on_demand(self):
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        i = Set(m, name="i", records=["i1"])
        p1 = Parameter(m, name="p1", domain=[i], records=[["i1", 1]])
        p2 = Parameter(m, name="p2", domain=[i])
        p2[i] = p1[i]
        m.write(f"tmp{os.sep}data.gdx")
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
            load_from=f"tmp{os.sep}data.gdx",
        )
        self.assertEqual(m["p2"].toList(), [("i1", 1.0)])

    def test_copy(self):
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
            working_directory=f"tmp{os.sep}copy",
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

        if not m._network_license:
            self.assertRaises(ValidationError, m.copy, f"tmp{os.sep}copy")
            new_cont = m.copy(working_directory=f"tmp{os.sep}test")
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
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )

        i = Set(m, "i")
        _ = Alias(m, "a", i)
        _ = Parameter(m, "p")
        _ = Variable(m, "v")
        _ = Equation(m, "e")

        self.assertEqual(
            m.generateGamsString(),
            f"$onMultiR\n$onUNDF\nSet i(*);\n$gdxIn {m._gdx_in}\n$loadDC i\n$gdxIn\n$offUNDF\n$onMultiR\n$onUNDF\nAlias(i,a);\n$gdxIn {m._gdx_in}\n$loadDC i\n$gdxIn\n$offUNDF\n$onMultiR\n$onUNDF\nParameter p;\n$gdxIn {m._gdx_in}\n$loadDC p\n$gdxIn\n$offUNDF\n$onMultiR\n$onUNDF\nfree Variable v;\n$gdxIn {m._gdx_in}\n$loadDC v\n$gdxIn\n$offUNDF\n$onMultiR\n$onUNDF\nEquation e;\n$gdxIn {m._gdx_in}\n$loadDC e\n$gdxIn\n$offUNDF\n",
        )

        self.assertEqual(
            m.generateGamsString(show_raw=True),
            """Set i(*);
Alias(i,a);
Parameter p;
free Variable v;
Equation e;
""",
        )

    def test_removal_of_autogenerated_symbols(self):
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
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
            [
                "i",
                "j",
                "a",
                "b",
                "d",
                "c",
                "x",
                "supply",
                "demand",
                "transport_objective_variable",
                "transport_objective",
            ],
        )

    def test_write(self):
        gdx_path = os.path.join("tmp", "test.gdx")

        from gamspy import SpecialValues

        _ = Parameter(self.m, "a", records=SpecialValues.EPS)
        self.m.write(gdx_path, eps_to_zero=True)

        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
            load_from=gdx_path,
        )
        self.assertEqual(int(m["a"].toValue()), 0)

    def test_read(self):
        gdx_path = os.path.join("tmp", "test.gdx")

        _ = Parameter(self.m, "a", records=5)
        self.m.write(gdx_path)

        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        m.read(gdx_path, load_records=False)
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
