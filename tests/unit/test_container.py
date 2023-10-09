import unittest

import pandas as pd

import gamspy.utils as utils
from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Problem
from gamspy import Sense
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


class ContainerSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(delayed_execution=True)

    def test_container(self):
        import gams.transfer as gt

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
        m = Container(delayed_execution=True)

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
        m = Container(delayed_execution=True)
        i1 = m.addSet("i")
        self.assertRaises(ValueError, m.addSet, "i", i1)
        self.assertTrue(isinstance(i1, Set))
        i2 = m.addSet("i")
        self.assertTrue(id(i1) == id(i2))
        i3 = m.addSet("i", records=["new_record"], description="new desc")
        self.assertTrue(id(i1) == id(i3))
        self.assertRaises(ValueError, m.addSet, "i", [j])
        self.assertRaises(TypeError, m.addSet, "i", None, 5)

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
        self.assertRaises(TypeError, m.addParameter, "a", None, None, 5)

        v1 = m.addVariable("v")
        self.assertRaises(ValueError, m.addVariable, "v", "free", domain=i1)
        self.assertTrue(isinstance(v1, Variable))
        v2 = m.addVariable("v", description="blabla", records=pd.DataFrame())
        self.assertTrue(id(v1) == id(v2))
        self.assertRaises(ValueError, m.addVariable, "v", "free", ["*"])
        self.assertRaises(ValueError, m.addVariable, "v", "dayum")

        e1 = m.addEquation("e")
        self.assertRaises(ValueError, m.addEquation, "e", "regular", i1)
        self.assertTrue(isinstance(e1, Equation))
        e2 = m.addEquation("e")
        self.assertTrue(id(e1) == id(e2))
        self.assertRaises(ValueError, m.addEquation, "e", "bla")
        self.assertRaises(ValueError, m.addEquation, "e", "leq")
        e3 = m.addEquation("e", records=pd.DataFrame())
        self.assertTrue(id(e3) == id(e1))

    def test_read_write(self):
        m = Container(delayed_execution=True)
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
        new_container = Container(delayed_execution=True)
        i = Set(new_container, name="i")
        a = Parameter(new_container, name="a", domain=[i])
        new_container.loadRecordsFromGdx("test.gdx")

        # Set
        self.assertEqual(i.records.values.tolist(), [["i1", ""], ["i2", ""]])

        # Parameter
        self.assertEqual(a.records.values.tolist(), [["i1", 1.0], ["i2", 2.0]])

        # Load specific symbols
        new_container2 = Container(delayed_execution=True)
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

    def test_options(self):
        options = {
            "limRow": 0,
            "limCol": 0,
            "solPrint": "silent",
            "solver": "cplex",
            "solveLink": "%solveLink.loadLibrary%",
        }
        self.m.addOptions(options)
        self.assertTrue(len(self.m._unsaved_statements), 5)
        self.assertEqual(
            list(self.m._unsaved_statements.values())[0], "limRow = 0"
        )
        self.assertEqual(
            list(self.m._unsaved_statements.values())[1], "limCol = 0"
        )
        self.assertEqual(
            list(self.m._unsaved_statements.values())[2], "solPrint = silent"
        )
        self.assertEqual(
            list(self.m._unsaved_statements.values())[3], "solver = cplex"
        )
        self.assertEqual(
            list(self.m._unsaved_statements.values())[4],
            "solveLink = %solveLink.loadLibrary%",
        )

        options = {"bla": 0}
        self.assertRaises(ValueError, self.m.addOptions, options)

    def test_arbitrary_gams_code(self):
        self.m._addGamsCode("Set i / i1*i3 /;")
        self.assertEqual(
            list(self.m._unsaved_statements.values())[-1], "Set i / i1*i3 /;"
        )

        m = Container(delayed_execution=True)
        m._addGamsCode("scalar piHalf / [pi/2] /;")
        m._run()
        self.assertTrue("piHalf" in m.data.keys())
        self.assertEqual(m["piHalf"].records.values[0][0], 1.5707963267948966)

    def test_system_directory(self):
        import gamspy_base

        expected_path = gamspy_base.__path__[0]

        m = Container(delayed_execution=True)
        self.assertEqual(m.system_directory.lower(), expected_path.lower())

        self.assertEqual(
            utils._getGAMSPyBaseDirectory().lower(), expected_path.lower()
        )

    def test_non_empty_working_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            # (1) run model once to fill the working directory
            m = Container(delayed_execution=True, working_directory=td)
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
            m = Container(delayed_execution=True, working_directory=td)
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
        m = Container(delayed_execution=True)
        i = Set(m, name="i", records=["i1"])
        p1 = Parameter(m, name="p1", domain=[i], records=[["i1", 1]])
        p2 = Parameter(m, name="p2", domain=[i])
        p2[i] = p1[i]
        m.write("data.gdx")
        m = Container(delayed_execution=True, load_from="data.gdx")
        self.assertEqual(m["p2"].toList(), [("i1", 1.0)])


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
