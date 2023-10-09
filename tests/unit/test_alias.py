import unittest

from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Parameter
from gamspy import Set
from gamspy import UniverseAlias
from gamspy.exceptions import GamspyException


class AliasSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(delayed_execution=True)

    def test_alias_string(self):
        # Set and Alias without domain
        i = Set(self.m, name="i", records=["a", "b", "c"])
        j = Alias(self.m, name="j", alias_with=i)
        self.assertEqual(j.gamsRepr(), "j")
        self.assertEqual(j.getStatement(), "Alias(i,j);")

        # Set and Alias with domain
        k = Set(self.m, name="k", domain=[i], records=["a", "b"])
        m = Alias(self.m, name="m", alias_with=k)
        self.assertEqual(m.gamsRepr(), "m")
        self.assertEqual(m.getStatement(), "Alias(k,m);")

        # Check if the name is reserved
        self.assertRaises(GamspyException, Alias, self.m, "set", i)

    def test_override(self):
        # Try to add the same Alias with non-Set alias_with
        u = Set(self.m, "u")
        v = Alias(self.m, "v", alias_with=u)
        eq = Equation(self.m, "eq", domain=[u, v])
        self.assertRaises(TypeError, self.m.addAlias, "v", eq)

        # Try to add the same alias
        self.assertRaises(ValueError, self.m.addAlias, "u", u)

    def test_universe_alias(self):
        h = UniverseAlias(self.m, "h")
        self.assertEqual(len(h), 0)
        _ = Set(self.m, "i", records=["i1", "i2"])
        _ = Set(self.m, "j", records=["j1", "j2"])

        self.assertEqual(
            h.records.values.tolist(), [["i1"], ["i2"], ["j1"], ["j2"]]
        )

        self.assertEqual(len(h), 4)

        self.m.write("test.gdx")

        bla = Container(delayed_execution=True)
        bla.read("test.gdx")
        self.assertEqual(
            bla.data["h"].records.values.tolist(), h.records.values.tolist()
        )

        m = Container(delayed_execution=True)
        _ = Set(m, name="i", records=["i1"])
        _ = Set(m, name="j", records=["j1"])
        p = Parameter(m, name="p", domain=["*"])
        universe1 = m.addUniverseAlias("universe")
        p[universe1] = 5

        self.assertEqual(
            list(m._unsaved_statements.values())[-1].getStatement(),
            "p(*) = 5;",
        )

        # Try to add it second time
        universe2 = m.addUniverseAlias("universe")
        self.assertEqual(id(universe1), id(universe2))

        # Try to add universe alias with existing name
        bla = Set(m, name="bla")
        self.assertRaises(ValueError, m.addUniverseAlias, "bla")


def alias_suite():
    suite = unittest.TestSuite()
    tests = [
        AliasSuite(name)
        for name in dir(AliasSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(alias_suite())
