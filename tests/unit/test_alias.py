import unittest

from gamspy import (
    Container,
    Set,
    Alias,
    Equation,
)


class AliasSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

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

    def test_override(self):
        # Try to add the same Alias with non-Set alias_with
        u = Set(self.m, "u")
        v = Alias(self.m, "v", alias_with=u)
        eq = Equation(self.m, "eq", domain=[u, v])
        self.assertRaises(TypeError, self.m.addAlias, "v", eq)

        # Try to add the same alias
        self.assertRaises(ValueError, self.m.addAlias, "u", u)


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
