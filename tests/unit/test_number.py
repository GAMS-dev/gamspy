import unittest

from gamspy import Container, Set, Domain


class NumberSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_domain(self):
        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        domain = Domain(i, j)
        self.assertEqual(domain.gamsRepr(), "(i,j)")

        # Domain with less than two sets
        self.assertRaises(Exception, Domain, i)

        # Domain with no set or alias symbols
        self.assertRaises(Exception, Domain, "i", "j")


def number_suite():
    suite = unittest.TestSuite()
    tests = [
        NumberSuite(name)
        for name in dir(NumberSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(number_suite())
