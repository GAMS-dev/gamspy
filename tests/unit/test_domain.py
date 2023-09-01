import unittest

from gamspy import Container, Set, Domain
from gamspy._algebra.domain import DomainException


class DomainSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_domain(self):
        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        domain = Domain(i, j)
        self.assertEqual(domain.gamsRepr(), "(i,j)")

        # Domain with less than two sets
        self.assertRaises(DomainException, Domain, i)

        # Domain with no set or alias symbols
        self.assertRaises(DomainException, Domain, "i", "j")


def domain_suite():
    suite = unittest.TestSuite()
    tests = [
        DomainSuite(name)
        for name in dir(DomainSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(domain_suite())
