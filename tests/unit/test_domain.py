from __future__ import annotations

import os
import unittest

from gamspy import Container
from gamspy import Domain
from gamspy import Parameter
from gamspy import Set
from gamspy.exceptions import ValidationError


class DomainSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
        )

    def test_domain(self):
        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        domain = Domain(i, j)
        self.assertEqual(domain.gamsRepr(), "(i,j)")

        # Domain with less than two sets
        self.assertRaises(ValidationError, Domain, i)

        # Domain with no set or alias symbols
        self.assertRaises(ValidationError, Domain, "i", "j")

    def test_domain_forwarding(self):
        m = Container(
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
        )
        i = Set(m, name="i")
        _ = Parameter(
            m,
            name="p",
            domain=[i],
            domain_forwarding=True,
            records=[["i1", 1]],
        )
        self.assertEqual(i.toList(), ["i1"])


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
