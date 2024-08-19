from __future__ import annotations

import unittest

from gamspy import Container, Domain, Parameter, Set, Sum
from gamspy.exceptions import GamspyException, ValidationError


class DomainSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_domain(self):
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        domain = Domain(i, j)
        self.assertEqual(domain.gamsRepr(), "(i,j)")

        # Domain with less than two sets
        self.assertRaises(ValidationError, Domain, i)

        # Domain with no set or alias symbols
        self.assertRaises(ValidationError, Domain, "i", "j")

    def test_domain_forwarding(self):
        i = Set(self.m, name="i")
        _ = Parameter(
            self.m,
            name="p",
            domain=[i],
            domain_forwarding=True,
            records=[["i1", 1]],
        )
        self.assertEqual(i.toList(), ["i1"])

        k = Set(self.m, name="k")
        j = Set(self.m, name="j")
        _ = Parameter(
            self.m,
            name="p2",
            domain=[k, j],
            domain_forwarding=[True, True],
            records=[["k1", "j1", 1]],
        )
        self.assertEqual(k.toList(), ["k1"])
        self.assertEqual(j.toList(), ["j1"])

        k2 = Set(self.m, name="k2")
        j2 = Set(self.m, name="j2")
        _ = Set(
            self.m,
            name="p3",
            domain=[k2, j2],
            domain_forwarding=[True, True],
            records=[("k2", "j2")],
        )
        self.assertEqual(k2.toList(), ["k2"])
        self.assertEqual(j2.toList(), ["j2"])

    def test_domain_validation(self):
        times = Set(self.m, "times", records=["release", "duration"])
        job = Set(self.m, "job", records=["job1", "job2"])
        data = Parameter(self.m, "data", domain=[times, job])

        M = self.m.addParameter("M")
        M[...] = Sum(job, data["release", job] + data["duration", job])
        with self.assertRaises(ValidationError):
            M[...] = Sum(job, data["rbla", job] + data["bla", job])

        job2 = Set(self.m, "job2", records=["job1", "job2"])
        data2 = Parameter(self.m, "data2", domain=["times", "job"])

        M2 = self.m.addParameter("M2")
        M2[...] = Sum(job2, data2["release", job2] + data2["duration", job2])

        with self.assertRaises(GamspyException):
            M[...] = Sum(job2, data2["rbla", job2] + data2["bla", job2])


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
