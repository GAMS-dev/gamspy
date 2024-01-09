from __future__ import annotations

import os
import unittest

import gamspy as gp
from gamspy import Container
from gamspy import Parameter
from gamspy import Set
from gamspy import Variable


class SpecialValuesSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )

    def test_parameter_special_values(self):
        i = Set(self.m, "i", records=["i1", "i2"])

        # Test special values in parameter
        a = Parameter(self.m, "a", records=gp.SpecialValues.EPS)
        self.assertEqual(a.toValue(), -0.0)

        b = Parameter(self.m, "b", domain=[i])
        b[...] = gp.SpecialValues.EPS

        if self.m.delayed_execution:
            self.assertEqual(
                self.m._unsaved_statements[-1].gamsRepr(), "b(i) = EPS;"
            )

        b[...] = gp.SpecialValues.NA
        if self.m.delayed_execution:
            self.assertEqual(
                self.m._unsaved_statements[-1].gamsRepr(), "b(i) = NA;"
            )

        b[...] = gp.SpecialValues.UNDEF
        if self.m.delayed_execution:
            self.assertEqual(
                self.m._unsaved_statements[-1].gamsRepr(), "b(i) = UNDF;"
            )

        b[...] = gp.SpecialValues.POSINF
        if self.m.delayed_execution:
            self.assertEqual(
                self.m._unsaved_statements[-1].gamsRepr(), "b(i) = INF;"
            )

        b[...] = gp.SpecialValues.NEGINF
        if self.m.delayed_execution:
            self.assertEqual(
                self.m._unsaved_statements[-1].gamsRepr(), "b(i) = -INF;"
            )

    def test_implicit_parameter_special_values(self):
        i = Set(self.m, "i", records=["i1", "i2"])

        b = Variable(self.m, "b", domain=[i])
        b.l[...] = gp.SpecialValues.EPS
        self.assertEqual(
            self.m._unsaved_statements[-1].gamsRepr(), "b.l(i) = EPS;"
        )

        b.l[...] = gp.SpecialValues.NA
        self.assertEqual(
            self.m._unsaved_statements[-1].gamsRepr(), "b.l(i) = NA;"
        )

        b.l[...] = gp.SpecialValues.UNDEF
        self.assertEqual(
            self.m._unsaved_statements[-1].gamsRepr(), "b.l(i) = UNDF;"
        )

        b.l[...] = gp.SpecialValues.POSINF
        self.assertEqual(
            self.m._unsaved_statements[-1].gamsRepr(), "b.l(i) = INF;"
        )

        b.l[...] = gp.SpecialValues.NEGINF
        self.assertEqual(
            self.m._unsaved_statements[-1].gamsRepr(), "b.l(i) = -INF;"
        )

    def test_operation_special_values(self):
        tax = Set(self.m, "tax", records=["i1", "i2"])
        bla = Set(self.m, "bla", records=["x"])
        results = Parameter(self.m, "results", domain=[tax, bla])

        x = Variable(self.m, "x", domain=tax)
        e = Variable(self.m, "e", domain=tax)
        results[tax, "x"] = gp.math.Max(
            x.l[tax] - e.l[tax], gp.SpecialValues.EPS
        )
        self.assertEqual(
            self.m._unsaved_statements[-1].gamsRepr(),
            'results(tax,"x") = ( max((x.l(tax) - e.l(tax)),EPS) );',
        )


def special_values_suite():
    suite = unittest.TestSuite()
    tests = [
        SpecialValuesSuite(name)
        for name in dir(SpecialValuesSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(special_values_suite())
