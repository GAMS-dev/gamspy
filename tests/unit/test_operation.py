from __future__ import annotations

import os
import unittest

from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
    Ord,
    Parameter,
    Product,
    Set,
    Smax,
    Smin,
    Sum,
    Variable,
)
import pandas as pd

from gamspy.exceptions import ValidationError


class OperationSuite(unittest.TestCase):
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

    def test_operations(self):
        i = Set(
            self.m,
            name="i",
            records=self.canning_plants,
            description="Canning Plants",
        )
        j = Set(self.m, name="j", records=self.markets, description="Markets")

        # Params
        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        d = Parameter(self.m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # SUM
        # Operation with one index
        sum_op = Sum(j, x[i, j]) <= a[i]
        self.assertEqual(sum_op.gamsRepr(), "sum(j,x(i,j)) =l= a(i)")

        expression = Sum(i, True)
        self.assertEqual(expression.gamsRepr(), "sum(i,yes)")

        # Operation with two indices
        sum_op = Sum((i, j), c[i, j] * x[i, j]) == z
        self.assertEqual(
            sum_op.gamsRepr(), "sum((i,j),(c(i,j) * x(i,j))) =e= z"
        )

        # PROD
        # Operation with one index
        sum_op = Product(j, x[i, j]) <= a[i]
        self.assertEqual(sum_op.gamsRepr(), "prod(j,x(i,j)) =l= a(i)")

        # Operation with two indices
        sum_op = Product((i, j), c[i, j] * x[i, j]) == z
        self.assertEqual(
            sum_op.gamsRepr(), "prod((i,j),(c(i,j) * x(i,j))) =e= z"
        )

        # Smin
        # Operation with one index
        sum_op = Smin(j, x[i, j]) <= a[i]
        self.assertEqual(sum_op.gamsRepr(), "smin(j,x(i,j)) =l= a(i)")

        # Operation with two indices
        sum_op = Smin((i, j), c[i, j] * x[i, j]) == z
        self.assertEqual(
            sum_op.gamsRepr(), "smin((i,j),(c(i,j) * x(i,j))) =e= z"
        )

        # Smax
        # Operation with one index
        sum_op = Smax(j, x[i, j]) <= a[i]
        self.assertEqual(sum_op.gamsRepr(), "smax(j,x(i,j)) =l= a(i)")

        # Operation with two indices
        sum_op = Smax((i, j), c[i, j] * x[i, j]) == z
        self.assertEqual(
            sum_op.gamsRepr(), "smax((i,j),(c(i,j) * x(i,j))) =e= z"
        )

        # Ord, Card
        expression = Ord(i) == Ord(j)
        self.assertEqual(expression.gamsRepr(), "(ord(i) eq ord(j))")
        expression = Ord(i) != Ord(j)
        self.assertEqual(expression.gamsRepr(), "(ord(i) ne ord(j))")
        expression = Card(i) == 5
        self.assertEqual(expression.gamsRepr(), "(card(i) eq 5)")
        expression = Card(i) != 5
        self.assertEqual(expression.gamsRepr(), "(card(i) ne 5)")
        expression = Card(i) <= 5
        self.assertEqual(expression.gamsRepr(), "(card(i) <= 5)")
        expression = Card(i) >= 5
        self.assertEqual(expression.gamsRepr(), "(card(i) >= 5)")

        sum_op = Sum((i, j), c[i, j] * x[i, j])
        expression = sum_op != sum_op
        self.assertEqual(
            expression.gamsRepr(),
            "(sum((i,j),(c(i,j) * x(i,j))) ne sum((i,j),(c(i,j) * x(i,j))))",
        )

    def test_operation_indices(self):
        # Test operation index
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        mt = 2016
        mg = 17
        maxdt = 40
        t = Set(
            m,
            name="t",
            records=[f"t{i}" for i in range(1, mt + 1)],
            description="hours",
        )
        g = Set(
            m,
            name="g",
            records=[f"g{i}" for i in range(1, mg + 1)],
            description="generators",
        )
        t1 = Alias(m, name="t1", alias_with=t)
        tt = Set(
            m,
            name="tt",
            domain=[t],
            records=[f"t{i}" for i in range(1, maxdt + 1)],
            description="max downtime hours",
        )
        pMinDown = Parameter(
            m, name="pMinDown", domain=[g, t], description="minimum downtime"
        )
        vStart = Variable(m, name="vStart", type="binary", domain=[g, t])
        eStartFast = Equation(m, name="eStartFast", domain=[g, t])
        eStartFast[g, t1] = (
            Sum(
                tt[t].where[Ord(t) <= pMinDown[g, t1]],
                vStart[g, t.lead(Ord(t1) - pMinDown[g, t1])],
            )
            <= 1
        )
        self.assertEqual(
            eStartFast.getDefinition(),
            "eStartFast(g,t1) .. sum(tt(t) $ (ord(t) <="
            " pMinDown(g,t1)),vStart(g,t + (ord(t1) - pMinDown(g,t1))))"
            " =l= 1;",
        )

    def test_operation_overloads(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        c = Set(m, "c")
        s = Set(m, "s")
        a = Parameter(m, "a", domain=[c, s])
        p = Variable(m, "p", type="Positive", domain=c)

        # test neq
        profit = Equation(m, "profit", domain=s)
        profit[s] = -Sum(c, a[c, s] * p[c]) >= 0
        self.assertEqual(
            profit.getDefinition(),
            "profit(s) .. ( - sum(c,(a(c,s) * p(c)))) =g= 0;",
        )

        # test ne
        bla = Parameter(m, "bla", domain=s)
        bla2 = Parameter(m, "bla2", domain=s)
        bla[...] = bla2[...] != 0

        self.assertEqual(
            bla.getAssignment(),
            "bla(s) = (bla2(s) ne 0);",
        )

    def test_truth_value(self):
        i_list = [f"i{i}" for i in range(10)]
        i = Set(self.m, "i", records=i_list)
        j = Alias(self.m, "j", alias_with=i)
        x = Variable(self.m, "x", domain=[i, j])
        eq = Equation(self.m, "eq", domain=[i, j])

        with self.assertRaises(ValidationError):
            eq[i, j].where[
                (Ord(i) < Card(i)) and (Ord(j) > 1) and (Ord(j) < Card(j))
            ] = x[i, j] >= 1

        with self.assertRaises(ValidationError):
            if Card(i):
                ...

        with self.assertRaises(ValidationError):
            if i:
                ...

        with self.assertRaises(ValidationError):
            if j:
                ...

        a = Parameter(self.m, "a")
        with self.assertRaises(ValidationError):
            if a:
                ...

        v = Variable(self.m, "v")
        with self.assertRaises(ValidationError):
            if v:
                ...

        e = Equation(self.m, "e")
        with self.assertRaises(ValidationError):
            if e:
                ...

    def test_operation_no_index(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))
        c = Set(m, "c")
        s = Set(m, "s")
        a = Parameter(m, "a", domain=[c, s])
        p = Variable(m, "p", type="Positive", domain=c)

        self.assertRaises(ValidationError, lambda: -Sum([], a[c, s] * p[c]))

    def test_operation_scalar_domain_update(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))
        c = Set(m, name="c")
        s = Set(m, name="s")
        s2 = Alias(m, name="s2", alias_with=s)
        self.assertRaises(ValidationError, lambda: Sum([c], 5.2)[s2])
        expr1 = Sum([c], 5.2)[:]  # this should be fine
        self.assertEqual(expr1.gamsRepr(), "sum(c,5.2)")

    def test_operation_extract_vars(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))
        s = Set(m, name="s")
        c = Set(m, name="c")
        p = Variable(m, "p", type="Positive", domain=[s, c])

        expr1 = Sum(c, p[s, c])
        self.assertIn("p", expr1._extract_variables())

        expr2 = Sum(c, p)
        self.assertIn("p", expr2._extract_variables())

        expr3 = Sum(s, Sum(c, p))
        self.assertIn("p", expr3._extract_variables())

        expr4 = Sum(s, Sum(c, p[s, c]))
        self.assertIn("p", expr4._extract_variables())

        expr5 = Sum(s, 2)
        self.assertEqual(expr5._extract_variables(), [])


def operation_suite():
    suite = unittest.TestSuite()
    tests = [
        OperationSuite(name)
        for name in dir(OperationSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(operation_suite())
