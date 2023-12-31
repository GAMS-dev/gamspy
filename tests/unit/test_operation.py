from __future__ import annotations

import os
import unittest

import pandas as pd

from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Ord
from gamspy import Parameter
from gamspy import Product
from gamspy import Set
from gamspy import Smax
from gamspy import Smin
from gamspy import Sum
from gamspy import Variable


class OperationSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            delayed_execution=os.getenv("DELAYED_EXECUTION", False)
        )

    def test_operations(self):
        # Prepare data
        distances = pd.DataFrame(
            [
                ["seattle", "new-york", 2.5],
                ["seattle", "chicago", 1.7],
                ["seattle", "topeka", 1.8],
                ["san-diego", "new-york", 2.5],
                ["san-diego", "chicago", 1.8],
                ["san-diego", "topeka", 1.4],
            ]
        )
        canning_plants = ["seattle", "san-diego"]
        markets = ["new-york", "chicago", "topeka"]
        capacities = pd.DataFrame([["seattle", 350], ["san-diego", 600]])

        # Sets
        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        j = Set(self.m, name="j", records=markets, description="Markets")

        # Params
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variables
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # SUM
        # Operation with one index
        sum_op = Sum(j, x[i, j]) <= a[i]
        self.assertEqual(sum_op.gamsRepr(), "sum(j,x(i,j)) =l= a(i)")

        expression = Sum(a[i], True)
        self.assertEqual(expression.gamsRepr(), "sum(a(i),yes)")

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

    def test_operation_indices(self):
        # Test operation index
        m = Container(
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
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
            eStartFast._definition.gamsRepr(),
            "eStartFast(g,t1) .. sum(tt(t) $ (ord(t) <="
            " pMinDown(g,t1)),vStart(g,t + (ord(t1) - pMinDown(g,t1))))"
            " =l= 1;",
        )

    def test_operation_overloads(self):
        m = Container(delayed_execution=os.getenv("DELAYED_EXECUTION", False))
        c = Set(m, "c")
        s = Set(m, "s")
        a = Parameter(m, "a", domain=[c, s])
        p = Variable(m, "p", type="Positive", domain=c)

        # test neq
        profit = Equation(m, "profit", domain=s)
        profit[s] = -Sum(c, a[c, s] * p[c]) >= 0
        self.assertEqual(
            profit._definition.getStatement(),
            "profit(s) .. ( - sum(c,(a(c,s) * p(c)))) =g= 0;",
        )

        # test ne
        bla = Parameter(m, "bla", domain=s)
        bla[...] = Sum(c, a[c, s] * p[c]) != 0
        self.assertEqual(
            m._unsaved_statements[-1].getStatement(),
            "bla(s) = (sum(c,(a(c,s) * p(c))) ne 0);",
        )


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
