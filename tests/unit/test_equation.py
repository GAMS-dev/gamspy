import unittest

import pandas as pd

import gamspy._symbols.implicits as implicits
from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import EquationType
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


class EquationSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_equation_types(self):
        # Prepare data
        canning_plants = ["seattle", "san-diego"]

        c = Parameter(self.m, name="c", domain=[], records=0.5)
        x = Variable(
            self.m,
            name="x",
            domain=[],
            records={"lower": 1.0, "level": 1.5, "upper": 3.75},
        )

        # Sets
        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )

        # Equations
        eq1 = Equation(self.m, "eq1", type="nonbinding")
        eq1.expr = x == c
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "eq1 .. x =n= c;",
        )
        self.assertEqual(eq1.type, "nonbinding")

        eq2 = Equation(self.m, "eq2", domain=[i], type="nonbinding")
        eq2[i] = x[i] == c[i]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "eq2(i) .. x(i) =n= c(i);",
        )

        eq2[i].expr = x[i] == c[i]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "eq2(i) .. x(i) =n= c(i);",
        )

        # eq
        eq3 = Equation(self.m, "eq3", domain=[i])
        eq3[i] = x == c
        self.assertEqual(eq3.type, "eq")

        # geq
        eq4 = Equation(self.m, "eq4", domain=[i])
        eq4[i] = x >= c
        self.assertEqual(eq4.type, "geq")

        # leq
        eq5 = Equation(self.m, "eq5", domain=[i])
        eq5[i] = x <= c
        self.assertEqual(eq5.type, "leq")

        self.assertEqual(str(EquationType.REGULAR), "REGULAR")
        eq6 = Equation(self.m, "eq6", type=EquationType.REGULAR, domain=[i])
        self.assertEqual(eq6.type, "eq")

        self.assertEqual(
            EquationType.values(),
            ["REGULAR", "NONBINDING", "EXTERNAL", "CONE", "BOOLEAN"],
        )

    def test_equation_declaration(self):
        # Prepare data
        canning_plants = ["seattle", "san-diego"]
        markets = ["new-york", "chicago", "topeka"]

        # Sets
        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        j = Set(self.m, name="j", records=markets, description="Markets")

        # Equation declaration without an index
        cost = Equation(
            self.m,
            name="cost",
            description="define objective function",
        )
        self.assertEqual(cost.gamsRepr(), "cost")
        self.assertEqual(
            cost.getStatement(),
            'Equation cost "define objective function";',
        )

        # Equation declaration with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        self.assertEqual(supply, "supply")
        self.assertEqual(
            supply.getStatement(),
            'Equation supply(i) "observe supply limit at plant i";',
        )

        # Equation declaration with more than one index
        bla = Equation(
            self.m,
            name="bla",
            domain=[i, j],
            description="some text",
        )
        self.assertEqual(bla.gamsRepr(), "bla")
        self.assertEqual(bla.getStatement(), 'Equation bla(i,j) "some text";')

        u = Set(self.m, "u")
        v = Alias(self.m, "v", alias_with=u)
        e = Set(self.m, "e", domain=[u, v])
        eq = Equation(self.m, "eq", domain=[u, v])
        self.assertEqual(eq[e[u, v]].gamsRepr(), "eq(e(u,v))")

    def test_equation_definition(self):
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

        # Equation definition without an index
        cost = Equation(
            self.m,
            name="cost",
            description="define objective function",
        )
        cost.expr = Sum((i, j), c[i, j] * x[i, j]) == z
        with self.assertRaises(TypeError):
            cost.records = 5

        self.assertIsNotNone(cost.expr)
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "cost .. sum((i,j),(c(i,j) * x(i,j))) =e= z;",
        )

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        supply[i] = Sum(j, x[i, j]) <= a[i]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "supply(i) .. sum(j,x(i,j)) =l= a(i);",
        )

        # Equation definition with more than one index
        bla = Equation(
            self.m,
            name="bla",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla[i, j].expr = Sum((i, j), x[i, j]) <= a[i]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "bla(i,j) .. sum((i,j),x(i,j)) =l= a(i);",
        )

        bla[i, "*"] = Sum((i, j), x[i, j]) <= a[i]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "bla(i,*) .. sum((i,j),x(i,j)) =l= a(i);",
        )

        # Equation definition in constructor
        _ = Equation(
            self.m,
            name="cost2",
            description="define objective function",
            expr=Sum((i, j), c[i, j] * x[i, j]) == z,
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "cost2 .. sum((i,j),(c(i,j) * x(i,j))) =e= z;",
        )

        # Equation definition in addEquation
        _ = self.m.addEquation(
            name="cost2",
            description="define objective function",
            expr=Sum((i, j), c[i, j] * x[i, j]) == z,
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "cost2 .. sum((i,j),(c(i,j) * x(i,j))) =e= z;",
        )

        # eq[bla].expr test
        bla2 = Equation(
            self.m,
            name="bla2",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla2[i, j].expr = Sum((i, j), x[i, j]) <= a[i]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "bla2(i,j) .. sum((i,j),x(i,j)) =l= a(i);",
        )

        # eq[bla] with different domain
        _ = Equation(
            self.m,
            name="bla3",
            domain=[i, j],
            description="observe supply limit at plant i",
            expr=Sum((i, j), x[i, j]) <= a[i],
            expr_domain=[i, "bla"],
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            'bla3(i,"bla") .. sum((i,j),x(i,j)) =l= a(i);',
        )

        m = Container()
        g = Set(m, name="g", records=[str(i) for i in range(1, 4)])
        t1 = Set(m, name="t1", records=[str(i) for i in range(1, 4)])
        t2 = Set(m, name="t2", records=[str(i) for i in range(1, 4)])

        eStartNaive = Equation(m, name="eStartNaive", domain=[g, t1])
        pMinDown = Parameter(m, name="pMinDown", domain=[g, t1])
        vStart = Parameter(m, name="vStart", domain=[g, t2])

        eStartNaive[g, t1] = (
            Sum(
                t2.where[
                    (Ord(t1) >= Ord(t2))
                    & (Ord(t2) > Ord(t1) - pMinDown[g, t1])
                ],
                vStart[g, t2],
            )
            <= 1
        )
        self.assertEqual(
            list(m._statements_dict.values())[-1].getStatement(),
            "eStartNaive(g,t1) .. sum(t2 $ ((ord(t1) >= ord(t2)) and"
            " (ord(t2) > (ord(t1) - pMinDown(g,t1)))),vStart(g,t2)) =l= 1;",
        )

    def test_equation_attributes(self):
        pi = Equation(self.m, "pi")

        self.assertTrue(
            hasattr(pi, "l") and isinstance(pi.l, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.l.gamsRepr(), "pi.l")

        self.assertTrue(
            hasattr(pi, "m") and isinstance(pi.m, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.m.gamsRepr(), "pi.m")

        self.assertTrue(
            hasattr(pi, "lo")
            and isinstance(pi.lo, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.lo.gamsRepr(), "pi.lo")

        self.assertTrue(
            hasattr(pi, "up")
            and isinstance(pi.up, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.up.gamsRepr(), "pi.up")

        self.assertTrue(
            hasattr(pi, "scale")
            and isinstance(pi.scale, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.scale.gamsRepr(), "pi.scale")

        self.assertTrue(
            hasattr(pi, "stage")
            and isinstance(pi.stage, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.stage.gamsRepr(), "pi.stage")

        self.assertTrue(
            hasattr(pi, "range")
            and isinstance(pi.range, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.range.gamsRepr(), "pi.range")

        self.assertTrue(
            hasattr(pi, "slacklo")
            and isinstance(pi.slacklo, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.slacklo.gamsRepr(), "pi.slacklo")

        self.assertTrue(
            hasattr(pi, "slackup")
            and isinstance(pi.slackup, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.slackup.gamsRepr(), "pi.slackup")

        self.assertTrue(
            hasattr(pi, "slack")
            and isinstance(pi.slack, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.slack.gamsRepr(), "pi.slack")

        self.assertTrue(
            hasattr(pi, "infeas")
            and isinstance(pi.infeas, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.infeas.gamsRepr(), "pi.infeas")

    def test_implicit_equation_attributes(self):
        i = Set(self.m, "i", records=[f"i{i}" for i in range(10)])
        a = Equation(self.m, "a", "regular", [i])

        self.assertTrue(
            hasattr(a[i], "l")
            and isinstance(a[i].l, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].l.gamsRepr(), "a(i).l")
        self.assertTrue(
            hasattr(a[i], "m")
            and isinstance(a[i].m, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].m.gamsRepr(), "a(i).m")
        self.assertTrue(
            hasattr(a[i], "lo")
            and isinstance(a[i].lo, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].lo.gamsRepr(), "a(i).lo")
        self.assertTrue(
            hasattr(a[i], "up")
            and isinstance(a[i].up, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].up.gamsRepr(), "a(i).up")
        self.assertTrue(
            hasattr(a[i], "scale")
            and isinstance(a[i].scale, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].scale.gamsRepr(), "a(i).scale")
        self.assertTrue(
            hasattr(a[i], "stage")
            and isinstance(a[i].stage, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].stage.gamsRepr(), "a(i).stage")
        self.assertTrue(
            hasattr(a[i], "range")
            and isinstance(a[i].range, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].range.gamsRepr(), "a(i).range")
        self.assertTrue(
            hasattr(a[i], "slacklo")
            and isinstance(a[i].slacklo, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].slacklo.gamsRepr(), "a(i).slacklo")
        self.assertTrue(
            hasattr(a[i], "slackup")
            and isinstance(a[i].slackup, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].slackup.gamsRepr(), "a(i).slackup")
        self.assertTrue(
            hasattr(a[i], "slack")
            and isinstance(a[i].slack, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].slack.gamsRepr(), "a(i).slack")
        self.assertTrue(
            hasattr(a[i], "infeas")
            and isinstance(a[i].infeas, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].infeas.gamsRepr(), "a(i).infeas")

    def test_mcp_equation(self):
        c = Parameter(self.m, name="c", domain=[], records=0.5)
        x = Variable(
            self.m,
            name="x",
            domain=[],
            records={"lower": 1.0, "level": 1.5, "upper": 3.75},
        )
        f = Equation(self.m, name="f", domain=[], type="nonbinding")
        f.expr = x - c

        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "f .. (x - c) =n= 0;",
        )


def equation_suite():
    suite = unittest.TestSuite()
    tests = [
        EquationSuite(name)
        for name in dir(EquationSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(equation_suite())
