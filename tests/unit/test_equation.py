from __future__ import annotations

import os
import unittest

import gamspy._symbols.implicits as implicits
import numpy as np
import pandas as pd
from gamspy import (
    Alias,
    Container,
    Equation,
    EquationType,
    Model,
    Ord,
    Parameter,
    Set,
    Sum,
    Variable,
)
from gamspy.exceptions import GamspyException, ValidationError
from gamspy.math import sqr


class EquationSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None)
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

    def test_equation_creation(self):
        # no name is fine now
        _ = Equation(self.m)

        # non-str type name
        self.assertRaises(TypeError, Equation, self.m, 5)

        # no container
        self.assertRaises(TypeError, Equation)

        # non-container type container
        self.assertRaises(TypeError, Equation, 5, "j")

        # try to create a symbol with same name but different type
        _ = Set(self.m, "i")
        self.assertRaises(TypeError, Equation, self.m, "i")

        # get already created symbol
        j1 = Equation(self.m, "j")
        j2 = Equation(self.m, "j")
        self.assertEqual(id(j1), id(j2))

        # Equation and domain containers are different
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        set1 = Set(self.m, "set1")
        with self.assertRaises(ValidationError):
            _ = Equation(m, "eq1", domain=[set1])

    def test_equation_types(self):
        # Prepare data
        canning_plants = ["seattle", "san-diego"]

        x = Variable(
            self.m,
            name="x",
            domain=[],
            records={"lower": 1.0, "level": 1.5, "upper": 3.75},
        )

        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )

        c = Parameter(
            self.m, name="c", domain=[i], records=np.array([0.5, 0.6])
        )

        d = Parameter(self.m, name="d", records=0.5)
        eq1 = Equation(self.m, "eq1", type="nonbinding")
        eq1[...] = (x - d) == 0
        self.assertEqual(
            eq1.getDefinition(),
            "eq1 .. (x - d) =n= 0;",
        )
        self.assertEqual(eq1.type, "nonbinding")

        y = Variable(self.m, "y", domain=[i])
        eq2 = Equation(self.m, "eq2", domain=[i], type="nonbinding")
        eq2[i] = (y[i] - c[i]) == 0
        self.assertEqual(
            eq2.getDefinition(),
            "eq2(i) .. (y(i) - c(i)) =n= 0;",
        )

        eq2[i] = (y[i] - c[i]) == 0
        self.assertEqual(
            eq2.getDefinition(),
            "eq2(i) .. (y(i) - c(i)) =n= 0;",
        )

        # eq
        eq3 = Equation(self.m, "eq3", domain=[i])
        eq3[i] = y[i] == c[i]
        self.assertEqual(eq3.type, "eq")

        # geq
        eq4 = Equation(self.m, "eq4", domain=[i])
        eq4[i] = y[i] >= c[i]
        self.assertEqual(eq4.type, "eq")

        # leq
        eq5 = Equation(self.m, "eq5", domain=[i])
        eq5[i] = y[i] <= c[i]
        self.assertEqual(eq5.type, "eq")

        self.assertEqual(str(EquationType.REGULAR), "REGULAR")
        eq6 = Equation(self.m, "eq6", type=EquationType.REGULAR, domain=[i])
        self.assertEqual(eq6.type, "eq")

        self.assertEqual(
            EquationType.values(),
            ["REGULAR", "NONBINDING", "EXTERNAL", "CONE", "BOOLEAN"],
        )

        eq6 = Equation(self.m, "eq6", domain=[i])
        with self.assertRaises(ValidationError):
            eq6[i] = y[i] - c[i]

    def test_nonbinding(self):
        x = Variable(self.m, "x")
        e = Equation(self.m, "e", definition=x == 0, type="NONBINDING")
        self.assertEqual(e.getDefinition(), "e .. x =n= 0;")

        x1 = Variable(self.m, "x1")
        e1 = Equation(self.m, "e1", definition=x1 >= 0, type="NONBINDING")
        self.assertEqual(e1.getDefinition(), "e1 .. x1 =n= 0;")

        x2 = Variable(self.m, "x2")
        e2 = Equation(self.m, "e2", definition=x2 <= 0, type="NONBINDING")
        self.assertEqual(e2.getDefinition(), "e2 .. x2 =n= 0;")

    def test_equation_declaration(self):
        # Check if the name is reserved
        self.assertRaises(ValidationError, Equation, self.m, "set")

        # Prepare data
        canning_plants = ["seattle", "san-diego"]
        markets = ["new-york", "chicago", "topeka"]

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
            cost.getDeclaration(),
            'Equation cost "define objective function";',
        )

        # Equation declaration with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        self.assertEqual(supply.gamsRepr(), "supply")
        self.assertEqual(
            supply.getDeclaration(),
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
        self.assertEqual(
            bla.getDeclaration(), 'Equation bla(i,j) "some text";'
        )

        u = Set(self.m, "u")
        v = Alias(self.m, "v", alias_with=u)
        e = Set(self.m, "e", domain=[u, v])
        eq = Equation(self.m, "eq", domain=[u, v])
        self.assertEqual(eq[e[u, v]].gamsRepr(), "eq(e(u,v))")

    def test_equation_definition(self):
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

        # Equation definition without an index
        cost = Equation(
            self.m,
            name="cost",
            description="define objective function",
        )
        cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z
        with self.assertRaises(TypeError):
            cost.records = 5

        self.assertIsNotNone(cost[...])
        self.assertEqual(
            cost.getDefinition(),
            "cost .. sum((i,j),(c(i,j) * x(i,j))) =e= z;",
        )

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
            definition=Sum(j, x[i, j]) <= a[i],
        )
        self.assertEqual(
            supply.getDefinition(),
            "supply(i) .. sum(j,x(i,j)) =l= a(i);",
        )

        # Equation definition with more than one index
        bla = Equation(
            self.m,
            name="bla",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla[i, j] = x[i, j] <= a[i]
        self.assertEqual(
            bla.getDefinition(),
            "bla(i,j) .. x(i,j) =l= a(i);",
        )

        bla[i, "topeka"] = x[i, "topeka"] <= a[i]
        self.assertEqual(
            bla.getDefinition(),
            'bla(i,"topeka") .. x(i,"topeka") =l= a(i);',
        )

        # Equation definition in constructor
        cost2 = Equation(
            self.m,
            name="cost2",
            description="define objective function",
            definition=Sum((i, j), c[i, j] * x[i, j]) == z,
        )
        self.assertEqual(
            cost2.getDefinition(),
            "cost2 .. sum((i,j),(c(i,j) * x(i,j))) =e= z;",
        )

        # Equation definition in addEquation
        cost3 = self.m.addEquation(
            name="cost3",
            description="define objective function",
            definition=Sum((i, j), c[i, j] * x[i, j]) == z,
        )
        self.assertEqual(
            cost3.getDefinition(),
            "cost3 .. sum((i,j),(c(i,j) * x(i,j))) =e= z;",
        )

        # eq[bla][...] test
        bla2 = Equation(
            self.m,
            name="bla2",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla2[i, j] = x[i, j] <= a[i]
        self.assertEqual(
            bla2.getDefinition(),
            "bla2(i,j) .. x(i,j) =l= a(i);",
        )

        # eq[bla] with different domain
        with self.assertRaises(GamspyException):
            bla3 = Equation(
                self.m,
                name="bla3",
                domain=[i, j],
                description="observe supply limit at plant i",
                definition=Sum((i, j), x[i, j]) <= a[i],
                definition_domain=[i, "bla"],
            )
            self.assertEqual(
                bla3.getDefinition(),
                'bla3(i,"bla") .. sum((i,j),x(i,j)) =l= a(i);',
            )

        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
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
            eStartNaive.getDefinition(),
            "eStartNaive(g,t1) .. sum(t2 $ ((ord(t1) >= ord(t2)) and"
            " (ord(t2) > (ord(t1) - pMinDown(g,t1)))),vStart(g,t2)) =l= 1;",
        )

        m = Container()
        i = Set(m, "i")
        j = Set(m, "j")

        a = Parameter(m, name="a", domain=[i, j])
        b = Parameter(m, name="b", domain=[i, j])
        c = Variable(m, name="c", domain=[i, j])
        assign_1 = Equation(m, name="assign_1", domain=[i, j])
        assign_1[...] = a[i, j] == b[i, j] + c[i, j]
        self.assertEqual(
            assign_1.getDefinition(),
            "assign_1(i,j) .. a(i,j) =e= (b(i,j) + c(i,j));",
        )

        m = Container()
        k = Set(m, "k")

        a = Parameter(m, name="a")
        b = Variable(m, name="b", domain=[k])
        c = Parameter(m, name="c", domain=[k])
        assign_1 = Equation(m, name="assign_1")

        assign_1[...] = a == Sum(k, b[k] * c[k])
        self.assertEqual(
            assign_1.getDefinition(), "assign_1 .. a =e= sum(k,(b(k) * c(k)));"
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

    def test_scalar_attr_assignment(self):
        a = Equation(self.m, "a")
        a.l = 5
        self.assertEqual(a._assignment.getDeclaration(), "a.l = 5;")

        a.m = 5
        self.assertEqual(a._assignment.getDeclaration(), "a.m = 5;")

        a.lo = 5
        self.assertEqual(a._assignment.getDeclaration(), "a.lo = 5;")

        a.up = 5
        self.assertEqual(a._assignment.getDeclaration(), "a.up = 5;")

        a.scale = 5
        self.assertEqual(a._assignment.getDeclaration(), "a.scale = 5;")

        a.stage = 5
        self.assertEqual(a._assignment.getDeclaration(), "a.stage = 5;")

    def test_mcp_equation(self):
        c = Parameter(self.m, name="c", domain=[], records=0.5)
        x = Variable(
            self.m,
            name="x",
            domain=[],
            records={"lower": 1.0, "level": 1.5, "upper": 3.75},
        )
        f = Equation(self.m, name="f", type="nonbinding")
        f[...] = (x - c) == 0

        self.assertEqual(
            f.getDefinition(),
            "f .. (x - c) =n= 0;",
        )

        f2 = Equation(
            self.m, name="f2", type="nonbinding", definition=(x - c) == 0
        )
        self.assertEqual(
            f2.getDefinition(),
            "f2 .. (x - c) =n= 0;",
        )

        f3 = Equation(
            self.m, name="f3", type="nonbinding", definition=(x - c) == 0
        )
        self.assertEqual(
            f3.getDefinition(),
            "f3 .. (x - c) =n= 0;",
        )

        f4 = Equation(self.m, name="f4", definition=(x - c) == 0)
        self.assertEqual(
            f4.getDefinition(),
            "f4 .. (x - c) =e= 0;",
        )

        model = Model(self.m, "mcp_model", "MCP", matches={f: x})
        model.solve()

    def test_changed_domain(self):
        cont = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )

        s = Set(cont, "s")
        m = Set(cont, "m")
        A = Equation(cont, "A", domain=[s, m])

        A.domain = ["s", "m"]
        self.assertEqual(A.getDeclaration(), "Equation A(*,*);")

    def test_equation_assignment(self):
        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )

        i = Set(self.m, "i")
        j = Set(m, "j")
        a = Equation(self.m, "a", domain=[i])

        with self.assertRaises(ValidationError):
            a[j] = 5

        m = Container(
            system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
        )
        N = Parameter(m, "N", records=20)
        L = Parameter(m, "L", records=int(N.toValue()) / 2)
        v = Set(m, "v", records=range(0, 1001))
        i = Set(m, "i", domain=[v])
        x = Variable(m, "x", "free", [v])
        y = Variable(m, "y", "free", [v])
        e = Equation(m, "e")
        e[...] = Sum(i.where[(i.val == L - 1)], sqr(x[i]) + sqr(y[i])) == 1
        self.assertEqual(
            e.getDefinition(),
            "e .. sum(i $ ((L - 1) eq i.val),(( sqr(x(i)) ) + ("
            " sqr(y(i)) ))) =e= 1;",
        )

    def test_assignment_dimensionality(self):
        j1 = Set(self.m, "j1")
        j2 = Set(self.m, "j2")
        j3 = Equation(self.m, "j3", domain=[j1, j2])
        with self.assertRaises(ValidationError):
            j3["bla"] = 5

        j4 = Set(self.m, "j4")

        with self.assertRaises(ValidationError):
            j3[j1, j2, j4] = 5

        i = Set(self.m, name="i")
        ii = Set(self.m, name="ii", domain=[i])
        j = Set(self.m, name="j")
        jj = Set(self.m, name="jj", domain=[j])
        k = Set(self.m, name="k")
        kk = Set(self.m, name="kk", domain=[k])
        TSAM = Variable(self.m, name="TSAM", domain=[i, j])
        A = Variable(self.m, name="A", domain=[i, j])
        Y = Variable(self.m, name="Y", domain=[i, j])
        NONZERO = Set(self.m, name="NONZERO")
        SAMCOEF = Equation(
            self.m,
            name="SAMCOEF",
            domain=[i, j],
            description="define SAM coefficients",
        )

        with self.assertRaises(ValidationError):
            SAMCOEF[ii, jj, kk].where[NONZERO[ii, jj]] = (
                TSAM[ii, jj] == A[ii, jj] * Y[jj]
            )

    def test_type(self):
        eq1 = Equation(self.m, "eq1")
        eq1.type = EquationType.REGULAR
        self.assertEqual(eq1.type, "eq")

        eq2 = Equation(self.m, "eq2")
        eq2.type = EquationType.BOOLEAN
        self.assertEqual(eq2.type, "boolean")

        eq3 = Equation(self.m, "eq3")
        eq3.type = EquationType.NONBINDING
        self.assertEqual(eq3.type, "nonbinding")

    def test_uels_on_axes(self):
        s = pd.Series(index=["a", "b", "c"], data=[i + 1 for i in range(3)])
        e = Equation(
            self.m, "e", "eq", domain=["*"], records=s, uels_on_axes=True
        )
        self.assertEqual(e.records.level.tolist(), [1, 2, 3])

    def test_expert_sync(self):
        m = Container()
        i = Set(m, "i", records=["i1", "i2"])
        e = Equation(m, "e", domain=i)
        e.l = 5
        e.synchronize = False
        e.l = e.l * 5
        self.assertEqual(e.records.level.tolist(), [5.0, 5.0])
        e.synchronize = True
        self.assertEqual(e.records.level.tolist(), [25.0, 25.0])


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
