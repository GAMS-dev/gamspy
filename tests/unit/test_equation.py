from __future__ import annotations

import os
import unittest

import gamspy._symbols.implicits as implicits
import numpy as np
import pandas as pd
from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
    EquationType,
    Model,
    Options,
    Ord,
    Parameter,
    Problem,
    Product,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.exceptions import ValidationError
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
        e1 = Equation(self.m)
        with self.assertRaises(ValidationError):
            _ = e1.getDefinition()

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
        with self.assertRaises(ValidationError):
            _ = Equation(
                self.m,
                name="bla3",
                domain=[i, j],
                description="observe supply limit at plant i",
                definition=x[i, j] <= a[i],
                definition_domain=[i, "bla"],
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

    def test_equation_listing(self):
        m = Container()

        td_data = pd.DataFrame(
            [
                ["icbm", "2", 0.05],
                ["icbm", "6", 0.15],
                ["icbm", "7", 0.10],
                ["icbm", "8", 0.15],
                ["icbm", "9", 0.20],
                ["icbm", "18", 0.05],
                ["mrbm-1", "1", 0.16],
                ["mrbm-1", "2", 0.17],
                ["mrbm-1", "3", 0.15],
                ["mrbm-1", "4", 0.16],
                ["mrbm-1", "5", 0.15],
                ["mrbm-1", "6", 0.19],
                ["mrbm-1", "7", 0.19],
                ["mrbm-1", "8", 0.18],
                ["mrbm-1", "9", 0.20],
                ["mrbm-1", "10", 0.14],
                ["mrbm-1", "12", 0.02],
                ["mrbm-1", "14", 0.12],
                ["mrbm-1", "15", 0.13],
                ["mrbm-1", "16", 0.12],
                ["mrbm-1", "17", 0.15],
                ["mrbm-1", "18", 0.16],
                ["mrbm-1", "19", 0.15],
                ["mrbm-1", "20", 0.15],
                ["lr-bomber", "1", 0.04],
                ["lr-bomber", "2", 0.05],
                ["lr-bomber", "3", 0.04],
                ["lr-bomber", "4", 0.04],
                ["lr-bomber", "5", 0.04],
                ["lr-bomber", "6", 0.10],
                ["lr-bomber", "7", 0.08],
                ["lr-bomber", "8", 0.09],
                ["lr-bomber", "9", 0.08],
                ["lr-bomber", "10", 0.05],
                ["lr-bomber", "11", 0.01],
                ["lr-bomber", "12", 0.02],
                ["lr-bomber", "13", 0.01],
                ["lr-bomber", "14", 0.02],
                ["lr-bomber", "15", 0.03],
                ["lr-bomber", "16", 0.02],
                ["lr-bomber", "17", 0.05],
                ["lr-bomber", "18", 0.08],
                ["lr-bomber", "19", 0.07],
                ["lr-bomber", "20", 0.08],
                ["f-bomber", "10", 0.04],
                ["f-bomber", "11", 0.09],
                ["f-bomber", "12", 0.08],
                ["f-bomber", "13", 0.09],
                ["f-bomber", "14", 0.08],
                ["f-bomber", "15", 0.02],
                ["f-bomber", "16", 0.07],
                ["mrbm-2", "1", 0.08],
                ["mrbm-2", "2", 0.06],
                ["mrbm-2", "3", 0.08],
                ["mrbm-2", "4", 0.05],
                ["mrbm-2", "5", 0.05],
                ["mrbm-2", "6", 0.02],
                ["mrbm-2", "7", 0.02],
                ["mrbm-2", "10", 0.10],
                ["mrbm-2", "11", 0.05],
                ["mrbm-2", "12", 0.04],
                ["mrbm-2", "13", 0.09],
                ["mrbm-2", "14", 0.02],
                ["mrbm-2", "15", 0.01],
                ["mrbm-2", "16", 0.01],
            ]
        )

        wa_data = pd.DataFrame(
            [
                ["icbm", 200],
                ["mrbm-1", 100],
                ["lr-bomber", 300],
                ["f-bomber", 150],
                ["mrbm-2", 250],
            ]
        )

        tm_data = pd.DataFrame(
            [
                ["1", 30],
                ["6", 100],
                ["10", 40],
                ["14", 50],
                ["15", 70],
                ["16", 35],
                ["20", 10],
            ]
        )

        mv_data = pd.DataFrame(
            [
                ["1", 60],
                ["2", 50],
                ["3", 50],
                ["4", 75],
                ["5", 40],
                ["6", 60],
                ["7", 35],
                ["8", 30],
                ["9", 25],
                ["10", 150],
                ["11", 30],
                ["12", 45],
                ["13", 125],
                ["14", 200],
                ["15", 200],
                ["16", 130],
                ["17", 100],
                ["18", 100],
                ["19", 100],
                ["20", 150],
            ]
        )

        # Sets
        w = Set(
            m,
            name="w",
            records=["icbm", "mrbm-1", "lr-bomber", "f-bomber", "mrbm-2"],
            description="weapons",
        )
        t = Set(
            m,
            name="t",
            records=[str(i) for i in range(1, 21)],
            description="targets",
        )

        # Parameters
        td = Parameter(
            m,
            name="td",
            domain=[w, t],
            records=td_data,
            description="target data",
        )
        wa = Parameter(
            m,
            name="wa",
            domain=w,
            records=wa_data,
            description="weapons availability",
        )
        tm = Parameter(
            m,
            name="tm",
            domain=t,
            records=tm_data,
            description="minimum number of weapons per target",
        )
        mv = Parameter(
            m,
            name="mv",
            domain=t,
            records=mv_data,
            description="military value of target",
        )

        # Variables
        x = Variable(
            m,
            name="x",
            domain=[w, t],
            type="Positive",
            description="weapons assignment",
        )
        prob = Variable(
            m, name="prob", domain=t, description="probability for each target"
        )

        # Equations
        maxw = Equation(
            m, name="maxw", domain=w, description="weapons balance"
        )
        minw = Equation(
            m,
            name="minw",
            domain=t,
            description="minimum number of weapons required per target",
        )
        probe = Equation(
            m, name="probe", domain=t, description="probability definition"
        )

        maxw[w] = Sum(t.where[td[w, t]], x[w, t]) <= wa[w]
        minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]
        probe[t] = prob[t] == 1 - Product(
            w.where[td[w, t]], (1 - td[w, t]) ** x[w, t]
        )

        _ = Sum(t, mv[t] * prob[t])
        etd = Sum(
            t,
            mv[t]
            * (1 - Product(w.where[td[w, t]], (1 - td[w, t]) ** x[w, t])),
        )

        war = Model(
            m,
            name="war",
            equations=[maxw, minw],
            problem=Problem.NLP,
            sense=Sense.MAX,
            objective=etd,
        )

        x.l[w, t].where[td[w, t]] = wa[w] / Card(t)

        war.solve()
        with self.assertRaises(ValidationError):
            _ = maxw.getEquationListing()

        war.solve(options=Options(equation_listing_limit=10))

        self.assertEqual(len(maxw.getEquationListing()), 5)

        self.assertEqual(
            len(maxw.getEquationListing(filters=[["f-bomber"]])), 1
        )

        self.assertEqual(len(maxw.getEquationListing(n=2)), 2)

    def test_equation_listing2(self):
        cont = Container()

        # Prepare data
        steel_plants = ["ahmsa", "fundidora", "sicartsa", "hylsa", "hylsap"]
        markets = ["mexico-df", "monterrey", "guadalaja"]
        commodities = [
            "pellets",
            "coke",
            "nat-gas",
            "electric",
            "scrap",
            "pig-iron",
            "sponge",
            "steel",
        ]
        final_products = ["steel"]
        intermediate_products = ["sponge", "pig-iron"]
        raw_materials = ["pellets", "coke", "nat-gas", "electric", "scrap"]
        processes = ["pig-iron", "sponge", "steel-oh", "steel-el", "steel-bof"]
        productive_units = [
            "blast-furn",
            "openhearth",
            "bof",
            "direct-red",
            "elec-arc",
        ]

        io_coefficients = pd.DataFrame(
            [
                ["pellets", "pig-iron", -1.58],
                ["pellets", "sponge", -1.38],
                ["coke", "pig-iron", -0.63],
                ["nat-gas", "sponge", -0.57],
                ["electric", "steel-el", -0.58],
                ["scrap", "steel-oh", -0.33],
                ["scrap", "steel-bof", -0.12],
                ["pig-iron", "pig-iron", 1.00],
                ["pig-iron", "steel-oh", -0.77],
                ["pig-iron", "steel-bof", -0.95],
                ["sponge", "sponge", 1.00],
                ["sponge", "steel-el", -1.09],
                ["steel", "steel-oh", 1.00],
                ["steel", "steel-el", 1.00],
                ["steel", "steel-bof", 1.00],
            ]
        )

        capacity_utilization = pd.DataFrame(
            [
                ["blast-furn", "pig-iron", 1.0],
                ["openhearth", "steel-oh", 1.0],
                ["bof", "steel-bof", 1.0],
                ["direct-red", "sponge", 1.0],
                ["elec-arc", "steel-el", 1.0],
            ]
        )

        capacities_of_units = pd.DataFrame(
            [
                ["blast-furn", "ahmsa", 3.25],
                ["blast-furn", "fundidora", 1.40],
                ["blast-furn", "sicartsa", 1.10],
                ["openhearth", "ahmsa", 1.50],
                ["openhearth", "fundidora", 0.85],
                ["bof", "ahmsa", 2.07],
                ["bof", "fundidora", 1.50],
                ["bof", "sicartsa", 1.30],
                ["direct-red", "hylsa", 0.98],
                ["direct-red", "hylsap", 1.00],
                ["elec-arc", "hylsa", 1.13],
                ["elec-arc", "hylsap", 0.56],
            ]
        )

        rail_distances = pd.DataFrame(
            [
                ["ahmsa", "mexico-df", 1204],
                ["ahmsa", "monterrey", 218],
                ["ahmsa", "guadalaja", 1125],
                ["ahmsa", "export", 739],
                ["fundidora", "mexico-df", 1017],
                ["fundidora", "guadalaja", 1030],
                ["fundidora", "export", 521],
                ["sicartsa", "mexico-df", 819],
                ["sicartsa", "monterrey", 1305],
                ["sicartsa", "guadalaja", 704],
                ["hylsa", "mexico-df", 1017],
                ["hylsa", "guadalaja", 1030],
                ["hylsa", "export", 521],
                ["hylsap", "mexico-df", 185],
                ["hylsap", "monterrey", 1085],
                ["hylsap", "guadalaja", 760],
                ["hylsap", "export", 315],
                ["import", "mexico-df", 428],
                ["import", "monterrey", 521],
                ["import", "guadalaja", 300],
            ]
        )

        product_prices = pd.DataFrame(
            [
                ["pellets", "domestic", 18.7],
                ["coke", "domestic", 52.17],
                ["nat-gas", "domestic", 14.0],
                ["electric", "domestic", 24.0],
                ["scrap", "domestic", 105.0],
                ["steel", "import", 150],
                ["steel", "export", 140],
            ]
        )

        demand_distribution = pd.DataFrame(
            [["mexico-df", 55], ["monterrey", 30], ["guadalaja", 15]]
        )

        dt = 5.209  # total demand for final goods in 1979
        rse = 40  # raw steel equivalence
        eb = 1.0  # export bound

        # Set
        i = Set(
            cont,
            name="i",
            records=pd.DataFrame(steel_plants),
            description="steel plants",
        )
        j = Set(
            cont,
            name="j",
            records=pd.DataFrame(markets),
            description="markets",
        )
        c = Set(
            cont,
            name="c",
            records=pd.DataFrame(commodities),
            description="commidities",
        )
        cf = Set(
            cont,
            name="cf",
            records=pd.DataFrame(final_products),
            domain=c,
            description="final products",
        )
        ci = Set(
            cont,
            name="ci",
            records=pd.DataFrame(intermediate_products),
            domain=c,
            description="intermediate products",
        )
        cr = Set(
            cont,
            name="cr",
            records=pd.DataFrame(raw_materials),
            domain=c,
            description="raw materials",
        )
        p = Set(
            cont,
            name="p",
            records=pd.DataFrame(processes),
            description="processes",
        )
        m = Set(
            cont,
            name="m",
            records=pd.DataFrame(productive_units),
            description="productive units",
        )

        # Data
        a = Parameter(
            cont,
            name="a",
            domain=[c, p],
            records=io_coefficients,
            description="input-output coefficients",
        )
        b = Parameter(
            cont,
            name="b",
            domain=[m, p],
            records=capacity_utilization,
            description="capacity utilization",
        )
        k = Parameter(
            cont,
            name="k",
            domain=[m, i],
            records=capacities_of_units,
            description="capacities of productive units",
        )
        dd = Parameter(
            cont,
            name="dd",
            domain=j,
            records=demand_distribution,
            description="distribution of demand",
        )
        d = Parameter(
            cont,
            name="d",
            domain=[c, j],
            description="demand for steel in 1979",
        )

        d["steel", j] = dt * (1 + rse / 100) * dd[j] / 100

        rd = Parameter(
            cont,
            name="rd",
            domain=["*", "*"],
            records=rail_distances,
            description="rail distances from plants to markets",
        )

        muf = Parameter(
            cont,
            name="muf",
            domain=[i, j],
            description="transport rate: final products",
        )
        muv = Parameter(
            cont, name="muv", domain=j, description="transport rate: imports"
        )
        mue = Parameter(
            cont, name="mue", domain=i, description="transport rate: exports"
        )

        muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]
        muv[j] = (2.48 + 0.0084 * rd["import", j]).where[rd["import", j]]
        mue[i] = (2.48 + 0.0084 * rd[i, "export"]).where[rd[i, "export"]]

        prices = Parameter(
            cont,
            name="prices",
            domain=[c, "*"],
            records=product_prices,
            description="product prices (us$ per unit)",
        )

        pdp = Parameter(
            cont, name="pd", domain=c, description="domestic prices"
        )
        pv = Parameter(cont, name="pv", domain=c, description="import prices")
        pe = Parameter(cont, name="pe", domain=c, description="export prices")

        pdp[c] = prices[c, "domestic"]
        pv[c] = prices[c, "import"]
        pe[c] = prices[c, "export"]

        # Variable
        z = Variable(
            cont,
            name="z",
            domain=[p, i],
            type="Positive",
            description="process level",
        )
        x = Variable(
            cont,
            name="x",
            domain=[c, i, j],
            type="Positive",
            description="shipment of final products",
        )
        u = Variable(
            cont,
            name="u",
            domain=[c, i],
            type="Positive",
            description="purchase of domestic materials",
        )
        v = Variable(
            cont,
            name="v",
            domain=[c, j],
            type="Positive",
            description="imports",
        )
        e = Variable(
            cont,
            name="e",
            domain=[c, i],
            type="Positive",
            description="exports",
        )
        phipsi = Variable(cont, name="phipsi", description="raw material cost")
        philam = Variable(cont, name="philam", description="transport cost")
        phipi = Variable(cont, name="phipi", description="import cost")
        phieps = Variable(cont, name="phieps", description="export revenue")

        # Equation declaration
        mbf = Equation(
            cont,
            name="mbf",
            domain=[c, i],
            description="material balances: final products",
        )
        mbi = Equation(
            cont,
            name="mbi",
            domain=[c, i],
            description="material balances: intermediates",
        )
        mbr = Equation(
            cont,
            name="mbr",
            domain=[c, i],
            description="material balances: raw materials",
        )
        cc = Equation(
            cont,
            name="cc",
            domain=[m, i],
            description="capacity constraint",
        )
        mr = Equation(
            cont,
            name="mr",
            domain=[c, j],
            description="market requirements",
        )
        me = Equation(
            cont,
            name="me",
            domain=c,
            description="maximum export",
        )
        apsi = Equation(
            cont,
            name="apsi",
            description="accounting: raw material cost",
        )
        alam = Equation(
            cont,
            name="alam",
            description="accounting: transport cost",
        )
        api = Equation(cont, name="api", description="accounting: import cost")
        aeps = Equation(
            cont,
            name="aeps",
            description="accounting: export cost",
        )

        # Equation definition
        obj = phipsi + philam + phipi - phieps  # Total Cost

        mbf[cf, i] = (
            Sum(p, a[cf, p] * z[p, i]) >= Sum(j, x[cf, i, j]) + e[cf, i]
        )
        mbi[ci, i] = Sum(p, a[ci, p] * z[p, i]) >= 0
        mbr[cr, i] = Sum(p, a[cr, p] * z[p, i]) + u[cr, i] >= 0
        cc[m, i] = Sum(p, b[m, p] * z[p, i]) <= k[m, i]
        mr[cf, j] = Sum(i, x[cf, i, j]) + v[cf, j] >= d[cf, j]
        me[cf] = Sum(i, e[cf, i]) <= eb
        apsi[...] = phipsi == Sum((cr, i), pdp[cr] * u[cr, i])
        alam[...] = philam == Sum((cf, i, j), muf[i, j] * x[cf, i, j]) + Sum(
            (cf, j), muv[j] * v[cf, j]
        ) + Sum((cf, i), mue[i] * e[cf, i])
        api[...] = phipi == Sum((cf, j), pv[cf] * v[cf, j])
        aeps[...] = phieps == Sum((cf, i), pe[cf] * e[cf, i])

        mexss = Model(
            cont,
            name="mexss",
            equations=cont.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=obj,
        )

        mexss.solve(options=Options(equation_listing_limit=100))
        self.assertEqual(
            len(mr.getEquationListing(filters=[["steel"], ["monterrey"]])), 1
        )

        self.assertEqual(
            len(mr.getEquationListing(filters=[["steel"], []])), 3
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
