import glob
import math
import os
import subprocess
import unittest
from pathlib import Path

import pandas as pd

import gamspy._algebra._expression as expression
import gamspy.math as gams_math
import gamspy._symbols._implicits as implicits
import gamspy.utils as utils
from gamspy.functions import ifthen
from gamspy import (
    Alias,
    Card,
    Container,
    Domain,
    Equation,
    Model,
    Number,
    Ord,
    Parameter,
    Product,
    Set,
    Smax,
    Smin,
    Sum,
    Variable,
    ModelStatus,
)


class GamspySuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_set_string(self):
        # Without records
        b = Set(self.m, "b")
        self.assertEqual(b.gamsRepr(), "b")
        self.assertEqual(
            b.getStatement(), f"Set b(*);\n$gdxLoad {self.m._gdx_path} b"
        )

        # Without domain
        i = Set(
            self.m,
            "i",
            records=["seattle", "san-diego"],
            description="dummy set",
        )
        self.assertEqual(i.gamsRepr(), "i")
        self.assertEqual(
            i.getStatement(),
            f'Set i(*) "dummy set";\n$gdxLoad {self.m._gdx_path} i',
        )

        # With one domain
        j = Set(self.m, "j", records=["seattle", "san-diego", "california"])
        k = Set(self.m, "k", domain=[j], records=["seattle", "san-diego"])
        self.assertEqual(k.gamsRepr(), "k")
        self.assertEqual(
            k.getStatement(), f"Set k(j);\n$gdxLoad {self.m._gdx_path} k"
        )

        # With two domain
        m = Set(self.m, "m", records=[f"i{i}" for i in range(2)])
        n = Set(self.m, "n", records=[f"j{i}" for i in range(2)])
        a = Set(self.m, "a", [m, n])
        a.generateRecords(density=1)
        self.assertEqual(a.gamsRepr(), "a")
        self.assertEqual(
            a.getStatement(), f"Set a(m,n);\n$gdxLoad {self.m._gdx_path} a"
        )

        s = Set(self.m, "s", is_singleton=True)
        self.assertEqual(
            s.getStatement(),
            f"Singleton Set s(*);\n$gdxLoad {self.m._gdx_path} s",
        )

        with self.assertRaises(TypeError):
            s.records = 5

    def test_set_operators(self):
        i = Set(self.m, "i", records=["seattle", "san-diego"])
        card = Card(i)
        self.assertEqual(card.gamsRepr(), "card(i)")

        ord = Ord(i)
        self.assertEqual(ord.gamsRepr(), "ord(i)")

    def test_set_operations(self):
        i = Set(self.m, "i", records=["seattle", "san-diego"])
        k = Set(self.m, "k", records=["seattle", "san-diego"])
        union = i + k
        self.assertEqual(union.gamsRepr(), "i + k")

        intersection = i * k
        self.assertEqual(intersection.gamsRepr(), "i * k")

        complement = ~i
        self.assertEqual(complement.gamsRepr(), "( not i)")

        difference = i - k
        self.assertEqual(difference.gamsRepr(), "i - k")

    def test_dynamic_sets(self):
        i = Set(self.m, name="i", records=[f"i{idx}" for idx in range(1, 4)])
        i["i1"] = False

        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            'i("i1") = no;',
        )

    def test_implicit_sets(self):
        j = Set(self.m, "j", records=["seattle", "san-diego", "california"])
        k = Set(self.m, "k", domain=[j], records=["seattle", "san-diego"])

        expr = k[j] <= k[j]
        self.assertEqual(expr.gamsRepr(), "(k(j) <= k(j))")
        expr = k[j] >= k[j]
        self.assertEqual(expr.gamsRepr(), "(k(j) >= k(j))")

        k[j] = ~k[j]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "k(j) = ( not k(j));",
        )

    def test_lag_and_lead(self):
        set = Set(
            self.m, name="S", records=["a", "b", "c"], description="Test text"
        )
        alias = Alias(self.m, "A", alias_with=set)

        # Circular lag
        new_set = set.lag(n=5, type="circular")
        self.assertEqual(new_set.name, "S -- 5")
        new_set = alias.lag(n=5, type="circular")
        self.assertEqual(new_set.name, "A -- 5")

        # Circular lead
        new_set = set.lead(n=5, type="circular")
        self.assertEqual(new_set.name, "S ++ 5")
        new_set = alias.lead(n=5, type="circular")
        self.assertEqual(new_set.name, "A ++ 5")

        # Linear lag
        new_set = set.lag(n=5, type="linear")
        self.assertEqual(new_set.name, "S - 5")
        new_set = alias.lag(n=5, type="linear")
        self.assertEqual(new_set.name, "A - 5")

        # Linear lead
        new_set = set.lead(n=5, type="linear")
        self.assertEqual(new_set.name, "S + 5")
        new_set = alias.lead(n=5, type="linear")
        self.assertEqual(new_set.name, "A + 5")

        # Incorrect type
        self.assertRaises(ValueError, set.lead, 5, "bla")
        self.assertRaises(ValueError, alias.lead, 5, "bla")
        self.assertRaises(ValueError, set.lag, 5, "bla")
        self.assertRaises(ValueError, alias.lag, 5, "bla")

        m = Container()
        s = Set(m, name="s", records=[f"s{i}" for i in range(1, 4)])
        t = Set(m, name="t", records=[f"t{i}" for i in range(1, 6)])

        sMinDown = Set(m, name="sMinDown", domain=[s, t])
        sMinDown[s, t.lead((Ord(t) - Ord(s)))] = 1
        self.assertEqual(
            list(m._statements_dict.values())[-1].gamsRepr(),
            "sMinDown(s,t + (ord(t) - ord(s))) = 1;",
        )

    def test_alias_string(self):
        # Set and Alias without domain
        i = Set(self.m, name="i", records=["a", "b", "c"])
        j = Alias(self.m, name="j", alias_with=i)
        self.assertEqual(j.gamsRepr(), "j")
        self.assertEqual(j.getStatement(), "Alias(i,j);")

        # Set and Alias with domain
        k = Set(self.m, name="k", domain=[i], records=["a", "b"])
        m = Alias(self.m, name="m", alias_with=k)
        self.assertEqual(m.gamsRepr(), "m")
        self.assertEqual(m.getStatement(), "Alias(k,m);")

    def test_parameter_string(self):
        canning_plants = pd.DataFrame(["seattle", "san-diego", "topeka"])

        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        a = Parameter(
            self.m,
            name="a",
            domain=[i],
            records=pd.DataFrame(
                [["seattle", 350], ["san-diego", 600], ["topeka", 500]]
            ),
            description="distances",
        )

        self.assertEqual(
            a.getStatement(),
            f'Parameter a(i) "distances";\n$gdxLoad {self.m._gdx_path} a',
        )

        b = Parameter(self.m, "b")
        self.assertEqual(
            b.getStatement(), f"Parameter b;\n$gdxLoad {self.m._gdx_path} b"
        )
        self.assertEqual((b == 5).gamsRepr(), "(b = 5)")
        self.assertEqual((-b).name, "-b")

    def test_implicit_parameter_string(self):
        canning_plants = pd.DataFrame(["seattle", "san-diego", "topeka"])

        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        a = Parameter(
            self.m,
            name="a",
            domain=[i],
            records=pd.DataFrame(
                [["seattle", 350], ["san-diego", 600], ["topeka", 500]]
            ),
        )

        self.assertEqual(a[i].gamsRepr(), "a(i)")
        a[i].assign = a * 5
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "a(i) = (a * 5);",
        )

        a[i] = -a[i] * 5
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "a(i) = (-a(i) * 5);",
        )

    def test_implicit_parameter_assignment(self):
        canning_plants = pd.DataFrame(["seattle", "san-diego", "topeka"])

        i = Set(
            self.m,
            name="i",
            records=canning_plants,
            description="Canning Plants",
        )
        a = implicits.ImplicitParameter(
            self.m,
            name="a",
            domain=[i],
            records=pd.DataFrame(
                [["seattle", 350], ["san-diego", 600], ["topeka", 500]]
            ),
        )

        a[i] = Sum(i, a[i])
        self.assertEqual(
            list(self.m._statements_dict.values())[-1], "a(i) = sum(i, a(i));"
        )

        a[i].assign = Sum(i, a[i])
        self.assertEqual(
            list(self.m._statements_dict.values())[-1], "a(i) = sum(i, a(i));"
        )

    def test_variable_string(self):
        # Set
        i = Set(self.m, name="i", records=["bla", "damn"])
        j = Set(self.m, name="j", records=["test", "test2"])

        # Variable without data
        v4 = Variable(self.m, "v4")
        self.assertEqual(v4.gamsRepr(), "v4")
        self.assertEqual(
            v4.getStatement(),
            f"free Variable v4;\n$gdxLoad {self.m._gdx_path} v4",
        )

        with self.assertRaises(TypeError):
            v4.records = 5

        # Variable without domain
        v0 = Variable(self.m, name="v0", description="some text")
        self.assertEqual(v0.gamsRepr(), "v0")
        self.assertEqual(
            v0.getStatement(),
            f'free Variable v0 "some text";\n$gdxLoad {self.m._gdx_path} v0',
        )

        expression = -v0
        self.assertEqual(expression.name, "-v0")

        # Variable one domain
        v1 = Variable(self.m, name="v1", domain=[i])
        self.assertEqual(v1.gamsRepr(), "v1")
        self.assertEqual(
            v1.getStatement(),
            f"free Variable v1(i);\n$gdxLoad {self.m._gdx_path} v1",
        )

        self.assertEqual((v1[i] == v1[i]).gamsRepr(), "v1(i) =e= v1(i)")

        # Variable two domain
        v2 = Variable(self.m, name="v2", domain=[i, j])
        self.assertEqual(v2.gamsRepr(), "v2")
        self.assertEqual(
            v2.getStatement(),
            f"free Variable v2(i,j);\n$gdxLoad {self.m._gdx_path} v2",
        )

        # Scalar variable with records
        pi = Variable(
            self.m,
            "pi",
            records=pd.DataFrame(data=[3.14159], columns=["level"]),
        )
        self.assertEqual(
            pi.getStatement(),
            f"free Variable pi;\n$gdxLoad {self.m._gdx_path} pi",
        )
        new_pi = -pi
        self.assertEqual(new_pi.gamsRepr(), "-pi")

        # 1D variable with records
        v = Variable(
            self.m,
            "v",
            "free",
            domain=["*"],
            records=pd.DataFrame(
                data=[("i" + str(i), i) for i in range(5)],
                columns=["domain", "marginal"],
            ),
        )
        self.assertEqual(
            v.getStatement(),
            f"free Variable v(*);\n$gdxLoad {self.m._gdx_path} v",
        )

        v3 = Variable(
            self.m,
            "v3",
            "positive",
            ["*", "*"],
            records=pd.DataFrame(
                [("seattle", "san-diego"), ("chicago", "madison")]
            ),
        )
        self.assertEqual(
            v3.getStatement(),
            f"positive Variable v3(*,*);\n$gdxLoad {self.m._gdx_path} v3",
        )

    def test_variable_types(self):
        i = Set(self.m, "i", records=["1", "2"])

        v = Variable(self.m, name="v", type="Positive")
        self.assertEqual(
            v.getStatement(),
            f"positive Variable v;\n$gdxLoad {self.m._gdx_path} v",
        )

        v1 = Variable(self.m, name="v1", type="Negative")
        self.assertEqual(
            v1.getStatement(),
            f"negative Variable v1;\n$gdxLoad {self.m._gdx_path} v1",
        )

        v2 = Variable(self.m, name="v2", type="Binary")
        self.assertEqual(
            v2.getStatement(),
            f"binary Variable v2;\n$gdxLoad {self.m._gdx_path} v2",
        )

        v3 = Variable(self.m, name="v3", domain=[i], type="Integer")
        self.assertEqual(
            v3.getStatement(),
            f"integer Variable v3(i);\n$gdxLoad {self.m._gdx_path} v3",
        )

    def test_variable_attributes(self):
        pi = Variable(
            self.m,
            "pi",
            records=pd.DataFrame(data=[3.14159], columns=["level"]),
        )

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
            hasattr(pi, "fx")
            and isinstance(pi.fx, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.fx.gamsRepr(), "pi.fx")

        self.assertTrue(
            hasattr(pi, "prior")
            and isinstance(pi.prior, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.prior.gamsRepr(), "pi.prior")

        self.assertTrue(
            hasattr(pi, "stage")
            and isinstance(pi.stage, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.stage.gamsRepr(), "pi.stage")

        i = Set(self.m, name="i", records=["bla", "damn"])
        test = Variable(self.m, "test", domain=[i])
        self.assertTrue(
            hasattr(test, "l")
            and isinstance(test.l, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "m")
            and isinstance(test.m, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "lo")
            and isinstance(test.lo, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "up")
            and isinstance(test.up, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "scale")
            and isinstance(test.scale, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "fx")
            and isinstance(test.fx, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "prior")
            and isinstance(test.prior, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "stage")
            and isinstance(test.stage, implicits.ImplicitParameter)
        )

    def test_implicit_variable(self):
        i = Set(self.m, "i", records=[f"i{i}" for i in range(10)])
        a = Variable(self.m, "a", "free", [i])
        a.generateRecords()
        self.assertTrue(a.isValid())

        expression = -a[i] * 5
        self.assertEqual(expression.gamsRepr(), "(-a(i) * 5)")

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
            hasattr(a[i], "fx")
            and isinstance(a[i].fx, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].fx.gamsRepr(), "a(i).fx")
        self.assertTrue(
            hasattr(a[i], "prior")
            and isinstance(a[i].prior, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].prior.gamsRepr(), "a(i).prior")
        self.assertTrue(
            hasattr(a[i], "stage")
            and isinstance(a[i].stage, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].stage.gamsRepr(), "a(i).stage")

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
        eq1.definition = x == c
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

        eq2[i].definition = x[i] == c[i]
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
            'Equation cost "define objective function" / /;',
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
        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        with self.assertRaises(TypeError):
            cost.records = 5

        self.assertIsNotNone(cost.definition)
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
        bla[i, j].definition = Sum((i, j), x[i, j]) <= a[i]
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
            definition=Sum((i, j), c[i, j] * x[i, j]) == z,
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "cost2 .. sum((i,j),(c(i,j) * x(i,j))) =e= z;",
        )

        # Equation definition in addEquation
        _ = self.m.addEquation(
            name="cost2",
            description="define objective function",
            definition=Sum((i, j), c[i, j] * x[i, j]) == z,
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "cost2 .. sum((i,j),(c(i,j) * x(i,j))) =e= z;",
        )

        # eq[bla].definition test
        bla2 = Equation(
            self.m,
            name="bla2",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla2[i, j].definition = Sum((i, j), x[i, j]) <= a[i]
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
            definition=Sum((i, j), x[i, j]) <= a[i],
            definition_domain=[i, "bla"],
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

    def test_model(self):
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
        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        supply[i] = Sum(j, x[i, j]) <= a[i]

        # Equation definition with more than one index
        bla = Equation(
            self.m,
            name="bla",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla[i, j] = Sum((i, j), x[i, j]) <= a[i]

        # Test model with specific equations
        test_model2 = Model(
            self.m,
            name="test_model2",
            equations=[cost, supply],
            problem="LP",
            sense="min",
            objective=z,
        )
        self.assertEqual(
            test_model2.getStatement(), "Model test_model2 / cost,supply /;"
        )
        self.assertEqual(test_model2.equations, [cost, supply])

        test_model3 = Model(
            self.m,
            name="test_model3",
            equations=[cost],
            problem="LP",
            sense="min",
            objective=z,
        )
        test_model3.equations = [cost, supply]
        self.assertEqual(test_model3.equations, [cost, supply])

        test_model4 = self.m.addModel(
            name="test_model4",
            equations=[cost, supply],
            problem="LP",
            sense="min",
            objective=z,
        )

        self.assertTrue(test_model4.equations == test_model3.equations)

        test_model5 = self.m.addModel(
            name="test_model5",
            equations=[cost, supply],
            problem="LP",
            sense="min",
            objective=z,
            matches={supply: x, cost: z},
        )
        self.assertEqual(
            test_model5.getStatement(),
            "Model test_model5 / supply.x,cost.z /;",
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
        self.assertEqual(expression.gamsRepr(), "(ord(i) = ord(j))")
        expression = Card(i) == 5
        self.assertEqual(expression.gamsRepr(), "(card(i) = 5)")

    def test_condition_on_expression(self):
        steel_plants = ["ahmsa", "fundidora", "sicartsa", "hylsa", "hylsap"]
        markets = ["mexico-df", "monterrey", "guadalaja"]

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

        # Set
        i = Set(
            self.m,
            name="i",
            records=pd.DataFrame(steel_plants),
            description="steel plants",
        )
        j = Set(
            self.m,
            name="j",
            records=pd.DataFrame(markets),
            description="markets",
        )

        # Data
        rd = Parameter(
            self.m,
            name="rd",
            domain=["*", "*"],
            records=rail_distances,
            description="rail distances from plants to markets",
        )
        muf = Parameter(
            self.m,
            name="muf",
            domain=[i, j],
            description="transport rate: final products",
        )

        # Condition
        muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]

        last_statement = list(self.m._statements_dict.values())[-1]
        self.assertEqual(
            last_statement.getStatement(),
            "muf(i,j) = ((2.48 + (0.0084 * rd(i,j))) $ (rd(i,j)));",
        )

        m = Container()

        p = Set(m, name="p", records=[f"pos{i}" for i in range(1, 11)])
        o = Set(m, name="o", records=[f"opt{i}" for i in range(1, 6)])

        sumc = Parameter(m, name="sumc", domain=[o, p])
        sumc[o, p] = gams_math.uniform(0, 1)

        op = Variable(m, name="op", type="free", domain=[o, p])

        # Equation
        defopLS = Equation(m, name="defopLS", domain=[o, p])
        defopLS[o, p].where[sumc[o, p] <= 0.5] = op[o, p] == Number(1)
        self.assertEqual(
            list(m._statements_dict.values())[-1].getStatement(),
            "defopLS(o,p) $ (sumc(o,p) <= 0.5) .. op(o,p) =e= 1;",
        )

        expression = Sum(i, muf[i, j]).where[muf[i, j] > 0]
        self.assertEqual(
            expression.getStatement(), "(sum(i,muf(i,j)) $ (muf(i,j) > 0))"
        )

        random_eq = Equation(m, "random", domain=[i, j])
        random_eq[i, j] = Sum(i, muf[i, j]).where[muf[i, j] > 0] >= 0
        self.assertEqual(
            list(m._statements_dict.values())[-1].getStatement(),
            "random(i,j) .. (sum(i,muf(i,j)) $ (muf(i,j) > 0)) =g= 0;",
        )

        i["ahmsa"] = True
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            'i("ahmsa") = yes;',
        )

        i["ahmsa"] = False
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            'i("ahmsa") = no;',
        )

    def test_condition_on_equation(self):
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

        # Sets
        w = Set(
            self.m,
            name="w",
            records=["icbm", "mrbm-1", "lr-bomber", "f-bomber", "mrbm-2"],
        )
        t = Set(self.m, name="t", records=[str(i) for i in range(1, 21)])

        # Parameters
        td = Parameter(self.m, name="td", domain=[w, t], records=td_data)
        wa = Parameter(self.m, name="wa", domain=[w], records=wa_data)
        tm = Parameter(self.m, name="tm", domain=[t], records=tm_data)

        # Variables
        x = Variable(self.m, name="x", domain=[w, t], type="Positive")

        # Equations
        maxw = Equation(self.m, name="maxw", domain=[w])
        minw = Equation(self.m, name="minw", domain=[t])

        maxw[w] = Sum(t.where[td[w, t]], x[w, t]) <= wa[w]
        minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]

        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "minw(t) $ (tm(t)) .. sum(w $ td(w,t),x(w,t)) =g= tm(t);",
        )

        m = Container()

        p = Set(m, name="p", records=[f"pos{i}" for i in range(1, 11)])
        o = Set(m, name="o", records=[f"opt{i}" for i in range(1, 6)])

        # Variables
        sumc = Variable(m, name="sumc", type="free", domain=[o, p])
        op = Variable(m, name="op", type="free", domain=[o, p])

        # Equation
        defopLS = Equation(m, name="defopLS", domain=[o, p])
        defopLS[o, p] = op[o, p] == Number(1).where[sumc[o, p] >= 0.5]
        self.assertEqual(
            list(m._statements_dict.values())[-1].getStatement(),
            "defopLS(o,p) .. op(o,p) =e= (1 $ (sumc(o,p) >= 0.5));",
        )

        k = Set(m, "k", domain=[p])
        k[p].where[k[p]] = True
        self.assertEqual(
            list(m._statements_dict.values())[-1].gamsRepr(),
            "k(p) $ (k(p)) = yes;",
        )

    def test_full_models(self):
        paths = glob.glob(
            str(Path(__file__).parent) + os.sep + "models" + os.sep + "*.py"
        )

        for idx, path in enumerate(paths):
            print(f"[{idx + 1}/{len(paths)}] {path.split(os.sep)[-1]}")
            try:
                process = subprocess.run(
                    ["python", path], check=True, capture_output=True
                )

                self.assertTrue(process.returncode == 0)
            except subprocess.CalledProcessError as e:
                print("(x)")
                print(f"Output: {e.stderr.decode('utf-8')}")
                exit(1)

    def test_operable_symbols(self):
        # Prepare data
        demands = pd.DataFrame(
            [["new-york", 325], ["chicago", 300], ["topeka", 275]]
        )

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        # Parameter
        b = Parameter(self.m, name="b", domain=[j], records=demands)

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")

        # ADD
        # Parameter + Variable, Variable + Parameter,
        # Parameter + builtin, builtin + Parameter
        op1 = b[i] + x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) + x(i))")
        op2 = x[i] + b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) + b(i))")
        op3 = b[i] + 5
        self.assertEqual(op3.gamsRepr(), "(b(i) + 5)")
        op4 = 5 + b[i]
        self.assertEqual(op4.gamsRepr(), "(5 + b(i))")

        # SUB
        # Parameter - Variable, Variable - Parameter,
        # Parameter - builtin, builtin - Parameter
        op1 = b[i] - x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) - x(i))")
        op2 = x[i] - b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) - b(i))")
        op3 = b[i] - 5
        self.assertEqual(op3.gamsRepr(), "(b(i) - 5)")
        op4 = 5 - b[i]
        self.assertEqual(op4.gamsRepr(), "(5 - b(i))")

        # MUL
        # Parameter * Variable, Variable * Parameter,
        # Parameter * builtin, builtin * Parameter
        op1 = b[i] * x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) * x(i))")
        op2 = x[i] * b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) * b(i))")
        op3 = b[i] * -5
        self.assertEqual(op3.gamsRepr(), "(b(i) * (-5))")
        op4 = -5 * b[i]
        self.assertEqual(op4.gamsRepr(), "((-5) * b(i))")

        # DIV
        # Parameter / Variable, Variable / Parameter,
        # Parameter / builtin, builtin / Parameter
        op1 = b[i] / x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) / x(i))")
        op2 = x[i] / b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) / b(i))")
        op3 = b[i] / 5
        self.assertEqual(op3.gamsRepr(), "(b(i) / 5)")
        op4 = 5 / b[i]
        self.assertEqual(op4.gamsRepr(), "(5 / b(i))")

        # POW
        # Parameter ** Variable, Variable ** Parameter
        op1 = b[i] ** x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) ** x(i))")
        op2 = x[i] ** b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) ** b(i))")

        # Set/Parameter/Variable ** 2
        op1 = i**2
        self.assertEqual(op1.gamsRepr(), "(sqr( i ))")
        op2 = b[i] ** 2
        self.assertEqual(op2.gamsRepr(), "(sqr( b(i) ))")
        op3 = x[i] ** 2
        self.assertEqual(op3.gamsRepr(), "(sqr( x(i) ))")

        # AND
        # Parameter and Variable, Variable and Parameter
        op1 = b[i] & x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) and x(i))")
        op2 = x[i] & b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) and b(i))")

        # RAND
        op1 = 5 & b[i]
        self.assertEqual(op1.gamsRepr(), "(5 and b(i))")

        # OR
        # Parameter or Variable, Variable or Parameter
        op1 = b[i] | x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) or x(i))")
        op2 = x[i] | b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) or b(i))")

        # ROR
        op1 = 5 | b[i]
        self.assertEqual(op1.gamsRepr(), "(5 or b(i))")

        # XOR
        # Parameter xor Variable, Variable xor Parameter
        op1 = b[i] ^ x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) xor x(i))")
        op2 = x[i] ^ b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) xor b(i))")

        # RXOR
        op1 = 5 ^ x[i]
        self.assertEqual(op1.gamsRepr(), "(5 xor x(i))")

        # LT
        # Parameter < Variable, Variable < Parameter
        op1 = b[i] < x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) < x(i))")
        op2 = x[i] < b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) < b(i))")

        # LE
        # Parameter <= Variable, Variable <= Parameter
        op1 = b[i] <= x[i]
        self.assertEqual(op1.gamsRepr(), "b(i) =l= x(i)")
        op2 = x[i] <= b[i]
        self.assertEqual(op2.gamsRepr(), "x(i) =l= b(i)")

        # GT
        # Parameter > Variable, Variable > Parameter
        op1 = b[i] > x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) > x(i))")
        op2 = x[i] > b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) > b(i))")

        # GE
        # Parameter >= Variable, Variable >= Parameter
        op1 = b[i] >= x[i]
        self.assertEqual(op1.gamsRepr(), "b(i) =g= x(i)")
        op2 = x[i] >= b[i]
        self.assertEqual(op2.gamsRepr(), "x(i) =g= b(i)")

        # NE
        # Parameter != Variable, Variable != Parameter
        op1 = b[i] != x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) ne x(i))")
        op2 = x[i] != b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) ne b(i))")

        # E
        # Parameter == Variable, Variable == Parameter
        op1 = b[i] == x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) = x(i))")
        op2 = x[i] == b[i]
        self.assertEqual(op2.gamsRepr(), "x(i) =e= b(i)")
        op3 = b[i] == b[i]
        self.assertEqual(op3.gamsRepr(), "(b(i) = b(i))")

        # not
        # not Parameter/Variable
        op1 = ~b[i]
        self.assertEqual(op1.gamsRepr(), "( not b(i))")
        op2 = ~x[i]
        self.assertEqual(op2.gamsRepr(), "( not x(i))")

    def test_math(self):
        # Prepare data
        demands = pd.DataFrame(
            [["new-york", 325], ["chicago", 300], ["topeka", 275]]
        )

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        # Parameter
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        s1 = Parameter(self.m, name="s1", records=5)
        s2 = Parameter(self.m, name="s2", records=3)
        s3 = Parameter(self.m, name="s3", records=6)

        # Variable
        v = Variable(self.m, name="v", domain=[i])

        # abs
        op1 = gams_math.abs(-5)
        self.assertEqual(op1, 5)
        op2 = gams_math.abs(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(abs( b(i) ))")

        # ceil
        op1 = gams_math.ceil(7.5)
        self.assertTrue(isinstance(op1, int) and op1 == 8)
        op2 = gams_math.ceil(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(ceil( b(i) ))")

        # centropy
        op2 = gams_math.centropy(v[i], b[i])
        self.assertEqual(op2.gamsRepr(), "(centropy( v(i),b(i),1e-20 ))")
        op2 = gams_math.centropy(v[i], b[i], 1e-15)
        self.assertEqual(op2.gamsRepr(), "(centropy( v(i),b(i),1e-15 ))")
        self.assertRaises(ValueError, gams_math.centropy, v[i], b[i], -1)

        # sqrt
        op1 = gams_math.sqrt(9)
        self.assertEqual(op1, 3)
        op2 = gams_math.sqrt(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(sqrt( b(i) ))")

        # exp
        op1 = gams_math.exp(3)
        self.assertEqual(op1, math.exp(3))
        op2 = gams_math.exp(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(exp( b(i) ))")

        # power
        op1 = gams_math.power(2, 3)
        self.assertEqual(op1, math.pow(2, 3))
        op2 = gams_math.power(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(power( b(i),3 ))")

        # sqr
        op1 = gams_math.sqr(4)
        self.assertEqual(op1, 4**2)
        op2 = gams_math.sqr(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(power( b(i),2 ))")

        # mod
        op1 = gams_math.mod(5, 2)
        self.assertEqual(op1, 1)
        op2 = gams_math.mod(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(mod(b(i) , 3))")

        # min
        op2 = gams_math.min(s1, s2, s3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(min( s1,s2,s3 ))")

        # max
        op2 = gams_math.max(s1, s2, s3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(max( s1,s2,s3 ))")

        # log
        op1 = gams_math.log(3)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.log(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(log( b(i) ))")

        # log2
        op1 = gams_math.log2(8)
        self.assertEqual(op1, 3)
        op2 = gams_math.log2(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(log2( b(i) ))")

        # log10
        op1 = gams_math.log10(100)
        self.assertEqual(op1, 2)
        op2 = gams_math.log10(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(log10( b(i) ))")

        # round
        op2 = gams_math.Round(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(round( b(i), 0 ))")

        # sin
        op1 = gams_math.sin(8)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.sin(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(sin( b(i) ))")

        # asin
        op1 = gams_math.asin(0.5)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.asin(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(arcsin( b(i) ))")

        # cos
        op1 = gams_math.cos(8)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.cos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(cos( b(i) ))")

        # arccos
        op1 = gams_math.acos(0.5)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.acos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(arccos( b(i) ))")

        # cos
        op1 = gams_math.cos(8)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.cos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(cos( b(i) ))")

        # tan
        op1 = gams_math.tan(0.5)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.tan(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(tan( b(i) ))")

        # arctan
        op1 = gams_math.atan(7.5)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.atan(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(arctan( b(i) ))")

        # floor
        op1 = gams_math.floor(7.5)
        self.assertTrue(isinstance(op1, int) and op1 == 7)
        op2 = gams_math.floor(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(floor( b(i) ))")

        # uniform
        op2 = gams_math.uniform(0, 1)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(uniform( 0,1 ))")

        # uniformInt
        op2 = gams_math.uniformInt(0, 1)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(uniformInt( 0,1 ))")

        # normal
        op2 = gams_math.normal(mean=0, dev=1)
        self.assertTrue(op2, expression.Expression)
        self.assertEqual(op2.gamsRepr(), "(normal( 0,1 ))")

        p = Parameter(self.m, "p", domain=[i])
        op2 = gams_math.sign(p[i])
        self.assertEqual(op2.gamsRepr(), "(sign( p(i) ))")

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

    def test_read_on_demand(self):
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
        capacities = [["seattle", 350], ["san-diego", 600]]
        demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])
        k = Set(
            self.m, name="k", records=["seattle", "san-diego", "california"]
        )
        k["seattle"] = False
        self.assertTrue(k._is_dirty)
        self.assertEqual(
            k.records.loc[0, :].values.tolist(), ["san-diego", ""]
        )
        self.assertFalse(k._is_dirty)

        # Data
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        e = Parameter(self.m, name="e")

        c[i, j] = 90 * d[i, j] / 1000
        self.assertTrue(c._is_dirty)
        self.assertEqual(
            c.records.values.tolist(),
            [
                ["seattle", "new-york", 0.225],
                ["seattle", "chicago", 0.153],
                ["seattle", "topeka", 0.162],
                ["san-diego", "new-york", 0.225],
                ["san-diego", "chicago", 0.162],
                ["san-diego", "topeka", 0.126],
            ],
        )
        self.assertFalse(c._is_dirty)

        e.assign = 5
        self.assertTrue(e._is_dirty)
        self.assertEqual(e.records.values.tolist(), [[5.0]])
        self.assertEqual(e.assign, 5)

        with self.assertRaises(TypeError):
            e.records = 5

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation
        cost = Equation(self.m, name="cost")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
        )
        transport.solve()

        # Test the columns of a set
        self.assertTrue(i.records.columns.tolist() == ["uni", "element_text"])

        # Test the columns of a parameter
        self.assertTrue(a.records.columns.tolist() == ["i", "value"])

        # Test the columns of scalar variable
        self.assertTrue(
            z.records.columns.tolist()
            == ["level", "marginal", "lower", "upper", "scale"]
        )

        # Test the columns of indexed variable
        self.assertTrue(
            x.records.columns.tolist()
            == ["i", "j", "level", "marginal", "lower", "upper", "scale"]
        )

        # Test the columns of equation
        self.assertTrue(
            cost.records.columns.tolist()
            == ["level", "marginal", "lower", "upper", "scale"]
        )

        # Test the columns of indexed equation
        self.assertTrue(
            supply.records.columns.tolist()
            == ["i", "level", "marginal", "lower", "upper", "scale"]
        )

    def test_number(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        d = Parameter(
            self.m,
            name="d",
            domain=[i],
            records=pd.DataFrame([["i1", 2], ["i2", 3]]),
        )
        c = Parameter(self.m, name="c", domain=[i])

        c[i] = Number(1).where[d[i] > 2]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "c(i) = (1 $ (d(i) > 2));",
        )

    def test_options(self):
        options = {
            "limRow": 0,
            "limCol": 0,
            "solPrint": "silent",
            "solver": "cplex",
            "solveLink": "%solveLink.loadLibrary%",
        }
        self.m.addOptions(options)
        self.assertTrue(len(self.m._statements_dict), 5)
        self.assertEqual(
            list(self.m._statements_dict.values())[0], "limRow = 0"
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[1], "limCol = 0"
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[2], "solPrint = silent"
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[3], "solver = cplex"
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[4],
            "solveLink = %solveLink.loadLibrary%",
        )

        options = {"bla": 0}
        self.assertRaises(ValueError, self.m.addOptions, options)

    def test_loadRecordsFromGdx(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        a = Parameter(
            self.m, name="a", domain=[i], records=[("i1", 1), ("i2", 2)]
        )
        self.m.write("test.gdx")

        # Load all
        new_container = Container()
        i = Set(new_container, name="i")
        a = Parameter(new_container, name="a", domain=[i])
        new_container.loadRecordsFromGdx("test.gdx")

        # Set
        self.assertEqual(i.records.values.tolist(), [["i1", ""], ["i2", ""]])

        # Parameter
        self.assertEqual(a.records.values.tolist(), [["i1", 1.0], ["i2", 2.0]])

        # Load specific symbols
        new_container2 = Container()
        i = Set(new_container2, name="i")
        a = Parameter(new_container2, name="a", domain=[i])
        new_container2.loadRecordsFromGdx("test.gdx", [i])

        self.assertEqual(i.records.values.tolist(), [["i1", ""], ["i2", ""]])
        self.assertIsNone(a.records)

    def test_read(self):
        m = Container()
        _ = Set(m, "i", records=["i1", "i2"])
        _ = Set(m, "j", records=["j1", "j2"])
        m.write("test.gdx")

        _ = Set(self.m, name="k", records=["k1", "k2"])
        self.m.read("test.gdx", ["i"])
        self.assertEqual(list(self.m.data.keys()), ["k", "i"])

    def test_iterable(self):
        # Set with no records
        i = Set(self.m, "i")
        count = 0
        for _ in i:
            count += 1

        self.assertEqual(count, 0)

        # Set with records
        k = Set(self.m, "k", records=[str(idx) for idx in range(1, 3)])
        count = 0
        for _ in k:
            count += 1

        self.assertEqual(count, 2)

        # Alias with no records
        x = Set(self.m, "x")
        a = Alias(self.m, "a", x)
        count = 0
        for _ in a:
            count += 1

        self.assertEqual(count, 0)

        # Alias with records
        b = Set(self.m, "b", records=[str(idx) for idx in range(1, 3)])
        c = Alias(self.m, "c", b)

        count = 0
        for _ in c:
            count += 1

        self.assertEqual(count, 2)

    def test_misc(self):
        u = Set(self.m, "u")
        v = Alias(self.m, "v", alias_with=u)
        e = Set(self.m, "e", domain=[u, v])
        eq = Equation(self.m, "eq", domain=[u, v])
        self.assertEqual(eq[e[u, v]].gamsRepr(), "eq(e(u,v))")

        m = Container()
        s = Set(m, name="s", records=[str(i) for i in range(1, 4)])
        c = Parameter(m, name="c", domain=[s])
        c[s].where[Ord(s) <= Ord(s)] = 1
        self.assertEqual(
            list(m._statements_dict.values())[-1].getStatement(),
            "c(s) $ (ord(s) <= ord(s)) = 1;",
        )

        # Parameter record override
        c = m.addParameter(
            name="c",
            domain=[s],
            records=[("1", 1), ("2", 2), ("3", 3)],
            description="new description",
        )
        self.assertEqual(c.description, "new description")

        # Try to add the same parameter
        self.assertRaises(ValueError, m.addParameter, "c", [s, s])

        # Try to add the same Alias with non-Set alias_with
        self.assertRaises(TypeError, self.m.addAlias, "v", eq)

        # Try to add the same alias
        self.assertRaises(ValueError, self.m.addAlias, "u", u)

        # Test operation index
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
            list(m._statements_dict.values())[-1].gamsRepr(),
            "eStartFast(g,t1) .. sum(tt(t) $ (ord(t) <="
            " pMinDown(g,t1)),vStart(g,t + (ord(t1) - pMinDown(g,t1))))"
            " =l= 1;",
        )

    def test_arbitrary_gams_code(self):
        self.m._addGamsCode("Set i / i1*i3 /;")
        self.assertEqual(
            list(self.m._unsaved_statements.values())[-1], "Set i / i1*i3 /;"
        )

    def test_isin(self):
        i = Set(self.m, "i")
        j = Set(self.m, "j")
        k = Set(self.m, "k")
        symbols = [i, j]

        self.assertTrue(utils.isin(i, symbols))
        self.assertFalse(utils.isin(k, symbols))

    def test_equals(self):
        i = Set(self.m, "i")
        j = Set(self.m, "j")

        self.assertFalse(i.equals(j))
        self.assertTrue(i.equals(i))

    def test_after_first_solve(self):
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
        capacities = pd.DataFrame([["seattle", 350], ["san-diego", 600]])
        demands = pd.DataFrame(
            [["new-york", 325], ["chicago", 300], ["topeka", 275]]
        )

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")
        z2 = Variable(self.m, name="z2")

        # Equation
        cost = Equation(self.m, name="cost")
        cost2 = Equation(self.m, name="cost2")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]
        cost2.definition = Sum((i, j), c[i, j] * x[i, j]) * 5 == z2

        transport = Model(
            self.m,
            name="transport",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
        )
        transport.solve()

        first_z2_value = z2.records["level"].values[0]
        self.assertEqual(first_z2_value, 0.0)

        transport2 = Model(
            self.m,
            name="transport2",
            equations=[cost2, supply, demand],
            problem="LP",
            sense="min",
            objective=z2,
        )
        transport2.solve()
        second_z2_value = z2.records["level"].values[0]
        self.assertAlmostEqual(second_z2_value, 768.375, 3)

    def test_solve(self):
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
        capacities = pd.DataFrame([["seattle", 350], ["san-diego", 600]])
        demands = pd.DataFrame(
            [["new-york", 325], ["chicago", 300], ["topeka", 275]]
        )

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation
        cost = Equation(self.m, name="cost")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
        )

        # Test output redirection
        with open("test.gms", "w") as file:
            _ = transport.solve(
                commandline_options={"resLim": 100},
                output=file,
            )

        self.assertTrue(os.path.exists("test.gms"))
        self.assertTrue(transport.status == ModelStatus.OptimalGlobal)
        for attr_name in transport._get_attribute_names().values():
            self.assertTrue(hasattr(transport, attr_name))

            # Make sure model attributes are not in the container
            self.assertFalse(attr_name in self.m.data.keys())

        # Make sure dummy variable and equation is not in the container
        self.assertFalse(any("dummy_" in name for name in self.m.data.keys()))

        # Test invalid problem
        self.assertRaises(ValueError, Model, self.m, "model", [cost], "bla")

        # Test invalid sense
        self.assertRaises(
            ValueError, Model, self.m, "model", [cost], "LP", "bla"
        )

        # Test invalid objective variable
        self.assertRaises(
            TypeError, Model, self.m, "model", [cost], "LP", "min", a
        )

        # Test invalid commandline options
        self.assertRaises(
            Exception,
            transport.solve,
            {"bla": 100},
        )

        self.assertRaises(Exception, transport.solve, 5)

        # Try to solve invalid model
        m = Container()
        cost = Equation(m, "cost")
        model = Model(m, "model", equations=[cost], problem="LP", sense="min")
        self.assertRaises(Exception, model.solve)

        # Test limited variables
        transport = Model(
            m,
            name="transport",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
            limited_variables=[x[i]],
        )

        self.assertEqual(
            transport.getStatement(),
            "Model transport / cost,supply,demand,x(i) /;",
        )

    def test_mcp_equation(self):
        c = Parameter(self.m, name="c", domain=[], records=0.5)
        x = Variable(
            self.m,
            name="x",
            domain=[],
            records={"lower": 1.0, "level": 1.5, "upper": 3.75},
        )
        f = Equation(self.m, name="f", domain=[], type="nonbinding")
        f.definition = x - c

        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "f .. (x - c) =n= 0;",
        )

    def test_equality_on_non_equation(self):
        j = Set(self.m, "j")
        h = Set(self.m, "h")
        hp = Alias(self.m, "hp", h)
        lamb = Parameter(self.m, "lambda", domain=[j, h])
        gamma = Parameter(self.m, "gamma", domain=[j, h])
        gamma[j, h] = Sum(hp.where[Ord(hp) >= Ord(h)], lamb[j, hp])
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "gamma(j,h) = sum(hp $ (ord(hp) >= ord(h)),lambda(j,hp));",
        )

    def test_container(self):
        import gams.transfer as gt

        i = gt.Set(self.m, "i")
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["i"], Set))

        j = gt.Alias(self.m, "j", i)
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["j"], Alias))

        _ = gt.Parameter(self.m, "a")
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["a"], Parameter))

        _ = gt.Variable(self.m, "v")
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["v"], Variable))

        _ = gt.Equation(self.m, "e", type="eq")
        self.m._cast_symbols()
        self.assertTrue(isinstance(self.m["e"], Equation))

        # test addX syntax
        m = Container()
        i1 = m.addSet("i")
        self.assertRaises(ValueError, m.addSet, "i", i1)
        self.assertTrue(isinstance(i1, Set))
        i2 = m.addSet("i")
        self.assertTrue(id(i1) == id(i2))
        i3 = m.addSet("i", records=["new_record"], description="new desc")
        self.assertTrue(id(i1) == id(i3))
        self.assertRaises(ValueError, m.addSet, "i", [j])
        self.assertRaises(TypeError, m.addSet, "i", None, 5)

        j1 = m.addAlias("j", i1)
        self.assertTrue(isinstance(j1, Alias))
        j2 = m.addAlias("j", i1)
        self.assertTrue(id(j1) == id(j2))
        j3 = m.addAlias("j", j2)
        self.assertTrue(id(j3) == id(j2))

        a1 = m.addParameter("a")
        self.assertRaises(ValueError, m.addParameter, "a", i1)
        self.assertTrue(isinstance(a1, Parameter))
        a2 = m.addParameter("a")
        self.assertTrue(id(a1) == id(a2))
        self.assertRaises(ValueError, m.addParameter, "a", ["*"])
        self.assertRaises(TypeError, m.addParameter, "a", None, None, 5)

        v1 = m.addVariable("v")
        self.assertRaises(ValueError, m.addVariable, "v", "free", domain=i1)
        self.assertTrue(isinstance(v1, Variable))
        v2 = m.addVariable("v", description="blabla", records=pd.DataFrame())
        self.assertTrue(id(v1) == id(v2))
        self.assertRaises(ValueError, m.addVariable, "v", "free", ["*"])
        self.assertRaises(ValueError, m.addVariable, "v", "dayum")

        e1 = m.addEquation("e")
        self.assertRaises(ValueError, m.addEquation, "e", "regular", i1)
        self.assertTrue(isinstance(e1, Equation))
        e2 = m.addEquation("e")
        self.assertTrue(id(e1) == id(e2))
        self.assertRaises(ValueError, m.addEquation, "e", "bla")
        self.assertRaises(ValueError, m.addEquation, "e", "leq")
        e3 = m.addEquation("e", records=pd.DataFrame())
        self.assertTrue(id(e3) == id(e1))

    def test_set_attributes(self):
        i = Set(self.m, "i")
        self.assertEqual(i.pos.gamsRepr(), "i.pos")
        self.assertEqual(i.ord.gamsRepr(), "i.ord")
        self.assertEqual(i.off.gamsRepr(), "i.off")
        self.assertEqual(i.rev.gamsRepr(), "i.rev")
        self.assertEqual(i.uel.gamsRepr(), "i.uel")
        self.assertEqual(i.len.gamsRepr(), "i.len")
        self.assertEqual(i.tlen.gamsRepr(), "i.tlen")
        self.assertEqual(i.val.gamsRepr(), "i.val")
        self.assertEqual(i.tval.gamsRepr(), "i.tval")
        self.assertEqual(i.first.gamsRepr(), "i.first")
        self.assertEqual(i.last.gamsRepr(), "i.last")

    def test_system_directory(self):
        import platform
        import gamspy

        gamspy_dir = os.path.dirname(gamspy.__file__)

        expected_path = ""
        user_os = platform.system().lower()
        expected_path += gamspy_dir + os.sep + "minigams" + os.sep + user_os
        if user_os == "darwin":
            expected_path += f"_{platform.machine()}"

        m = Container()
        self.assertEqual(m.system_directory.lower(), expected_path.lower())

        self.assertEqual(utils._getSystemDirectory("bla/bla"), "bla/bla")

    def test_sameas(self):
        i = Set(self.m, "i")
        j = Alias(self.m, "j", i)
        self.assertEqual(i.sameAs(j).gamsRepr(), "(sameAs( i,j ))")
        self.assertEqual(j.sameAs(i).gamsRepr(), "(sameAs( j,i ))")

    def test_utils(self):
        string = "(bla))"
        self.assertRaises(
            Exception, utils._getMatchingParanthesisIndices, string
        )

        string2 = "((bla)"
        self.assertRaises(
            Exception, utils._getMatchingParanthesisIndices, string2
        )

        i = Set(self.m, "i", records=["i1", "i2"])
        self.assertEqual(utils._getDomainStr([i, "b", "*"]), '(i,"b",*)')
        self.assertRaises(Exception, utils._getDomainStr, [5])

        # invalid system directory
        self.assertRaises(Exception, utils._openGdxFile, "bla", "bla")

        self.assertFalse(utils.checkAllSame([1, 2], [2]))
        self.assertFalse(utils.checkAllSame([1, 2], [2, 3]))
        self.assertTrue(utils.checkAllSame([1, 2], [1, 2]))

        # invalid load from path
        self.assertRaises(
            Exception, utils._openGdxFile, self.m.system_directory, "bla.gdx"
        )

        # invalid symbol
        self.assertRaises(Exception, utils._getSymbolData, None, None, "i")

    def test_functions(self):
        m = Container()

        o = Set(m, "o", records=[f"pos{idx}" for idx in range(1, 11)])
        p = Set(m, "p", records=[f"opt{idx}" for idx in range(1, 6)])
        sumc = Variable(m, "sumc", domain=[o, p])
        op = Variable(m, "op", domain=[o, p])
        defopLS = Equation(m, "defopLS", domain=[o, p])
        defopLS[o, p] = op[o, p] == ifthen(sumc[o, p] >= 0.5, 1, 0)
        self.assertEqual(
            list(m._statements_dict.values())[-1].gamsRepr(),
            "defopLS(o,p) .. op(o,p) =e= (ifthen(sumc(o,p) >= 0.5, 1, 0)  );",
        )


def suite():
    suite = unittest.TestSuite()
    tests = [
        GamspySuite(name)
        for name in dir(GamspySuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    result = runner.run(suite())

    if result.wasSuccessful():
        exit(0)
    exit(1)
