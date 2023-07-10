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
)


class GamspySuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_set_string(self):
        # Without domain
        i = Set(
            self.m,
            "i",
            records=["seattle", "san-diego"],
            description="dummy set",
        )
        self.assertEqual(i.gamsRepr(), "i")
        self.assertEqual(
            i.getStatement(), 'Set i(*) "dummy set" / seattle,san-diego /;'
        )

        # With one domain
        j = Set(self.m, "j", records=["seattle", "san-diego", "california"])
        k = Set(self.m, "k", domain=[j], records=["seattle", "san-diego"])
        self.assertEqual(k.gamsRepr(), "k")
        self.assertEqual(k.getStatement(), "Set k(j) / seattle,san-diego /;")

        # With two domain
        m = Set(self.m, "m", records=[f"i{i}" for i in range(2)])
        n = Set(self.m, "n", records=[f"j{i}" for i in range(2)])
        a = Set(self.m, "a", [m, n])
        a.generateRecords(density=1)
        self.assertEqual(a.gamsRepr(), "a")
        self.assertEqual(
            a.getStatement(), "Set a(m,n) / \ni0.j0\ni0.j1\ni1.j0\ni1.j1 /;"
        )

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
            (
                'Parameter a(i) "distances" / \nseattle 350.0\nsan-diego'
                " 600.0\ntopeka 500.0 /;"
            ),
        )

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

        # Variable without domain
        v0 = Variable(self.m, name="v0", description="some text")
        self.assertEqual(v0.gamsRepr(), "v0")
        self.assertEqual(v0.getStatement(), 'free Variable v0 "some text";')

        # Variable one domain
        v1 = Variable(self.m, name="v1", domain=[i])
        self.assertEqual(v1.gamsRepr(), "v1")
        self.assertEqual(v1.getStatement(), "free Variable v1(i);")

        # Variable two domain
        v2 = Variable(self.m, name="v2", domain=[i, j])
        self.assertEqual(v2.gamsRepr(), "v2")
        self.assertEqual(v2.getStatement(), "free Variable v2(i,j);")

        # Scalar variable with records
        pi = Variable(
            self.m,
            "pi",
            records=pd.DataFrame(data=[3.14159], columns=["level"]),
        )
        self.assertEqual(
            pi.getStatement(),
            "free Variable pi / L 3.14159,M 0.0,LO -inf,UP inf,scale 1.0/;",
        )

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
            (
                "free Variable v(*) / \ni0.L 0.0\ni0.M 0.0\ni0.LO -inf\ni0.UP"
                " inf\ni0.scale 1.0\ni1.L 0.0\ni1.M 1.0\ni1.LO -inf\ni1.UP"
                " inf\ni1.scale 1.0\ni2.L 0.0\ni2.M 2.0\ni2.LO -inf\ni2.UP"
                " inf\ni2.scale 1.0\ni3.L 0.0\ni3.M 3.0\ni3.LO -inf\ni3.UP"
                " inf\ni3.scale 1.0\ni4.L 0.0\ni4.M 4.0\ni4.LO -inf\ni4.UP"
                " inf\ni4.scale 1.0/;"
            ),
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
            (
                "positive Variable v3(*,*) / \nseattle.san-diego.L"
                " 0.0\nseattle.san-diego.M 0.0\nseattle.san-diego.LO"
                " 0.0\nseattle.san-diego.UP inf\nseattle.san-diego.scale"
                " 1.0\nchicago.madison.L 0.0\nchicago.madison.M"
                " 0.0\nchicago.madison.LO 0.0\nchicago.madison.UP"
                " inf\nchicago.madison.scale 1.0/;"
            ),
        )

    def test_variable_types(self):
        i = Set(self.m, "i", records=["1", "2"])

        v = Variable(self.m, name="v", type="Positive")
        self.assertEqual(v.getStatement(), "positive Variable v;")

        v1 = Variable(self.m, name="v1", type="Negative")
        self.assertEqual(v1.getStatement(), "negative Variable v1;")

        v2 = Variable(self.m, name="v2", type="Binary")
        self.assertEqual(v2.getStatement(), "binary Variable v2;")

        v3 = Variable(self.m, name="v3", domain=[i], type="Integer")
        self.assertEqual(v3.getStatement(), "integer Variable v3(i);")

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

    def test_implicit_variable_attributes(self):
        i = Set(self.m, "i", records=[f"i{i}" for i in range(10)])
        a = Variable(self.m, "a", "free", [i])
        a.generateRecords()
        self.assertTrue(a.isValid())

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

        eq2 = Equation(self.m, "eq2", domain=[i], type="nonbinding")
        eq2[i] = x[i] == c[i]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "eq2(i) .. x(i) =n= c(i);",
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
            type="eq",
            description="define objective function",
        )
        self.assertEqual(cost.gamsRepr(), "cost")
        self.assertEqual(
            cost.getStatement(), 'Equation cost "define objective function";'
        )

        # Equation declaration with an index
        supply = Equation(
            self.m,
            name="supply",
            type="geq",
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
            type="leq",
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
            type="eq",
            description="define objective function",
        )
        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "cost .. sum((i,j),(c(i,j) * x(i,j))) =e= z;",
        )

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            type="leq",
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
            type="geq",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla[i, j] = Sum((i, j), x[i, j]) <= a[i]
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "bla(i,j) .. sum((i,j),x(i,j)) =l= a(i);",
        )

        # Equation definition in constructor
        _ = Equation(
            self.m,
            name="cost2",
            type="eq",
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
            type="eq",
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
            type="geq",
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
            type="geq",
            domain=[i, j],
            description="observe supply limit at plant i",
            definition=Sum((i, j), x[i, j]) <= a[i],
            definition_domain=[i, "bla"],
        )
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            'bla3(i,"bla") .. sum((i,j),x(i,j)) =l= a(i);',
        )

    def test_equation_attributes(self):
        pi = Equation(self.m, "pi", type="eq")

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
        a = Equation(self.m, "a", "eq", [i])

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

    def test_model_string(self):
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
            type="eq",
            description="define objective function",
        )
        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            type="leq",
            domain=[i],
            description="observe supply limit at plant i",
        )
        supply[i] = Sum(j, x[i, j]) <= a[i]

        # Equation definition with more than one index
        bla = Equation(
            self.m,
            name="bla",
            type="geq",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla[i, j] = Sum((i, j), x[i, j]) <= a[i]

        # Test all
        test_model = Model(self.m, name="test_model", equations="all")
        self.assertEqual(
            test_model.getStatement(), "\nModel test_model / all /;"
        )

        # Test model with specific equations
        test_model2 = Model(
            self.m, name="test_model2", equations=[cost, supply]
        )
        self.assertEqual(
            test_model2.getStatement(), "\nModel test_model2 / cost,supply /;"
        )

        # Test limited variables
        test_model3 = Model(
            self.m,
            name="test_model3",
            equations="all",
            limited_variables=[x[i]],
        )
        self.assertEqual(
            test_model3.getStatement(), "\nModel test_model3 / all,x(i) /;"
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

    def test_condition_onexpression(self):
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
            "muf(i,j) = ((2.48 + (0.0084 * rd(i,j))) $ rd(i,j));",
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
        maxw = Equation(self.m, name="maxw", type="leq", domain=[w])
        minw = Equation(self.m, name="minw", type="geq", domain=[t])

        maxw[w] = Sum(t.where[td[w, t]], x[w, t]) <= wa[w]
        minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]

        self.assertEqual(
            list(self.m._statements_dict.values())[-1].getStatement(),
            "minw(t) $ tm(t).. sum(w $ td(w,t),x(w,t)) =g= tm(t);",
        )

    def test_full_models(self):
        paths = glob.glob(
            str(Path(__file__).parent.absolute()) + "/models/*.py"
        )

        for idx, path in enumerate(paths):
            print(
                f"[{idx + 1}/{len(paths)}] {path.split(os.sep)[-1]}", end=" "
            )
            process = subprocess.run(
                [
                    os.environ["PYTHON38"]
                    if "PYTHON38" in os.environ
                    else "python3",
                    path,
                ],
                check=True,
            )

            self.assertTrue(process.returncode == 0)
            _ = print("PASSED") if process.returncode == 0 else print("FAILED")

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

        # OR
        # Parameter or Variable, Variable or Parameter
        op1 = b[i] | x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) or x(i))")
        op2 = x[i] | b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) or b(i))")

        # XOR
        # Parameter xor Variable, Variable xor Parameter
        op1 = b[i] ^ x[i]
        self.assertEqual(op1.gamsRepr(), "(b(i) xor x(i))")
        op2 = x[i] ^ b[i]
        self.assertEqual(op2.gamsRepr(), "(x(i) xor b(i))")

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

        # abs
        op1 = gams_math.abs(-5)
        self.assertEqual(op1, 5)
        op2 = gams_math.abs(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(abs( b(i) ))")

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

        op1 = gams_math.power(2, 3)
        self.assertEqual(op1, math.pow(2, 3))
        op2 = gams_math.power(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(power( b(i),3 ))")

        # mod
        op1 = gams_math.mod(5, 2)
        self.assertEqual(op1, 1)
        op2 = gams_math.mod(b[i], 3)
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(mod(b(i) , 3))")

        # min
        op2 = gams_math.min([s1, s2, s3])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(min( s1,s2,s3 ))")

        # max
        op2 = gams_math.max([s1, s2, s3])
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

        # cos
        op1 = gams_math.cos(8)
        self.assertTrue(isinstance(op1, float))
        op2 = gams_math.cos(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(cos( b(i) ))")

        # ceil
        op1 = gams_math.ceil(7.5)
        self.assertTrue(isinstance(op1, int) and op1 == 8)
        op2 = gams_math.ceil(b[i])
        self.assertTrue(isinstance(op2, expression.Expression))
        self.assertEqual(op2.gamsRepr(), "(ceil( b(i) ))")

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

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation
        cost = Equation(self.m, name="cost", type="eq")
        supply = Equation(self.m, name="supply", domain=[i], type="leq")
        demand = Equation(self.m, name="demand", domain=[j], type="geq")

        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(self.m, name="transport", equations="all")

        self.m.solve(
            transport, problem="LP", sense="min", objective_variable=z
        )

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

    def test_loadFromGdx(self):
        i = Set(self.m, name="i", records=["i1", "i2"])
        a = Parameter(
            self.m, name="a", domain=[i], records=[("i1", 1), ("i2", 2)]
        )
        self.m.write("test.gdx")

        # Load all
        new_container = Container()
        i = Set(new_container, name="i")
        a = Parameter(new_container, name="a", domain=[i])
        new_container.loadFromGdx("test.gdx")

        # Set
        self.assertEqual(i.records.values.tolist(), [["i1", ""], ["i2", ""]])

        # Parameter
        self.assertEqual(a.records.values.tolist(), [["i1", 1.0], ["i2", 2.0]])

        # Load specific symbols
        new_container2 = Container()
        i = Set(new_container2, name="i")
        a = Parameter(new_container2, name="a", domain=[i])
        new_container2.loadFromGdx("test.gdx", [i])

        self.assertEqual(i.records.values.tolist(), [["i1", ""], ["i2", ""]])
        self.assertIsNone(a.records)

    def test_iterable(self):
        # Set with no records
        def iterate_over_set():
            i = Set(self.m, "i")
            for elem in i:
                print(elem)

        self.assertRaises(AssertionError, iterate_over_set)

        # Set with records
        k = Set(self.m, "k", records=[str(idx) for idx in range(1, 3)])
        for elem in k:
            self.assertTrue(isinstance(elem, tuple))
            self.assertTrue(isinstance(elem[0], int))
            self.assertTrue(isinstance(elem[1], pd.Series))

        # Alias with no records
        def iterate_over_alias():
            x = Set(self.m, "x")
            a = Alias(self.m, "a", x)
            for elem in a:
                print(elem)

        self.assertRaises(AssertionError, iterate_over_alias)

        # Alias with records
        b = Set(self.m, "b", records=[str(idx) for idx in range(1, 3)])
        c = Alias(self.m, "c", b)

        for elem in c:
            self.assertTrue(isinstance(elem, tuple))
            self.assertTrue(isinstance(elem[0], int))
            self.assertTrue(isinstance(elem[1], pd.Series))

        # Parameter with no records
        def iterate_over_parameter():
            d = Parameter(self.m, "d")
            for elem in d:
                print(elem)

        self.assertRaises(AssertionError, iterate_over_parameter)

        # Parameter with records
        e = Parameter(self.m, "e", domain=[k], records=[("1", 1), ("2", 2)])
        for elem in e:
            self.assertTrue(isinstance(elem, tuple))
            self.assertTrue(isinstance(elem[0], int))
            self.assertTrue(isinstance(elem[1], pd.Series))

        # Variable with no records
        def iterate_over_variable():
            f = Variable(self.m, "f")
            for elem in f:
                print(elem)

        self.assertRaises(AssertionError, iterate_over_variable)

        # Variable with records
        g = Variable(
            self.m,
            "g",
            domain=[k],
            records=pd.DataFrame(
                data=[(str(i), i) for i in range(1, 3)],
                columns=["domain", "marginal"],
            ),
        )
        for elem in g:
            self.assertTrue(isinstance(elem, tuple))
            self.assertTrue(isinstance(elem[0], int))
            self.assertTrue(isinstance(elem[1], pd.Series))

        # Equation with no records
        def iterate_over_equation():
            h = Equation(self.m, "h", type="eq")
            for elem in h:
                print(elem)

        self.assertRaises(AssertionError, iterate_over_equation)

    def test_misc(self):
        u = Set(self.m, "u")
        v = Alias(self.m, "v", alias_with=u)
        e = Set(self.m, "e", domain=[u, v])
        eq = Equation(self.m, "eq", domain=[u, v], type="leq")
        self.assertEqual(eq[e[u, v]].gamsRepr(), "eq(e(u,v))")

    def test_arbitrary_gams_code(self):
        self.m.addGamsCode("Set i / i1*i3 /;")
        self.assertEqual(
            list(self.m._statements_dict.values())[-1], "Set i / i1*i3 /;  "
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
        cost = Equation(self.m, name="cost", type="eq")
        cost2 = Equation(self.m, name="cost2", type="eq")
        supply = Equation(self.m, name="supply", domain=[i], type="leq")
        demand = Equation(self.m, name="demand", domain=[j], type="geq")

        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]
        cost2.definition = Sum((i, j), c[i, j] * x[i, j]) * 5 == z2

        transport = Model(
            self.m, name="transport", equations=[cost, supply, demand]
        )

        self.m.solve(
            transport, problem="LP", sense="min", objective_variable=z
        )
        first_z2_value = z2.records["level"].values[0]
        self.assertEqual(first_z2_value, 0.0)

        transport2 = Model(
            self.m, name="transport2", equations=[cost2, supply, demand]
        )
        self.m.solve(
            transport2, problem="LP", sense="min", objective_variable=z2
        )
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
        cost = Equation(self.m, name="cost", type="eq")
        supply = Equation(self.m, name="supply", domain=[i], type="leq")
        demand = Equation(self.m, name="demand", domain=[j], type="geq")

        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(self.m, name="transport", equations="all")

        # Test output redirection
        output = self.m.solve(
            transport,
            problem="LP",
            sense="min",
            objective_variable=z,
            stdout="test.gms",
        )

        self.assertTrue(os.path.exists("test.gms"))
        self.assertTrue(output is not None and type(output) == str)

        # Test invalid problem
        self.assertRaises(ValueError, self.m.solve, transport, "bla", "min", z)

        # Test invalid sense
        self.assertRaises(ValueError, self.m.solve, transport, "LP", "bla", z)

        # Test invalid stdout options
        self.assertRaises(
            ValueError, self.m.solve, transport, "LP", "bla", z, None, "bla"
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

        _ = gt.Alias(self.m, "j", i)
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
        self.assertTrue(isinstance(i1, Set))
        i2 = m.addSet("i")
        self.assertTrue(id(i1) == id(i2))

        j1 = m.addAlias("j", i1)
        self.assertTrue(isinstance(j1, Alias))
        j2 = m.addAlias("j", i1)
        self.assertTrue(id(j1) == id(j2))

        a1 = m.addParameter("a")
        self.assertTrue(isinstance(a1, Parameter))
        a2 = m.addParameter("a")
        self.assertTrue(id(a1) == id(a2))

        v1 = m.addVariable("v")
        self.assertTrue(isinstance(v1, Variable))
        v2 = m.addVariable("v")
        self.assertTrue(id(v1) == id(v2))

        e1 = m.addEquation("e", type="eq")
        self.assertTrue(isinstance(e1, Equation))
        e2 = m.addEquation("e", type="eq")
        self.assertTrue(id(e1) == id(e2))

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

    def test_setitem_errors(self):
        distances = [
            ["seattle", "new-york", 2.5],
            ["seattle", "chicago", 1.7],
            ["seattle", "topeka", 1.8],
            ["san-diego", "new-york", 2.5],
            ["san-diego", "chicago", 1.8],
            ["san-diego", "topeka", 1.4],
        ]

        capacities = [["seattle", 350], ["san-diego", 600]]
        demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

        # Set
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(self.m, name="a", domain=[i], records=capacities)
        b = Parameter(self.m, name="b", domain=[j], records=demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        with self.assertRaises(Exception):
            c[i] = 5
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation
        cost = Equation(self.m, name="cost", type="eq")
        supply = Equation(self.m, name="supply", domain=[i], type="leq")
        demand = Equation(self.m, name="demand", type="geq")

        with self.assertRaises(Exception):
            cost[i] = a * b
        cost.definition = Sum((i, j), c[i, j] * x[i, j]) == z

        with self.assertRaises(Exception):
            supply[i, j] = c * d
        supply[i] = Sum(j, x[i, j]) <= a[i]

        with self.assertRaises(Exception):
            demand[j] = Sum(i, x[i, j]) >= b[j]

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
        self.assertEqual(m.system_directory, expected_path)


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
