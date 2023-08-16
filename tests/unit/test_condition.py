import unittest

import pandas as pd
import gamspy.math as gamspy_math
from gamspy import (
    Container,
    Set,
    Parameter,
    Variable,
    Equation,
    Number,
    Sum,
    Ord,
)


class ConditionSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

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
        sumc[o, p] = gamspy_math.uniform(0, 1)

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

    def test_operator_comparison_in_condition(self):
        m = Container()
        s = Set(m, name="s", records=[str(i) for i in range(1, 4)])
        c = Parameter(m, name="c", domain=[s])
        c[s].where[Ord(s) <= Ord(s)] = 1
        self.assertEqual(
            list(m._statements_dict.values())[-1].getStatement(),
            "c(s) $ (ord(s) <= ord(s)) = 1;",
        )


def condition_suite():
    suite = unittest.TestSuite()
    tests = [
        ConditionSuite(name)
        for name in dir(ConditionSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(condition_suite())
