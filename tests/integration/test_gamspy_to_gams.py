import math
import os
import subprocess
import unittest

import gamspy.math as gams_math
import numpy as np
import pandas as pd
from gamspy import (
    Alias,
    Card,
    Container,
    Domain,
    Equation,
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
from gamspy.math import sqrt


def reformat_df(dataframe):
    return dataframe.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )


def data_records():
    # data records table
    cols = ["a", "b", "c", "Pmin", "Pmax"]
    inds = [f"g{i}" for i in range(1, 6)]
    data = [
        [3.0, 20.0, 100.0, 28, 206],
        [4.05, 18.07, 98.87, 90, 284],
        [4.05, 15.55, 104.26, 68, 189],
        [3.99, 19.21, 107.21, 76, 266],
        [3.88, 26.18, 95.31, 19, 53],
    ]
    data_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    return data_recs


class GamspyToGamsSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_lp_transport(self):
        i = Set(
            self.m,
            name="i",
            records=["seattle", "san-diego"],
            description="canning plants",
        )
        j = Set(
            self.m,
            name="j",
            records=["new-york", "chicago", "topeka"],
            description="markets",
        )

        # Data
        a = Parameter(
            self.m,
            name="a",
            domain=i,
            records=[["seattle", 350], ["san-diego", 600]],
            description="capacity of plant i in cases",
        )
        b = Parameter(
            self.m,
            name="b",
            domain=j,
            records=[["new-york", 325], ["chicago", 300], ["topeka", 275]],
            description="demand at market j in cases",
        )
        d = Parameter(
            self.m,
            name="d",
            domain=[i, j],
            records=[
                ["seattle", "new-york", 2.5],
                ["seattle", "chicago", 1.7],
                ["seattle", "topeka", 1.8],
                ["san-diego", "new-york", 2.5],
                ["san-diego", "chicago", 1.8],
                ["san-diego", "topeka", 1.4],
            ],
            description="distance in thousands of miles",
        )
        c = Parameter(
            self.m,
            name="c",
            domain=[i, j],
            description="transport cost in thousands of dollars per case",
        )
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(
            self.m,
            name="x",
            domain=[i, j],
            type="Positive",
            description="shipment quantities in cases",
        )

        # Equation
        supply = Equation(
            self.m,
            name="supply",
            domain=i,
            description="observe supply limit at plant i",
        )
        demand = Equation(
            self.m,
            name="demand",
            domain=j,
            description="satisfy demand at market j",
        )

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=self.m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        with self.assertRaises(ValidationError):
            transport.toGams(
                os.path.join("tmp", "to_gams"), options={"lp": "cplex"}
            )

        transport.toGams(
            os.path.join("tmp", "to_gams"),
            options=Options(generate_name_dict=False, lp="CPLEX"),
        )

        process = subprocess.run(
            [
                os.path.join(self.m.system_directory, "gams"),
                os.path.join("tmp", "to_gams", "transport.gms"),
                "traceopt=2",
                "trace=trace.txt",
                f'output={os.path.join("tmp", "to_gams", "transport.lst")}',
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(process.returncode, 0)

        with open("trace.txt") as trace:
            lines = trace.read().splitlines()
            objective = lines[-1].split("//")[0].split(" ")[-3]
            self.assertTrue(objective.startswith("153.675"))

    def test_mip_cutstock(self):
        i = Set(
            self.m,
            "i",
            records=[f"w{idx}" for idx in range(1, 5)],
            description="widths",
        )
        p = Set(
            self.m,
            "p",
            records=[f"p{idx}" for idx in range(1, 1001)],
            description="possible patterns",
        )
        pp = Set(self.m, "pp", domain=p, description="dynamic subset of p")

        # Parameters
        r = Parameter(self.m, "r", records=100, description="raw width")
        w = Parameter(
            self.m,
            "w",
            domain=i,
            records=[["w1", 45], ["w2", 36], ["w3", 31], ["w4", 14]],
            description="width",
        )
        d = Parameter(
            self.m,
            "d",
            domain=i,
            records=[["w1", 97], ["w2", 610], ["w3", 395], ["w4", 211]],
            description="demand",
        )
        aip = Parameter(
            self.m,
            "aip",
            domain=[i, p],
            description="number of width i in pattern growing in p",
        )

        # Master model variables
        xp = Variable(
            self.m, "xp", domain=p, type="integer", description="patterns used"
        )
        z = Variable(self.m, "z", description="objective variable")
        xp.up[p] = Sum(i, d[i])

        # Master model equations
        numpat = Equation(
            self.m,
            "numpat",
            definition=z == Sum(pp, xp[pp]),
            description="number of patterns used",
        )
        demand = Equation(
            self.m, "demand", domain=i, description="meet demand"
        )
        demand[i] = Sum(pp, aip[i, pp] * xp[pp]) >= d[i]

        master = Model(
            self.m,
            "master",
            equations=[numpat, demand],
            problem="rmip",
            sense=Sense.MIN,
            objective=z,
        )

        # Pricing model variables
        y = Variable(
            self.m, "y", domain=i, type="integer", description="new pattern"
        )
        y.up[i] = gams_math.ceil(r / w[i])

        defobj = Equation(
            self.m, "defobj", definition=z == (1 - Sum(i, demand.m[i] * y[i]))
        )
        knapsack = Equation(
            self.m,
            "knapsack",
            description="knapsack constraint",
            definition=Sum(i, w[i] * y[i]) <= r,
        )

        pricing = Model(
            self.m,
            "pricing",
            equations=[defobj, knapsack],
            problem="mip",
            sense=Sense.MIN,
            objective=z,
        )

        pp[p] = Ord(p) <= Card(i)
        aip[i, pp[p]].where[Ord(i) == Ord(p)] = gams_math.floor(r / w[i])

        pi = Set(self.m, "pi", domain=p, description="set of the last pattern")
        pi[p] = Ord(p) == Card(pp) + 1

        while len(pp) < len(p):
            master.solve(options=Options(relative_optimality_gap=0))
            pricing.solve(options=Options(relative_optimality_gap=0))

            if z.toValue() >= -0.001:
                break

            aip[i, pi] = gams_math.Round(y.l[i])
            pp[pi] = True
            pi[p] = pi[p.lag(1)]

        master.solve(options=Options(relative_optimality_gap=0))

        master.toGams(os.path.join("tmp", "to_gams"))

        process = subprocess.run(
            [
                os.path.join(self.m.system_directory, "gams"),
                os.path.join("tmp", "to_gams", "master.gms"),
                "traceopt=2",
                "trace=trace.txt",
                f'output={os.path.join("tmp", "to_gams", "master.lst")}',
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(process.returncode, 0)
        with open("trace.txt") as trace:
            lines = trace.read().splitlines()
            objective = lines[-1].split("//")[0].split(" ")[-3]
            self.assertTrue(objective.startswith("452.25"))

    def test_nlp_weapons(self):
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
            self.m,
            name="w",
            records=["icbm", "mrbm-1", "lr-bomber", "f-bomber", "mrbm-2"],
            description="weapons",
        )
        t = Set(
            self.m,
            name="t",
            records=[str(i) for i in range(1, 21)],
            description="targets",
        )

        # Parameters
        td = Parameter(
            self.m,
            name="td",
            domain=[w, t],
            records=td_data,
            description="target data",
        )
        wa = Parameter(
            self.m,
            name="wa",
            domain=w,
            records=wa_data,
            description="weapons availability",
        )
        tm = Parameter(
            self.m,
            name="tm",
            domain=t,
            records=tm_data,
            description="minimum number of weapons per target",
        )
        mv = Parameter(
            self.m,
            name="mv",
            domain=t,
            records=mv_data,
            description="military value of target",
        )

        # Variables
        x = Variable(
            self.m,
            name="x",
            domain=[w, t],
            type="Positive",
            description="weapons assignment",
        )
        prob = Variable(
            self.m,
            name="prob",
            domain=t,
            description="probability for each target",
        )

        # Equations
        maxw = Equation(
            self.m, name="maxw", domain=w, description="weapons balance"
        )
        minw = Equation(
            self.m,
            name="minw",
            domain=t,
            description="minimum number of weapons required per target",
        )
        probe = Equation(
            self.m,
            name="probe",
            domain=t,
            description="probability definition",
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
            self.m,
            name="war",
            equations=[maxw, minw],
            problem=Problem.NLP,
            sense=Sense.MAX,
            objective=etd,
        )

        x.l[w, t].where[td[w, t]] = wa[w] / Card(t)

        war.solve()

        war.toGams(os.path.join("tmp", "to_gams"))

        process = subprocess.run(
            [
                os.path.join(self.m.system_directory, "gams"),
                os.path.join("tmp", "to_gams", "war.gms"),
                "traceopt=2",
                "trace=trace.txt",
                f'output={os.path.join("tmp", "to_gams", "war.lst")}',
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(process.returncode, 0)
        with open("trace.txt") as trace:
            lines = trace.read().splitlines()
            objective = lines[-1].split("//")[0].split(" ")[-3]
            self.assertTrue(objective.startswith("1735.569579"))

    def test_mcp_qp6(self):
        this = os.path.abspath(os.path.dirname(__file__))
        gdx_file = os.path.join(this, "models", "qp6.gdx")

        cont = Container(
            load_from=gdx_file,
        )

        # Sets
        days, stocks = cont.getSymbols(["days", "stocks"])

        # Parameters
        returns, val = cont.getSymbols(["return", "val"])

        # Set
        d = Set(cont, name="d", domain=[days], description="selected days")
        s = Set(cont, name="s", domain=[stocks], description="selected stocks")

        # select subset of stocks and periods
        d[days] = (Ord(days) > 1) & (Ord(days) < 31)
        s[stocks] = Ord(stocks) < 51

        # Parameter
        mean = Parameter(
            cont,
            name="mean",
            domain=stocks,
            description="mean of daily return",
        )
        dev = Parameter(
            cont, name="dev", domain=[stocks, days], description="deviations"
        )
        totmean = Parameter(
            cont, name="totmean", description="total mean return"
        )

        mean[s] = Sum(d, returns[s, d]) / Card(d)
        dev[s, d] = returns[s, d] - mean[s]
        totmean[...] = Sum(s, mean[s]) / (Card(s))

        # Variable
        x = Variable(
            cont,
            name="x",
            type="positive",
            domain=stocks,
            description="investments",
        )
        w = Variable(
            cont,
            name="w",
            type="free",
            domain=days,
            description="intermediate variables",
        )

        # Equation
        budget = Equation(cont, name="budget")
        retcon = Equation(
            cont, name="retcon", description="returns constraint"
        )
        wdef = Equation(cont, name="wdef", domain=days)

        wdef[d] = w[d] == Sum(s, x[s] * dev[s, d])

        budget[...] = Sum(s, x[s]) == 1.0

        retcon[...] = Sum(s, mean[s] * x[s]) >= totmean * 1.25

        # Equation
        d_x = Equation(cont, name="d_x", domain=stocks)
        d_w = Equation(cont, name="d_w", domain=days)

        # Variable
        m_budget = Variable(cont, name="m_budget", type="free")
        m_wdef = Variable(cont, name="m_wdef", type="free", domain=days)

        # Positive Variable
        m_retcon = Variable(cont, name="m_retcon", type="positive")

        m_wdef.fx[days].where[~d[days]] = 0

        d_x[s] = Sum(d, m_wdef[d] * dev[s, d]) >= m_retcon * mean[s] + m_budget

        d_w[d] = 2 * w[d] / (Card(d) - 1) == m_wdef[d]

        qp6 = Model(
            cont,
            name="qp6",
            matches={
                d_x: x,
                d_w: w,
                retcon: m_retcon,
                budget: m_budget,
                wdef: m_wdef,
            },
            problem="mcp",
        )

        qp6.solve()

        qp6.toGams(os.path.join("tmp", "to_gams"))

        process = subprocess.run(
            [
                os.path.join(self.m.system_directory, "gams"),
                os.path.join("tmp", "to_gams", "qp6.gms"),
                "traceopt=2",
                "trace=trace.txt",
                f'output={os.path.join("tmp", "to_gams", "qp6.lst")}',
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(process.returncode, 0)

        with open("trace.txt") as trace:
            lines = trace.read().splitlines()
            objective = lines[-1].split("//")[0].split(" ")[-3]
            self.assertTrue(objective.startswith("8.499300"))

    def test_dnlp_inscribedsquare(self):
        def fx(t):
            return gams_math.sin(t) * gams_math.cos(t - t * t)

        def fy(t):
            return t * gams_math.sin(t)

        i = Set(
            self.m,
            name="i",
            records=["1", "2", "3", "4"],
            description="corner points of square",
        )

        # Variable
        t = Variable(
            self.m,
            name="t",
            domain=i,
            description="position of square corner points on curve",
        )
        x = Variable(
            self.m,
            name="x",
            description=(
                "x-coordinate of lower-left corner of square (=fx(t('1')))"
            ),
        )
        y = Variable(
            self.m,
            name="y",
            description=(
                "y-coordinate of lower-left corner of square (=fy(t('1')))"
            ),
        )
        a = Variable(
            self.m,
            name="a",
            type="Positive",
            description=(
                "horizontal distance between lower-left and lower-right corner of"
                " square"
            ),
        )
        b = Variable(
            self.m,
            name="b",
            type="Positive",
            description=(
                "vertical distance between lower-left and lower-right corner of"
                " square"
            ),
        )

        t.lo[i] = -math.pi
        t.up[i] = math.pi

        # Equation
        e1x = Equation(
            self.m,
            name="e1x",
            description="define x-coordinate of lower-left corner",
        )
        e1y = Equation(
            self.m,
            name="e1y",
            description="define y-coordinate of lower-left corner",
        )
        e2x = Equation(
            self.m,
            name="e2x",
            description="define x-coordinate of lower-right corner",
        )
        e2y = Equation(
            self.m,
            name="e2y",
            description="define y-coordinate of lower-right corner",
        )
        e3x = Equation(
            self.m,
            name="e3x",
            description="define x-coordinate of upper-left corner",
        )
        e3y = Equation(
            self.m,
            name="e3y",
            description="define y-coordinate of upper-left corner",
        )
        e4x = Equation(
            self.m,
            name="e4x",
            description="define x-coordinate of upper-right corner",
        )
        e4y = Equation(
            self.m,
            name="e4y",
            description="define y-coordinate of upper-right corner",
        )

        obj = a**2 + b**2  # Area of square to be maximized

        e1x[...] = fx(t["1"]) == x
        e1y[...] = fy(t["1"]) == y
        e2x[...] = fx(t["2"]) == x + a
        e2y[...] = fy(t["2"]) == y + b
        e3x[...] = fx(t["3"]) == x - b
        e3y[...] = fy(t["3"]) == y + a
        e4x[...] = fx(t["4"]) == x + a - b
        e4y[...] = fy(t["4"]) == y + a + b

        square = Model(
            self.m,
            name="square",
            equations=self.m.getEquations(),
            problem="DNLP",
            sense=Sense.MAX,
            objective=obj,
        )

        t.l[i] = -math.pi + (Ord(i) - 1) * 2 * math.pi / Card(i)
        x.l[...] = fx(t.l["1"])
        y.l[...] = fy(t.l["1"])
        a.l[...] = 1
        b.l[...] = 1

        square.solve()

        square.toGams(os.path.join("tmp", "to_gams"))

        process = subprocess.run(
            [
                os.path.join(self.m.system_directory, "gams"),
                os.path.join("tmp", "to_gams", "square.gms"),
                "traceopt=2",
                "trace=trace.txt",
                f'output={os.path.join("tmp", "to_gams", "square.lst")}',
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(process.returncode, 0)

        with open("trace.txt") as trace:
            lines = trace.read().splitlines()
            objective = lines[-1].split("//")[0].split(" ")[-3]
            self.assertTrue(objective.startswith("9.80076"))

    def test_minlp_minlphix(self):
        # Set
        i = Set(
            self.m,
            name="i",
            records=[f"c-{i}" for i in range(1, 5)],
            description="condensers-columns",
        )
        j = Set(
            self.m,
            name="j",
            records=[f"r-{i}" for i in range(1, 5)],
            description="reboilers",
        )
        hu = Set(
            self.m,
            name="hu",
            records=["lp", "ex"],
            description="hot utilities",
        )
        cu = Set(
            self.m, name="cu", records=["cw"], description="cold utilities"
        )
        n = Set(self.m, name="n", records=["a", "b"], description="index")
        m = Set(
            self.m, name="m", records=["ab", "bc"], description="intermediates"
        )
        pm = Set(
            self.m,
            name="pm",
            domain=[i, m],
            records=[("c-1", "bc"), ("c-2", "ab")],
            description="products",
        )
        fm = Set(
            self.m,
            name="fm",
            domain=[i, m],
            records=[("c-3", "bc"), ("c-4", "ab")],
            description="feeds",
        )

        ip = Alias(self.m, name="ip", alias_with=i)
        jp = Alias(self.m, name="jp", alias_with=j)

        # ====================================================================
        # Definition of "z" sets for conditional control of model
        # used to map permissible matches between condensers and reboilers
        # and the position of columns in the superstructure
        # =====================================================================

        # Set
        zlead = Set(
            self.m,
            name="zlead",
            domain=i,
            records=["c-1", "c-2"],
            description="leading columns in superstructure",
        )
        zcrhx = Set(
            self.m,
            name="zcrhx",
            domain=[i, j],
            records=[
                ("c-1", "r-3"),
                ("c-2", "r-4"),
                ("c-3", "r-1"),
                ("c-4", "r-2"),
            ],
            description="condenser to reboiler allowable matches",
        )
        zlim = Set(
            self.m,
            name="zlim",
            domain=[i, j],
            description="direction of heat integration",
        )
        zcr = Set(
            self.m,
            name="zcr",
            domain=[i, j],
            description="reboiler-condenser pairs",
        )

        zlim[i, j] = zcrhx[i, j] & (Ord(i) < Ord(j))
        zcr[i, j] = Ord(i) == Ord(j)

        # Parameter
        spltfrc = Parameter(
            self.m,
            name="spltfrc",
            domain=[i, m],
            records=pd.DataFrame([["c-1", "bc", 0.20], ["c-2", "ab", 0.90]]),
            description="split fraction of distillation columns",
        )

        tcmin = Parameter(
            self.m,
            name="tcmin",
            domain=i,
            records=np.array([341.92, 343.01, 353.54, 341.92]),
            description="minimum condenser temperatures",
        )
        trmax = Parameter(
            self.m,
            name="trmax",
            domain=j,
            description="maximum reboiler temperatures",
        )
        trmax[j] = 1000

        # ====================================================================
        # scaled cost coefficients for distillation column fits
        # nonlinear fixed-charge cost model
        #   cost = fc*y + vc*flow*temp
        # scaling factor = 1000
        # ====================================================================

        # Parameter
        fc = Parameter(
            self.m,
            name="fc",
            domain=i,
            records=np.array([151.125, 180.003, 4.2286, 213.42]),
            description="fixed charge for distillation columns",
        )
        vc = Parameter(
            self.m,
            name="vc",
            domain=i,
            records=np.array([0.003375, 0.000893, 0.004458, 0.003176]),
            description="variable charge for distillation columns",
        )
        thu = Parameter(
            self.m,
            name="thu",
            domain=hu,
            records=np.array([421.0, 373.0]),
            description="hot utility temperatures",
        )

        # hot utility cost coeff - gives cost in thousands of dollars per year
        # ucost = q(10e+6 kj/hr)*costhu(hu)

        costhu = Parameter(
            self.m,
            name="costhu",
            domain=hu,
            records=np.array([24.908, 9.139]),
            description="hot utility cost coefficients",
        )

        kf = Parameter(
            self.m,
            name="kf",
            domain=[i, n],
            records=np.array(
                [
                    [32.4, 0.0225],
                    [25.0, 0.0130],
                    [3.76, 0.0043],
                    [35.1, 0.0156],
                ]
            ),
            description="coeff. for heat duty temperature fits",
        )

        af = Parameter(
            self.m,
            name="af",
            domain=[i, n],
            records=np.array(
                [
                    [9.541, 1.028],
                    [12.24, 1.050],
                    [8.756, 1.029],
                    [9.181, 1.005],
                ]
            ),
            description="coeff. for column temperature fits",
        )

        # Scalar
        totflow = Parameter(
            self.m,
            name="totflow",
            records=396,
            description="total flow to superstructure",
        )
        fchx = Parameter(
            self.m,
            name="fchx",
            records=3.392,
            description="fixed charge for heat exchangers scaled",
        )
        vchx = Parameter(
            self.m,
            name="vchx",
            records=0.0893,
            description="variable charge for heat exchangers scaled",
        )
        htc = Parameter(
            self.m,
            name="htc",
            records=0.0028,
            description="overall heat transfer coefficient",
        )
        dtmin = Parameter(
            self.m,
            name="dtmin",
            records=10.0,
            description="minimum temperature approach",
        )
        tcin = Parameter(
            self.m,
            name="tcin",
            records=305.0,
            description="inlet temperature of cold water",
        )
        tcout = Parameter(
            self.m,
            name="tcout",
            records=325.0,
            description="outlet temperature of cold water",
        )
        costcw = Parameter(
            self.m,
            name="costcw",
            records=4.65,
            description="cooling water cost coefficient",
        )
        beta = Parameter(
            self.m,
            name="beta",
            records=0.52,
            description="income tax correction factor",
        )
        alpha = Parameter(
            self.m,
            name="alpha",
            records=0.40,
            description="one over payout time factor in years",
        )
        u = Parameter(
            self.m,
            name="u",
            records=1500,
            description="large number for logical constraints",
        )
        uint = Parameter(
            self.m,
            name="uint",
            records=20,
            description="upper bound for integer logical",
        )

        # Positive Variables
        f = Variable(
            self.m,
            name="f",
            type="positive",
            domain=i,
            description="flowrates to columns",
        )
        qr = Variable(
            self.m,
            name="qr",
            type="positive",
            domain=j,
            description="reboiler duties for column with reboiler j",
        )
        qc = Variable(
            self.m,
            name="qc",
            type="positive",
            domain=i,
            description="condenser duties for column i",
        )
        qcr = Variable(
            self.m,
            name="qcr",
            type="positive",
            domain=[i, j],
            description="heat integration heat transfer",
        )
        qhu = Variable(
            self.m,
            name="qhu",
            type="positive",
            domain=[hu, j],
            description="hot utility heat transfer",
        )
        qcu = Variable(
            self.m,
            name="qcu",
            type="positive",
            domain=[i, cu],
            description="cold utility heat transfer",
        )
        tc = Variable(
            self.m,
            name="tc",
            type="positive",
            domain=i,
            description="condenser temperature for column with cond. i",
        )
        tr = Variable(
            self.m,
            name="tr",
            type="positive",
            domain=j,
            description="reboiler temperature for column with reb. j",
        )
        lmtd = Variable(
            self.m,
            name="lmtd",
            type="positive",
            domain=i,
            description="lmtd for cooling water exchanges",
        )
        sl1 = Variable(
            self.m,
            name="sl1",
            type="positive",
            domain=i,
            description="artificial slack variable for lmtd equalities",
        )
        sl2 = Variable(
            self.m,
            name="sl2",
            type="positive",
            domain=i,
            description="artificial slack variable for lmtd equalities",
        )
        s1 = Variable(
            self.m,
            name="s1",
            type="positive",
            domain=i,
            description="artificial slack variable for reb-con equalities",
        )
        s2 = Variable(
            self.m,
            name="s2",
            type="positive",
            domain=i,
            description="artificial slack variable for reb-con equalities",
        )
        s3 = Variable(
            self.m,
            name="s3",
            type="positive",
            domain=i,
            description="artificial slack variable for duty equalities",
        )
        s4 = Variable(
            self.m,
            name="s4",
            type="positive",
            domain=i,
            description="artificial slack variable for duty equalities",
        )

        # Binary Variable
        yhx = Variable(
            self.m,
            name="yhx",
            type="binary",
            domain=[i, j],
            description="heat integration matches condenser i reboiler j",
        )
        yhu = Variable(
            self.m,
            name="yhu",
            type="binary",
            domain=[hu, j],
            description="hot utility matches hot utility hu reboiler j",
        )
        ycu = Variable(
            self.m,
            name="ycu",
            type="binary",
            domain=[i, cu],
            description="cold utility matches condenser i cold util cu",
        )
        ycol = Variable(
            self.m,
            name="ycol",
            type="binary",
            domain=i,
            description="columns in superstructure",
        )

        # Equation
        tctrlo = Equation(
            self.m,
            name="tctrlo",
            domain=[i, j],
            description="prevent division by 0 in the objective",
        )
        lmtdlo = Equation(
            self.m,
            name="lmtdlo",
            domain=i,
            description="prevent division by 0 in the objective",
        )
        lmtdsn = Equation(
            self.m,
            name="lmtdsn",
            domain=i,
            description="nonlinear form of lmtd definition",
        )
        tempset = Equation(
            self.m,
            name="tempset",
            domain=i,
            description="sets temperatures of inactive columns to 0 (milp)",
        )
        artrex1 = Equation(
            self.m,
            name="artrex1",
            domain=i,
            description="relaxes artificial slack variables (nlp)",
        )
        artrex2 = Equation(
            self.m,
            name="artrex2",
            domain=i,
            description="relaxes artificial slack variables (nlp)",
        )
        material = Equation(
            self.m,
            name="material",
            domain=m,
            description="material balances for each intermediate product",
        )
        feed = Equation(
            self.m, name="feed", description="feed to superstructure"
        )
        matlog = Equation(
            self.m,
            name="matlog",
            domain=i,
            description="material balance logical constraints",
        )
        duty = Equation(
            self.m,
            name="duty",
            domain=i,
            description="heat duty definition of condenser i",
        )
        rebcon = Equation(
            self.m,
            name="rebcon",
            domain=[i, j],
            description="equates condenser and reboiler duties",
        )
        conheat = Equation(
            self.m,
            name="conheat",
            domain=i,
            description="condenser heat balances",
        )
        rebheat = Equation(
            self.m,
            name="rebheat",
            domain=j,
            description="reboiler heat balances",
        )
        dtminlp = Equation(
            self.m,
            name="dtminlp",
            domain=j,
            description="minimum temp approach for low pressure steam",
        )
        dtminc = Equation(
            self.m,
            name="dtminc",
            domain=i,
            description="minimum temp allowable for each condenser",
        )
        trtcdef = Equation(
            self.m,
            name="trtcdef",
            domain=[i, j],
            description="relates reboiler and condenser temps of columns",
        )
        dtmincr = Equation(
            self.m,
            name="dtmincr",
            domain=[i, j],
            description="minimum temp approach for heat integration",
        )
        dtminex = Equation(
            self.m,
            name="dtminex",
            domain=j,
            description="minimum temp approach for exhaust steam",
        )
        hxclog = Equation(
            self.m,
            name="hxclog",
            domain=[i, j],
            description="logical constraint for heat balances",
        )
        hxhulog = Equation(
            self.m,
            name="hxhulog",
            domain=[hu, j],
            description="logical constraint for heat balances",
        )
        hxculog = Equation(
            self.m,
            name="hxculog",
            domain=[i, cu],
            description="logical constraint for heat balances",
        )
        qcqrlog = Equation(
            self.m,
            name="qcqrlog",
            domain=i,
            description="logical constraint for con-reb duties",
        )

        # these are the pure binary constraints of the minlp
        sequen = Equation(
            self.m,
            name="sequen",
            domain=m,
            description="restricts superstructure to a single sequence",
        )
        lead = Equation(self.m, name="lead", description="sequence control")
        limutil = Equation(
            self.m,
            name="limutil",
            domain=j,
            description="limits columns to have a single hot utility",
        )
        hidirect = Equation(
            self.m,
            name="hidirect",
            domain=[i, j],
            description="requires a single direction of heat integration",
        )
        heat = Equation(
            self.m,
            name="heat",
            domain=i,
            description="logical integer constraint",
        )

        # nlp subproblems objective
        zoau = alpha * (
            Sum(i, fc[i] * ycol[i] + vc[i] * (tc[i] - tcmin[i]) * f[i])
            + Sum(
                zcrhx[i, j],
                fchx * yhx[i, j]
                + (vchx / htc) * (qcr[i, j] / (tc[i] - tr[j] + 1 - ycol[i])),
            )
            + Sum(
                [i, cu],
                fchx * ycu[i, cu]
                + (vchx / htc) * (qcu[i, cu] / (lmtd[i] + 1 - ycol[i])),
            )
            + Sum(
                [hu, j],
                fchx * yhu[hu, j]
                + (vchx / htc) * (qhu[hu, j] / (thu[hu] - tr[j])),
            )
        ) + beta * (
            Sum([i, cu], costcw * qcu[i, cu])
            + Sum([hu, j], costhu[hu] * qhu[hu, j])
        )

        # limit the denominator in the second line of the objective away from zero
        tctrlo[zcrhx[i, j]] = tc[i] - tr[j] + 1 - ycol[i] >= 1

        # lmtd and ycol from being 0 and 1 at the same time to prevent divding
        # by 0 in the objective
        lmtdlo[i] = lmtd[i] >= 2 * ycol[i]

        lmtdsn[i] = lmtd[i] == (
            (2 / 3) * sqrt((tc[i] - tcin) * (tc[i] - tcout))
            + (1 / 6) * ((tc[i] - tcin) + (tc[i] - tcout))
            + sl1[i]
            - sl2[i]
        )

        artrex1[i] = s1[i] + s2[i] + sl1[i] <= u * (1 - ycol[i])

        artrex2[i] = s3[i] + s4[i] + sl2[i] <= u * (1 - ycol[i])

        material[m] = Sum(pm[i, m], spltfrc[i, m] * f[i]) == Sum(
            fm[i, m], f[i]
        )

        feed[...] = Sum(zlead[i], f[i]) == totflow

        duty[i] = (
            qc[i]
            == (kf[i, "a"] + kf[i, "b"] * (tc[i] - tcmin[i])) + s3[i] - s4[i]
        )

        rebcon[zcr[i, j]] = qr[j] == qc[i]

        conheat[i] = qc[i] == Sum(zcrhx[i, j], qcr[i, j]) + Sum(cu, qcu[i, cu])

        rebheat[j] = qr[j] == Sum(zcrhx[i, j], qcr[i, j]) + Sum(hu, qhu[hu, j])

        trtcdef[zcr[i, j]] = (
            tr[j]
            == (af[i, "a"] + af[i, "b"] * (tc[i] - tcmin[i])) + s1[i] - s2[i]
        )

        dtminlp[j] = dtmin - (thu["lp"] - tr[j]) <= 0

        dtminex[j] = dtmin - (thu["ex"] - tr[j]) - u * (1 - yhu["ex", j]) <= 0

        tempset[i] = tc[i] + lmtd[i] + Sum(zcr[i, j], tr[j]) <= u * ycol[i]

        matlog[i] = f[i] <= u * ycol[i]

        dtminc[i] = tcmin[i] - tc[i] <= u * (1 - ycol[i])

        dtmincr[zcrhx[i, j]] = tr[j] - tc[i] - u * (1 - yhx[i, j]) + dtmin <= 0

        hxclog[zcrhx[i, j]] = qcr[i, j] <= u * yhx[i, j]

        hxhulog[hu, j] = qhu[hu, j] <= u * yhu[hu, j]

        hxculog[i, cu] = qcu[i, cu] <= u * ycu[i, cu]

        qcqrlog[i] = qc[i] + Sum(j.where[zcr[i, j]], qr[j]) <= u * ycol[i]

        sequen[m] = Sum(pm[i, m], ycol[i]) == Sum(fm[i, m], ycol[i])

        lead[...] = Sum(zlead[i], ycol[i]) == 1

        limutil[j] = Sum(hu, yhu[hu, j]) <= 1

        # only one of the mutual heat integration binaries can be 1
        hidirect[zlim[i, j]] = (
            yhx[i, j]
            + Sum(
                Domain(ip, jp).where[
                    (Ord(ip) == Ord(j)) & (Ord(jp) == Ord(i))
                ],
                yhx[ip, jp],
            )
            <= 1
        )

        # if a column doesn't exist then all binary variables associated
        # with it must also be set to zero
        heat[i] = (
            Sum(
                zcrhx[i, j],
                yhx[i, j]
                + Sum(
                    Domain(ip, jp).where[
                        (Ord(ip) == Ord(j)) & (Ord(jp) == Ord(i))
                    ],
                    yhx[ip, jp],
                ),
            )
            + Sum((hu, zcr[i, j]), yhu[hu, j])
            + Sum(cu, ycu[i, cu])
        ) <= uint * ycol[i]

        tc.lo["c-1"] = tcout + 1
        tc.up["c-2"] = tcin - 1
        tc.lo["c-3"] = tcout + 1
        tc.up["c-4"] = tcin - 1
        tr.up[j] = trmax[j]

        skip = Model(
            self.m,
            name="skip",
            equations=self.m.getEquations(),
            problem="minlp",
            sense=Sense.MIN,
            objective=zoau,
        )

        skip.solve(options=Options(domain_violation_limit=100))

        skip.toGams(os.path.join("tmp", "to_gams"))

        process = subprocess.run(
            [
                os.path.join(self.m.system_directory, "gams"),
                os.path.join("tmp", "to_gams", "skip.gms"),
                "traceopt=2",
                "trace=trace.txt",
                "domlim=100",
                f'output={os.path.join("tmp", "to_gams", "skip.lst")}',
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(process.returncode, 0)

        with open("trace.txt") as trace:
            lines = trace.read().splitlines()
            objective = lines[-1].split("//")[0].split(" ")[-3]
            print(f"{objective=}")

    def test_qcp_EDsensitivity(self):
        gen = Set(self.m, name="gen", records=[f"g{i}" for i in range(1, 6)])
        counter = Set(
            self.m, name="counter", records=[f"c{i}" for i in range(1, 12)]
        )

        report = Parameter(self.m, name="report", domain=[counter, "*"])
        repGen = Parameter(self.m, name="repGen", domain=[counter, gen])
        load = Parameter(self.m, name="load", records=400)
        data = Parameter(
            self.m, name="data", domain=[gen, "*"], records=data_records()
        )

        P = Variable(self.m, name="P", domain=gen)

        eq1 = Sum(
            gen,
            data[gen, "a"] * P[gen] * P[gen]
            + data[gen, "b"] * P[gen]
            + data[gen, "c"],
        )

        eq2 = Equation(self.m, name="eq2", type="regular")
        eq2[...] = Sum(gen, P[gen]) >= load

        P.lo[gen] = data[gen, "Pmin"]
        P.up[gen] = data[gen, "Pmax"]

        ECD = Model(
            self.m,
            name="ECD",
            equations=[eq2],
            problem="qcp",
            sense="min",
            objective=eq1,
        )

        for idx, cc in enumerate(counter.toList()):
            load[...] = Sum(gen, data[gen, "Pmin"]) + (
                (idx) / (Card(counter) - 1)
            ) * Sum(gen, data[gen, "Pmax"] - data[gen, "Pmin"])
            ECD.solve()
            repGen[cc, gen] = P.l[gen]
            report[cc, "OF"] = ECD.objective_value
            report[cc, "load"] = load

        ECD.toGams(os.path.join("tmp", "to_gams"))

        process = subprocess.run(
            [
                os.path.join(self.m.system_directory, "gams"),
                os.path.join("tmp", "to_gams", "ECD.gms"),
                "traceopt=2",
                "trace=trace.txt",
                f'output={os.path.join("tmp", "to_gams", "ECD.lst")}',
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(process.returncode, 0)

        with open("trace.txt") as trace:
            lines = trace.read().splitlines()
            objective = lines[-1].split("//")[0].split(" ")[-3]
            self.assertTrue(objective.startswith("911044.089"))

    def test_set_attributes(self):
        s = self.m.addSet("s", records=[1, 2])
        x = self.m.addVariable("x", type="positive", domain=s)
        eq = self.m.addEquation("eq", domain=s)
        eq[s].where[~s.first] = x[s] >= 1

        m = self.m.addModel(
            problem="LP",
            name="test",
            sense=Sense.MIN,
            equations=[eq],
            objective=Sum(s, x[s]),
        )

        m.toGams(os.path.join("tmp", "to_gams"))

        process = subprocess.run(
            [
                os.path.join(self.m.system_directory, "gams"),
                os.path.join("tmp", "to_gams", "test.gms"),
                "traceopt=2",
                "trace=trace.txt",
                f'output={os.path.join("tmp", "to_gams", "test.lst")}',
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(process.returncode, 0)

        with open("trace.txt") as trace:
            lines = trace.read().splitlines()
            objective = lines[-1].split("//")[0].split(" ")[-3]
            self.assertTrue(float(objective) == 1)


def gamspy_to_gams_suite():
    suite = unittest.TestSuite()
    tests = [
        GamspyToGamsSuite(name)
        for name in dir(GamspyToGamsSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(gamspy_to_gams_suite())
