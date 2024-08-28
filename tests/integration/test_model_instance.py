from __future__ import annotations

import os
import unittest

import pandas as pd
from gamspy import (
    Container,
    Equation,
    Model,
    ModelInstanceOptions,
    Options,
    Parameter,
    Problem,
    Product,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.exceptions import ValidationError


class ModelInstanceSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()
        self.canning_plants = ["seattle", "san-diego"]
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

    def test_parameter_change(self):
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        b = Parameter(self.m, name="b", domain=[j], records=self.demands)
        d = Parameter(
            self.m,
            name="d",
            domain=[i, j],
            records=self.distances,
            is_miro_input=True,
        )
        c = Parameter(self.m, name="c", domain=[i, j])
        bmult = Parameter(self.m, name="bmult", records=1)
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        cost = Equation(self.m, name="cost")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=self.m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=z,
        )

        bmult_list = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        results = [
            92.204,
            107.572,
            122.940,
            138.307,
            153.675,
            169.94250000000002,
            185.58,
            201.21750000000003,
        ]

        transport.freeze(modifiables=[bmult])

        for b_value, result in zip(bmult_list, results):
            bmult[...] = b_value
            transport.solve(solver="conopt")
            self.assertTrue("bmult_var" in self.m.data)
            self.assertTrue(
                x.records.columns.to_list()
                == ["i", "j", "level", "marginal", "lower", "upper", "scale"]
            )
            self.assertAlmostEqual(z.toValue(), result, places=2)
            self.assertAlmostEqual(transport.objective_value, result, places=2)

        # different solver
        transport.solve(solver="cplex")
        self.assertAlmostEqual(
            transport.objective_value, 199.77750000000003, places=2
        )

        # invalid solver
        with self.assertRaises(ValidationError):
            transport.solve(solver="blablabla")

        transport.unfreeze()
        self.assertFalse(transport._is_frozen)

    def test_variable_change(self):
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        b = Parameter(self.m, name="b", domain=[j], records=self.demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        bmult = Parameter(self.m, name="bmult", records=1)
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        cost = Equation(self.m, name="cost")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=self.m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=z,
        )

        transport.freeze(modifiables=[x.up])
        transport.solve(solver="conopt")
        self.assertAlmostEqual(transport.objective_value, 153.675, places=3)

        x.records.loc[1, "upper"] = 0
        transport.solve(solver="conopt")
        self.assertAlmostEqual(transport.objective_value, 156.375, places=3)

        transport.unfreeze()

    def test_fx(self):
        INCOME0 = Parameter(
            self.m,
            name="INCOME0",
            description="notional income level",
            records=3.5,
        )

        IADJ = Variable(
            self.m,
            name="IADJ",
            description=(
                "investment scaling factor (for fixed capital formation)"
            ),
            type="Free",
        )
        MPSADJ = Variable(
            self.m,
            name="MPSADJ",
            description="savings rate scaling factor",
            type="Free",
        )

        BALANCE = Equation(
            self.m,
            name="BALANCE",
            description="notional balance constraint",
            definition=(1 + IADJ) + (1 + MPSADJ) == INCOME0,
        )

        mm = Model(self.m, name="mm", equations=[BALANCE], problem="MCP")
        mm.freeze(modifiables=[INCOME0, IADJ.fx, MPSADJ.fx])
        IADJ.setRecords({"lower": 0, "upper": 0, "scale": 1})
        mm.solve()

        self.assertEqual(MPSADJ.records["level"].tolist()[0], 1.5)

        MPSADJ.setRecords({"lower": 0, "upper": 0, "scale": 1})
        mm.solve()

        self.assertEqual(MPSADJ.records["level"].tolist()[0], 0)
        mm.unfreeze()

    def test_validations(self):
        i = Set(self.m, name="i", records=["seattle", "san-diego"])
        j = Set(self.m, name="j", records=["new-york", "chicago", "topeka"])

        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        b = Parameter(self.m, name="b", domain=[j], records=self.demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        bmult = Parameter(self.m, name="bmult", records=1)
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        cost = Equation(self.m, name="cost")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost[...] = z == Sum((i, j), c[i, j] * x[i, j])
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= bmult * b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=self.m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=z,
        )

        # modifiables is not an iterable.
        with self.assertRaises(ValidationError):
            transport.freeze(modifiables=bmult)

        # provide a set as a modifiable
        with self.assertRaises(ValidationError):
            transport.freeze(modifiables=[i])

        transport.freeze(modifiables=[x.up], options=Options(lp="conopt"))

        # Test model instance options
        transport.solve(
            solver="conopt",
            model_instance_options=ModelInstanceOptions(debug=True),
        )
        self.assertAlmostEqual(transport.objective_value, 153.675, places=3)
        self.assertTrue(os.path.exists("dict.txt"))
        self.assertTrue(os.path.exists("gams.gms"))

        # Test solver options
        transport.solve(solver="conopt", solver_options={"rtmaxv": "1.e12"})
        self.assertTrue(
            os.path.exists(
                os.path.join(self.m.working_directory, "conopt.opt")
            )
        )

    def test_modifiable_in_condition(self):
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
            w.where[x.l[w, t]], (1 - td[w, t]) ** x[w, t]
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
            equations=[maxw, minw, probe],
            problem=Problem.NLP,
            sense=Sense.MAX,
            objective=etd,
        )

        with self.assertRaises(ValidationError):
            war.freeze(modifiables=[td])

        with self.assertRaises(ValidationError):
            war.freeze(modifiables=[tm])

        with self.assertRaises(ValidationError):
            war.freeze(modifiables=[x.l])


def model_instance_suite():
    suite = unittest.TestSuite()
    tests = [
        ModelInstanceSuite(name)
        for name in dir(ModelInstanceSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(model_instance_suite())
