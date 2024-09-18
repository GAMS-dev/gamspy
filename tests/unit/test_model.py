from __future__ import annotations

import unittest

import pandas as pd
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Options,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
    VariableType,
)
from gamspy.exceptions import ValidationError


class ModelSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()
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

    def test_model(self):
        i = Set(
            self.m,
            name="i",
            records=self.canning_plants,
            description="Canning Plants",
        )
        j = Set(self.m, name="j", records=self.markets, description="Markets")

        # Params
        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        b = Parameter(self.m, name="b", domain=[j], records=self.demands)
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

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        supply[i] = Sum(j, x[i, j]) <= a[i]

        demand = Equation(self.m, name="demand", domain=[j])
        demand[j] = Sum(i, x[i, j]) >= b[j]

        # Model with implicit objective
        test_model = Model(
            self.m,
            name="test_model",
            equations=[supply, demand],
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        test_model.solve(solver="CPLEX")
        self.assertEqual(
            list(self.m.data.keys()),
            [
                "i",
                "j",
                "a",
                "b",
                "d",
                "c",
                "x",
                "z",
                "cost",
                "supply",
                "demand",
                "test_model_objective_variable",
                "test_model_objective",
            ],
        )
        self.assertEqual(test_model.objective_value, 153.675)

        # Check if the name is reserved
        self.assertRaises(ValidationError, Model, self.m, "set", "LP")

        # Equation definition with more than one index
        bla = Equation(
            self.m,
            name="bla",
            domain=[i, j],
            description="observe supply limit at plant i",
        )
        bla[i, j] = x[i, j] <= a[i]

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
            test_model2.getDeclaration(),
            "Model test_model2 / cost,supply /;",
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
            test_model5.getDeclaration(),
            "Model test_model5 / supply.x,cost.z /;",
        )

        # Equations provided as strings
        self.assertRaises(
            TypeError, Model, self.m, "test_model5", "LP", ["cost", "supply"]
        )

        # Test matches
        test_model6 = Model(
            self.m,
            name="test_model6",
            equations=[supply],
            matches={demand: x},
            problem="LP",
            sense="min",
        )
        self.assertEqual(
            test_model6.getDeclaration(),
            "Model test_model6 / supply,demand.x /;",
        )

        # Test no name
        _ = Model(
            self.m,
            equations=[supply, demand],
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        self.m.addModel(
            equations=[supply, demand],
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        # Test repr and str
        self.assertTrue(
            str(test_model6).startswith(
                f"Model {test_model6.name}:\n  Problem Type: LP\n  Sense: MIN\n  Equations:"
            )
        )

        # empty model name
        self.assertRaises(
            ValueError,
            Model,
            self.m,
            "test_model7",
            "",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

        # model name too long
        self.assertRaises(
            ValueError,
            Model,
            self.m,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "LP",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

        # model name is not an str
        self.assertRaises(
            TypeError,
            Model,
            self.m,
            5,
            "LP",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

        # model name contains empty space
        self.assertRaises(
            ValidationError,
            Model,
            self.m,
            "test_model 8",
            "LP",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

        # model name begins with underscore
        self.assertRaises(
            ValidationError,
            Model,
            self.m,
            "_test_model7",
            "LP",
            self.m.getEquations(),
            "min",
            Sum((i, j), c[i, j] * x[i, j]),
        )

    def test_feasibility(self):
        m = Container()

        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        a = Parameter(m, name="a", domain=[i], records=self.capacities)
        b = Parameter(m, name="b", domain=[j], records=self.demands)
        d = Parameter(m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(m, name="x", domain=[i, j], type="Positive")

        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            m,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense="feasibility",
        )
        self.assertEqual(
            transport._generate_solve_string(),
            "solve transport using LP MIN transport_objective_variable;",
        )
        transport.solve()
        self.assertIsNotNone(x.records)

        self.assertRaises(
            ValidationError,
            Model,
            m,
            "transport2",
            "LP",
            m.getEquations(),
            "feasibility",
            Sum((i, j), c[i, j] * x[i, j]),
        )

    def test_tuple_equations(self):
        i = Set(
            self.m,
            name="i",
            records=self.canning_plants,
            description="Canning Plants",
        )
        j = Set(self.m, name="j", records=self.markets, description="Markets")

        # Params
        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        b = Parameter(self.m, name="b", domain=[j], records=self.demands)
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

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        supply[i] = Sum(j, x[i, j]) <= a[i]

        demand = Equation(self.m, name="demand", domain=[j])
        demand[j] = Sum(i, x[i, j]) >= b[j]

        test_model = Model(
            self.m,
            name="test_model",
            equations=(supply, demand),
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        test_model.solve()

        test_model2 = Model(
            self.m,
            name="test_model2",
            equations=set(self.m.getEquations()) - {cost},
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        test_model2.solve()

    def test_computeInfeasibilities(self):
        m = Container()

        i = Set(
            m,
            name="i",
            records=self.canning_plants,
            description="canning plants",
        )
        j = Set(
            m,
            name="j",
            records=self.markets,
            description="markets",
        )

        a = Parameter(
            m,
            name="a",
            domain=i,
            records=self.capacities,
            description="capacity of plant i in cases",
        )
        b = Parameter(
            m,
            name="b",
            domain=j,
            records=self.demands,
            description="demand at market j in cases",
        )
        d = Parameter(
            m,
            name="d",
            domain=[i, j],
            records=self.distances,
            description="distance in thousands of miles",
        )
        c = Parameter(
            m,
            name="c",
            domain=[i, j],
            description="transport cost in thousands of dollars per case",
        )
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(
            m,
            name="x",
            domain=[i, j],
            type="Positive",
            description="shipment quantities in cases",
        )

        supply = Equation(
            m,
            name="supply",
            domain=i,
            description="observe supply limit at plant i",
        )
        demand = Equation(
            m,
            name="demand",
            domain=j,
            description="satisfy demand at market j",
        )

        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            m,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        b[j] = 1.5 * b[j]
        transport.solve()

        infeasibilities = transport.computeInfeasibilities()
        columns = [
            "i",
            "level",
            "marginal",
            "lower",
            "upper",
            "scale",
            "infeasibility",
        ]
        self.assertEqual(
            list(infeasibilities.keys()),
            [
                "supply",
                "x",
                "demand",
                "transport_objective",
                "transport_objective_variable",
            ],
        )
        self.assertEqual(list(infeasibilities["supply"].columns), columns)
        self.assertEqual(
            infeasibilities["supply"].values.tolist(),
            [["san-diego", 1000.0, 0.0, float("-inf"), 600.0, 1.0, 400.0]],
        )

        self.assertEqual(
            x.computeInfeasibilities().values.tolist(),
            [
                [
                    "seattle",
                    "new-york",
                    -100.0,
                    0.0,
                    0.0,
                    float("inf"),
                    1.0,
                    100.0,
                ]
            ],
        )

        self.assertEqual(
            supply.computeInfeasibilities().values.tolist(),
            [["san-diego", 1000.0, 0.0, float("-inf"), 600.0, 1.0, 400.0]],
        )

        all_infeasibilities = transport.computeInfeasibilities()
        self.assertEqual(
            list(all_infeasibilities.keys()),
            [
                "supply",
                "x",
                "demand",
                "transport_objective",
                "transport_objective_variable",
            ],
        )

        self.assertEqual(
            [elem.values.tolist() for elem in list(infeasibilities.values())],
            [
                [["san-diego", 1000.0, 0.0, -float("inf"), 600.0, 1.0, 400.0]],
                [
                    [
                        "seattle",
                        "new-york",
                        -100.0,
                        0.0,
                        0.0,
                        float("inf"),
                        1.0,
                        100.0,
                    ]
                ],
                [],
                [],
                [],
            ],
        )

    def test_equations(self):
        e = Equation(self.m, "e")
        e.l[...] = -10
        e.lo[...] = 5
        model = Model(
            self.m,
            "my",
            problem=Problem.LP,
            equations=[e],
            sense=Sense.FEASIBILITY,
        )

        with self.assertRaises(ValidationError):
            model.solve()

    def test_equation_listing(self):
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
        self.assertEqual(len(mexss.getEquationListing().split("\n")), 74)
        self.assertEqual(
            len(
                mexss.getEquationListing(infeasibility_threshold=2.5).split(
                    "\n"
                )
            ),
            2,
        )

    def test_jupyter_behaviour(self):
        i = Set(self.m, name="i", records=self.canning_plants)
        i = Set(self.m, name="i", records=self.canning_plants)
        j = Set(self.m, name="j", records=self.markets)

        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        a = Parameter(self.m, name="a", domain=[i], records=self.capacities)
        b = Parameter(self.m, name="b", domain=[j], records=self.demands)
        d = Parameter(self.m, name="d", domain=[i, j], records=self.distances)
        c = Parameter(self.m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")

        supply = Equation(self.m, name="supply", domain=[i])
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])
        demand = Equation(self.m, name="demand", domain=[j])

        supply[i] = Sum(j, x[i, j]) <= a[i]
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            self.m,
            name="transport",
            equations=[supply, demand],
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )

        transport = Model(
            self.m,
            name="transport",
            equations=[supply, demand],
            problem="LP",
            sense="min",
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        transport.solve()
        transport.solve()

    def test_solve_string_lp(self):
        i = Set(self.m, name="i")
        j = Set(self.m, name="j")

        # Params
        a = Parameter(self.m, name="a", domain=[i])
        b = Parameter(self.m, name="b", domain=[j])
        c = Parameter(self.m, name="c", domain=[i, j])

        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation definition without an index
        cost = Equation(
            self.m,
            name="cost",
            description="define objective function",
        )
        cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z

        # Equation definition with an index
        supply = Equation(
            self.m,
            name="supply",
            domain=[i],
            description="observe supply limit at plant i",
        )
        supply[i] = Sum(j, x[i, j]) <= a[i]

        demand = Equation(self.m, name="demand", domain=[j])
        demand[j] = Sum(i, x[i, j]) >= b[j]

        test_model = Model(
            self.m,
            name="test_model",
            equations=[supply, demand],
            problem="LP",
            sense="min",
            objective=z,
        )
        self.assertEqual(
            test_model._generate_solve_string(),
            "solve test_model using LP MIN z;",
        )

    def test_solve_string_mcp(self):
        c = Set(self.m, "c")
        h = Set(self.m, "h")
        s = Set(self.m, "s")

        cc = Alias(self.m, "cc", c)

        e = Parameter(self.m, "e", domain=[c, h])
        esub = Parameter(self.m, "esub", domain=h)

        alpha = Parameter(self.m, "alpha", domain=[c, h])
        a = Parameter(self.m, "a", domain=[c, s])

        p = Variable(self.m, "p", type=VariableType.POSITIVE, domain=c)
        y = Variable(self.m, "y", type=VariableType.POSITIVE, domain=s)
        i = Variable(self.m, "i", type=VariableType.POSITIVE, domain=h)

        mkt = Equation(self.m, "mkt", domain=c, description="commodity market")
        profit = Equation(
            self.m, "profit", domain=s, description="zero profit"
        )
        income = Equation(
            self.m, "income", domain=h, description="income index"
        )

        mkt[c] = Sum(s, a[c, s] * y[s]) + Sum(h, e[c, h]) >= Sum(
            h.where[esub[h] != 1],
            (i[h] / Sum(cc, alpha[cc, h] * p[cc] ** (1 - esub[h])))
            * alpha[c, h]
            * (1 / p[c]) ** esub[h],
        ) + Sum(h.where[esub[h] == 1], i[h] * alpha[c, h] / p[c])

        profit[s] = -Sum(c, a[c, s] * p[c]) >= 0
        income[h] = i[h] >= Sum(c, p[c] * e[c, h])

        hansen = Model(
            self.m,
            "hansen",
            problem=Problem.MCP,
            matches={mkt: p, profit: y, income: i},
        )
        self.assertEqual(
            hansen._generate_solve_string(), "solve hansen using MCP;"
        )

    def test_solve_string_cns(self):
        x = Variable(self.m, "x")
        f = Equation(self.m, "f")
        f[...] = x * x == 4
        x.l = 1
        m = Model(
            self.m, name="m", equations=[f], problem="CNS", sense="FEASIBILITY"
        )
        self.assertEqual(m._generate_solve_string(), "solve m using CNS;")


def model_suite():
    suite = unittest.TestSuite()
    tests = [
        ModelSuite(name)
        for name in dir(ModelSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(model_suite())
