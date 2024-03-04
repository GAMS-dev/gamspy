# flake8: noqa
# fmt: off
from __future__ import annotations

import os
import sys
import time
import unittest

import numpy as np
import pandas as pd

import gamspy._validation as validation
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import ModelStatus
from gamspy import SolveStatus
from gamspy import Options
from gamspy import Ord
from gamspy import Parameter
from gamspy import Problem
from gamspy import Sense
from gamspy import Set
from gamspy import Smax
from gamspy import Sum
from gamspy import Variable
from gamspy.exceptions import GamspyException
from gamspy.exceptions import ValidationError
from gamspy.math import sqr


class SolveSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))
        
    def test_uel_order(self):
        i = Set(self.m,'i')
        p = self.m.addParameter('base',[i])
        d = Parameter(self.m,'d')
        i.setRecords(['i1','i2'])
        d[...] = 0 # trigger the execution of GAMS
        i.setRecords(['i0','i1'])
        p[i] = i.ord
        self.assertEqual(p.records.values.tolist(), [['i1', 1.0], ['i0', 2.0]])


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
        self.assertFalse(k._is_dirty)
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
        self.assertFalse(c._is_dirty)
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

        e[...] = 5
        self.assertFalse(e._is_dirty)
        self.assertEqual(e.records.values.tolist(), [[5.0]])
        self.assertEqual(e[...], 5)

        with self.assertRaises(TypeError):
            e.records = 5

        # Variable
        x = Variable(self.m, name="x", domain=[i, j], type="Positive")
        z = Variable(self.m, name="z")

        # Equation
        cost = Equation(self.m, name="cost")
        supply = Equation(self.m, name="supply", domain=[i])
        demand = Equation(self.m, name="demand", domain=[j])

        cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z
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

        cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z
        supply[i] = Sum(j, x[i, j]) <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]
        cost2[...] = Sum((i, j), c[i, j] * x[i, j]) * 5 == z2

        transport = Model(
            self.m,
            name="transport",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
        )
        transport.solve()

        self.assertIsNotNone(z.records)
        self.assertIsNotNone(x.records)
        self.assertIsNotNone(cost.records)
        self.assertIsNotNone(supply.records)
        self.assertIsNotNone(demand.records)

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

        cost[...] = Sum((i, j), c[i, j] * x[i, j]) == z
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
        
        
        freeLinks = Set(self.m, "freeLinks", domain=[i,j], records=[('seattle', 'chicago')])
        # Test limited variables
        transport2 = Model(
            self.m,
            name="transport2",
            equations=[cost, supply, demand],
            problem="LP",
            sense="min",
            objective=z,
            limited_variables=[x[freeLinks]],
        )

        self.assertEqual(
            transport2.getStatement(),
            "Model transport2 / cost,supply,demand,x(freeLinks) /;",
        )

        # Test output redirection
        with open("test.gms", "w") as file:
            _ = transport.solve(
                options=Options(time_limit=100),
                output=file,
            )

        self.assertTrue(os.path.exists("test.gms"))
        self.assertTrue(transport.status == ModelStatus.OptimalGlobal)
        self.assertTrue(transport.solve_status == SolveStatus.NormalCompletion)

        self.assertRaises(
            ValidationError,
            transport.solve,
            None,
            None,
            None,
            None,
            None,
            "bla",
        )

        from gamspy._model import attribute_map

        for attr_name in attribute_map.values():
            self.assertTrue(hasattr(transport, attr_name))

            # Make sure model attributes are not in the container
            self.assertFalse(attr_name in self.m.data.keys())

        # Make sure dummy variable and equation is not in the container
        self.assertFalse(any("dummy_" in name for name in self.m.data.keys()))

        # Test invalid problem
        self.assertRaises(ValueError, Model, self.m, "dummy", "bla", [cost])

        # Test invalid sense
        self.assertRaises(
            ValueError, Model, self.m, "dummy", "LP", [cost], "bla"
        )

        # Test invalid objective variable
        self.assertRaises(
            TypeError, Model, self.m, "dummy", "LP", [cost], "min", a
        )

        # Test invalid commandline options
        self.assertRaises(
            TypeError,
            transport.solve,
            None,
            {"bla": 100},
        )

        self.assertRaises(TypeError, transport.solve, None, 5)

        # Try to solve invalid model
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))
        cost = Equation(m, "cost")
        model = Model(m, "dummy", equations=[cost], problem="LP", sense="min")
        self.assertRaises(Exception, model.solve)

    def test_interrupt(self):
        cont = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))

        power_forecast_recs = np.array(
            [
                287,
                275,
                262,
                250,
                255,
                260,
                265,
                270,
                267,
                265,
                262,
                260,
                262,
                265,
                267,
                270,
                277,
                285,
                292,
                300,
                310,
                320,
                330,
                340,
                357,
                375,
                392,
                410,
                405,
                400,
                395,
                390,
                400,
                410,
                420,
                430,
                428,
                427,
                426,
                425,
                432,
                440,
                447,
                455,
                458,
                462,
                466,
                470,
                466,
                462,
                458,
                455,
                446,
                437,
                428,
                420,
                416,
                412,
                408,
                405,
                396,
                387,
                378,
                370,
                375,
                380,
                385,
                390,
                383,
                377,
                371,
                365,
                368,
                372,
                376,
                380,
                386,
                392,
                398,
                405,
                408,
                412,
                416,
                420,
                413,
                407,
                401,
                395,
                386,
                377,
                368,
                360,
                345,
                330,
                315,
                300,
            ]
        )

        # Energy
        t = Set(
            cont,
            name="t",
            records=[f"t{i}" for i in range(1, 97)],
            description="time slices (quarter-hour)",
        )

        # Parameters
        PowerForecast = Parameter(
            cont,
            name="PowerForecast",
            domain=[t],
            records=power_forecast_recs,
            description="electric power forecast",
        )

        # Power Plant (PP)
        # Scalars
        cPPvar = Parameter(
            cont,
            name="cPPvar",
            records=25,
            description="variable cost of power plant [euro / MWh]",
        )
        pPPMax = Parameter(
            cont,
            name="pPPMax",
            records=300,
            description="maximal capacity of power plant      [MW]",
        )

        # Sets
        m = Set(
            cont,
            name="m",
            records=[f"m{i}" for i in range(1, 9)],
            description="'stage of the power plant",
        )
        iS = Set(
            cont,
            name="iS",
            records=[f"iS{i}" for i in range(9)],
            description="interval for constant PP operation",
        )
        iI = Set(
            cont,
            name="iI",
            records=[f"iI{i}" for i in range(17)],
            description="length of idle time period",
        )

        # Spot Market (SM)
        # Scalars
        cBL = Parameter(
            cont,
            name="cBL",
            records=32,
            description="cost for one base load contract [euro / MWh]",
        )
        cPL = Parameter(
            cont,
            name="cPL",
            records=41,
            description="cost for one peak load contract [euro / MWh]",
        )

        # Parameter
        IPL = Parameter(
            cont,
            name="IPL",
            domain=[t],
            description="indicator function for peak load contracts",
        )
        IPL[t] = (Ord(t) >= 33) & (Ord(t) <= 80)

        # Load following Contract (LFC)
        # Scalars
        pLFCref = Parameter(
            cont,
            name="pLFCref",
            records=400,
            description="power reference level for the LFC",
        )

        b = Set(
            cont,
            name="b",
            records=[f"b{i}" for i in range(1, 4)],
            description="support points of the zone prices",
        )

        # Parameters
        eLFCbY = Parameter(
            cont,
            name="eLFCbY",
            domain=[b],
            records=np.array([54750, 182500, 9000000]),
            description="amount of energy at support point b",
        )
        cLFCvar = Parameter(
            cont,
            name="cLFCvar",
            domain=[b],
            records=np.array([80.0, 65.0, 52.0]),
            description="specific energy price in segment b",
        )
        eLFCb = Parameter(
            cont,
            name="eLFCb",
            domain=[b],
            description="daily border of energy volumes for LFC",
        )
        cLFCs = Parameter(
            cont,
            name="cLFCs",
            domain=[b],
            description="accumulated cost for LFC up to segment b",
        )

        # calculate the daily borders of the energy volumes for the zones
        eLFCb[b] = eLFCbY[b] / 365

        # calculate the accumulated cost
        cLFCs["b1"] = 0
        cLFCs["b2"] = cLFCvar["b1"] * eLFCb["b1"]
        cLFCs[b].where[Ord(b) > 2] = cLFCs[b.lag(1)] + cLFCvar[b.lag(1)] * (
            eLFCb[b.lag(1)] - eLFCb[b.lag(2)]
        )

        # Variables
        c = Variable(cont, name="c", type="free", description="total cost")
        cPP = Variable(
            cont, name="cPP", type="positive", description="cost of PP usage"
        )
        pPP = Variable(
            cont,
            name="pPP",
            type="positive",
            domain=[t],
            description="power withdrawn from power plant",
        )
        delta = Variable(
            cont,
            name="delta",
            type="binary",
            domain=[m, t],
            description="indicate if the PP is in stage m at time t",
        )
        chiS = Variable(
            cont,
            name="chiS",
            type="positive",
            domain=[t],
            description="indicate if there is a PP stage change",
        )
        chiI = Variable(
            cont,
            name="chiI",
            type="positive",
            domain=[t],
            description="indicate if the PP left the idle stage",
        )
        cSM = Variable(
            cont,
            name="cSM",
            type="positive",
            description="cost of energy from SM",
        )
        pSM = Variable(
            cont,
            name="pSM",
            type="positive",
            domain=[t],
            description="power from the spot market",
        )
        alpha = Variable(
            cont,
            name="alpha",
            type="integer",
            description="quantity of base load contracts",
        )
        beta = Variable(
            cont,
            name="beta",
            type="integer",
            description="quantity of peak load contracts",
        )
        cLFC = Variable(
            cont,
            name="cLFC",
            type="positive",
            description="cost of LFC which is the enery rate",
        )
        eLFCtot = Variable(
            cont,
            name="eLFCtot",
            type="positive",
            description="total energy amount of LFC",
        )
        eLFCs = Variable(
            cont,
            name="eLFCs",
            type="positive",
            domain=[b],
            description="energy from LFC in segment b",
        )
        pLFC = Variable(
            cont,
            name="pLFC",
            type="positive",
            domain=[t],
            description="power from the LFC",
        )
        mu = Variable(
            cont,
            name="mu",
            type="binary",
            domain=[b],
            description="indicator for segment b (for zone prices)",
        )

        alpha.up[...] = Smax(t, PowerForecast[t])
        beta.up[...] = alpha.up
        pLFC.up[t] = pLFCref

        # Equations
        obj = Equation(cont, name="obj", description="objective function")
        demand = Equation(
            cont,
            name="demand",
            domain=[t],
            description="demand constraint for energy forcast",
        )
        PPcost = Equation(cont, name="PPcost", description="power plant cost")
        PPpower = Equation(
            cont,
            name="PPpower",
            domain=[t],
            description="power of power plant at time t",
        )
        PPstage = Equation(
            cont,
            name="PPstage",
            domain=[t],
            description="exactly one stage of power plant at any time",
        )
        PPchiS1 = Equation(
            cont,
            name="PPchiS1",
            domain=[t, m],
            description="relate chi and delta variables first constraint",
        )
        PPchiS2 = Equation(
            cont,
            name="PPchiS2",
            domain=[t, m],
            description="relate chi and delta variables second constraint",
        )
        PPstageChange = Equation(
            cont,
            name="PPstageChange",
            domain=[t],
            description="restrict the number of stage changes",
        )
        PPstarted = Equation(
            cont,
            name="PPstarted",
            domain=[t],
            description="connect chiZ and chi variables",
        )
        PPidleTime = Equation(
            cont,
            name="PPidleTime",
            domain=[t],
            description="control the idle time of the plant",
        )
        SMcost = Equation(
            cont,
            name="SMcost",
            description="cost associated with spot market",
        )
        SMpower = Equation(
            cont,
            name="SMpower",
            domain=[t],
            description="power from the spot market",
        )
        LFCcost = Equation(
            cont, name="LFCcost", description="cost for the LFC"
        )
        LFCenergy = Equation(
            cont,
            name="LFCenergy",
            description="total energy from the LFC",
        )
        LFCmu = Equation(
            cont,
            name="LFCmu",
            description="exactly one price segment b",
        )
        LFCenergyS = Equation(
            cont,
            name="LFCenergyS",
            description="connect the mu variables with the total energy",
        )
        LFCemuo = Equation(
            cont,
            name="LFCemuo",
            description="accumulated energy amount for segement b1",
        )
        LFCemug = Equation(
            cont,
            name="LFCemug",
            domain=[b],
            description="accumulated energy amount for all other segements",
        )

        # the objective function: total cost eq. (6)
        obj[...] = c == cPP + cSM + cLFC

        # meet the power demand for each time period exactly eq. (23)
        demand[t] = pPP[t] + pSM[t] + pLFC[t] == PowerForecast[t]

        # (fix cost +) variable cost * energy amount produced eq. (7) & (8)
        PPcost[...] = cPP == cPPvar * Sum(t, 0.25 * pPP[t])

        # power produced by the power plant eq. (26)
        PPpower[t] = pPP[t] == pPPMax * Sum(
            m.where[Ord(m) > 1], 0.1 * (Ord(m) + 2) * delta[m, t]
        )

        # the power plant is in exactly one stage at any time eq. (25)
        PPstage[t] = Sum(m, delta[m, t]) == 1

        # next constraints model the minimum time period a power plant is in
        # the same state and the constraint of the minimum idle time
        # we need variable 'chiS' to find out when a status change takes place
        # eq. (27)
        PPchiS1[t, m].where[Ord(t) > 1] = (
            chiS[t] >= delta[m, t] - delta[m, t.lag(1)]
        )

        # second constraint for 'chiS' variable eq. (28)
        PPchiS2[t, m].where[Ord(t) > 1] = (
            chiS[t] >= delta[m, t.lag(1)] - delta[m, t]
        )

        # control the minimum change time period eq. (29)
        PPstageChange[t].where[Ord(t) < Card(t) - Card(iS) + 2] = (
            Sum(iS, chiS[t.lead(Ord(iS))]) <= 1
        )

        # indicate if the plant left the idle state eq. (30)
        PPstarted[t] = chiI[t] >= delta["m1", t.lag(1)] - delta["m1", t]

        # control the minimum idle time period:
        # it has to be at least Nk2 time periods long eq. (31)
        PPidleTime[t].where[Ord(t) < Card(t) - Card(iI) + 2] = (
            Sum(iI, chiI[t.lead(Ord(iI))]) <= 1
        )

        # cost for the spot market eq. (12)
        # consistent of the base load (alpha) and peak load (beta) contracts
        SMcost[...] = cSM == 24 * cBL * alpha + 12 * cPL * beta

        # Spot Market power contribution eq. (9)
        SMpower[t] = pSM[t] == alpha + IPL[t] * beta

        # cost of the LFC is given by the energy rate eq. (14) & (21)
        LFCcost[...] = cLFC == Sum(b, cLFCs[b] * mu[b] + cLFCvar[b] * eLFCs[b])

        # total energy from the LFC eq. (16)
        # connect the eLFC[t] variables with eLFCtot
        LFCenergy[...] = eLFCtot == Sum(t, 0.25 * pLFC[t])

        # indicator variable 'mu':
        # we are in exactly one price segment b eq. (18)
        LFCmu[...] = Sum(b, mu[b]) == 1

        # connect the 'mu' variables with the total energy amount eq. (19)
        LFCenergyS[...] = eLFCtot == Sum(
            b.where[Ord(b) > 1], eLFCb[b.lag(1)] * mu[b]
        ) + Sum(b, eLFCs[b])

        # accumulated energy amount for segment "b1" eq. (20)
        LFCemuo[...] = eLFCs["b1"] <= eLFCb["b1"] * mu["b1"]

        # accumulated energy amount for all other segments (then "b1") eq. (20)
        LFCemug[b].where[Ord(b) > 1] = (
            eLFCs[b] <= (eLFCb[b] - eLFCb[b.lag(1)]) * mu[b]
        )

        energy = Model(
            cont,
            name="energy",
            equations=cont.getEquations(),
            problem="MIP",
            sense=Sense.MIN,
            objective=c,
        )
        
        def interrupt_gams(model):
            time.sleep(1)
            model.interrupt()

        import threading

        threading.Thread(target=interrupt_gams, args=(energy,)).start()
        
        try:
            energy.solve(options=Options(relative_optimality_gap=0.000001))
            self.assertIsNotNone(energy.objective_value)
        except GamspyException:
            pass

    def test_solver_options(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))

        # Prepare data
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
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=capacities)
        b = Parameter(m, name="b", domain=[j], records=demands)
        d = Parameter(m, name="d", domain=[i, j], records=distances)
        c = Parameter(m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(m, name="x", domain=[i, j], type="Positive")

        # Equation
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

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

        # Test solver change
        transport.solve(solver="conopt", solver_options={"rtmaxv": "1.e12"})

        self.assertTrue(
            os.path.exists(
                f"{m.workspace.working_directory}{os.sep}conopt.123"
            )
        )

        self.assertRaises(
            ValidationError, transport.solve, None, None, {"rtmaxv": "1.e12"}
        )

    def test_ellipsis(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))

        # Prepare data
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
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=capacities)
        b = Parameter(m, name="b", domain=[j], records=demands)
        d = Parameter(m, name="d", domain=[i, j], records=distances)
        c = Parameter(m, name="c", domain=[i, j])
        c[...] = 90 * d[...] / 1000

        # Variable
        x = Variable(m, name="x", domain=[i, j], type="Positive")

        # Equation
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        supply[...] = Sum(j, x[...]) <= a[...]
        demand[...] = Sum(i, x[...]) >= b[...]

        transport = Model(
            m,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        transport.solve()
        self.assertEqual(transport.objective_value, 153.675)

        supply.l[...] = 5
        self.assertEqual(supply.records.level.to_list(), [5.0, 5.0])
        
        domain = validation._transform_given_indices(["a", "b", "c"], ["a", ...])
        self.assertEqual(domain, ["a", "b", "c"])

        domain = validation._transform_given_indices(["a", "b", "c"], ["a", ..., "c"])
        self.assertEqual(domain, ["a", "b", "c"])

        domain = validation._transform_given_indices(["a", "b", "c"], [..., "c"])
        self.assertEqual(domain, ["a", "b", "c"])
        
        with self.assertRaises(ValidationError):
            c[..., ...] = 5
        
    def test_slice(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))

        # Prepare data
        distances = [
            ["seattle", "new-york", 2.5],
            ["seattle", "chicago", 1.7],
            ["seattle", "topeka", 1.8],
            ["san-diego", "new-york", 2.5],
            ["san-diego", "chicago", 1.8],
            ["san-diego", "topeka", 1.4],
        ]

        capacities = [["seattle", 350], ["san-diego", 600]]

        # Set
        i = Set(m, name="i", records=["seattle", "san-diego"])
        i2 = Set(m, name="i2", records=["seattle", "san-diego"])
        i3 = Set(m, name="i3", records=["seattle", "san-diego"])
        i4 = Set(m, name="i4", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])
        k = Set(m, "k", domain=[i, i2, i3, i4])
        self.assertEqual(k[..., i4].gamsRepr(), "k(i,i2,i3,i4)")
        self.assertEqual(k[i, ..., i4].gamsRepr(), "k(i,i2,i3,i4)")
        self.assertEqual(k[i, ...].gamsRepr(), "k(i,i2,i3,i4)")
        self.assertEqual(k[..., i3, :].gamsRepr(), "k(i,i2,i3,i4)")

        # Data
        a = Parameter(m, name="a", domain=[i], records=capacities)
        self.assertEqual(a[:].gamsRepr(), "a(i)")
        d = Parameter(m, name="d", domain=[i, j], records=distances)
        c = Parameter(m, name="c", domain=[i, j])
        self.assertEqual(c[:, :].gamsRepr(), "c(i,j)")
        c[:, :] = 90 * d[:, :] / 1000

        # Variable
        x = Variable(m, name="x", domain=[i, j], type="Positive")
        self.assertEqual(x[:, :].gamsRepr(), "x(i,j)")

        # Equation
        supply = Equation(m, name="supply", domain=[i])
        self.assertEqual(supply[:].gamsRepr(), "supply(i)")

        supply.l[:] = 5
        self.assertEqual(supply.l[:].gamsRepr(), "supply.l(i)")
        
        date = Set(m, "date", description="trading date")
        ntd = Set(m, "ntd", domain=[date], description="none-training days")
        error = Parameter(m, "error", domain=[date], description="Absolute error")
        error_test = Parameter(
            m,
            "error_test",
            description="Absolute error in entire testing phase",
            is_miro_output=True,
        )
        error_test[:] = Sum(ntd, error[ntd])
        
    def test_max_line_length(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))

        # Prepare data
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
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=capacities)
        b = Parameter(m, name="b", domain=[j], records=demands)
        d = Parameter(m, name="d", domain=[i, j], records=distances)
        c = Parameter(m, name="c", domain=[i, j])
        f = Parameter(m, "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", records=1)
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(m, name="x", domain=[i, j], type="Positive")

        # Equation
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])
        
        # This generates an equation with length > 80000
        supply[i] = Sum(j, x[i, j])*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f*f <= a[i]
        demand[j] = Sum(i, x[i, j]) >= b[j]

        transport = Model(
            m,
            name="transport",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        transport.solve()
        
    def test_summary(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))

        # Prepare data
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
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=capacities)
        b = Parameter(m, name="b", domain=[j], records=demands)
        d = Parameter(m, name="d", domain=[i, j], records=distances)
        c = Parameter(m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(m, name="x", domain=[i, j], type="Positive")

        # Equation
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

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
        summary = transport.solve()
        self.assertTrue(summary['Solver Status'].tolist()[0], 'Normal')
        
        summary = transport.solve(options=Options(trace_file_format=5))
        self.assertIsNone(summary)
        
        transport2 = Model(
            m,
            name="transport2",
            equations=m.getEquations(),
            problem="LP",
            sense=Sense.MIN,
            objective=Sum((i, j), c[i, j] * x[i, j]),
        )
        summary = transport2.solve(options=Options(trace_file="different_path.txt"))
        self.assertTrue(summary['Solver Status'].tolist()[0], 'Normal')
        
    def test_validation(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))
        
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
        i = Set(m, name="i", records=["seattle", "san-diego"])
        j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

        # Data
        a = Parameter(m, name="a", domain=[i], records=capacities)
        b = Parameter(m, name="b", domain=[j], records=demands)
        d = Parameter(m, name="d", domain=[i, j], records=distances)
        c = Parameter(m, name="c", domain=[i, j])
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(m, name="x", domain=[i, j], type="Positive")

        # Equation
        supply = Equation(m, name="supply", domain=[i])
        demand = Equation(m, name="demand", domain=[j])

        with self.assertRaises(ValidationError):
            supply[j] = Sum(j, x[i, j]) <= a[i]
            
        with self.assertRaises(ValidationError):
            demand[i, j] = Sum(i, x[i, j]) >= b[j]
            
        with self.assertRaises(TypeError):
            c[b[j]] = 90 * d[i, j] / 1000
            
    def test_after_exception(self):
        m = Container(system_directory=os.getenv("SYSTEM_DIRECTORY", None))
        x = Variable(m, "x", type="positive")
        e = Equation(m, "e", definition=x <= x + 1)
        t = Model(
            m,
            name="t",
            equations=[e],
            problem="LP",
            sense=Sense.MIN,
            objective=x,
        )
        x.lo[...] = 0  # triggers GAMS
        try:
            t.solve()  # this fails with GAMS compilation error `Objective variable is not a free variable`
        except GamspyException:
            pass
        f = Parameter(m, "f")
        f[...] = 5
        
        self.assertEqual(f._assignment.getStatement(), "f = 5;")
            
    def test_variable_discovery(self):
        x = self.m.addVariable('x')
        l2 = self.m.addModel('l2', problem=Problem.QCP, sense=Sense.MIN, objective=sqr(x - 1) + sqr(x-2))
        self.assertTrue('x' in l2.equations[0]._definition.find_variables())
        
        e = Equation(self.m, "e", definition=x+5)
        self.assertTrue('x' in e._definition.find_variables())
        
    def test_invalid_arguments(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            
        )

        # Prepare data
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
        i = Set(
            m,
            name="i",
            records=["seattle", "san-diego"],
            description="canning plants",
        )
        j = Set(
            m,
            name="j",
            records=["new-york", "chicago", "topeka"],
            description="markets",
        )

        # Data
        a = Parameter(
            m,
            name="a",
            domain=i,
            records=capacities,
            description="capacity of plant i in cases",
        )
        b = Parameter(
            m,
            name="b",
            domain=j,
            records=demands,
            description="demand at market j in cases",
        )
        d = Parameter(
            m,
            name="d",
            domain=[i, j],
            records=distances,
            description="distance in thousands of miles",
        )
        c = Parameter(
            m,
            name="c",
            domain=[i, j],
            description="transport cost in thousands of dollars per case",
        )
        c[i, j] = 90 * d[i, j] / 1000

        # Variable
        x = Variable(
            m,
            name="x",
            domain=[i, j],
            type="Positive",
            description="shipment quantities in cases",
        )

        # Equation
        supply = Equation(
            m,
            name="supply",
            domain=i,
            description="observe supply limit at plant i",
        )
        demand = Equation(
            m, name="demand", domain=j, description="satisfy demand at market j"
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
        
        with self.assertRaises(TypeError):
            transport.solve(solver=sys.stdout)
            
        with self.assertRaises(ValidationError):
            transport.solve(solver="sadwqeq")
            
        with self.assertRaises(ValidationError):
            transport.solve(solver="SNOPT")
            
        with self.assertRaises(TypeError):
            transport.solve(options={"bla": "bla"})
            
        with self.assertRaises(TypeError):
            transport.solve(output={"bla": "bla"})
            
    def test_marking_updated_symbols(self):
        import gamspy as gp
        import numpy as np

        height_data = [
            1.47,
            1.50,
            1.52,
            1.55,
            1.57,
            1.60,
            1.63,
            1.65,
            1.68,
            1.70,
            1.73,
            1.75,
            1.78,
            1.80,
            1.83,
        ]

        weight_data = [
            52.21,
            53.12,
            54.48,
            55.84,
            57.20,
            58.57,
            59.93,
            61.29,
            63.11,
            64.47,
            66.28,
            68.10,
            69.92,
            72.19,
            74.46,
        ]

        m = gp.Container()

        num_recs = len(height_data)

        x_recs = np.ones((num_recs, 2))
        x_recs[:, 0] = height_data


        set_15 = gp.Set(m, name="set15", records=range(15))
        set_2 = gp.Set(m, name="set2", records=range(2))

        X = gp.Parameter(
            m, name="X", domain=[set_15, set_2], records=x_recs, uels_on_axes=True
        )

        y_recs = np.array(weight_data)

        y = gp.Parameter(
            m, name="y", domain=[set_15], records=y_recs, uels_on_axes=True
        )

        w = gp.Variable(m, name="w", domain=[set_2])

        loss = gp.Variable(m, name="loss", domain=[])

        loss_eq = gp.Equation(m, name="set_loss", domain=[])

        loss_eq[...] = loss == gp.Sum(
            set_15, (y[set_15] - gp.Sum(set_2, X[set_15, set_2] * w[set_2])) ** 2
        )

        model = gp.Model(
            m,
            name="OLS",
            problem="QCP",
            equations=m.getEquations(),
            sense="MIN",
            objective=loss,
        )

        model.solve()
        self.assertIsNotNone(w.records)
        self.assertIsNotNone(loss.records)

    

def solve_suite():
    suite = unittest.TestSuite()
    tests = [
        SolveSuite(name)
        for name in dir(SolveSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(solve_suite())
