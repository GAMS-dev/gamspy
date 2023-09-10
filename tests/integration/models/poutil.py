"""
Portfolio Optimization for Electric Utilities (POUTIL)

We discuss a portfolio optimization problem occurring in the energy
market. Energy distributing public services have to decide how much
of the requested energy demand has to be produced in their own power
plant, and which complementary amount has to be bought from the spot
market and from load following contracts.

This problem is formulated as a mixed-integer linear programming
problem and implemented in GAMS. The formulation is applied to real data
of a German electricity distributor.

Most equations contain the reference number of the formula in the
publication.


Rebennack, S, Kallrath, J, and Pardalos, P M, Energy Portfolio
Optimization for Electric Utilities: Case Study for Germany. In
Bj�rndal, E, Bj�rndal, M, Pardalos, P.M. and R�nnqvist, M Eds,.
Springer, pp. 221-246, 2010.

Keywords: mixed integer linear programming, energy economics, portfolio
optimization,
          unit commitment, economic dispatch, power plant control,
          day-ahead market
"""

from gamspy import (
    Set,
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Smax,
    Card,
    Ord,
    Sense,
)
import numpy as np


def main():
    cont = Container()

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
        cont, name="cSM", type="positive", description="cost of energy from SM"
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

    alpha.up.assign = Smax(t, PowerForecast[t])
    beta.up.assign = alpha.up
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
    LFCcost = Equation(cont, name="LFCcost", description="cost for the LFC")
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
    obj.expr = c == cPP + cSM + cLFC

    # meet the power demand for each time period exactly eq. (23)
    demand[t] = pPP[t] + pSM[t] + pLFC[t] == PowerForecast[t]

    # (fix cost +) variable cost * energy amount produced eq. (7) & (8)
    PPcost.expr = cPP == cPPvar * Sum(t, 0.25 * pPP[t])

    # power produced by the power plant eq. (26)
    PPpower[t] = pPP[t] == pPPMax * Sum(
        m.where[Ord(m) > 1], 0.1 * (Ord(m) + 2) * delta[m, t]
    )

    # the power plant is in exactly one stage at any time eq. (25)
    PPstage[t] = Sum(m, delta[m, t]) == 1

    # next constraints model the minimum time period a power plant is in the
    # same state and the constraint of the minimum idle time
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
    SMcost.expr = cSM == 24 * cBL * alpha + 12 * cPL * beta

    # Spot Market power contribution eq. (9)
    SMpower[t] = pSM[t] == alpha + IPL[t] * beta

    # cost of the LFC is given by the energy rate eq. (14) & (21)
    LFCcost.expr = cLFC == Sum(b, cLFCs[b] * mu[b] + cLFCvar[b] * eLFCs[b])

    # total energy from the LFC eq. (16)
    # connect the eLFC[t] variables with eLFCtot
    LFCenergy.expr = eLFCtot == Sum(t, 0.25 * pLFC[t])

    # indicator variable 'mu':
    # we are in exactly one price segment b eq. (18)
    LFCmu.expr = Sum(b, mu[b]) == 1

    # connect the 'mu' variables with the total energy amount eq. (19)
    LFCenergyS.expr = eLFCtot == Sum(
        b.where[Ord(b) > 1], eLFCb[b.lag(1)] * mu[b]
    ) + Sum(b, eLFCs[b])

    # accumulated energy amount for segment "b1" eq. (20)
    LFCemuo.expr = eLFCs["b1"] <= eLFCb["b1"] * mu["b1"]

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

    # relative termination criterion for MIP (relative gap)
    # termination criterion is decreased to 0.1 from 0.000001
    cont.addOptions({"optCr": 0.1})

    energy.solve()

    print("Objective Function Value: ", energy.objective_value)


if __name__ == "__main__":
    main()
