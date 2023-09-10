"""
Indexation model with selective hedging

* SelectiveHedging.gms: Indexation model with selective hedging
* Consiglio, Nielsen and Zenios.
* PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 7.2.3
* Last modified: Apr 2008.
"""

from pathlib import Path
from gamspy import (
    Set,
    Alias,
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Card,
    Number,
    ModelStatus,
    Sense,
)


def main(output=None):
    # Define container
    m = Container(
        load_from=str(Path(__file__).parent.absolute())
        + "/SelectiveHedging.gdx",
    )

    # SETS #
    BB, EE, BxE, SS = m.getSymbols(["BB2", "EE", "BxE2", "SS"])

    i = Alias(m, name="i", alias_with=BB)
    l = Alias(m, name="l", alias_with=SS)
    s = Alias(m, name="s", alias_with=SS)
    e = Alias(m, name="e", alias_with=EE)

    # PARAMETERS #
    (
        BondPrices1,
        BondPrices0,
        ExchangeRates0,
        ExchangeRates1,
        IndexReturns,
    ) = m.getSymbols(
        ["data2", "data3", "ExchangeRates0", "ExchangeRates1", "IndexReturns"]
    )

    mu = Parameter(m, name="mu", description="Target expected value")
    USDDEMForwardRate = Parameter(
        m, name="USDDEMForwardRate", description="USD-DEM forward rate"
    )
    USDCHFForwardRate = Parameter(
        m, name="USDCHFForwardRate", description="USD-CHF forward rate"
    )
    pr = Parameter(
        m, name="pr", domain=[l], description="Scenario probability"
    )

    USDDEMForwardRate.assign = -0.005
    USDCHFForwardRate.assign = 0.001

    BondReturns = Parameter(
        m, name="BondReturns", domain=[SS, BB], description="Bond returns"
    )
    UnhedgedBondReturns = Parameter(
        m,
        name="UnhedgedBondReturns",
        domain=[SS, BB],
        description="Unhedged bond returns",
    )
    HedgedBondReturns = Parameter(
        m,
        name="HedgedBondReturns",
        domain=[SS, BB],
        description="Hedged bond returns",
    )
    ExchangeRatesReturns = Parameter(
        m, name="ExchangeRatesReturns", domain=[SS, EE]
    )

    BondReturns[l, i] = (BondPrices1[l, i] - BondPrices0[i]) / BondPrices0[i]
    ExchangeRatesReturns[l, e] = (
        ExchangeRates1[l, e] - ExchangeRates0[e]
    ) / ExchangeRates0[e]

    # Unhedged bond returns in USD currency

    UnhedgedBondReturns[l, i].where[BxE[i, "USD"]] = BondReturns[l, i].where[
        BxE[i, "USD"]
    ]
    UnhedgedBondReturns[l, i].where[BxE[i, "DEM"]] = (
        BondReturns[l, i].where[BxE[i, "DEM"]] + ExchangeRatesReturns[l, "DEM"]
    )
    UnhedgedBondReturns[l, i].where[BxE[i, "CHF"]] = (
        BondReturns[l, i].where[BxE[i, "CHF"]] + ExchangeRatesReturns[l, "CHF"]
    )

    # Hedged bond returns

    HedgedBondReturns[l, i].where[BxE[i, "DEM"]] = (
        BondReturns[l, i].where[BxE[i, "DEM"]] + USDDEMForwardRate
    )
    HedgedBondReturns[l, i].where[BxE[i, "CHF"]] = (
        BondReturns[l, i].where[BxE[i, "CHF"]] + USDCHFForwardRate
    )

    pr[l] = 1.0 / Card(l)

    # VARIABLES #
    z = Variable(m, name="z")
    h = Variable(m, name="h", type="positive", domain=[i])
    u = Variable(m, name="u", type="positive", domain=[i])
    y = Variable(m, name="y", type="positive", domain=[l])

    # EQUATIONS #
    ObjDef = Equation(m, name="ObjDef")
    ReturnCon = Equation(m, name="ReturnCon")
    NormalCon = Equation(m, name="NormalCon")
    yPosDef = Equation(m, name="yPosDef", domain=[l])
    yNegDef = Equation(m, name="yNegDef", domain=[l])

    ObjDef.expr = z == Sum(l, pr[l] * y[l])

    yPosDef[l] = y[l] >= Sum(
        i, UnhedgedBondReturns[l, i] * u[i] + HedgedBondReturns[l, i] * h[i]
    ) - Sum(
        s,
        pr[s]
        * Sum(
            i,
            UnhedgedBondReturns[s, i] * u[i] + HedgedBondReturns[s, i] * h[i],
        ),
    )

    ReturnCon.expr = (
        Sum(
            l,
            pr[l]
            * Sum(
                i,
                UnhedgedBondReturns[l, i] * u[i]
                + HedgedBondReturns[l, i] * h[i],
            ),
        )
        >= mu
    )

    yNegDef[l] = y[l] >= Sum(
        s,
        pr[s]
        * Sum(
            i,
            UnhedgedBondReturns[s, i] * u[i] + HedgedBondReturns[s, i] * h[i],
        ),
    ) - Sum(
        i, UnhedgedBondReturns[l, i] * u[i] + HedgedBondReturns[l, i] * h[i]
    )

    NormalCon.expr = Sum(i, h[i] + u[i]) == 1.0

    IndexFund = Model(
        m,
        name="IndexFund",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )

    FrontierPoints = Set(
        m, name="FrontierPoints", records=[f"P_{i}" for i in range(1, 51)]
    )
    p = Alias(m, name="p", alias_with=FrontierPoints)

    Frontiers = Parameter(
        m, name="Frontiers", domain=[p, "*"], description="Frontiers"
    )

    # We assign to each point a return level mu

    Frontiers["P_1", "mu"] = 0.0

    for idx, pp, _ in p.records.itertuples():
        if idx == 0:
            continue
        if idx < 20:
            Frontiers[pp, "mu"] = Frontiers[f"P_{idx}", "mu"] + Number(0.0005)
        elif idx >= 20:
            Frontiers[pp, "mu"] = Frontiers[f"P_{idx}", "mu"] + Number(0.001)

    for pp, _ in p.records.itertuples(index=False):
        if IndexFund.status not in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            continue

        mu.assign = Frontiers[pp, "mu"]

        IndexFund.solve()
        print("Objective: ", round(z.records.level[0], 3))

        if IndexFund.status in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            Frontiers[pp, "Partial Hedge"] = z.l

    # Fully hedged model
    u.fx[i] = 0.0

    IndexFund.status = None
    for pp, _ in p.records.itertuples(index=False):
        if IndexFund.status not in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            continue

        mu.assign = Frontiers[pp, "mu"]

        IndexFund.solve()
        print("Objective: ", round(z.records.level[0], 3))

        if IndexFund.status in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            Frontiers[pp, "Fully Hedged"] = z.l

    # Unhedged model
    u.lo[i] = 0.0
    u.up[i] = 1.0
    h.fx[i] = 0.0

    IndexFund.status = None
    for pp, _ in p.records.itertuples(index=False):
        if IndexFund.status not in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            continue

        mu.assign = Frontiers[pp, "mu"]

        IndexFund.solve()
        print("Objective: ", round(z.records.level[0], 3))

        if IndexFund.status in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            Frontiers[pp, "Unhedged"] = z.l

    # Create an excel file with the information stored in Frontiers
    # To activate it, add the name of the output file to "main" function below
    if output is not None:
        Frontiers.pivot().to_csv(f"{output}.csv")


if __name__ == "__main__":
    main()
