"""
Tracking international bond index - GDX input

* BondIndexGDX.gms: Tracking international bond index - GDX input.
* Consiglio, Nielsen and Zenios.
* PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 8.2
* Last modified: Apr 2008.
"""
from pathlib import Path

import numpy as np

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import ModelStatus
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container()
    m.read(
        load_from=str(Path(__file__).parent.absolute()) + "/BondIndex.gdx",
        symbol_names=[
            "Bonds",
            "Currencies",
            "Scenarios",
            "j",
            "i",
            "l",
            "SS",
            "JxI",
            "ExchangeRates0",
            "ExchangeRates1",
            "Price0",
            "Price1",
            "InitialHoldings",
            "Accruals0",
            "Accruals1",
            "Outstanding",
            "ReinvestmentRate",
            "IndexReturns",
            "pr",
        ],
    )

    # SETS #
    j, i, l = m.getSymbols(["j", "i", "l"])
    JxI = m.getSymbols(["JxI"])[0]

    i_recs = i.records.element_text.tolist()

    # PARAMETERS #
    (
        ExchangeRates0,
        ExchangeRates1,
        Price0,
        Price1,
        InitialHoldings,
        Accruals0,
        Accruals1,
        Outstanding,
        ReinvestmentRate,
        IndexReturns,
        pr,
    ) = m.getSymbols(
        [
            "ExchangeRates0",
            "ExchangeRates1",
            "Price0",
            "Price1",
            "InitialHoldings",
            "Accruals0",
            "Accruals1",
            "Outstanding",
            "ReinvestmentRate",
            "IndexReturns",
            "pr",
        ]
    )

    # SCALARS #
    TrnCstB = Parameter(
        m,
        name="TrnCstB",
        records=0.0025,
        description="Buying transaction costs",
    )
    TrnCstS = Parameter(
        m,
        name="TrnCstS",
        records=0.0015,
        description="Selling transaction costs",
    )
    CashInfusion = Parameter(
        m,
        name="CashInfusion",
        records=100000,
        description="Available budget infused",
    )
    EpsTolerance = Parameter(
        m, name="EpsTolerance", records=0.10, description="Tolerance"
    )
    UpprBnd = Parameter(
        m,
        name="UpprBnd",
        records=0.1,
        description="Maximum holding percentage for each bond",
    )
    CHFtrade = Parameter(
        m,
        name="CHFtrade",
        records=15000000,
        description="Maximum trading value allowed for Swiss bonds (in CHF)",
    )
    HoldVal = Parameter(
        m, name="HoldVal", description="Value of the initial holdings"
    )
    InitAccrCash = Parameter(
        m,
        name="InitAccrCash",
        description="Accrued cash originated by the initial holdings",
    )
    InitVal = Parameter(
        m, name="InitVal", description="Initial portfolio value"
    )

    # Calculate initial portfolio value
    HoldVal.assign = Sum(
        JxI[j, i], ExchangeRates0[j] * InitialHoldings[i] * Price0[i]
    )
    InitAccrCash.assign = Sum(
        JxI[j, i], ExchangeRates0[j] * InitialHoldings[i] * Accruals0[i]
    )
    InitVal.assign = CashInfusion + InitAccrCash + HoldVal

    # VARIABLES #
    X0 = Variable(
        m,
        name="X0",
        type="positive",
        domain=["*"],
        description="Face value bought today.",
    )
    Y0 = Variable(
        m,
        name="Y0",
        type="positive",
        domain=["*"],
        description="Face value sold today.",
    )
    Z0 = Variable(
        m,
        name="Z0",
        type="positive",
        domain=["*"],
        description="Face value hold today for the next period.",
    )
    Cash = Variable(
        m,
        name="Cash",
        type="positive",
        description=(
            "Amount of cash resulting from trading (sell and buy) today."
        ),
    )
    FinalCash = Variable(
        m,
        name="FinalCash",
        type="positive",
        domain=[l],
        description="Amount of cash resulting from portfolio liquidation",
    )
    z = Variable(m, name="z", description="Objective function value")

    # Set the upper bound on the holdings
    for jj, ii, _ in JxI.records.itertuples(index=False):
        Z0.up[ii] = InitVal / ExchangeRates0[jj] / Price0[ii] * UpprBnd

    # Set the limit on trading (sell or buy)
    # CHF bonds for liquidity reasons
    X0.up[i].where[JxI["CHF", i]] = CHFtrade / Price0[i]
    Y0.up[i].where[JxI["CHF", i]] = CHFtrade / Price0[i]

    # EQUATIONS #
    ObjDef = Equation(
        m,
        name="ObjDef",
        type="regular",
        description="Objective function definition (Expected return)",
    )
    CashInventoryCon = Equation(
        m,
        name="CashInventoryCon",
        type="regular",
        description="Cash balance equation today.",
    )
    FinalCashCon = Equation(
        m,
        name="FinalCashCon",
        type="regular",
        domain=[l],
        description="Cash balance equations at the end of first stage.",
    )
    InventoryCon = Equation(
        m,
        name="InventoryCon",
        type="regular",
        domain=[i],
        description="Constraints defining the asset inventory balance",
    )
    MADCon = Equation(
        m,
        name="MADCon",
        type="regular",
        domain=[l],
        description="MAD constraints",
    )

    ObjDef.expr = z == 1000 * Sum(l, pr[l] * (FinalCash[l] / InitVal - 1))

    CashInventoryCon.expr = (
        CashInfusion
        + Sum(JxI[j, i], ExchangeRates0[j] * Y0[i] * Price0[i] * (1 - TrnCstS))
        == Sum(
            JxI[j, i], ExchangeRates0[j] * X0[i] * Price0[i] * (1 + TrnCstB)
        )
        + Cash
    )

    FinalCashCon[l] = (
        Sum(JxI[j, i], ExchangeRates1[j, l] * Accruals1[i, l] * Z0[i])
        + ReinvestmentRate[l] * Cash
        + Sum(
            JxI[j, i],
            ExchangeRates1[j, l]
            * Z0[i]
            * Outstanding[i, l]
            * Price1[i, l]
            * (1 - TrnCstS),
        )
        == FinalCash[l]
    )

    InventoryCon[i] = InitialHoldings[i] + X0[i] == Y0[i] + Z0[i]

    MADCon[l] = (FinalCash[l] / InitVal - 1) - IndexReturns[l] >= -EpsTolerance

    BondIndex = Model(
        m,
        name="BondIndex",
        equations=m.getEquations(),
        problem="LP",
        sense="MAX",
        objective=z,
    )
    m.addOptions(
        {"LIMROW": 0, "LIMCOL": 0, "SOLPRINT": "off"}
    )  # 'Turn off row and colum listing'

    # Find a feasible EpsTolerance
    low = Parameter(m, name="low", description="Lower bisection value")
    high = Parameter(m, name="high", description="Upper bisection value")

    high.assign = -np.inf
    low.assign = 0
    EpsTolerance.assign = 0.01

    while high.records.value[0] <= 0:
        BondIndex.solve()
        if BondIndex.status in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            high.assign = EpsTolerance
        else:
            EpsTolerance.assign = 2 * EpsTolerance

    # Find a small feasible EpsTolerance via bisection
    while True:
        EpsTolerance.assign = (
            low.records.value[0] + high.records.value[0]
        ) / 2
        BondIndex.solve()
        if BondIndex.status in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            high.assign = EpsTolerance
        else:
            low.assign = EpsTolerance

        if (
            (high.records.value[0] - low.records.value[0]) < 0.005
        ) and BondIndex.status in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            break

    CurrentValue = Parameter(
        m, name="CurrentValue", domain=[i], description="Holdings in USD"
    )
    CurrentValue[i] = Sum(JxI[j, i], ExchangeRates0[j]) * Price0[i] * Z0.l[i]

    ColHeaders = Set(
        m,
        name="ColHeaders",
        records=["FaceValue", "USDValue", "Percent"],
        description="Column headers",
    )

    SummaryReport = Parameter(
        m, name="SummaryReport", domain=["*", ColHeaders]
    )

    SummaryReport[i, "FaceValue"] = Z0.l[i]
    SummaryReport[i, "USDValue"] = CurrentValue[i]
    SummaryReport[i, "Percent"] = CurrentValue[i] / InitVal * 100
    SummaryReport["Total", ColHeaders] = Sum(i, SummaryReport[i, ColHeaders])

    print("Summary Report: \n", SummaryReport.pivot().round(3))
    print("\nEpsTolerance: \n", round(EpsTolerance.records.value[0], 3))
    print("\nObjective Function Value: \n", round(z.records.level[0], 3))
    print("\nInitVal: \n", round(InitVal.records.value[0], 3))
    print(CurrentValue.records)

    ResultHandle = open("BondIndex.csv", "w", encoding="UTF-8")
    ResultHandle.write(
        f'"Objective Function", {round(z.records.level[0],3)}\n'
    )
    ResultHandle.write(
        f'"Final Epsilon", {round(EpsTolerance.records.value[0],3)}\n'
    )
    ResultHandle.write(
        '"Initial Portfolio Value in USD",'
        f" {round(InitVal.records.value[0],3)}\n"
    )
    ResultHandle.write("\n")
    ResultHandle.write(
        '"Bond number","CUSIP code","Holdings in unit of face value","Holdings'
        ' in USD","Percentage of the portfolio value"\n'
    )
    for ii in Z0.records.itertuples():
        if ii.level == 0:
            continue
        cv_value = CurrentValue.records.loc[
            CurrentValue.records["i"] == ii.uni, "value"
        ].values[0]
        ResultHandle.write(
            f'"{ii.uni}","{i_recs[ii.Index]}",{round(ii.level,3)},{round(cv_value,2)},{round(100*(cv_value/InitVal.records.value[0]),2)}\n'
        )

    ResultHandle.write(
        '"Cash in US'
        f' dollar",{Cash.records.level[0]},{((Cash.records.level[0]/InitVal.records.value[0])*100)}'
    )
    ResultHandle.close()


if __name__ == "__main__":
    main()
