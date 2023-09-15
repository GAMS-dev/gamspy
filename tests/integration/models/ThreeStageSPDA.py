"""
* ThreeStageSPDA.gms: A three stage stochastic programming model for SPDA
* Consiglio, Nielsen, Vladimirou and Zenios: A Library of Financial Optimization Models, Section 5.4
* See also Zenios: Practical Financial Optimization, Section 6.4.
* Last modified: Nov. 2005.
"""
import numpy as np
import pandas as pd

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def prepare_yield():
    cols = ["UU", "UD", "DD", "DU"]
    idxs = [
        ("IO2", "T0"),
        ("IO2", "T1"),
        ("PO7", "T0"),
        ("PO7", "T1"),
        ("PO70", "T0"),
        ("PO70", "T1"),
        ("IO90", "T0"),
        ("IO90", "T1"),
    ]

    data = np.array(
        [
            [1.104439, 1.104439, 0.959238, 0.959238],
            [1.110009, 0.975907, 0.935106, 1.167817],
            [0.938159, 0.938159, 1.166825, 1.166825],
            [0.933668, 1.154590, 1.156536, 0.903233],
            [0.924840, 0.924840, 1.167546, 1.167546],
            [0.891527, 1.200802, 1.141917, 0.907837],
            [1.107461, 1.107461, 0.908728, 0.908728],
            [1.105168, 0.925925, 0.877669, 1.187143],
        ]
    )

    idxs = pd.MultiIndex.from_tuples(idxs, names=["Index1", "Index2"])
    data = pd.DataFrame(data, columns=cols, index=idxs)
    data.reset_index(inplace=True)
    melted_data = data.melt(
        id_vars=["Index1", "Index2"], value_vars=["UU", "UD", "DD", "DU"]
    )
    return np.array(melted_data).tolist()


def main():
    m = Container()

    # SETS #
    Scenarios = Set(
        m,
        name="Scenarios",
        records=["uu", "ud", "dd", "du"],
        description="Set of scenarios",
    )
    Assets = Set(
        m,
        name="Assets",
        records=["io2", "po7", "po70", "io90"],
        description="Available assets",
    )
    Time = Set(
        m, name="Time", records=["t0", "t1", "t2"], description="Time steps"
    )

    # ALIASES #
    l = Alias(m, name="l", alias_with=Scenarios)
    i = Alias(m, name="i", alias_with=Assets)
    t = Alias(m, name="t", alias_with=Time)

    # PARAMETERS #
    Yield = Parameter(
        m,
        name="Yield",
        domain=[i, t, l],
        records=prepare_yield(),
        description="Asset yields",
    )
    CashYield = Parameter(
        m,
        name="CashYield",
        domain=[t, l],
        records=np.array(
            [
                [1.030414, 1.030414, 1.012735, 1.012735],
                [1.032623, 1.014298, 1.009788, 1.030481],
                [0, 0, 0, 0],
            ]
        ),
        description="Risk free (cash) yield",
    )
    Liability = Parameter(
        m,
        name="Liability",
        domain=[t, l],
        records=np.array(
            [
                [0, 0, 0, 0],
                [26.474340, 26.474340, 10.953843, 10.953843],
                [31.264791, 26.044541, 10.757200, 13.608207],
            ]
        ),
        description="Liabilities due to annuitant lapses",
    )
    FinalLiability = Parameter(
        m,
        name="FinalLiability",
        domain=[l],
        records=np.array([47.284751, 49.094838, 86.111238, 83.290085]),
        description="Final liabilities",
    )
    Output = Parameter(
        m,
        name="Output",
        domain=["*", i],
        description=(
            "Parameter used to save the optimal holdings for each model"
        ),
    )
    PropCost = Parameter(
        m, name="PropCost", description="Proportional transaction cost"
    )

    # VARIABLES #
    buy = Variable(
        m,
        name="buy",
        type="positive",
        domain=[t, i, l],
        description="Amount purchased",
    )
    sell = Variable(
        m,
        name="sell",
        type="positive",
        domain=[t, i, l],
        description="Amount sold",
    )
    hold = Variable(
        m,
        name="hold",
        type="positive",
        domain=[t, i, l],
        description="Holdings",
    )
    cash = Variable(
        m,
        name="cash",
        type="positive",
        domain=[t, l],
        description="Holding in cash",
    )
    wealth = Variable(m, name="wealth", domain=[l], description="Final wealth")
    z = Variable(m, name="z", description="Objective function value")

    # EQUATIONS #
    AssetInventoryCon = Equation(
        m,
        name="AssetInventoryCon",
        type="regular",
        domain=[t, i, l],
        description="Constraints defining the asset inventory balance",
    )
    CashInventoryCon = Equation(
        m,
        name="CashInventoryCon",
        type="regular",
        domain=[t, l],
        description="Constraint defining the inventory balance",
    )
    WealthRatioDef = Equation(
        m,
        name="WealthRatioDef",
        type="regular",
        domain=[l],
        description="Equations defining the final asset-liability ratio",
    )
    NonAnticConOne = Equation(
        m,
        name="NonAnticConOne",
        type="regular",
        domain=[i, l],
        description="Constraints defining the first nonanticipativity set",
    )
    NonAnticConTwo = Equation(
        m,
        name="NonAnticConTwo",
        type="regular",
        domain=[i, l],
        description="Constraints defining the second nonanticipativity set",
    )
    ExpWealthObjDef = Equation(
        m,
        name="ExpWealthObjDef",
        type="regular",
        description="Expected wealth objective function definition",
    )

    AssetInventoryCon[t, i, l] = (
        buy[t, i, l].where[Ord(t) < Card(t)]
        + (Yield[i, t.lag(1), l] * hold[t.lag(1), i, l]).where[
            Ord(t) > Number(1)
        ]
        == sell[t, i, l].where[Ord(t) > Number(1)]
        + hold[t, i, l].where[Ord(t) < Card(t)]
    )

    CashInventoryCon[t, l] = (
        Sum(i, sell[t, i, l] * (1 - PropCost)).where[Ord(t) > Number(1)]
        + (CashYield[t.lag(1), l] * cash[t.lag(1), l]).where[
            Ord(t) > Number(1)
        ]
        + Number(100).where[Ord(t) == Number(1)]
        == Sum(i, buy[t, i, l]).where[Ord(t) < Card(t)]
        + cash[t, l]
        + Liability[t, l]
    )

    NonAnticConOne[i, l].where[Ord(l) < Card(l)] = (
        hold["t0", i, l] == hold["t0", i, l.lead(1)]
    )

    NonAnticConTwo[i, l].where[
        (Ord(l) == Number(1)) | (Ord(l) == Number(3))
    ] = (hold["t1", i, l] == hold["t1", i, l.lead(1)])

    WealthRatioDef[l] = wealth[l] == cash["t2", l] / FinalLiability[l]

    ExpWealthObjDef.expr = z == Sum(l, wealth[l]) / Card(l)

    ThreeStageExpWealth = Model(
        m,
        name="ThreeStageExpWealth",
        equations=[
            AssetInventoryCon,
            CashInventoryCon,
            WealthRatioDef,
            ExpWealthObjDef,
            NonAnticConOne,
            NonAnticConTwo,
        ],
        problem="LP",
        sense="MAX",
        objective=z,
    )

    # Model 1: Maximize the expected wealth, without transaction cost

    PropCost.assign = 0.0

    ThreeStageExpWealth.solve()

    print("\n\n ### Model 1 ### \n")
    print("\nbuy: \n", buy.pivot().round(3))
    print("\nsell: \n", sell.pivot().round(3))
    print("\nhold: \n", hold.pivot().round(3))
    print("\nwealth: \n", wealth.records.iloc[:, :2].round(3))

    Output["Exp Wealth no TC", i] = hold.l["t0", i, "uu"]

    # Model 2: Maximize the expected wealth, with transaction cost

    PropCost.assign = 0.01

    ThreeStageExpWealth.solve()

    print("\n\n ### Model 2 ### ")
    print("\nbuy: \n", buy.pivot().round(3))
    print("\nsell: \n", sell.pivot().round(3))
    print("\nhold: \n", hold.pivot().round(3))
    print("\nwealth: \n", wealth.records.iloc[:, :2].round(3))
    Output["Exp Wealth with TC", i] = hold.l["t0", i, "uu"]

    # Model 3: Maximize the worst-cast outcome.

    WorstCase = Variable(m, name="WorstCase", description="Worst case outcome")
    WorstCaseDef = Equation(
        m,
        name="WorstCaseDef",
        type="regular",
        domain=[l],
        description="Equations defining the worst case outcome",
    )

    WorstCaseDef[l] = WorstCase <= wealth[l]

    ThreeStageWorstCase = Model(
        m,
        name="ThreeStageWorstCase",
        equations=[
            AssetInventoryCon,
            CashInventoryCon,
            WealthRatioDef,
            WorstCaseDef,
            NonAnticConOne,
            NonAnticConTwo,
        ],
        problem="LP",
        sense="MAX",
        objective=WorstCase,
    )

    ThreeStageWorstCase.solve()

    print("\n\n ### Model 3 ### ")
    print("\nbuy: \n", buy.pivot().round(3))
    print("\nsell: \n", sell.pivot().round(3))
    print("\nhold: \n", hold.pivot().round(3))
    print("\nwealth: \n", wealth.records.iloc[:, :2].round(3))
    print("\nWorstCase: \n", round(WorstCase.records.level[0], 3))

    Output["Worst Case", i] = hold.l["t0", i, "uu"]

    # Model 4: Maximize expected utility:

    UtilityObjDef = Equation(
        m,
        name="UtilityObjDef",
        type="regular",
        description="Utility objective function definition",
    )

    UtilityObjDef.expr = z == Sum(l, gams_math.log(wealth[l])) / Card(l)

    ThreeStageUtility = Model(
        m,
        name="ThreeStageUtility",
        equations=[
            AssetInventoryCon,
            CashInventoryCon,
            WealthRatioDef,
            UtilityObjDef,
            NonAnticConOne,
            NonAnticConTwo,
        ],
        problem="NLP",
        sense="MAX",
        objective=z,
    )

    ThreeStageUtility.solve()

    print("\n\n ### Model 4 ### ")
    print("\nbuy: \n", buy.pivot().round(3))
    print("\nsell: \n", sell.pivot().round(3))
    print("\nhold: \n", hold.pivot().round(3))
    print("\nwealth: \n", wealth.records.iloc[:, :2].round(3))
    print("\nz: \n", round(z.records.level[0], 3))

    Output["Utility", i] = hold.l["t0", i, "uu"]

    # Model 5: Maximize expected wealth with MAD constraints such that A/L > 1.1

    EpsTolerance = Parameter(m, name="EpsTolerance", description="Tolerance")

    MADCon = Equation(
        m,
        name="MADCon",
        type="regular",
        domain=[l],
        description="MAD contraints",
    )

    MADCon[l] = wealth[l] >= 1.1 - EpsTolerance

    ThreeStageMAD = Model(
        m,
        name="ThreeStageMAD",
        equations=[
            AssetInventoryCon,
            CashInventoryCon,
            WealthRatioDef,
            MADCon,
            ExpWealthObjDef,
            NonAnticConOne,
            NonAnticConTwo,
        ],
        problem="LP",
        sense="MAX",
        objective=z,
    )

    EpsTolerance.assign = 0.09

    ThreeStageMAD.solve()

    print("\n\n ### Model 5 ### ")
    print("\nbuy: \n", buy.pivot().round(3))
    print("\nsell: \n", sell.pivot().round(3))
    print("\nhold: \n", hold.pivot().round(3))
    print("\nwealth: \n", wealth.records.iloc[:, :2].round(3))
    print("\nz: \n", round(z.records.level[0], 3))

    Output["MAD", i] = hold.l["t0", i, "uu"]

    # EXPORT RESULT SUMMARY TO EXCEL #

    writer = pd.ExcelWriter("ThreeStage.xlsx", engine="openpyxl")

    # Write each DataFrame to a separate sheet in the Excel file
    Output.pivot().to_excel(writer, sheet_name="Holdings")
    buy.pivot().to_excel(writer, sheet_name="Purchase")
    sell.pivot().to_excel(writer, sheet_name="Sell")

    # Save and close the ExcelWriter
    writer.close()


if __name__ == "__main__":
    main()
