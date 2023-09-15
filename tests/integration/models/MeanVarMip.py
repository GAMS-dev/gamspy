"""
Mean-variance model with diversification constraints

MeanVarMip.gms:  Mean-variance model with diversification constraints.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 3.4
Last modified: Apr 2008.
"""
from pathlib import Path

from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container()

    # Read from MeanVarMip.gdx the data needed to run the mean-variance model
    m.read(
        str(Path(__file__).parent.absolute()) + "/MeanVarMip.gdx",
        [
            "assets",
            "SUBSET",
            "s1",
            "s2",
            "MeanRiskFreeReturn",
            "VarCov",
            "ExpectedReturns",
        ],
    )

    # SETS #
    Assets = m.getSymbols(["subset"])[0]

    # ALIASES #
    i = Alias(m, name="i", alias_with=Assets)
    j = Alias(m, name="j", alias_with=Assets)

    # PARAMETERS #
    ExpectedReturns, VarCov = m.getSymbols(["ExpectedReturns", "VarCov"])

    # Risk attitude: 0 is risk-neutral, 1 is very risk-averse.
    StockMax = Parameter(
        m, name="StockMax", records=3, description="Maximum number of stocks"
    )
    lamda = Parameter(m, name="lamda", description="Risk attitude")
    xlow = Parameter(
        m,
        name="xlow",
        domain=[i],
        description="lower bound for active variables",
    )

    # VARIABLES #
    x = Variable(m, name="x", domain=[i], description="Holdings of assets")
    Y = Variable(
        m,
        name="Y",
        type="binary",
        domain=[i],
        description="Indicator variable for assets included in the portfolio",
    )
    PortVariance = Variable(
        m, name="PortVariance", description="Portfolio variance"
    )
    PortReturn = Variable(m, name="PortReturn", description="Portfolio return")
    z = Variable(m, name="z", description="Objective function value")

    # In case short sales are allowed these bounds must be set properly.
    xlow[i] = 0.0
    x.up[i] = 1.0

    # EQUATIONS #
    ReturnDef = Equation(
        m,
        name="ReturnDef",
        type="regular",
        description="Equation defining the portfolio return",
    )
    VarDef = Equation(
        m,
        name="VarDef",
        type="regular",
        description="Equation defining the portfolio variance",
    )
    NormalCon = Equation(
        m,
        name="NormalCon",
        type="regular",
        description="Equation defining the normalization contraint",
    )
    LimitCon = Equation(
        m,
        name="LimitCon",
        type="regular",
        description="Constraint defining the maximum number of assets allowed",
    )
    UpBounds = Equation(
        m,
        name="UpBounds",
        type="regular",
        domain=[i],
        description="Upper bounds for each variable",
    )
    LoBounds = Equation(
        m,
        name="LoBounds",
        type="regular",
        domain=[i],
        description="Lower bounds for each variable",
    )
    ObjDef = Equation(
        m,
        name="ObjDef",
        type="regular",
        description="Objective function definition",
    )

    ReturnDef.expr = PortReturn == Sum(i, ExpectedReturns[i] * x[i])

    VarDef.expr = PortVariance == Sum([i, j], x[i] * VarCov[i, j] * x[j])

    LimitCon.expr = Sum(i, Y[i]) <= StockMax

    UpBounds[i] = x[i] <= x.up[i] * Y[i]

    LoBounds[i] = x[i] >= xlow[i] * Y[i]

    NormalCon.expr = Sum(i, x[i]) == 1

    ObjDef.expr = z == (1 - lamda) * PortReturn - lamda * PortVariance

    MeanVarMip = Model(
        m,
        name="MeanVarMip",
        equations=[
            ReturnDef,
            VarDef,
            LimitCon,
            UpBounds,
            LoBounds,
            NormalCon,
            ObjDef,
        ],
        problem="MINLP",
        sense="MAX",
        objective=z,
    )

    m.addOptions({"MINLP": "SBB", "optcr": 0})

    MeanVarianceMIP = '"Lambda","z","Variance","ExpReturn",'
    i_recs = [f'"{i_rec}"' for i_rec in i.records.ASSETS.tolist()]
    MeanVarianceMIP += ",".join(i_recs)
    MeanVarianceMIP += "\n"

    lamda_loop = 0
    while True:
        if lamda_loop > 1:
            break
        lamda.assign = lamda_loop
        MeanVarMip.solve()
        MeanVarianceMIP += f"{round(lamda_loop,1)},{round(z.records.level[0],4)},{round(PortVariance.records.level[0],4)},{round(PortReturn.records.level[0],4)},"
        x_recs = [str(round(x_rec, 4)) for x_rec in x.records.level.tolist()]
        MeanVarianceMIP += ",".join(x_recs)
        MeanVarianceMIP += "\n"
        lamda_loop += 0.1

    FrontierHandle = open("MeanVarianceMIP.csv", "w", encoding="UTF-8")
    FrontierHandle.write(MeanVarianceMIP)
    FrontierHandle.close()

    # ***** Transaction Cost *****
    # In this section the MeanVar.gms model is modified by imposing transaction
    # costs. We consider a more realistic setting with fixed and proportional costs.

    FlatCost = Parameter(
        m, name="FlatCost", records=0.001, description="Flat transaction cost"
    )
    PropCost = Parameter(
        m,
        name="PropCost",
        records=0.005,
        description="Proportional transaction cost",
    )

    x_0 = Variable(
        m,
        name="x_0",
        type="positive",
        domain=[i],
        description="Holdings for the flat cost regime",
    )
    x_1 = Variable(
        m,
        name="x_1",
        type="positive",
        domain=[i],
        description="Holdings for the linear cost regime",
    )

    # Amount at which is possible to make transactions at the flat fee.

    x_0.up[i] = 0.1

    HoldingCon = Equation(
        m,
        name="HoldingCon",
        type="regular",
        domain=[i],
        description="Constraint defining the holdings",
    )
    ReturnDefWithCost = Equation(
        m,
        name="ReturnDefWithCost",
        type="regular",
        description="Equation defining the portfolio return with cost",
    )
    FlatCostBounds = Equation(
        m,
        name="FlatCostBounds",
        type="regular",
        domain=[i],
        description="Upper bounds for flat transaction fee",
    )
    LinCostBounds = Equation(
        m,
        name="LinCostBounds",
        type="regular",
        domain=[i],
        description="Upper bonds for linear transaction fee",
    )

    HoldingCon[i] = x[i] == x_0[i] + x_1[i]

    ReturnDefWithCost.expr = PortReturn == Sum(
        i, (ExpectedReturns[i] * x_0[i] - FlatCost * Y[i])
    ) + Sum(i, (ExpectedReturns[i] - PropCost) * x_1[i])

    FlatCostBounds[i] = x_0[i] <= x_0.up[i] * Y[i]

    LinCostBounds[i] = x_1[i] <= Y[i]

    MeanVarWithCost = Model(
        m,
        name="MeanVarWithCost",
        equations=[
            ReturnDefWithCost,
            VarDef,
            HoldingCon,
            NormalCon,
            FlatCostBounds,
            LinCostBounds,
            ObjDef,
        ],
        problem="MINLP",
        sense="MAX",
        objective=z,
    )

    m.addOptions({"MINLP": "SBB", "optcr": 0})

    MeanVarianceWithCost = '"Lambda","z","Variance","ExpReturn",'
    MeanVarianceWithCost += ",".join(i_recs) + ","
    MeanVarianceWithCost += ",".join(i_recs)
    MeanVarianceWithCost += "\n"

    lamda_loop = 0
    while True:
        if lamda_loop > 1:
            break
        lamda.assign = lamda_loop
        MeanVarWithCost.solve()
        MeanVarianceWithCost += f"{round(lamda_loop,1)},{round(z.records.level[0],4)},{round(PortVariance.records.level[0],4)},{round(PortReturn.records.level[0],4)},"
        x0_recs = [
            str(round(x_rec, 4)) for x_rec in x_0.records.level.tolist()
        ]
        x1_recs = [
            str(round(x_rec, 4)) for x_rec in x_1.records.level.tolist()
        ]
        MeanVarianceWithCost += ",".join(x0_recs) + ","
        MeanVarianceWithCost += ",".join(x1_recs) + "\n"
        lamda_loop += 0.1

    FrontierHandleTwo = open("MeanVarianceWithCost.csv", "w", encoding="UTF-8")
    FrontierHandleTwo.write(MeanVarianceWithCost)
    FrontierHandleTwo.close()

    # ***** Portfolio Revision *****
    # In this section the MeanVar.gms model is modified by imposing zero-or-range
    # variable to cope with portfolio revision.

    Bound = Set(m, name="Bound", records=["Lower", "Upper"])

    BuyLimits = Parameter(m, name="BuyLimits", domain=[Bound, i])
    SellLimits = Parameter(m, name="SellLimits", domain=[Bound, i])
    InitHold = Parameter(
        m, name="InitHold", domain=[i], description="Current holdings"
    )

    # We set the curret holding to the optimal unconstrained mean-variance portfolio
    # with lamda = 0.5

    InitHold["Cash_EU"] = 0.3686
    InitHold["YRS_1_3"] = 0.3597
    InitHold["EMU"] = 0.0
    InitHold["EU_EX"] = 0.0
    InitHold["PACIFIC"] = 0.0
    InitHold["EMERGT"] = 0.0591
    InitHold["NOR_AM"] = 0.2126
    InitHold["ITMHIST"] = 0.0

    BuyLimits["Lower", i] = InitHold[i] * 0.9
    BuyLimits["Upper", i] = InitHold[i] * 1.10

    SellLimits["Lower", i] = InitHold[i] * 0.75
    SellLimits["Upper", i] = InitHold[i] * 1.25

    buy = Variable(
        m,
        name="buy",
        type="positive",
        domain=[i],
        description="Amount to be purchased",
    )
    sell = Variable(
        m,
        name="sell",
        type="positive",
        domain=[i],
        description="Amount to be sold",
    )
    Yb = Variable(
        m,
        name="Yb",
        type="binary",
        domain=[i],
        description="Indicator variable for assets to be purchased",
    )
    Ys = Variable(
        m,
        name="Ys",
        type="binary",
        domain=[i],
        description="Indicator variable for assets to be sold",
    )

    BuyTurnover = Equation(m, name="BuyTurnover", type="regular")
    LoBuyLimits = Equation(m, name="LoBuyLimits", type="regular", domain=[i])
    UpBuyLimits = Equation(m, name="UpBuyLimits", type="regular", domain=[i])
    UpSellLimits = Equation(m, name="UpSellLimits", type="regular", domain=[i])
    LoSellLimits = Equation(m, name="LoSellLimits", type="regular", domain=[i])
    BinBuyLimits = Equation(m, name="BinBuyLimits", type="regular", domain=[i])
    BinSellLimits = Equation(
        m, name="BinSellLimits", type="regular", domain=[i]
    )
    InventoryCon = Equation(
        m,
        name="InventoryCon",
        type="regular",
        domain=[i],
        description="Inventory constraints",
    )

    InventoryCon[i] = x[i] - buy[i] + sell[i] == InitHold[i]

    UpBuyLimits[i] = InitHold[i] + buy[i] <= BuyLimits["Upper", i]

    LoBuyLimits[i] = InitHold[i] + buy[i] >= BuyLimits["Lower", i]

    UpSellLimits[i] = InitHold[i] - sell[i] <= SellLimits["Upper", i]

    LoSellLimits[i] = InitHold[i] - sell[i] >= SellLimits["Lower", i]

    BinBuyLimits[i] = buy[i] <= Yb[i]

    BinSellLimits[i] = sell[i] <= Ys[i]

    BuyTurnover.expr = Sum(i, buy[i]) <= 0.05

    MeanVarRevision = Model(
        m,
        name="MeanVarRevision",
        equations=[
            NormalCon,
            HoldingCon,
            InventoryCon,
            ReturnDef,
            UpBuyLimits,
            LoBuyLimits,
            UpSellLimits,
            LoSellLimits,
            BinBuyLimits,
            BinSellLimits,
            BuyTurnover,
            VarDef,
            ObjDef,
        ],
        problem="MINLP",
        sense="MAX",
        objective=z,
    )

    m.addOptions({"MINLP": "SBB", "optcr": 0})

    MeanVarianceRevision = (
        '"Model status","Lambda","z","Variance","ExpReturn",'
    )

    MeanVarianceRevision += ",".join(i_recs) + ","
    MeanVarianceRevision += ",".join(i_recs) + ","
    MeanVarianceRevision += ",".join(i_recs) + "\n"

    lamda_loop = 0
    while True:
        if lamda_loop > 1:
            break
        lamda.assign = lamda_loop
        MeanVarRevision.solve()
        MeanVarianceRevision += f"{MeanVarRevision.status},{round(lamda_loop,1)},{round(z.records.level[0],4)},{round(PortVariance.records.level[0],4)},{round(PortReturn.records.level[0],4)},"
        x_recs = [str(round(x_rec, 4)) for x_rec in x.records.level.tolist()]
        buy_recs = [
            str(round(x_rec, 4)) for x_rec in buy.records.level.tolist()
        ]
        sell_recs = [
            str(round(x_rec, 4)) for x_rec in sell.records.level.tolist()
        ]
        MeanVarianceRevision += ",".join(x_recs) + ","
        MeanVarianceRevision += ",".join(buy_recs) + ","
        MeanVarianceRevision += ",".join(sell_recs) + "\n"
        lamda_loop += 0.1

    FrontierHandleThree = open(
        "MeanVarianceRevision.csv", "w", encoding="UTF-8"
    )
    FrontierHandleThree.write(MeanVarianceRevision)
    FrontierHandleThree.close()


if __name__ == "__main__":
    main()
