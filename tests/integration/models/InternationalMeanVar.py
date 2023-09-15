"""
International asset allocation model

InternationalMeanVar.gms:  International asset allocation model.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 3.5
Last modified: Apr 2008.


We use real data for the 10-year period 1990-01-01 to 2000-01-01,

       23 Italian Stock indices
       3 Italian Bond indices (1-3yr, 3-7yr, 5-7yr)
       Italian risk-free rate (3-month cash)

       7 international Govt. bond indices
       5 Regions Stock Indices: (EMU, Eur-ex-emu, PACIF, EMER, NORAM)
       3 risk-free rates (3-mth cash) for EUR, US, JP

       US Corporate Bond Sector Indices (Finance, Energy, Life Ins.)

       Exchange rates, ITL to: (FRF, DEM, ESP, GBP, US, YEN, EUR)
       Also US to EUR.
"""
from pathlib import Path

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    gdx_file = (
        str(Path(__file__).parent.absolute()) + "/InternationalMeanVar.gdx"
    )
    m = Container(load_from=gdx_file)

    # SETS #
    ASSETS = m.getSymbols(["ASSETS"])[0]

    # ALIASES #
    i, j = m.getSymbols(["i", "j"])

    # SUBSETS #
    IT_STOCK, IT_ALL, INT_STOCK, INT_ALL = m.getSymbols(
        ["IT_STOCK", "IT_ALL", "INT_STOCK", "INT_ALL"]
    )

    # PARAMETERS #
    MAX_MU, ExpectedReturns, VarCov, RiskFree = m.getSymbols(
        ["MAX_MU", "MU", "Q", "RiskFreeRt"]
    )

    # Build more symbols
    # SETS #
    ACTIVE = Set(m, name="ACTIVE", domain=[ASSETS])

    a = Alias(m, name="a", alias_with=ACTIVE)
    a1 = Alias(m, name="a1", alias_with=ACTIVE)
    a2 = Alias(m, name="a2", alias_with=ACTIVE)

    # Target return

    # SCALARS #
    MU_TARGET = Parameter(
        m, name="MU_TARGET", description="Target portfolio return"
    )
    MU_STEP = Parameter(m, name="MU_STEP", description="Target return step")

    # Assume we want 20 portfolios in the frontier

    MU_STEP.assign = MAX_MU / 20

    # VARIABLES #
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i],
        description="Holdings of assets",
    )
    PortVariance = Variable(
        m, name="PortVariance", description="Portfolio variance"
    )

    # EQUATIONS #
    ReturnCon = Equation(
        m,
        name="ReturnCon",
        type="regular",
        description="Equation defining the portfolio return constraint",
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

    ReturnCon.expr = Sum(a, ExpectedReturns[a] * x[a]) == MU_TARGET

    VarDef.expr = PortVariance == Sum([a1, a2], x[a1] * VarCov[a1, a2] * x[a2])

    NormalCon.expr = Sum(a, x[a]) == 1

    m.addOptions({"SOLVEOPT": "REPLACE"})

    MeanVar = Model(
        m,
        name="MeanVar",
        equations=[ReturnCon, VarDef, NormalCon],
        problem="nlp",
        sense="MIN",
        objective=PortVariance,
    )

    FrontierHandle = open(
        "InternationalMeanVarFrontier.csv", "w", encoding="UTF-8"
    )

    # Step 1: First solve only for Italian stocks:

    ACTIVE[i] = IT_STOCK[i]
    print("\nStep 1: Italian stock assets\n")

    FrontierHandle.write('"Step 1: Italian stock assets"\n')
    FrontierHandle.write('"Variance","ExpReturn",')

    # Asset labels
    i_recs = [f'"{i_rec}"' for i_rec in ACTIVE.toList()]
    FrontierHandle.write(",".join(i_recs) + "\n")

    mu = 0
    while round(mu, 7) < round(MAX_MU.toList()[0], 7):
        MU_TARGET.assign = mu
        MeanVar.solve()
        print("PortVariance: ", round(PortVariance.toValue(), 3))

        FrontierHandle.write(
            f"{round(PortVariance.toValue(),4)},{round(MU_TARGET.toValue(),4)},"
        )

        x_recs = [str(round(x_rec, 4)) for x_rec in x.toDict().values()]
        FrontierHandle.write(",".join(x_recs))

        FrontierHandle.write("\n")

        mu += MU_STEP.toList()[0]

    #
    # Step 2: Now solve for Italian stock and government indices:
    #
    ACTIVE[i] = IT_ALL[i]
    print("\nStep 2: Italian stock and government assets\n")

    FrontierHandle.write('"Step 2: Italian stock and government assets"\n')
    FrontierHandle.write('"Variance","ExpReturn",')

    # Asset labels
    i_recs = [f'"{i_rec}"' for i_rec in ACTIVE.toList()]
    FrontierHandle.write(",".join(i_recs) + "\n")

    mu = 0
    while round(mu, 7) < round(MAX_MU.toList()[0], 7):
        MU_TARGET.assign = mu
        MeanVar.solve()
        print("PortVariance: ", round(PortVariance.toValue(), 3))

        FrontierHandle.write(
            f"{round(PortVariance.toValue(),4)},{round(MU_TARGET.toValue(),4)},"
        )

        x_recs = [str(round(x_rec, 4)) for x_rec in x.toDict().values()]
        FrontierHandle.write(",".join(x_recs))

        FrontierHandle.write("\n")

        mu += MU_STEP.toList()[0]

    #
    # Step 3: Italian stock plus international stock indices
    #
    ACTIVE[i] = INT_STOCK[i]
    print("\nStep 3: Italian and international stock indices\n")

    FrontierHandle.write('"Step 3: Italian and international stock indices"\n')
    FrontierHandle.write('"Variance","ExpReturn",')

    # Asset labels
    i_recs = [f'"{i_rec}"' for i_rec in ACTIVE.toList()]
    FrontierHandle.write(",".join(i_recs) + "\n")

    mu = 0
    while round(mu, 7) < round(MAX_MU.toList()[0], 7):
        MU_TARGET.assign = mu
        MeanVar.solve()
        print("PortVariance: ", round(PortVariance.toValue(), 3))

        FrontierHandle.write(
            f"{round(PortVariance.toValue(),4)},{round(MU_TARGET.toValue(),4)},"
        )

        x_recs = [str(round(x_rec, 4)) for x_rec in x.toDict().values()]
        FrontierHandle.write(",".join(x_recs))

        FrontierHandle.write("\n")

        mu += MU_STEP.toList()[0]

    #
    # Step 4: Italian stock and government indices, international stock and government
    # indices, plus corporate indices.
    #

    ACTIVE[i] = INT_ALL[i]
    print("\nStep 4: All indices\n")

    FrontierHandle.write('"Step 4: All indices"\n')
    FrontierHandle.write('"Variance","ExpReturn",')

    # Asset labels
    i_recs = [f'"{i_rec}"' for i_rec in ACTIVE.toList()]
    FrontierHandle.write(",".join(i_recs) + "\n")

    mu = 0
    while round(mu, 7) < round(MAX_MU.toList()[0], 7):
        MU_TARGET.assign = mu
        MeanVar.solve()
        print("PortVariance: ", round(PortVariance.toValue(), 3))

        FrontierHandle.write(
            f"{round(PortVariance.toValue(),4)},{round(MU_TARGET.toValue(),4)},"
        )

        x_recs = [str(round(x_rec, 4)) for x_rec in x.toDict().values()]
        FrontierHandle.write(",".join(x_recs))

        FrontierHandle.write("\n")

        mu += MU_STEP.toList()[0]

    #
    # Step 5: All italian stock indices plus  risk free
    #

    # VARIABLES #
    z = Variable(m, name="z", type="free")
    d_bar = Variable(m, name="d_bar", type="free")

    # EQUATIONS #
    RiskFreeReturnDef = Equation(m, name="RiskFreeReturnDef", type="regular")
    SharpeRatio = Equation(m, name="SharpeRatio", type="regular")

    RiskFreeReturnDef.expr = (
        d_bar == Sum(a, ExpectedReturns[a] * x[a]) - RiskFree
    )

    SharpeRatio.expr = z == d_bar / gams_math.sqrt(PortVariance)

    Sharpe = Model(
        m,
        name="Sharpe",
        equations=[RiskFreeReturnDef, VarDef, NormalCon, SharpeRatio],
        problem="nlp",
        sense="MAX",
        objective=z,
    )

    Sharpe.solve()
    print("\nStep 5: Tangent portfolio\n")
    print("z: ", round(z.toValue(), 3))

    # Write the variance and expected return for the tangent portfolio

    FrontierHandle.write('"Step 5: Tangent portfolio"\n')
    FrontierHandle.write('"Variance","RiskFree","z",')

    # Asset labels
    i_recs = [f'"{i_rec}"' for i_rec in ACTIVE.toList()]
    FrontierHandle.write(",".join(i_recs) + "\n")

    FrontierHandle.write(
        f"{round(PortVariance.toValue(),4)},{round((d_bar.toValue()+RiskFree.toValue()),4)},{round(z.toValue(),4)},"
    )

    # Write the tangent portfolio.

    x_recs = [str(round(x_rec, 4)) for x_rec in x.toDict().values()]
    FrontierHandle.write(",".join(x_recs))
    FrontierHandle.write("\n")

    #
    # Step 6: Include the total Italian stock index as a liability
    #
    #
    # Build a model (very similar to the previous one)
    # which attempts to track (synthesize) the Italian total stock index,
    # ITMHIST, using the 23 Italian stock indices and 3 Italian bond indices
    # plus the Italian risk-free asset.
    #
    # This is done by including ITMHIST as an asset but fixing its weight
    # in the portfolio at -1. The 26 other assets then must try to balance
    # out the variance of ITMHIST. In addition, we pursue different levels
    # of expected return (over and above the ITMHIST return).

    # Create a convenient subset containing only the general Italian stock index:

    It_general = Set(
        m, name="It_general", domain=[ASSETS], records=["ITMHIST"]
    )

    # The only constraint which need to be redefined is the
    # normalization constraint. Indeed, it must be se to 0.

    NormalConTrack = Equation(
        m,
        name="NormalConTrack",
        type="regular",
        description=(
            "Equation defining the normalization contraint for tracking"
        ),
    )

    NormalConTrack.expr = Sum(a, x[a]) == 0

    m.addOptions({"SOLVEOPT": "REPLACE"})

    MeanVarTrack = Model(
        m,
        name="MeanVarTrack",
        equations=[ReturnCon, VarDef, NormalConTrack],
        problem="nlp",
        sense="MIN",
        objective=PortVariance,
    )

    x.fx[It_general] = -1

    FrontierHandle.write('"Step 6: Index tracking"\n')
    print("\nStep 6: Index tracking\n")

    ACTIVE[i] = IT_STOCK[i] | It_general[i]

    FrontierHandle.write('"Status","Variance","ExpReturn",')

    # Asset labels
    i_recs = [f'"{i_rec}"' for i_rec in ACTIVE.toList()]
    FrontierHandle.write(",".join(i_recs) + "\n")

    # Re-estimate MU_STEP as MAX_MU is different for the tracking problem
    MAX_MU.assign = 0.1587
    MU_STEP.assign = MAX_MU / 20

    mu = 0
    while round(mu, 7) < round(MAX_MU.toList()[0], 7):
        MU_TARGET.assign = mu
        MeanVarTrack.solve()
        print("PortVariance: ", round(PortVariance.toValue(), 3))

        FrontierHandle.write(
            f"{MeanVarTrack.status},{round(PortVariance.toValue(),4)},{round(MU_TARGET.toValue(),4)},"
        )

        x_recs = [str(round(x_rec, 4)) for x_rec in x.toDict().values()]
        FrontierHandle.write(",".join(x_recs))

        FrontierHandle.write("\n")

        mu += MU_STEP.toList()[0]

    FrontierHandle.close()


if __name__ == "__main__":
    main()
