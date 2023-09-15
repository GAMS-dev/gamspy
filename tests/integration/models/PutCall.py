"""
Put/Call efficient frontier model

MAD.gms: Put/Call efficient frontier model.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 5.7
Last modified: Apr 2008.
"""
from pathlib import Path
from sys import argv

import numpy as np
import pandas as pd

from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def index_data():
    data = np.array(
        [
            1.034769211,
            1.024362083,
            1.005721076,
            1.014359531,
            1.008024402,
            1.015164086,
            1.011776948,
            0.999367312,
            1.022337255,
            1.014084294,
            1.002478561,
            1.004830772,
            1.022048865,
            1.001584376,
            1.014789245,
            1.018465908,
            1.02023507,
            1.001142139,
            1.028357519,
            1.012274672,
            1.005320247,
            0.991451365,
            1.007344745,
            1.017165464,
            0.997931432,
            0.990041853,
            0.9933217,
            0.991912481,
            1.036980768,
            0.994451813,
            1.007412617,
            0.955150659,
            0.958840232,
            1.033624988,
            1.018079311,
            1.010736753,
            1.016700717,
            1.048121923,
            1.015146812,
            1.0074283,
            1.025778693,
            0.983151788,
            1.022476292,
            1.004850227,
            1.0091122,
            0.995361207,
            0.992432266,
            1.024212556,
            1.022181138,
            1.006772127,
            0.990354861,
            1.0067649,
            1.008943286,
            0.983974563,
            1.000347655,
            1.001230771,
            1.020880921,
            1.019415483,
            1.008638044,
            1.017173042,
            1.011831605,
            1.018260066,
            1.014054495,
            1.011841085,
            0.988155751,
            1.009224769,
            1.010321982,
            1.033008126,
            0.999626315,
            1.022733809,
            1.00083866,
            1.039925522,
            1.0144168,
            0.974402144,
            0.971108237,
            1.002198078,
            1.001221179,
            0.988342113,
            1.01750901,
            1.019968377,
            0.992895551,
            1.000905761,
            0.997333999,
            0.994982525,
            0.987972315,
            1.000443735,
            1.01583315,
            1.018338346,
            1.026183302,
            0.999209138,
            1.018809687,
            1.009775599,
            1.01722367,
            0.999752239,
            1.015986198,
            1.019400469,
            1.017624299,
            0.996067065,
            1.00947457,
            1.0164916,
            0.997929107,
            1.008892447,
            0.990498799,
            1.011843222,
            1.023338977,
            1.001536085,
            1.023978065,
            0.994238826,
            1.022819962,
            1.012142345,
            0.989862601,
            1.021050206,
            1.019165002,
            1.023260624,
            1.027413626,
            0.971346297,
            1.026087902,
            0.959990261,
            1.008484984,
            1.014365951,
            1.010347985,
            1.029266037,
            1.01989954,
            0.999383722,
            0.992191473,
            0.993782093,
            1.010730888,
            0.918149916,
            1.008729179,
            1.031641438,
            1.032632304,
            1.004115995,
            1.007398109,
            0.99859111,
            1.032604061,
            1.027883261,
            0.979268923,
            1.024656356,
            0.989556024,
            1.001926578,
            0.991713793,
            1.020697126,
            1.022268183,
            1.031273723,
            0.992301368,
            1.011547524,
            1.02121809,
            0.988701862,
            0.991540642,
            1.014546705,
            0.997009882,
            1.01168258,
            0.982543898,
            0.992197467,
        ]
    )
    return data


def main():
    gdx_file = str(Path(__file__).parent.absolute()) + "/WorldIndices.gdx"
    m = Container(load_from=gdx_file)

    output = argv[1] if len(argv) > 1 else "PutCallModel"

    m.addOptions({"LIMROW": 0, "LIMCOL": 0, "SOLVELINK": 2})

    # SETS #
    i, l = m.getSymbols(["i", "l"])

    # SCALARS #
    Budget = Parameter(
        m, name="Budget", description="Nominal investment budget"
    )
    Omega = Parameter(
        m, name="Omega", description="Bound on the expected shortfalls"
    )

    Budget.assign = 100.0

    # PARAMETERS #
    pr = Parameter(
        m, name="pr", domain=[l], description="Scenario probability"
    )
    P = Parameter(m, name="P", domain=[i, l], description="Final values")
    EP = Parameter(
        m, name="EP", domain=[i], description="Expected final values"
    )
    TargetIndex = Parameter(
        m, name="TargetIndex", domain=[l], description="Target index returns"
    )
    AssetReturns = m.getSymbols(["AssetReturns"])[0]

    pr[l] = 1.0 / Card(l)

    P[i, l] = 1 + AssetReturns[i, l]

    EP[i] = Sum(l, pr[l] * P[i, l])

    # To test the model with a market index, uncomment the following line two lines.
    # Note that, this index is consistent only when using WorldIndexes.inc.
    Index = Parameter(
        m,
        name="Index",
        domain=[l],
        records=index_data(),
        description="Index returns",
    )
    TargetIndex[l] = Index[l]

    # VARIABLES
    yPos = Variable(
        m,
        name="yPos",
        type="positive",
        domain=[l],
        description="Positive deviations",
    )
    yNeg = Variable(
        m,
        name="yNeg",
        type="positive",
        domain=[l],
        description="Negative deviations",
    )
    x = Variable(
        m,
        name="x",
        type="free",
        domain=[i],
        description="Holdings of assets in monetary units (not proportions)",
    )
    z = Variable(
        m, name="z", type="free", description="Objective function value"
    )

    # EQUATIONS #
    BudgetCon = Equation(
        m,
        name="BudgetCon",
        type="regular",
        description="Equation defining the budget contraint",
    )
    ObjDef = Equation(
        m,
        name="ObjDef",
        type="regular",
        description="Objective function definition for MAD",
    )
    TargetDevDef = Equation(
        m,
        name="TargetDevDef",
        type="regular",
        domain=[l],
        description="Equations defining the positive and negative deviations",
    )
    PutCon = Equation(
        m,
        name="PutCon",
        type="regular",
        description=(
            "Constraint to bound the expected value of the negative deviations"
        ),
    )

    BudgetCon.expr = Sum(i, x[i]) == Budget

    PutCon.expr = Sum(l, pr[l] * yNeg[l]) <= Omega

    TargetDevDef[l] = (
        Sum(i, (P[i, l] - TargetIndex[l]) * x[i]) == yPos[l] - yNeg[l]
    )

    ObjDef.expr = z == Sum(l, pr[l] * yPos[l])

    UnConPutCallModel = Model(
        m,
        name="UnConPutCallModel",
        equations=[PutCon, TargetDevDef, ObjDef],
        problem="LP",
        sense="MAX",
        objective=z,
    )

    # Set the average level of downside (risk) allowed

    Omega.assign = 0.1
    UnConPutCallModel.solve()

    # Dual of the UnConstrained Put/Call model

    # VARIABLES #
    Pi = Variable(m, name="Pi", type="positive", domain=[l])
    PiOmega = Variable(m, name="PiOmega", type="positive")

    # EQUATIONS #
    DualObjDef = Equation(m, name="DualObjDef", type="regular")
    DualTrackingDef = Equation(
        m, name="DualTrackingDef", type="regular", domain=[i]
    )
    MeasureDef = Equation(m, name="MeasureDef", type="regular", domain=[l])

    DualObjDef.expr = z == Omega * PiOmega

    DualTrackingDef[i] = Sum(l, (P[i, l] - TargetIndex[l]) * Pi[l]) == 0.0

    MeasureDef[l] = pr[l] * PiOmega - Pi[l] >= 0

    Pi.lo[l] = pr[l]

    UnConDualPutCallModel = Model(
        m,
        name="UnConDualPutCallModel",
        equations=[DualObjDef, DualTrackingDef, MeasureDef],
        problem="LP",
        sense="MIN",
        objective=z,
    )

    UnConDualPutCallModel.solve()

    # Display PiOmega.l and Pi.l and check that they are, respectively, equal
    # to TargetDevDef.m and PutCon.m
    # GAMS provides the dual prices directly, so it is not
    # really necessary to build explicitly the dual model.

    # PARAMETER
    PrimalDual = Parameter(
        m,
        name="PrimalDual",
        domain=[l, "*"],
        description="Compare primal and dual soultions",
    )

    PrimalDual[l, "pi.l"] = -Pi.l[l]
    PrimalDual[l, "TargetDevDef.m"] = TargetDevDef.m[l]
    PrimalDual[l, "Difference"] = TargetDevDef.m[l] + Pi.l[l]

    # DISPLAY z.l,PiOmega.l,PutCon.m,PrimalDual

    # We propose an alternative way to build a frontier using
    # the loop statement. Such a structure is suitable for the
    # GDX utility (for details, see gdxutility.pdf included in the doc folder )

    m.addOptions({"SOLPRINT": "OFF"})

    # SET
    FrontierPoints = Set(
        m,
        name="FrontierPoints",
        records=[f"P_{fp}" for fp in range(1, 51)],
        description="Number of points in the frontier",
    )
    j = Alias(m, name="j", alias_with=FrontierPoints)

    # PARAMETER
    FrontierPortfolios = Parameter(
        m,
        name="FrontierPortfolios",
        domain=[j, i],
        description="Frontier portfolios",
    )
    CallValues = Parameter(
        m, name="CallValues", domain=[j, "*"], description="Call values"
    )
    DualPrices = Parameter(
        m, name="DualPrices", domain=[j, "*"], description="Dual prices"
    )
    PutCall = Parameter(
        m, name="PutCall", domain=[j, "*"], description="Put and Call values"
    )
    OmegaLevels = Parameter(
        m, name="OmegaLevels", domain=[j], description="Risk levels (Omega)"
    )

    # We assign to each point a risk level Omega

    OmegaLevels["P_1"] = 0.01

    for idx, jj in enumerate(j.toList()):
        if idx == 0:
            continue
        elif (idx + 1) <= 10:
            OmegaLevels[jj] = OmegaLevels[j.toList()[idx - 1]] + Number(0.01)
        elif (idx + 1) > 10:
            OmegaLevels[jj] = OmegaLevels[j.toList()[idx - 1]] + Number(0.025)

    DualPrices[j, "Omega"] = OmegaLevels[j]

    CallValues[j, "Omega"] = OmegaLevels[j]

    # Set some liquidity constraints

    x.lo[i] = -100.0
    x.up[i] = 100.0

    for jj in j.toList():
        Omega.assign = OmegaLevels[jj]
        UnConPutCallModel.solve()

        FrontierPortfolios[jj, i] = x.l[i]
        CallValues[jj, "Mild Constraint"] = z.l
        DualPrices[jj, "Mild Constraint"] = PutCon.m

    # EXPORT RESULT SUMMARY TO EXCEL SHEET #
    writer = pd.ExcelWriter(f"{output}.xlsx", engine="openpyxl")
    FrontierPortfolios.pivot().to_excel(writer, sheet_name="MildPortfolios")

    # Set tight liquidity constraints

    x.lo[i] = -20.0
    x.up[i] = 20.0

    for jj in j.toList():
        Omega.assign = OmegaLevels[jj]
        UnConPutCallModel.solve()

        FrontierPortfolios[jj, i] = x.l[i]
        CallValues[jj, "Tight Constraint"] = z.l
        DualPrices[jj, "Tight Constraint"] = PutCon.m

    DualPrices.pivot().to_excel(writer, sheet_name="DualPrices")
    CallValues.pivot().to_excel(writer, sheet_name="CallValues")
    FrontierPortfolios.pivot().to_excel(writer, sheet_name="TightPortfolios")

    # Determine the liquidity and discount premium
    # for one put/call efficient portfolio.

    Omega.assign = 0.475

    UnConPutCallModel.solve()

    # SCALAR
    Df = Parameter(m, name="Df")

    # PARAMETERS
    Price = Parameter(m, name="Price", domain=[i])
    Discount = Parameter(m, name="Discount", domain=[i])
    Premium = Parameter(m, name="Premium", domain=[i])
    BenchMarkNeutralPrice = Parameter(
        m, name="BenchMarkNeutralPrice", domain=[i]
    )
    Psi = Parameter(m, name="Psi", domain=[l])
    liquidity = Parameter(
        m, name="liquidity", domain=[i, "*"], description="Liquidity report"
    )

    Discount[i] = 0.0

    Premium[i] = 0.0

    Df.assign = Sum(l, -TargetDevDef.m[l])

    Psi[l] = -TargetDevDef.m[l] / Df

    BenchMarkNeutralPrice[i] = Sum(l, Psi[l] * P[i, l]) / Sum(
        l, Psi[l] * TargetIndex[l]
    )

    Discount[i].where[x.m[i] > 0] = x.m[i] / Sum(
        l, (-TargetDevDef.m[l]) * TargetIndex[l]
    )

    Premium[i].where[x.m[i] < 0] = (-x.m[i]) / Sum(
        l, (-TargetDevDef.m[l]) * TargetIndex[l]
    )

    Price[i] = BenchMarkNeutralPrice[i] + Premium[i] - Discount[i]

    liquidity[i, "Premium"] = Premium[i]
    liquidity[i, "Discount"] = Discount[i]

    liquidity.pivot().to_excel(writer, sheet_name="Liquidity")

    # Put/call model with balance constraint

    PutCallModel = Model(
        m,
        name="PutCallModel",
        equations=[BudgetCon, PutCon, TargetDevDef, ObjDef],
        problem="LP",
        sense="MAX",
        objective=z,
    )

    for jj in j.toList():
        Omega.assign = OmegaLevels[jj]
        PutCallModel.solve()
        FrontierPortfolios[jj, i] = x.l[i]
        PutCall[jj, "Put side"] = PutCon.l
        PutCall[jj, "Call side"] = z.l

    PutCall.pivot().to_excel(writer, sheet_name="Frontiers")
    FrontierPortfolios.pivot().to_excel(writer, sheet_name="Portfolios")
    writer.close()


if __name__ == "__main__":
    main()
