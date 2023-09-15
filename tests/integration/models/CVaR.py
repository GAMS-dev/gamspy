"""
Conditional Value at Risk models

CVaR.gms: Conditional Value at Risk models.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 5.5
Last modified: Apr 2008.
"""
from pathlib import Path

import numpy as np

from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Parameter
from gamspy import Smax
from gamspy import Smin
from gamspy import Sum
from gamspy import Variable


def index_data():
    np.array(
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


def main():
    gdx_file = str(Path(__file__).parent.absolute()) + "/WorldIndices.gdx"
    m = Container(load_from=gdx_file)

    # SETS #
    i, l = m.getSymbols(["i", "l"])

    # SCALARS #
    Budget = Parameter(
        m, name="Budget", description="Nominal investment budget"
    )
    alpha = Parameter(m, name="alpha", description="Confidence level")
    MU_TARGET = Parameter(
        m, name="MU_TARGET", description="Target portfolio return"
    )
    MU_STEP = Parameter(m, name="MU_STEP", description="Target return step")
    MIN_MU = Parameter(
        m, name="MIN_MU", description="Minimum return in universe"
    )
    MAX_MU = Parameter(
        m, name="MAX_MU", description="Maximum return in universe"
    )
    RISK_TARGET = Parameter(
        m, name="RISK_TARGET", description="Bound on CVaR (risk)"
    )
    LossFlag = Parameter(
        m, name="LossFlag", description="Flag selecting the type of loss"
    )

    Budget.assign = 100.0
    alpha.assign = 0.99

    # PARAMETERS #
    pr = Parameter(
        m, name="pr", domain=[l], description="Scenario probability"
    )
    P = Parameter(m, name="P", domain=[i, l], description="Final values")
    EP = Parameter(
        m, name="EP", domain=[i], description="Expected final values"
    )
    AssetReturns = m.getSymbols(["AssetReturns"])[0]

    pr[l] = 1.0 / Card(l)

    P[i, l] = 1 + AssetReturns[i, l]

    EP[i] = Sum(l, pr[l] * P[i, l])

    MIN_MU.assign = Smin(i, EP[i])
    MAX_MU.assign = Smax(i, EP[i])

    # Assume we want 20 portfolios in the frontier

    MU_STEP.assign = (MAX_MU - MIN_MU) / 20

    TargetIndex = Parameter(
        m, name="TargetIndex", domain=[l], description="Target index returns"
    )

    # To test the model with a market index, uncomment the following two lines.
    # Note that, this index can be used only with WorldIndexes.inc.
    # Index = Parameter(m, name="Index", domain=[l], records=index_data(), description="Index returns")
    # TargetIndex[l] = Index[l]

    # VARIABLES #
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i],
        description="Holdings of assets in monetary units (not proportions)",
    )
    VaRDev = Variable(
        m,
        name="VaRDev",
        type="positive",
        domain=[l],
        description="Measures of the deviations from the VaR",
    )
    VaR = Variable(m, name="VaR", description="Value-at-Risk")
    z = Variable(m, name="z", description="Objective function value")
    Losses = Variable(
        m, name="Losses", domain=[l], description="Measures of the losses"
    )

    # EQUATIONS #
    BudgetCon = Equation(
        m,
        name="BudgetCon",
        type="regular",
        description="Equation defining the budget contraint",
    )
    ReturnCon = Equation(
        m,
        name="ReturnCon",
        type="regular",
        description="Equation defining the portfolio return constraint",
    )
    CVaRCon = Equation(
        m,
        name="CVaRCon",
        type="regular",
        description="Equation defining the CVaR allowed",
    )
    ObjDefCVaR = Equation(
        m,
        name="ObjDefCVaR",
        type="regular",
        description="Objective function definition for CVaR minimization",
    )
    ObjDefReturn = Equation(
        m,
        name="ObjDefReturn",
        type="regular",
        description="Objective function definition for return mazimization",
    )
    LossDef = Equation(
        m,
        name="LossDef",
        type="regular",
        domain=[l],
        description="Equations defining the losses",
    )
    VaRDevCon = Equation(
        m,
        name="VaRDevCon",
        type="regular",
        domain=[l],
        description="Equations defining the VaR deviation constraints",
    )

    BudgetCon.expr = Sum(i, x[i]) == Budget

    ReturnCon.expr = Sum(i, EP[i] * x[i]) >= MU_TARGET * Budget

    CVaRCon.expr = VaR + Sum(l, pr[l] * VaRDev[l]) / (1 - alpha) <= RISK_TARGET

    VaRDevCon[l] = VaRDev[l] >= Losses[l] - VaR

    LossDef[l] = (
        Losses[l]
        == (Budget - Sum(i, P[i, l] * x[i])).where[LossFlag == Number(1)]
        + (TargetIndex[l] * Budget - Sum(i, P[i, l] * x[i])).where[
            LossFlag == Number(2)
        ]
        + (Sum(i, EP[i] * x[i]) - Sum(i, P[i, l] * x[i])).where[
            LossFlag == Number(3)
        ]
    )

    ObjDefCVaR.expr = z == VaR + Sum(l, pr[l] * VaRDev[l]) / (1 - alpha)

    ObjDefReturn.expr = z == Sum(i, EP[i] * x[i])

    MinCVaR = Model(
        m,
        name="MinCVaR",
        equations=[BudgetCon, ReturnCon, LossDef, VaRDevCon, ObjDefCVaR],
        problem="LP",
        sense="MIN",
        objective=z,
    )

    output_csv = '"Status","VaR","CVaR","Mean",'
    i_recs = [f'"{i_rec}"' for i_rec in i.records.uni.tolist()]
    output_csv += ",".join(i_recs) + "\n"

    LossFlag.assign = 2

    # Comment the following line if you want to track the market index.
    TargetIndex[l] = 1.01

    mu_target = MIN_MU.records.value[0]
    while mu_target <= MAX_MU.records.value[0]:
        MU_TARGET.assign = mu_target

        MinCVaR.solve()

        output_csv += (
            f"{str(MinCVaR.status).split('.')[-1]},{VaR.records.level.round(3)[0]},{z.records.level.round(3)[0]},{round(MU_TARGET.records.value[0] * Budget.records.value[0],3)},"
        )
        x_recs = [str(x_rec) for x_rec in x.records.level.round(2).tolist()]
        output_csv += ",".join(x_recs) + "\n"

        mu_target += MU_STEP.records.value[0]

    FrontierHandle = open("CVaRFrontiers_new.csv", "w", encoding="UTF-8")
    FrontierHandle.write(output_csv)
    FrontierHandle.close()


if __name__ == "__main__":
    main()
