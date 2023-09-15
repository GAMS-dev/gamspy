"""
Mean absolute deviation model

MAD.gms: Mean absolute deviation model.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 5.3
Last modified: Apr 2008.
"""
from pathlib import Path

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
from gamspy import Parameter
from gamspy import Smax
from gamspy import Smin
from gamspy import Sum
from gamspy import Variable


def sqr(x):
    return gams_math.power(x, 2)


def main():
    gdx_file = str(Path(__file__).parent.absolute()) + "/WorldIndices.gdx"
    m = Container(load_from=gdx_file)

    # SETS #
    i, l = m.getSymbols(["i", "l"])

    # SCALARS #
    Budget = Parameter(
        m, name="Budget", description="Nominal investment budget"
    )
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

    Budget.assign = 100.0

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

    print("MAX_MU: ", MAX_MU.records.value.round(3)[0])

    # VARIABLES #
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i],
        description="Holdings of assets in monetary units (not proportions)",
    )
    y = Variable(
        m,
        name="y",
        type="positive",
        domain=[l],
        description="Measures of the absolute deviation",
    )
    z = Variable(m, name="z", description="Objective function value")

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
    ObjDef = Equation(
        m,
        name="ObjDef",
        type="regular",
        description="Objective function definition for MAD",
    )
    yPosDef = Equation(
        m,
        name="yPosDef",
        type="regular",
        domain=[l],
        description="Equations defining the positive deviations",
    )
    yNegDef = Equation(
        m,
        name="yNegDef",
        type="regular",
        domain=[l],
        description="Equations defining the negative deviations",
    )

    BudgetCon.expr = Sum(i, x[i]) == Budget

    ReturnCon.expr = Sum(i, EP[i] * x[i]) >= MU_TARGET * Budget

    yPosDef[l] = y[l] >= Sum(i, P[i, l] * x[i]) - Sum(i, EP[i] * x[i])

    yNegDef[l] = y[l] >= Sum(i, EP[i] * x[i]) - Sum(i, P[i, l] * x[i])

    ObjDef.expr = z == Sum(l, pr[l] * y[l])

    MeanAbsoluteDeviation = Model(
        m,
        name="MeanAbsoluteDeviation",
        equations=[BudgetCon, ReturnCon, yPosDef, yNegDef, ObjDef],
        problem="LP",
        sense="MIN",
        objective=z,
    )

    m.addOptions({"SOLVEOPT": "REPLACE"})

    output_csv = '"MAD","Mean"\n'

    mu_target = MIN_MU.records.value[0]
    while mu_target <= MAX_MU.records.value[0]:
        # MU_TARGET = MIN_MU TO MAX_MU BY MU_STEP,
        MU_TARGET.assign = mu_target

        MeanAbsoluteDeviation.solve()

        output_csv += (
            f"{z.records.level.round(3)[0]},{round(MU_TARGET.records.value[0] * Budget.records.value[0],3)},"
        )
        x_recs = [str(x_rec) for x_rec in x.records.level.round(3).tolist()]
        output_csv += ",".join(x_recs) + "\n"

        mu_target += MU_STEP.records.value[0]

    # Compute variances and covariances
    # for comparison between Mean Variance and Mean Absolute Deviation

    # ALIAS
    i1 = Alias(m, name="i1", alias_with=i)
    i2 = Alias(m, name="i2", alias_with=i)

    # PARAMETERS
    VP = Parameter(m, name="VP", domain=[i, i])

    VP[i, i] = Sum(l, sqr(P[i, l] - EP[i])) / (Card(l) - 1)

    VP[i1, i2].where[Ord(i1) > Ord(i2)] = Sum(
        l, (P[i1, l] - EP[i1]) * (P[i2, l] - EP[i2])
    ) / (Card(l) - 1)

    print("VP: ", VP.records.value.round(3).tolist())

    # EQUATION
    ObjDefMV = Equation(
        m,
        name="ObjDefMV",
        type="regular",
        description="Objective function definition for Mean-Variance",
    )

    ObjDefMV.expr = z == Sum([i1, i2], x[i1] * VP[i1, i2] * x[i2])

    MeanVariance = Model(
        m,
        name="MeanVariance",
        equations=[BudgetCon, ReturnCon, ObjDefMV],
        problem="NLP",
        sense="MIN",
        objective=z,
    )
    MeanVariance.solve()

    output_csv += '"SD","Mean"\n'

    mu_target = MIN_MU.records.value[0]
    while mu_target <= MAX_MU.records.value[0]:
        MU_TARGET.assign = mu_target

        MeanVariance.solve()
        z.l.assign = gams_math.sqrt(z.l)

        output_csv += (
            f"{z.records.level.round(3)[0]},{round(MU_TARGET.records.value[0] * Budget.records.value[0],3)},"
        )
        x_recs = [str(x_rec) for x_rec in x.records.level.round(3).tolist()]
        output_csv += ",".join(x_recs) + "\n"

        mu_target += MU_STEP.records.value[0]

    # SCALARS
    lambdaPos = Parameter(
        m,
        name="lambdaPos",
        description="Weight attached to positive deviations",
    )
    lambdaNeg = Parameter(
        m,
        name="lambdaNeg",
        description="Weight attached to negative deviations",
    )

    lambdaPos.assign = 0.5
    lambdaNeg.assign = 0.5

    # EQUATIONS
    yPosWeightDef = Equation(
        m,
        name="yPosWeightDef",
        type="regular",
        domain=[l],
        description=(
            "Equations defining the positive deviations with weight attached"
        ),
    )
    yNegWeightDef = Equation(
        m,
        name="yNegWeightDef",
        type="regular",
        domain=[l],
        description=(
            "Equations defining the positive deviations with weight attached"
        ),
    )

    yPosWeightDef[l] = y[l] >= lambdaPos * (
        Sum(i, P[i, l] * x[i]) - Sum(i, EP[i] * x[i])
    )

    yNegWeightDef[l] = y[l] >= lambdaNeg * (
        Sum(i, EP[i] * x[i]) - Sum(i, P[i, l] * x[i])
    )

    MeanAbsoluteDeviationWeighted = Model(
        m,
        name="MeanAbsoluteDeviationWeighted",
        equations=[BudgetCon, ReturnCon, yPosWeightDef, yNegWeightDef, ObjDef],
        problem="LP",
        sense="MIN",
        objective=z,
    )

    output_csv += '"MADWeighted","Mean"\n'

    mu_target = MIN_MU.records.value[0]
    while mu_target <= MAX_MU.records.value[0]:
        MU_TARGET.assign = mu_target

        MeanAbsoluteDeviationWeighted.solve()

        output_csv += (
            f"{z.records.level.round(3)[0]},{round(MU_TARGET.records.value[0] * Budget.records.value[0],3)},"
        )
        x_recs = [str(x_rec) for x_rec in x.records.level.round(3).tolist()]
        output_csv += ",".join(x_recs) + "\n"

        mu_target += MU_STEP.records.value[0]

    lambdaPos.assign = 0.2
    lambdaNeg.assign = 0.8

    output_csv += '"MADWeighted","Mean"\n'

    mu_target = MIN_MU.records.value[0]
    while mu_target <= MAX_MU.records.value[0]:
        MU_TARGET.assign = mu_target

        MeanAbsoluteDeviationWeighted.solve()

        output_csv += (
            f"{z.records.level.round(3)[0]},{round(MU_TARGET.records.value[0] * Budget.records.value[0],3)},"
        )
        x_recs = [str(x_rec) for x_rec in x.records.level.round(3).tolist()]
        output_csv += ",".join(x_recs) + "\n"

        mu_target += MU_STEP.records.value[0]

    FrontierHandle = open("MADvsMV.csv", "w", encoding="UTF-8")
    FrontierHandle.write(output_csv)
    FrontierHandle.close()

    # Note that, the last two models will yield the same portfolios! See PFO Section 5.2.2.


if __name__ == "__main__":
    main()
