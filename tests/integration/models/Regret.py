"""
Regret models

Regret.gms: Regret models.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 5.4
Last modified: Apr 2008.
"""

from pathlib import Path
from gamspy import (
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Card,
    Smin,
    Smax,
    Sense,
)
import pandas as pd


def main():
    # Define container
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/Regret.gdx",
    )

    # SETS #
    Assets, Time = m.getSymbols(["Assets", "Time"])

    # ALIASES #
    i, l = m.getSymbols(["i", "l"])

    AssetReturns = m.getSymbols(["AssetReturns"])[0]

    # SCALARS #
    Budget = Parameter(
        m, name="Budget", description="Nominal investment budget"
    )
    EpsRegret = Parameter(
        m,
        name="EpsRegret",
        description="Tolerance allowed for epsilon regret models",
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
    RISK_TARGET = Parameter(
        m, name="RISK_TARGET", description="Bound on expected regret (risk)"
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

    # VARIABLES #
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i],
        description="Holdings of assets in monetary units (not proportions)",
    )
    Regrets = Variable(
        m,
        name="Regrets",
        type="positive",
        domain=[l],
        description="Measures of the negative deviations or regrets",
    )
    z = Variable(m, name="z", description="Objective function value")

    # EQUATIONS #
    BudgetCon = Equation(
        m,
        name="BudgetCon",
        description="Equation defining the budget contraint",
    )
    ReturnCon = Equation(
        m,
        name="ReturnCon",
        description="Equation defining the portfolio return constraint",
    )
    ExpRegretCon = Equation(
        m,
        name="ExpRegretCon",
        description="Equation defining the expected regret allowed",
    )
    ObjDefRegret = Equation(
        m,
        name="ObjDefRegret",
        description="Objective function definition for regret minimization",
    )
    ObjDefReturn = Equation(
        m,
        name="ObjDefReturn",
        description="Objective function definition for return mazimization",
    )
    RegretCon = Equation(
        m,
        name="RegretCon",
        domain=[l],
        description="Equations defining the regret constraints",
    )
    EpsRegretCon = Equation(
        m,
        name="EpsRegretCon",
        domain=[l],
        description=(
            "Equations defining the regret constraints with tolerance"
            " threshold"
        ),
    )

    BudgetCon.expr = Sum(i, x[i]) == Budget

    ReturnCon.expr = Sum(i, EP[i] * x[i]) >= MU_TARGET * Budget

    ExpRegretCon.expr = Sum(l, pr[l] * Regrets[l]) <= RISK_TARGET

    RegretCon[l] = Regrets[l] >= TargetIndex[l] * Budget - Sum(
        i, P[i, l] * x[i]
    )

    EpsRegretCon[l] = Regrets[l] >= (
        TargetIndex[l] - EpsRegret
    ) * Budget - Sum(i, P[i, l] * x[i])

    ObjDefRegret.expr = z == Sum(l, pr[l] * Regrets[l])

    ObjDefReturn.expr = z == Sum(i, EP[i] * x[i])

    MinRegret = Model(
        m,
        name="MinRegret",
        equations=[BudgetCon, ReturnCon, RegretCon, ObjDefRegret],
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )
    MaxReturn = Model(
        m,
        name="MaxReturn",
        equations=[BudgetCon, ExpRegretCon, EpsRegretCon, ObjDefReturn],
        problem="LP",
        sense=Sense.MAX,
        objective=z,
    )

    TargetIndex[l] = 1.01
    EpsRegret.assign = 0.0

    # Write a code for displaying the result into a table
    result = []
    mu_iter = MIN_MU.records.value[0]
    while mu_iter <= MAX_MU.records.value[0]:
        MU_TARGET.assign = mu_iter
        MinRegret.solve()

        result.append(
            [
                z.records.level[0],
                (MU_TARGET.records.value[0] * Budget.records.value[0]),
            ]
        )
        result[-1] += x.records.level.tolist()

        RISK_TARGET.assign = z.l

        result[-1].append("")

        MaxReturn.solve()

        result[-1] += [RISK_TARGET.records.value[0], z.records.level[0]]
        result[-1] += x.records.level.tolist()

        mu_iter += MU_STEP.records.value[0]

    res_columns = [
        "Regret",
        "Mean",
        *i.records.uni,
        "",
        "Regret",
        "Mean",
        *i.records.uni,
    ]
    RegretFrontiers = pd.DataFrame(result, columns=res_columns)
    RegretFrontiers.to_csv("RegretFrontiers.csv")


if __name__ == "__main__":
    main()
