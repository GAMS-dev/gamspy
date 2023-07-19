"""
Sharpe model

Sharpe.gms: Sharpe model.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 3.3
Last modified: Apr 2008.
"""

from pathlib import Path
from gamspy import (
    Alias,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
)
import gamspy.math as gams_math
import numpy as np
import pandas as pd


def main():
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/Sharpe.gdx",
    )

    # SETS #
    Assets = m.getSymbols(["subset"])[0]
    ii = Alias(m, name="ii", alias_with=Assets)
    j = Alias(m, name="j", alias_with=Assets)

    # PARAMETERS #
    RiskFreeRate, ExExpectedReturns, ExVarCov = m.getSymbols(
        ["MeanRiskFreeReturn", "MeanExcessRet", "ExcessCov"]
    )

    # VARIABLES #
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[ii],
        description="Holdings of assets",
    )
    PortVariance = Variable(
        m, name="PortVariance", description="Portfolio variance"
    )
    d_bar = Variable(
        m, name="d_bar", description="Portfolio expected excess return"
    )
    z = Variable(m, name="z", description="Objective function value")

    # EQUATIONS #
    ReturnDef = Equation(
        m,
        name="ReturnDef",
        type="eq",
        description="Equation defining the portfolio excess return",
    )
    VarDef = Equation(
        m,
        name="VarDef",
        type="eq",
        description="Equation defining the portfolio excess variance",
    )
    NormalCon = Equation(
        m,
        name="NormalCon",
        type="eq",
        description="Equation defining the normalization contraint",
    )
    ObjDef = Equation(
        m,
        name="ObjDef",
        type="eq",
        description="Objective function definition",
    )

    ReturnDef.definition = d_bar == Sum(ii, ExExpectedReturns[ii] * x[ii])

    VarDef.definition = PortVariance == Sum(
        [ii, j], x[ii] * ExVarCov[ii, j] * x[j]
    )

    NormalCon.definition = Sum(ii, x[ii]) == 1

    ObjDef.definition = z == d_bar / gams_math.sqrt(PortVariance)

    # Put strictly positive bound on Variance to keep the model out of trouble:
    PortVariance.lo.assign = 0.001

    Sharpe = Model(
        m, name="Sharpe", equations=[ReturnDef, VarDef, NormalCon, ObjDef]
    )
    m.solve(Sharpe, problem="nlp", sense="MAX", objective_variable=z)

    print("Objective Function Variable: ", round(z.records.level[0], 3))

    current_port_variance = 0
    results = []
    while current_port_variance <= 1:
        theta = np.sqrt(current_port_variance / PortVariance.records.level[0])
        current_port_return = (
            RiskFreeRate.records.value[0] + theta * d_bar.records.level[0]
        )
        results.append(
            [np.sqrt(current_port_variance), current_port_return, theta]
        )
        current_port_variance += 0.1

    # Also plot the tangent portfolio
    theta = 1
    results.append(
        [
            np.sqrt(PortVariance.records.level[0]),
            RiskFreeRate.records.value[0] + theta * d_bar.records.level[0],
            theta,
        ]
    )
    SharpeFrontier = pd.DataFrame(
        results, columns=["Standard Deviations", "Expected Return", "Theta"]
    )
    SharpeFrontier.to_csv("SharpeFrontier.csv")


if __name__ == "__main__":
    main()
