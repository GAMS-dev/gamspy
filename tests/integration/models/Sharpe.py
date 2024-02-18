"""
## GAMSSOURCE: https://www.gams.com/latest/finlib_ml/libhtml/finlib_Sharpe.html
## LICENSETYPE: Demo
## MODELTYPE: NLP
## DATAFILES: Sharpe.gdx


Sharpe model

Sharpe.gms: Sharpe model.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 3.3
Last modified: Apr 2008.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Problem
from gamspy import Sense
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
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
        domain=ii,
        description="Holdings of assets",
    )
    PortVariance = Variable(
        m, name="PortVariance", description="Portfolio variance"
    )
    d_bar = Variable(
        m, name="d_bar", description="Portfolio expected excess return"
    )

    # EQUATIONS #
    ReturnDef = Equation(
        m,
        name="ReturnDef",
        description="Equation defining the portfolio excess return",
    )
    VarDef = Equation(
        m,
        name="VarDef",
        description="Equation defining the portfolio excess variance",
    )
    NormalCon = Equation(
        m,
        name="NormalCon",
        description="Equation defining the normalization contraint",
    )

    ReturnDef[...] = d_bar == Sum(ii, ExExpectedReturns[ii] * x[ii])

    VarDef[...] = PortVariance == Sum([ii, j], x[ii] * ExVarCov[ii, j] * x[j])

    NormalCon[...] = Sum(ii, x[ii]) == 1

    # Objective Function
    ObjDef = d_bar / gams_math.sqrt(PortVariance)

    # Put strictly positive bound on Variance to keep the model out of trouble:
    PortVariance.lo[...] = 0.001

    Sharpe = Model(
        m,
        name="Sharpe",
        equations=[ReturnDef, VarDef, NormalCon],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=ObjDef,
    )
    Sharpe.solve()

    print("Objective Function Variable: ", round(Sharpe.objective_value, 3))

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
    results.append([
        np.sqrt(PortVariance.records.level[0]),
        RiskFreeRate.records.value[0] + theta * d_bar.records.level[0],
        theta,
    ])
    SharpeFrontier = pd.DataFrame(
        results, columns=["Standard Deviations", "Expected Return", "Theta"]
    )
    SharpeFrontier.to_csv("SharpeFrontier.csv")


if __name__ == "__main__":
    main()
