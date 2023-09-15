"""
Indexation model using the structural approach

StructuralModel.gms: Indexation model using the structural approach
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 7.2.1
Last modified: Apr 2008.
"""
from pathlib import Path

import numpy as np

from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    gdx_file = str(Path(__file__).parent.absolute()) + "/InputData.gdx"
    m = Container(load_from=gdx_file)

    # SETS #
    BB, EE, BxE = m.getSymbols(["BB", "EE", "BxE"])

    # PARAMETERS #
    data = m.getSymbols("data")[0]

    # SETS
    DD = Set(
        m,
        name="DD",
        records=["LOW", "MEDIUM", "HIGH"],
        description="Duration levels",
    )
    BxD = Set(
        m, name="BxD", domain=[BB, DD], description="Bonds by Duration levels"
    )

    i = Alias(m, name="i", alias_with=BB)
    e = Alias(m, name="e", alias_with=EE)
    k = Alias(m, name="k", alias_with=DD)

    # Partition the set of bonds by duration levels

    BxD[i, "LOW"] = data[i, "Duration"] <= 3

    BxD[i, "MEDIUM"] = (data[i, "Duration"] > 3) & (data[i, "Duration"] <= 7)

    BxD[i, "HIGH"] = data[i, "Duration"] > 7

    # The index dollar duration

    # SCALARS #
    IndexDollarDuration = Parameter(m, name="IndexDollarDuration", records=820)

    # PARAMETERS #
    DurationWeights = Parameter(
        m,
        name="DurationWeights",
        domain=[k],
        records=np.array([0.3, 0.2, 0.5]),
    )
    CurrencyWeights = Parameter(
        m,
        name="CurrencyWeights",
        domain=[e],
        records=np.array([0.6, 0.3, 0.1]),
    )

    # VARIABLE #
    z = Variable(m, name="z", type="free")
    x = Variable(m, name="x", type="positive", domain=[i])

    # EQUATIONS #
    ObjDef = Equation(
        m,
        name="ObjDef",
        type="regular",
        description="Linear approximation of the portfolio yield",
    )
    ObjDefTwo = Equation(
        m,
        name="ObjDefTwo",
        type="regular",
        description="NonLinear approximation of the portfolio yield",
    )
    NormalCon = Equation(
        m,
        name="NormalCon",
        type="regular",
        description="Equation defining the budget contraint",
    )
    DurationMatch = Equation(
        m,
        name="DurationMatch",
        type="regular",
        description=(
            "Equation matching the dollar duration of the portfolio and of the"
            " index"
        ),
    )
    CurCons = Equation(
        m,
        name="CurCons",
        type="regular",
        domain=[e],
        description="Equation matching the index currency allocation",
    )
    DurCons = Equation(
        m,
        name="DurCons",
        type="regular",
        domain=[k],
        description="Equation matching the index duration allocation",
    )

    ObjDef.expr = (
        z
        == Number(-1)
        * Sum(
            i, data[i, "Duration"] * data[i, "Price"] * data[i, "Yield"] * x[i]
        )
        / IndexDollarDuration
    )

    DurationMatch.expr = (
        Sum(i, data[i, "Duration"] * data[i, "Price"] * x[i])
        == IndexDollarDuration
    )

    CurCons[e] = Sum(i.where[BxE[i, e]], x[i]) == CurrencyWeights[e]

    DurCons[k] = Sum(i.where[BxD[i, k]], x[i]) == DurationWeights[k]

    NormalCon.expr = Sum(i, x[i]) == 1.0

    IndexFund = Model(
        m,
        name="IndexFund",
        equations=[ObjDef, DurationMatch, CurCons, DurCons, NormalCon],
        problem="LP",
        sense="MAX",
        objective=z,
    )

    IndexFund.solve()

    x_l = {i: round(j, 3) for i, j in x.toDict().items() if j != 0}
    print("\nSolving 'IndexFund' Model:")
    print("x: ", x_l)
    print("z: ", round(z.toList()[0], 3))

    # If we let the duration of the portfolio unconstrained, the objective
    # function turns to nonlinear as the variable x[i] will appear in the denominator

    ObjDefTwo.expr = z == Number(-1) * Sum(
        i, data[i, "Duration"] * data[i, "Price"] * data[i, "Yield"] * x[i]
    ) / Sum(i, data[i, "Duration"] * data[i, "Price"] * x[i])

    NonLinearIndexFund = Model(
        m,
        name="NonLinearIndexFund",
        equations=[ObjDefTwo, CurCons, DurCons, NormalCon],
        problem="NLP",
        sense="MAX",
        objective=z,
    )

    NonLinearIndexFund.solve()

    x_l = {i: round(j, 3) for i, j in x.toDict().items() if j != 0}
    print(f"\n{'-'*25}\n")
    print("Solving 'NonLinearIndexFund' Model:")
    print("x: ", x_l)
    print("z: ", round(z.toList()[0], 3), "\n")


if __name__ == "__main__":
    main()
