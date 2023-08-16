"""
Stochastic Dedication model with borrowing and lending variables

* StochDedicationBL.gms: Stochastic Dedication model with borrowing
* and lending variables.
* Consiglio, Nielsen and Zenios.
* PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 6.2.2
* Last modified: Apr 2008.
"""

from pathlib import Path
from gamspy import (
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Sense,
)


def main():
    # Define container
    m = Container(
        load_from=str(Path(__file__).parent.absolute())
        + "/StochDedicationBL.gdx"
    )

    # Aliases
    l, t = m.getSymbols(["l", "t"])

    # Scalars
    Horizon = m.getSymbols(["Horizon"])[0]

    # Parameters
    tau = m.getSymbols(["tau"])[0]

    # Aliases
    i = m.getSymbols(["i"])[0]

    # Scalars
    spread = m.getSymbols(["spread"])[0]

    # Parameters
    Price = m.getSymbols(["Price"])[0]

    # Parameters
    Srf, SF, SLiability = m.getSymbols(
        [
            "Srf",
            "SF",
            "SLiability",
        ]
    )

    # Variables
    x = Variable(m, "x", domain=[i], type="Positive")
    surplus = Variable(m, "surplus", domain=[t, l], type="Positive")
    borrow = Variable(m, "borrow", domain=[t, l], type="Positive")
    v0 = Variable(m, "v0")

    # Equations
    CashFlowCon = Equation(m, "CashFlowCon", domain=[t, l])

    CashFlowCon[t, l] = (
        Sum(i, SF[t, i, l] * x[i]).where[tau[t] > 0]
        + (v0 - Sum(i, Price[i] * x[i])).where[tau[t] == 0]
        + ((1 + Srf[t.lag(1), l]) * surplus[t.lag(1), l]).where[tau[t] > 0]
        + borrow[t, l].where[tau[t] < Horizon]
        == surplus[t, l]
        + SLiability[t, l].where[tau[t] > 0]
        + ((1 + Srf[t.lag(1), l] + spread) * borrow[t.lag(1), l]).where[
            tau[t] > 0
        ]
    )

    StochDedicationBL = Model(
        m,
        name="StochDedicationBL",
        equations=[CashFlowCon],
        problem="LP",
        sense=Sense.MIN,
        objective=v0,
    )

    StochDedicationBL.solve()
    print("Objective Function Value: ", round(v0.records.level[0], 3))

    print(x.description, ": \n", x.records.loc[:, ["i", "level"]])


if __name__ == "__main__":
    main()
