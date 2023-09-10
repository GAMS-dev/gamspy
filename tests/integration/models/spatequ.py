"""
Spatial Equilibrium (SPATEQU)

This program is written for the spatial equilibrium model with linear supply
and demand having two products and three regions.

The model contains multiple approaches (LP, NLP, and MCP) for solving this
problem.


Phan, S H, Policy option to promote the wood-processing industry in
northern Vietnam, forth coming. PhD thesis,
The University of Queensland, Australia, 2011.

Phan, S H, and Harrison, S, A Review of the Formulation and
Application of the Spatial Equilibrium Models to Analyze
Policy. Journal of Forestry Research 22, 4 (2011).

The numerical example has been taken from:
Takayama, T, and Judge, G G, Spatial Equilibrium and Quadratic
Programming. Journal of Farm Economics 46, 1 (1964), 67-93

Contributed by: Phan Sy Hieu, November 2010

Keywords: linear programming, nonlinear programming, mixed complementarity
          problem, spatial equilibrium model
"""

from pathlib import Path
from gamspy import Sum
import gamspy.math as gams_math
from gamspy import Model, Container
from gamspy import Problem, Sense


def main():
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/spatequ.gdx"
    )

    # Sets
    c, r, rr, cc = m.getSymbols(["c", "r", "rr", "cc"])

    # Parameters
    (
        AlphaD,
        BetaD,
        BetadSq,
        AlphaS,
        BetaS,
        BetasSq,
        TCost,
    ) = m.getSymbols(
        [
            "AlphaD",
            "BetaD",
            "BetadSq",
            "AlphaS",
            "BetaS",
            "BetasSq",
            "TCost",
        ]
    )

    # Variables
    DINT, SINT, TC, Qd, Qs, X, P, OBJ = m.getSymbols(
        ["DINT", "SINT", "TC", "Qd", "Qs", "X", "P", "OBJ"]
    )

    # Equations
    (
        DEM,
        DEMLOG,
        DEMINT,
        SUP,
        SUPLOG,
        SUPINT,
        SDBAL,
        PDIF,
        TRANSCOST,
        SX,
        DX,
        OBJECT,
        IN_OUT,
        DOM_TRAD,
    ) = m.getSymbols(
        [
            "DEM",
            "DEMLOG",
            "DEMINT",
            "SUP",
            "SUPLOG",
            "SUPINT",
            "SDBAL",
            "PDIF",
            "TRANSCOST",
            "SX",
            "DX",
            "OBJECT",
            "IN_OUT",
            "DOM_TRAD",
        ]
    )

    DEM[r, c] = AlphaD[r, c] + Sum(cc, (BetaD[r, c, cc] * P[r, c])) == Qd[r, c]

    DEMLOG[r, c] = (
        AlphaD[r, c] + Sum(cc, (BetaD[r, c, cc] * gams_math.log(P[r, c])))
        == Qd[r, c]
    )

    DEMINT[r, c] = (
        DINT[r, c]
        == AlphaD[r, c] * P[r, c]
        + Sum(cc, BetadSq[r, c, cc] * P[r, cc]) * P[r, c]
    )

    SUP[r, c] = AlphaS[r, c] + Sum(cc, (BetaS[r, c, cc] * P[r, c])) == Qs[r, c]
    SUPLOG[r, c] = (
        AlphaS[r, c] + Sum(cc, (BetaS[r, c, cc] * gams_math.log(P[r, c])))
        == Qs[r, c]
    )

    SUPINT[r, c] = (
        SINT[r, c]
        == AlphaS[r, c] * P[r, c]
        + Sum(cc, BetasSq[r, c, cc] * P[r, cc]) * P[r, c]
    )

    SDBAL[c] = Sum(r, Qd[r, c]) == Sum(r, Qs[r, c])

    TRANSCOST.expr = TC == Sum((r, rr, c), X[r, rr, c] * TCost[r, rr, c])

    OBJECT.expr = OBJ == Sum([r, c], DINT[r, c] - SINT[r, c]) - TC

    PDIF[r, rr, c] = P[r, c] - P[rr, c] <= TCost[r, rr, c]

    SX[r, c] = Sum(rr, X[r, rr, c]) == Qs[r, c]

    DX[r, c] = Sum(rr, X[rr, r, c]) == Qd[r, c]

    IN_OUT[r, c] = Qs[r, c] + Sum(rr, X[rr, r, c] - X[r, rr, c]) == Qd[r, c]

    DOM_TRAD[r, rr, c] = P[r, c] + TCost[r, rr, c] >= P[rr, c]

    P2R3_Linear = Model(
        m,
        name="P2R3_Linear",
        equations=[DEM, SUP, SDBAL, PDIF, TRANSCOST, SX, DX],
        problem="LP",
        sense=Sense.MIN,
        objective=TC,
    )
    P2R3_LinearLog = Model(
        m,
        name="P2R3_LinearLog",
        equations=[DEMLOG, SUPLOG, SDBAL, PDIF, TRANSCOST, SX, DX],
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=TC,
    )
    P2R3_NonLinear = Model(
        m,
        name="P2R3_NonLinear",
        equations=P2R3_Linear.equations + [DEMINT, SUPINT, OBJECT],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=OBJ,
    )
    P2R3_MCP = Model(
        m,
        name="P2R3_MCP",
        equations=[DEM, SUP, IN_OUT, DOM_TRAD],
        matches={IN_OUT: P, DOM_TRAD: X},
        problem="MCP",
    )

    P2R3_Linear.solve()
    P2R3_LinearLog.solve()
    P2R3_NonLinear.solve()

    X.fx[r, r, c] = 0

    P2R3_MCP.solve()


if __name__ == "__main__":
    main()
