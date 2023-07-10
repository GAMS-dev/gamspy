from pathlib import Path
from gamspy import Sum
import gamspy.math as gams_math
from gamspy import Model, Container


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

    TRANSCOST.definition = TC == Sum((r, rr, c), X[r, rr, c] * TCost[r, rr, c])

    OBJECT.definition = OBJ == Sum([r, c], DINT[r, c] - SINT[r, c]) - TC

    PDIF[r, rr, c] = P[r, c] - P[rr, c] <= TCost[r, rr, c]

    SX[r, c] = Sum(rr, X[r, rr, c]) == Qs[r, c]

    DX[r, c] = Sum(rr, X[rr, r, c]) == Qd[r, c]

    IN_OUT[r, c] = Qs[r, c] + Sum(rr, X[rr, r, c] - X[r, rr, c]) == Qd[r, c]

    DOM_TRAD[r, rr, c] = P[r, c] + TCost[r, rr, c] >= P[rr, c]

    P2R3_Linear = Model(
        m,
        name="P2R3_Linear",
        equations=[DEM, SUP, SDBAL, PDIF, TRANSCOST, SX, DX],
    )
    P2R3_LinearLog = Model(
        m,
        name="P2R3_LinearLog",
        equations=[DEMLOG, SUPLOG, SDBAL, PDIF, TRANSCOST, SX, DX],
    )
    P2R3_NonLinear = Model(
        m,
        name="P2R3_NonLinear",
        equations="P2R3_Linear, DEMINT, SUPINT, OBJECT",
    )
    P2R3_MCP = Model(
        m, name="P2R3_MCP", equations="DEM, SUP, IN_OUT.P, DOM_TRAD.X"
    )

    m.solve(P2R3_Linear, problem="LP", sense="min", objective_variable=TC)
    m.solve(P2R3_LinearLog, problem="NLP", sense="min", objective_variable=TC)
    m.solve(P2R3_NonLinear, problem="NLP", sense="max", objective_variable=OBJ)

    X.fx[r, r, c] = 0

    m.solve(P2R3_MCP, problem="MCP")


if __name__ == "__main__":
    main()
