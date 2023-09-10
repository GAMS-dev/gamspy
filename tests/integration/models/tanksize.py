"""
Tank Size Design Problem - (TANKSIZE)

We discuss a tank design problem for a multi product plant, in which the
optimal cycle time and the optimal campaign size are unknown. A mixed in-
teger nonlinear programming formulation is presented, where non-convexities
are due to the tank investment cost, storage cost, campaign setup cost and
variable production rates. The objective of the optimization model is to
minimize the sum of the production cost per ton per product produced. A
continuous-time mathematical programming formulation for the problem is
implemented with a fixed number of event points.


Rebennack, S, Kallrath, J, and Pardalos, P M, Optimal Storage Design
for a Multi-Product Plant: A Non-Convex MINLP Formulation. Tech. rep.,
University of Florida, 2009. Submitted to Computers and Chemical
Engineering

Keywords: mixed integer nonlinear programming, storage design, global
optimization
          continuous-time model, chemical engineering
"""

from pathlib import Path
from gamspy import Sum
import gamspy.math as gams_math
from gamspy import Model, Container, Sense


def main():
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/tanksize.gdx"
    )

    # Sets
    p, n, pp = m.getSymbols(["p", "n", "pp"])

    # Parameters
    (
        PRMIN,
        PRMAX,
        SLB,
        SUB,
        SI,
        DLB,
        DUB,
        DEMAND,
        TS,
        CSTI,
        CSTC,
        B,
        pdata,
        DPD,
        L,
        CAL,
        PRL,
        CSTCMin,
        CSTCMax,
    ) = m.getSymbols(
        [
            "PRMIN",
            "PRMAX",
            "SLB",
            "SUB",
            "SI",
            "DLB",
            "DUB",
            "DEMAND",
            "TS",
            "CSTI",
            "CSTC",
            "B",
            "pdata",
            "DPD",
            "L",
            "CAL",
            "PRL",
            "CSTCMin",
            "CSTCMax",
        ]
    )

    # Variables
    d, pC, s, sM, sH, cI, cC, cS, T, omega, cPT = m.getSymbols(
        ["d", "pC", "s", "sM", "sH", "cI", "cC", "cS", "T", "omega", "cPT"]
    )

    # Equations
    (
        TIMECAP,
        UNIQUE,
        MATBAL,
        TANKCAP,
        PPN1,
        PPN2,
        SCCam1,
        SCCam2,
        DEFcC,
        DEFcI,
        DEFcS,
        DefsH,
        DEFcPT,
        NONIDLE,
        SEQUENCE,
        SYMMETRY,
    ) = m.getSymbols(
        [
            "TIMECAP",
            "UNIQUE",
            "MATBAL",
            "TANKCAP",
            "PPN1",
            "PPN2",
            "SCCam1",
            "SCCam2",
            "DEFcC",
            "DEFcI",
            "DEFcS",
            "DefsH",
            "DEFcPT",
            "NONIDLE",
            "SEQUENCE",
            "SYMMETRY",
        ]
    )

    TIMECAP.expr = Sum(n, d[n] + Sum(p, TS[p] * omega[p, n])) == T
    UNIQUE[n] = Sum(p, omega[p, n]) <= 1
    NONIDLE[n] = Sum(p, DUB[p] * omega[p, n]) >= d[n]
    MATBAL[p, n] = s[p, n.lead(1)] == s[p, n] + pC[p, n] - DPD[p] * (
        d[n] + Sum(pp, TS[pp] * omega[pp, n])
    )
    TANKCAP[p, n] = s[p, n] <= sM[p]
    PPN1[p, n] = pC[p, n] <= PRMAX[p] * d[n] * omega[p, n]
    PPN2[p, n] = pC[p, n] >= PRMIN[p] * d[n] * omega[p, n]
    SCCam2[n] = d[n] >= Sum(p, DLB[p] * omega[p, n])
    SCCam1[n] = d[n] <= Sum(p, DUB[p] * omega[p, n])
    DEFcPT.expr = (cPT * L - cI) * T == cC + cS
    DEFcC.expr = cC == Sum([p, n], CSTC[p] * omega[p, n])
    DEFcI.expr = cI == B * Sum(p, gams_math.sqrt(sM[p]))
    DEFcS.expr = cS == Sum(
        [p, n], CSTI[p] * sH[p, n] * (d[n] + Sum(pp, TS[pp] * omega[pp, n]))
    )
    DefsH[p, n] = sH[p, n] == 0.5 * (s[p, n.lead(1)] + s[p, n]) - SLB[p]
    SEQUENCE[p, n] = 1 - omega[p, n] >= omega[p, n.lead(1, "linear")]
    SYMMETRY[n] = Sum(p, omega[p, n]) >= Sum(p, omega[p, n.lead(1, "linear")])

    s.lo[p, n] = SLB[p]
    s.up[p, n] = SUB[p]
    s.fx["P1", "N1"] = SLB["P1"]
    omega.fx[p, "N1"] = 0
    omega.fx["P1", "N1"] = 1
    omega.fx["P1", "N2"] = 0
    sM.lo[p] = SLB[p]
    sM.up[p] = SUB[p]

    Sequenz = Model(
        m,
        name="Sequenz",
        equations=m.getEquations(),
        problem="MINLP",
        sense=Sense.MIN,
        objective=cPT,
    )
    omega.l[p, n] = gams_math.uniform(0, 1)

    Sequenz.solve()


if __name__ == "__main__":
    main()
