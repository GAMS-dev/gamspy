"""
Heat Integrated Distillation Sequences - MINLP (MINLPHIX)

This is a direct MINLP formulation of the model MINLPHI.


Morari, M, and Grossmann, I E, Eds, Chemical Engineering
Optimization Models with GAMS. Computer Aids for Chemical
Engineering Corporation, 1991.

Floudas, C A, and Paules IV, G E, A Mixed-Integer Nonlinear
Programming Formulation for the Synthesis of Heat Integrated
Distillation Sequence. Computers and Chemical Engineering 12,
6 (1988), 531-546.


This formulation provides the Optimal Heat Integrated
Distillation Sequence with Pressure as a continuous variable
for a three component separation.

           Components:     a == Hexane
                           b == Benzene
                           c == Heptane

total feed to superstructure == 396 kgmol/hr

multicomponent feed composition: a = 0.80
                                 b = 0.10
                                 c = 0.10


A Superstructure of the form ...

                        _______               _______
                       _|_    |              _|_    |
                      /   \  ( )            /   \  ( )
                      |   |___|__ A         |   |___|___ B
                      |   |                 |   |
            |---------| 1 |                 | 3 |
            |         |   |       ----------|   |
            |         |   |       |         |   |
            |         |   |_______|         |   |
            |         \___/  |  BC          \___/_______ C
     F      |           |   ( )               |     |
   -------->|           |____|                |----( )
   (ABC)    |
            |           _______               _______
            |          _|_    |              _|_    |
            |         /   \  ( )            /   \  ( )
            |         |   |___| AB          |   |___|___ A
            |         |   |   |_____________|   |
            |---------| 2 |                 | 4 |
                      |   |                 |   |
                      |   |                 |   |
                      |   |______ C         |   |_______ B
                      \___/  |              \___/   |
                        |   ( )               |    ( )
                        |____|                |_____|


is used with binary variables representing:
   a_  the existence of columns in the sequence.
   b_  the selection of heat exchangers for heat integration.
   c_  the selection of hot and cold utilities.


Associated Reference:

"A Mixed-Integer Nonlinear Programming formulation for the
 synthesis of Heat-Integrated Distillation Sequences"

 C.A. Floudas and G.E. Paules IV,  1988.
 Computers and Chemical Engineering vol. 12 no. 6 pp. 531-546

Keywords: mixed integer nonlinear programming, chemical engineering,
distillation
          sequences, heat integrated distillation
"""

from gamspy import (
    Set,
    Parameter,
    Variable,
    Equation,
    Container,
    Alias,
    Model,
    Sum,
    Ord,
    Domain,
    Sense,
)
from gamspy.math import sqrt
import pandas as pd
import numpy as np


def main():
    cont = Container()

    # Set
    i = Set(cont, name="i", records=[f"c-{i}" for i in range(1, 5)])
    j = Set(cont, name="j", records=[f"r-{i}" for i in range(1, 5)])
    hu = Set(cont, name="hu", records=["lp", "ex"])
    cu = Set(cont, name="cu", records=["cw"])
    n = Set(cont, name="n", records=["a", "b"])
    m = Set(cont, name="m", records=["ab", "bc"])
    pm = Set(
        cont, name="pm", domain=[i, m], records=[("c-1", "bc"), ("c-2", "ab")]
    )
    fm = Set(
        cont, name="fm", domain=[i, m], records=[("c-3", "bc"), ("c-4", "ab")]
    )

    ip = Alias(cont, name="ip", alias_with=i)
    jp = Alias(cont, name="jp", alias_with=j)

    # ====================================================================
    # Definition of "z" sets for conditional control of model
    # used to map permissible matches between condensers and reboilers
    # and the position of columns in the superstructure
    # =====================================================================

    # Set
    zlead = Set(cont, name="zlead", domain=[i], records=["c-1", "c-2"])
    zcrhx = Set(
        cont,
        name="zcrhx",
        domain=[i, j],
        records=[
            ("c-1", "r-3"),
            ("c-2", "r-4"),
            ("c-3", "r-1"),
            ("c-4", "r-2"),
        ],
    )
    zlim = Set(cont, name="zlim", domain=[i, j])
    zcr = Set(cont, name="zcr", domain=[i, j])

    zlim[i, j] = zcrhx[i, j] & (Ord(i) < Ord(j))
    zcr[i, j] = Ord(i) == Ord(j)

    # Parameter
    spltfrc = Parameter(
        cont,
        name="spltfrc",
        domain=[i, m],
        records=pd.DataFrame([["c-1", "bc", 0.20], ["c-2", "ab", 0.90]]),
    )

    tcmin = Parameter(
        cont,
        name="tcmin",
        domain=[i],
        records=np.array([341.92, 343.01, 353.54, 341.92]),
    )
    trmax = Parameter(cont, name="trmax", domain=[j])
    trmax[j] = 1000

    # ====================================================================
    # scaled cost coefficients for distillation column fits
    # nonlinear fixed-charge cost model
    #   cost = fc*y + vc*flow*temp
    # scaling factor = 1000
    # ====================================================================

    # Parameter
    fc = Parameter(
        cont,
        name="fc",
        domain=[i],
        records=np.array([151.125, 180.003, 4.2286, 213.42]),
    )
    vc = Parameter(
        cont,
        name="vc",
        domain=[i],
        records=np.array([0.003375, 0.000893, 0.004458, 0.003176]),
    )
    thu = Parameter(
        cont, name="thu", domain=[hu], records=np.array([421.0, 373.0])
    )

    # hot utility cost coeff - gives cost in thousands of dollars per year
    # ucost = q(10e+6 kj/hr)*costhu(hu)

    costhu = Parameter(
        cont, name="costhu", domain=[hu], records=np.array([24.908, 9.139])
    )

    kf = Parameter(
        cont,
        name="kf",
        domain=[i, n],
        records=np.array(
            [
                [32.4, 0.0225],
                [25.0, 0.0130],
                [3.76, 0.0043],
                [35.1, 0.0156],
            ]
        ),
    )

    af = Parameter(
        cont,
        name="af",
        domain=[i, n],
        records=np.array(
            [
                [9.541, 1.028],
                [12.24, 1.050],
                [8.756, 1.029],
                [9.181, 1.005],
            ]
        ),
    )

    # Scalar
    totflow = Parameter(cont, name="totflow", records=396)
    fchx = Parameter(cont, name="fchx", records=3.392)
    vchx = Parameter(cont, name="vchx", records=0.0893)
    htc = Parameter(cont, name="htc", records=0.0028)
    dtmin = Parameter(cont, name="dtmin", records=10.0)
    tcin = Parameter(cont, name="tcin", records=305.0)
    tcout = Parameter(cont, name="tcout", records=325.0)
    costcw = Parameter(cont, name="costcw", records=4.65)
    beta = Parameter(cont, name="beta", records=0.52)
    alpha = Parameter(cont, name="alpha", records=0.40)
    u = Parameter(cont, name="u", records=1500)
    uint = Parameter(cont, name="uint", records=20)

    # Variable
    zoau = Variable(cont, name="zoau", type="free")

    # Positive Variables
    f = Variable(cont, name="f", type="positive", domain=[i])
    qr = Variable(cont, name="qr", type="positive", domain=[j])
    qc = Variable(cont, name="qc", type="positive", domain=[i])
    qcr = Variable(cont, name="qcr", type="positive", domain=[i, j])
    qhu = Variable(cont, name="qhu", type="positive", domain=[hu, j])
    qcu = Variable(cont, name="qcu", type="positive", domain=[i, cu])
    tc = Variable(cont, name="tc", type="positive", domain=[i])
    tr = Variable(cont, name="tr", type="positive", domain=[j])
    lmtd = Variable(cont, name="lmtd", type="positive", domain=[i])
    sl1 = Variable(cont, name="sl1", type="positive", domain=[i])
    sl2 = Variable(cont, name="sl2", type="positive", domain=[i])
    s1 = Variable(cont, name="s1", type="positive", domain=[i])
    s2 = Variable(cont, name="s2", type="positive", domain=[i])
    s3 = Variable(cont, name="s3", type="positive", domain=[i])
    s4 = Variable(cont, name="s4", type="positive", domain=[i])

    # Binary Variable
    yhx = Variable(cont, name="yhx", type="binary", domain=[i, j])
    yhu = Variable(cont, name="yhu", type="binary", domain=[hu, j])
    ycu = Variable(cont, name="ycu", type="binary", domain=[i, cu])
    ycol = Variable(cont, name="ycol", type="binary", domain=[i])

    # Equation
    nlpobj = Equation(cont, name="nlpobj")
    tctrlo = Equation(cont, name="tctrlo", domain=[i, j])
    lmtdlo = Equation(cont, name="lmtdlo", domain=[i])
    lmtdsn = Equation(cont, name="lmtdsn", domain=[i])
    tempset = Equation(cont, name="tempset", domain=[i])
    artrex1 = Equation(cont, name="artrex1", domain=[i])
    artrex2 = Equation(cont, name="artrex2", domain=[i])
    material = Equation(cont, name="material", domain=[m])
    feed = Equation(cont, name="feed")
    matlog = Equation(cont, name="matlog", domain=[i])
    duty = Equation(cont, name="duty", domain=[i])
    rebcon = Equation(cont, name="rebcon", domain=[i, j])
    conheat = Equation(cont, name="conheat", domain=[i])
    rebheat = Equation(cont, name="rebheat", domain=[j])
    dtminlp = Equation(cont, name="dtminlp", domain=[j])
    dtminc = Equation(cont, name="dtminc", domain=[i])
    trtcdef = Equation(cont, name="trtcdef", domain=[i, j])
    dtmincr = Equation(cont, name="dtmincr", domain=[i, j])
    dtminex = Equation(cont, name="dtminex", domain=[j])
    hxclog = Equation(cont, name="hxclog", domain=[i, j])
    hxhulog = Equation(cont, name="hxhulog", domain=[hu, j])
    hxculog = Equation(cont, name="hxculog", domain=[i, cu])
    qcqrlog = Equation(cont, name="qcqrlog", domain=[i])

    # these are the pure binary constraints of the minlp
    sequen = Equation(cont, name="sequen", domain=[m])
    lead = Equation(cont, name="lead")
    limutil = Equation(cont, name="limutil", domain=[j])
    hidirect = Equation(cont, name="hidirect", domain=[i, j])
    heat = Equation(cont, name="heat", domain=[i])

    nlpobj.expr = zoau == (
        alpha
        * (
            Sum(i, fc[i] * ycol[i] + vc[i] * (tc[i] - tcmin[i]) * f[i])
            + Sum(
                zcrhx[i, j],
                fchx * yhx[i, j]
                + (vchx / htc) * (qcr[i, j] / (tc[i] - tr[j] + 1 - ycol[i])),
            )
            + Sum(
                [i, cu],
                fchx * ycu[i, cu]
                + (vchx / htc) * (qcu[i, cu] / (lmtd[i] + 1 - ycol[i])),
            )
            + Sum(
                [hu, j],
                fchx * yhu[hu, j]
                + (vchx / htc) * (qhu[hu, j] / (thu[hu] - tr[j])),
            )
        )
        + beta
        * (
            Sum([i, cu], costcw * qcu[i, cu])
            + Sum([hu, j], costhu[hu] * qhu[hu, j])
        )
    )

    # limit the denominator in the second line of the objective away from zero
    tctrlo[zcrhx[i, j]] = tc[i] - tr[j] + 1 - ycol[i] >= 1

    # lmtd and ycol from being 0 and 1 at the same time to prevent divding
    # by 0 in the objective
    lmtdlo[i] = lmtd[i] >= 2 * ycol[i]

    lmtdsn[i] = lmtd[i] == (
        (2 / 3) * sqrt((tc[i] - tcin) * (tc[i] - tcout))
        + (1 / 6) * ((tc[i] - tcin) + (tc[i] - tcout))
        + sl1[i]
        - sl2[i]
    )

    artrex1[i] = s1[i] + s2[i] + sl1[i] <= u * (1 - ycol[i])

    artrex2[i] = s3[i] + s4[i] + sl2[i] <= u * (1 - ycol[i])

    material[m] = Sum(pm[i, m], spltfrc[i, m] * f[i]) == Sum(fm[i, m], f[i])

    feed.expr = Sum(zlead[i], f[i]) == totflow

    duty[i] = (
        qc[i] == (kf[i, "a"] + kf[i, "b"] * (tc[i] - tcmin[i])) + s3[i] - s4[i]
    )

    rebcon[zcr[i, j]] = qr[j] == qc[i]

    conheat[i] = qc[i] == Sum(zcrhx[i, j], qcr[i, j]) + Sum(cu, qcu[i, cu])

    rebheat[j] = qr[j] == Sum(zcrhx[i, j], qcr[i, j]) + Sum(hu, qhu[hu, j])

    trtcdef[zcr[i, j]] = (
        tr[j] == (af[i, "a"] + af[i, "b"] * (tc[i] - tcmin[i])) + s1[i] - s2[i]
    )

    dtminlp[j] = dtmin - (thu["lp"] - tr[j]) <= 0

    dtminex[j] = dtmin - (thu["ex"] - tr[j]) - u * (1 - yhu["ex", j]) <= 0

    tempset[i] = tc[i] + lmtd[i] + Sum(zcr[i, j], tr[j]) <= u * ycol[i]

    matlog[i] = f[i] <= u * ycol[i]

    dtminc[i] = tcmin[i] - tc[i] <= u * (1 - ycol[i])

    dtmincr[zcrhx[i, j]] = tr[j] - tc[i] - u * (1 - yhx[i, j]) + dtmin <= 0

    hxclog[zcrhx[i, j]] = qcr[i, j] <= u * yhx[i, j]

    hxhulog[hu, j] = qhu[hu, j] <= u * yhu[hu, j]

    hxculog[i, cu] = qcu[i, cu] <= u * ycu[i, cu]

    qcqrlog[i] = qc[i] + Sum(j.where[zcr[i, j]], qr[j]) <= u * ycol[i]

    sequen[m] = Sum(pm[i, m], ycol[i]) == Sum(fm[i, m], ycol[i])

    lead.expr = Sum(zlead[i], ycol[i]) == 1

    limutil[j] = Sum(hu, yhu[hu, j]) <= 1

    # only one of the mutual heat integration binaries can be 1
    hidirect[zlim[i, j]] = (
        yhx[i, j]
        + Sum(
            Domain(ip, jp).where[(Ord(ip) == Ord(j)) & (Ord(jp) == Ord(i))],
            yhx[ip, jp],
        )
        <= 1
    )

    # if a column doesn't exist then all binary variables associated
    # with it must also be set to zero
    heat[i] = (
        Sum(
            zcrhx[i, j],
            yhx[i, j]
            + Sum(
                Domain(ip, jp).where[
                    (Ord(ip) == Ord(j)) & (Ord(jp) == Ord(i))
                ],
                yhx[ip, jp],
            ),
        )
        + Sum((hu, zcr[i, j]), yhu[hu, j])
        + Sum(cu, ycu[i, cu])
    ) <= uint * ycol[i]

    tc.lo["c-1"] = tcout + 1
    tc.up["c-2"] = tcin - 1
    tc.lo["c-3"] = tcout + 1
    tc.up["c-4"] = tcin - 1
    tr.up[j] = trmax[j]

    skip = Model(
        cont,
        name="skip",
        equations=cont.getEquations(),
        problem="minlp",
        sense=Sense.MIN,
        objective=zoau,
    )
    cont.addOptions({"domLim": 100})

    skip.solve()

    print("Best integer solution found:", round(zoau.records.level[0], 3))


if __name__ == "__main__":
    main()
