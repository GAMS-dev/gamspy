"""
General Equilibrium Model for Korea (KORCNS)

This mini equilibrium model of Korea for the year 1963 is used to
illustrate the basic use of CGE models. This version follows closely
Chapter 11 of the reference.

The original model (KORCGE) is formulated as an optimization
model, but it is really a square system of nonlinear equations.
In this version, we formulate the model directly as a square system
using the model type CNS = Constrained Nonlinear System.

An MCP version exist under the name (KORMCP).


Lewis, J, and Robinson, S, Chapter 11. In Chenery, H B, Robinson, S,
and Syrquin, S, Eds, Industrialization and Growth: A Comparative
Study. Oxford University Press, London, 1986.

Keywords: constrained nonlinear system, general equilibrium model, economic
growth,
          industrialization, economic policy, Korean economy
"""

from gamspy import (
    Set,
    Alias,
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Number,
    Sum,
    Product,
)
import numpy as np
import pandas as pd


def main():
    # Data

    data = [
        [0.00000, 0.00000, 0.00000],
        [0.01000, 0.03920, 0.05000],
        [0.02000, 0.07000, 0.91000],
        [0.13000, 0.29000, 0.58000],
        [0.00000, 0.00000, 0.00000],
        [0.00000, 0.00000, 0.00000],
        [0.10000, 0.22751, 0.08084],
        [0.61447, 1.60111, 0.52019],
        [0.33263, 0.43486, 0.23251],
        [0.90909, 0.81466, 0.92521],
        [1.00000, 1.00000, 1.00000],
        [2.00000, 0.66000, 0.40000],
        [0.24820, 0.05111, 0.00001],
        [1.59539, 1.34652, 1.01839],
        [2.00000, 2.00000, 2.00000],
        [0.86628, 0.84602, 0.82436],
        [3.85424, 3.51886, 3.23592],
    ]

    columns = ["agricult", "industry", "services"]
    indexes = [
        "depr",
        "itax",
        "gles",
        "kio",
        "dstr",
        "te",
        "tm",
        "ad",
        "pwts",
        "pwm",
        "pwe",
        "sigc",
        "delta",
        "ac",
        "sigt",
        "gamma",
        "at",
    ]
    zz_df = pd.DataFrame(data=data, columns=columns, index=indexes)
    zz_df = zz_df.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )

    # sectres
    data = [
        [1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000],
        [0.7370, 0.2911, 0.6625],
        [711.6443, 930.3509, 497.4428],
        [657.3677, 840.0500, 515.4296],
        [641.7037, 812.2222, 492.0307],
        [15.6639, 27.8278, 23.3988],
        [69.9406, 118.1287, 5.4120],
        [657.5754, 338.7076, 1548.5192],
        [256.6450, 464.1656, 156.2598],
        [452.1765, 307.8561, 202.0416],
        [2.8230, 9.8806, 128.4482],
        [0.0000, 148.4488, 10.6931],
        [0.0000, 0.0000, 0.0000],
        [20.6884, 46.1511, 92.3023],
        [1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000],
    ]
    columns = ["agricult", "industry", "services"]
    indexes = [
        "pd1",
        "pk",
        "pva",
        "x",
        "xd",
        "xxd",
        "e",
        "m",
        "k",
        "intr",
        "cd",
        "gd",
        "id",
        "dst",
        "dk",
        "pm",
        "pe",
        "px",
        "p",
    ]
    sectres_df = pd.DataFrame(data=data, columns=columns, index=indexes)
    sectres_df = sectres_df.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )

    # Model's container
    cont = Container()

    # Sets
    i = Set(cont, name="i", records=["agricult", "industry", "services"])
    hh = Set(cont, name="hh", records=["lab_hh", "cap_hh"])
    lc = Set(cont, name="lc", records=["labor1", "labor2", "labor3"])
    it = Set(cont, name="it", domain=[i])
    inn = Set(cont, name="inn", domain=[i])

    j = Alias(cont, name="j", alias_with=i)

    # Parameters
    delta = Parameter(cont, name="delta", domain=[i])
    ac = Parameter(cont, name="ac", domain=[i])
    rhoc = Parameter(cont, name="rhoc", domain=[i])
    rhot = Parameter(cont, name="rhot", domain=[i])
    at = Parameter(cont, name="at", domain=[i])
    gamma = Parameter(cont, name="gamma", domain=[i])
    ad = Parameter(cont, name="ad", domain=[i])
    gles = Parameter(cont, name="gles", domain=[i])
    depr = Parameter(cont, name="depr", domain=[i])
    dstr = Parameter(cont, name="dstr", domain=[i])
    kio = Parameter(cont, name="kio", domain=[i])
    te = Parameter(cont, name="te", domain=[i])
    itax = Parameter(cont, name="itax", domain=[i])
    htax = Parameter(cont, name="htax", domain=[hh])
    pwm = Parameter(cont, name="pwm", domain=[i])
    pwe = Parameter(cont, name="pwe", domain=[i])
    tm = Parameter(cont, name="tm", domain=[i])
    pwts = Parameter(cont, name="pwts", domain=[i])

    htax["lab_hh"] = 0.08910
    htax["cap_hh"] = 0.08910

    alphl = Parameter(
        cont,
        name="alphl",
        domain=[i, lc],
        records=np.array(
            [
                [0.38258, 0.06740, 0.00000],
                [0.00000, 0.53476, 0.00000],
                [0.00000, 0.16234, 0.42326],
            ]
        ),
    )

    io = Parameter(
        cont,
        name="io",
        domain=[i, j],
        records=np.array(
            [
                [0.12591, 0.19834, 0.01407],
                [0.10353, 0.35524, 0.18954],
                [0.02358, 0.11608, 0.08390],
            ]
        ),
    )

    imat = Parameter(
        cont,
        name="imat",
        domain=[i, j],
        records=np.array(
            [
                [0.00000, 0.00000, 0.00000],
                [0.93076, 0.93774, 0.93080],
                [0.06924, 0.06226, 0.06920],
            ]
        ),
    )

    wdist = Parameter(
        cont,
        name="wdist",
        domain=[i, lc],
        records=np.array(
            [
                [1.00000, 0.52780, 0.00000],
                [0.00000, 1.21879, 0.00000],
                [0.00000, 1.11541, 1.00000],
            ]
        ),
    )

    cles = Parameter(
        cont,
        name="cles",
        domain=[i, hh],
        records=np.array(
            [[0.47000, 0.47000], [0.31999, 0.31999], [0.21001, 0.21001]]
        ),
    )

    zz = Parameter(cont, name="zz", domain=["*", i], records=zz_df)

    depr[i] = zz["depr", i]
    itax[i] = zz["itax", i]
    gles[i] = zz["gles", i]
    kio[i] = zz["kio", i]
    dstr[i] = zz["dstr", i]
    te[i] = zz["te", i]
    tm[i] = zz["tm", i]
    ad[i] = zz["ad", i]
    pwts[i] = zz["pwts", i]
    pwm[i] = zz["pwm", i]
    pwe[i] = zz["pwe", i]
    rhoc[i] = (1 / zz["sigc", i]) - 1
    delta[i] = zz["delta", i]
    ac[i] = zz["ac", i]
    rhot[i] = (1 / zz["sigt", i]) + 1
    gamma[i] = zz["gamma", i]
    at[i] = zz["at", i]

    # Model Definition
    # Variables
    # prices block
    er = Variable(cont, name="er", type="free")
    pd1 = Variable(cont, name="pd1", type="free", domain=[i])
    pm = Variable(cont, name="pm", type="free", domain=[i])
    pe = Variable(cont, name="pe", type="free", domain=[i])
    pk = Variable(cont, name="pk", type="free", domain=[i])
    px = Variable(cont, name="px", type="free", domain=[i])
    p = Variable(cont, name="p", type="free", domain=[i])
    pva = Variable(cont, name="pva", type="free", domain=[i])
    pr = Variable(cont, name="pr", type="free")
    pindex = Variable(cont, name="pindex", type="free")

    # production block
    x = Variable(cont, name="x", type="free", domain=[i])
    xd = Variable(cont, name="xd", type="free", domain=[i])
    xxd = Variable(cont, name="xxd", type="free", domain=[i])
    e = Variable(cont, name="e", type="free", domain=[i])
    m = Variable(cont, name="m", type="free", domain=[i])

    # factors block
    k = Variable(cont, name="k", type="free", domain=[i])
    wa = Variable(cont, name="wa", type="free", domain=[lc])
    ls = Variable(cont, name="ls", type="free", domain=[lc])
    l = Variable(cont, name="l", type="free", domain=[i, lc])

    # demand block
    intr = Variable(cont, name="intr", type="free", domain=[i])
    cd = Variable(cont, name="cd", type="free", domain=[i])
    gd = Variable(cont, name="gd", type="free", domain=[i])
    id = Variable(cont, name="id", type="free", domain=[i])
    dst = Variable(cont, name="dst", type="free", domain=[i])
    y = Variable(cont, name="y", type="free")
    gr = Variable(cont, name="gr", type="free")
    tariff = Variable(cont, name="tariff", type="free")
    indtax = Variable(cont, name="indtax", type="free")
    netsub = Variable(cont, name="netsub", type="free")
    gdtot = Variable(cont, name="gdtot", type="free")
    hhsav = Variable(cont, name="hhsav", type="free")
    govsav = Variable(cont, name="govsav", type="free")
    deprecia = Variable(cont, name="deprecia", type="free")
    invest = Variable(cont, name="invest", type="free")
    savings = Variable(cont, name="savings", type="free")
    mps = Variable(cont, name="mps", type="free", domain=[hh])
    fsav = Variable(cont, name="fsav", type="free")
    dk = Variable(cont, name="dk", type="free", domain=[i])
    ypr = Variable(cont, name="ypr", type="free")
    remit = Variable(cont, name="remit", type="free")
    fbor = Variable(cont, name="fbor", type="free")
    yh = Variable(cont, name="yh", type="free", domain=[hh])
    tothhtax = Variable(cont, name="tothhtax", type="free")

    # welfare indicator for objective function
    omega = Variable(cont, name="omega", type="free")

    er.l.assign = 1.0000
    pr.l.assign = 0.0000
    pindex.l.assign = 1.0000
    gr.l.assign = 194.0449
    tariff.l.assign = 28.6572
    indtax.l.assign = 65.2754
    netsub.l.assign = 0.0000
    gdtot.l.assign = 141.1519
    hhsav.l.assign = 61.4089
    govsav.l.assign = 52.8930
    deprecia.l.assign = 0.0000
    savings.l.assign = 159.1419
    invest.l.assign = 159.1419
    fsav.l.assign = 39.1744
    fbor.l.assign = 58.7590
    remit.l.assign = 0.0000
    tothhtax.l.assign = 100.1122
    y.l.assign = 1123.5941

    labres1 = Parameter(
        cont,
        name="labres1",
        domain=[i, lc],
        records=np.array(
            [
                [2515.900, 442.643, 0.000],
                [0.000, 767.776, 0.000],
                [0.000, 355.568, 948.100],
            ]
        ),
    )

    labres2 = Parameter(
        cont,
        name="labres2",
        domain=["*", lc],
        records=pd.DataFrame(
            [
                ["wa", "labor1", 0.074],
                ["ls", "labor1", 2515.9],
                ["wa", "labor2", 0.14],
                ["ls", "labor2", 1565.987],
                ["wa", "labor3", 0.152],
                ["ls", "labor3", 948.1],
            ]
        ),
    )

    hhres = Parameter(
        cont,
        name="hhres",
        domain=["*", hh],
        records=pd.DataFrame(
            [
                ["yh", "lab_hh", 548.7478],
                ["mps", "lab_hh", 0.06],
                ["yh", "cap_hh", 574.8463],
                ["mps", "cap_hh", 0.06],
            ]
        ),
    )

    l.l[i, lc] = labres1[i, lc]
    ls.l[lc] = labres2["ls", lc]
    wa.l[lc] = labres2["wa", lc]
    mps.l[hh] = hhres["mps", hh]
    yh.l[hh] = hhres["yh", hh]

    sectres = Parameter(
        cont, name="sectres", domain=["*", i], records=sectres_df
    )

    pd1.l[i] = sectres["pd1", i]
    pm.l[i] = sectres["pm", i]
    pe.l[i] = sectres["pe", i]
    pk.l[i] = sectres["pk", i]
    px.l[i] = sectres["px", i]
    p.l[i] = sectres["p", i]
    pva.l[i] = sectres["pva", i]
    x.l[i] = sectres["x", i]
    xd.l[i] = sectres["xd", i]
    xxd.l[i] = sectres["xxd", i]
    e.l[i] = sectres["e", i]
    m.l[i] = sectres["m", i]
    k.l[i] = sectres["k", i]
    intr.l[i] = sectres["intr", i]
    cd.l[i] = sectres["cd", i]
    gd.l[i] = sectres["gd", i]
    id.l[i] = sectres["id", i]
    dst.l[i] = sectres["dst", i]
    dk.l[i] = sectres["dk", i]
    it[i] = Number(1).where[e.l[i] | m.l[i]]
    inn[i] = not it[i]
    k.fx[i] = k.l[i]
    m.fx[inn] = 0
    e.fx[inn] = 0
    l.fx[i, lc].where[l.l[i, lc] == 0] = 0

    p.lo[i] = 0.01
    pd1.lo[i] = 0.01
    pm.lo[it] = 0.01
    pk.lo[i] = 0.01
    px.lo[i] = 0.01
    x.lo[i] = 0.01
    xd.lo[i] = 0.01
    m.lo[it] = 0.01
    xxd.lo[it] = 0.01
    wa.lo[lc] = 0.01
    intr.lo[i] = 0.01
    y.lo.assign = 0.01
    e.lo[it] = 0.01
    l.lo[i, lc].where[l.l[i, lc] != 0] = 0.01

    # Equation Definitions
    # price block
    pmdef = Equation(cont, name="pmdef", domain=[i])
    pedef = Equation(cont, name="pedef", domain=[i])
    absorption = Equation(cont, name="absorption", domain=[i])
    sales = Equation(cont, name="sales", domain=[i])
    actp = Equation(cont, name="actp", domain=[i])
    pkdef = Equation(cont, name="pkdef", domain=[i])
    pindexdef = Equation(cont, name="pindexdef")

    # output block
    activity = Equation(cont, name="activity", domain=[i])
    profitmax = Equation(cont, name="profitmax", domain=[i, lc])
    lmequil = Equation(cont, name="lmequil", domain=[lc])
    cet = Equation(cont, name="cet", domain=[i])
    esupply = Equation(cont, name="esupply", domain=[i])
    armington = Equation(cont, name="armington", domain=[i])
    costmin = Equation(cont, name="costmin", domain=[i])
    xxdsn = Equation(cont, name="xxdsn", domain=[i])
    xsn = Equation(cont, name="xsn", domain=[i])

    # demand block
    inteq = Equation(cont, name="inteq", domain=[i])
    cdeq = Equation(cont, name="cdeq", domain=[i])
    dsteq = Equation(cont, name="dsteq", domain=[i])
    gdp = Equation(cont, name="gdp")
    labory = Equation(cont, name="labory")
    capitaly = Equation(cont, name="capitaly")
    hhtaxdef = Equation(cont, name="hhtaxdef")
    gdeq = Equation(cont, name="gdeq", domain=[i])
    greq = Equation(cont, name="greq")
    tariffdef = Equation(cont, name="tariffdef")
    premium = Equation(cont, name="premium")
    indtaxdef = Equation(cont, name="indtaxdef")
    netsubdef = Equation(cont, name="netsubdef")

    # savings-investment block
    hhsaveq = Equation(cont, name="hhsaveq")
    gruse = Equation(cont, name="gruse")
    depreq = Equation(cont, name="depreq")
    totsav = Equation(cont, name="totsav")
    prodinv = Equation(cont, name="prodinv", domain=[i])
    ieq = Equation(cont, name="ieq", domain=[i])

    # balance of payments
    caeq = Equation(cont, name="caeq")

    # market clearing
    equil = Equation(cont, name="equil", domain=[i])

    # objective function
    obj = Equation(cont, name="obj")

    # price block
    pmdef[it] = pm[it] == pwm[it] * er * (1 + tm[it] + pr)

    pedef[it] = pe[it] == pwe[it] * (1 + te[it]) * er

    absorption[i] = (
        p[i] * x[i] == pd1[i] * xxd[i] + (pm[i] * m[i]).where[it[i]]
    )

    sales[i] = px[i] * xd[i] == pd1[i] * xxd[i] + (pe[i] * e[i]).where[it[i]]

    actp[i] = px[i] * (1 - itax[i]) == pva[i] + Sum(j, io[j, i] * p[j])

    pkdef[i] = pk[i] == Sum(j, p[j] * imat[j, i])

    pindexdef.expr = pindex == Sum(i, pwts[i] * p[i])

    # output and factors of production block
    activity[i] = xd[i] == ad[i] * Product(
        lc.where[wdist[i, lc]], l[i, lc] ** alphl[i, lc]
    ) * k[i] ** (1 - Sum(lc, alphl[i, lc]))

    profitmax[i, lc].where[wdist[i, lc]] = (
        wa[lc] * wdist[i, lc] * l[i, lc] == xd[i] * pva[i] * alphl[i, lc]
    )

    lmequil[lc] = Sum(i, l[i, lc]) == ls[lc]

    cet[it] = xd[it] == at[it] * (
        gamma[it] * e[it] ** rhot[it] + (1 - gamma[it]) * xxd[it] ** rhot[it]
    ) ** (1 / rhot[it])

    esupply[it] = e[it] / xxd[it] == (
        pe[it] / pd1[it] * (1 - gamma[it]) / gamma[it]
    ) ** (1 / (rhot[it] - 1))

    armington[it] = x[it] == ac[it] * (
        delta[it] * m[it] ** (rhoc[it] * (-1))
        + (1 - delta[it]) * xxd[it] ** (rhoc[it] * (-1))
    ) ** (-1 / rhoc[it])

    costmin[it] = m[it] / xxd[it] == (
        pd1[it] / pm[it] * delta[it] / (1 - delta[it])
    ) ** (1 / (1 + rhoc[it]))

    xxdsn[inn] = xxd[inn] == xd[inn]

    xsn[inn] = x[inn] == xxd[inn]

    # demand block
    inteq[i] = intr[i] == Sum(j, io[i, j] * xd[j])

    dsteq[i] = dst[i] == dstr[i] * xd[i]

    cdeq[i] = p[i] * cd[i] == Sum(
        hh, cles[i, hh] * (1 - mps[hh]) * yh[hh] * (1 - htax[hh])
    )

    gdp.expr = y == Sum(hh, yh[hh])

    labory.expr = yh["lab_hh"] == Sum(lc, wa[lc] * ls[lc]) + remit * er

    capitaly.expr = (
        yh["cap_hh"]
        == Sum(i, pva[i] * xd[i])
        - deprecia
        - Sum(lc, wa[lc] * ls[lc])
        + fbor * er
        + ypr
    )

    hhsaveq.expr = hhsav == Sum(hh, mps[hh] * yh[hh] * (1 - htax[hh]))

    greq.expr = gr == tariff - netsub + indtax + tothhtax

    gruse.expr = gr == Sum(i, p[i] * gd[i]) + govsav

    gdeq[i] = gd[i] == gles[i] * gdtot

    tariffdef.expr = tariff == Sum(it, tm[it] * m[it] * pwm[it]) * er

    indtaxdef.expr = indtax == Sum(i, itax[i] * px[i] * xd[i])

    netsubdef.expr = netsub == Sum(it, te[it] * e[it] * pwe[it]) * er

    premium.expr = ypr == Sum(it, pwm[it] * m[it]) * er * pr

    hhtaxdef.expr = tothhtax == Sum(hh, htax[hh] * yh[hh])

    depreq.expr = deprecia == Sum(i, depr[i] * pk[i] * k[i])

    totsav.expr = savings == hhsav + govsav + deprecia + fsav * er

    prodinv[i] = pk[i] * dk[i] == kio[i] * invest - kio[i] * Sum(
        j, dst[j] * p[j]
    )

    ieq[i] = id[i] == Sum(j, imat[i, j] * dk[j])

    # balance of payments
    caeq.expr = (
        Sum(it, pwm[it] * m[it])
        == Sum(it, pwe[it] * e[it]) + fsav + remit + fbor
    )
    # market clearing
    equil[i] = x[i] == intr[i] + cd[i] + gd[i] + id[i] + dst[i]

    # objective function
    obj.expr = omega == Product(
        i.where[cles[i, "lab_hh"]], cd[i] ** cles[i, "lab_hh"]
    )

    er.fx.assign = er.l
    fsav.fx.assign = fsav.l
    remit.fx.assign = remit.l
    fbor.fx.assign = fbor.l
    pindex.fx.assign = pindex.l
    mps.fx[hh] = mps.l[hh]
    gdtot.fx.assign = gdtot.l
    ls.fx[lc] = ls.l[lc]

    model1 = Model(
        cont, name="model1", equations=cont.getEquations(), problem="cns"
    )

    model1.solve()

    print(
        "\nObjective Function Variable <omega>: ",
        round(omega.records.level.tolist()[0], 2),
    )
    print("\nDomestic prices:\n", pd1.records.set_index("i").level)


if __name__ == "__main__":
    main()
