"""
A Recursive-Dynamic Standard CGE Model (DYNCGE)

This model is featured in the following book.
Hosoe, N., Gasawa, K., Hashimoto, H. Textbook of Computable General
Equilibrium Modeling: Programming and Simulations, 2nd Edition,
University of Tokyo Press. (in Japanese)

Keywords: nonlinear programming, general equilibrium model, social accounting
          matrix
"""

from gamspy import (
    Set,
    Alias,
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Ord,
    Product,
)
import numpy as np
from gamspy import Problem, Sense


def main():
    m = Container()

    m.addOptions({"limRow": 0, "limCol": 0})

    # ===============================================================
    # Definition of sets for suffix ---------------------------------
    # ===============================================================
    # Sets
    u = Set(
        m,
        name="u",
        records=[
            "AGR",
            "LMN",
            "HMN",
            "SRV",
            "CAP",
            "LAB",
            "HOH",
            "GOV",
            "INV",
            "EXT",
            "IDT",
            "TRF",
        ],
    )
    i = Set(m, name="i", domain=[u], records=["AGR", "LMN", "HMN", "SRV"])
    h = Set(m, name="h", domain=[u], records=["CAP", "LAB"])
    h_mob = Set(m, name="h_mob", domain=[h], records=["LAB"])
    t = Set(m, name="t", records=[str(i) for i in range(31)])

    # Alias
    v = Alias(m, name="v", alias_with=u)
    j = Alias(m, name="j", alias_with=i)
    k = Alias(m, name="k", alias_with=h)

    # ===============================================================
    # Data for Dynamics ---------------------------------------------
    # ===============================================================
    # Scalar
    ror = Parameter(m, name="ror")
    dep = Parameter(m, name="dep")
    pop = Parameter(m, name="pop")
    zeta = Parameter(m, name="zeta")

    ror.assign = 0.05
    dep.assign = 0.04
    pop.assign = 0.02
    zeta.assign = 1

    sam_data = np.array(
        [
            [
                1643.017,
                7560.896,
                237.841,
                1409.202,
                0,
                0,
                3563.257,
                0,
                919.745,
                62.464,
                0,
                0,
            ],
            [
                1485.854,
                10803.527,
                15330.764,
                18597.270,
                0,
                0,
                32220.169,
                329.469,
                802.026,
                1196.525,
                0,
                0,
            ],
            [
                1071.954,
                4277.721,
                113390.269,
                48734.424,
                0,
                0,
                27648.678,
                4.931,
                34979.803,
                55083.516,
                0,
                0,
            ],
            [
                2002.380,
                11406.260,
                50513.476,
                177675.714,
                0,
                0,
                234243.865,
                90707.177,
                79169.426,
                17426.156,
                0,
                0,
            ],
            [
                5082.506,
                7042.697,
                21058.821,
                163045.396,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1435.010,
                8942.365,
                42510.123,
                222732.700,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, 196229.420, 275620.198, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 52243.041, 0, 0, 0, 34024.445, 4774.091],
            [0, 0, 0, 0, 0, 0, 121930.608, 0, 0, -6059.608, 0, 0],
            [
                2092.569,
                23796.669,
                30982.559,
                10837.256,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [433.854, 4068.616, 9418.058, 20103.917, 0, 0, 0, 0, 0, 0, 0, 0],
            [149.278, 2866.853, 1749.385, 8.575, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    # ===============================================================
    # SAM Data
    # ===============================================================
    SAM = Parameter(m, name="SAM", domain=[u, v], records=sam_data)

    # Source: compiled by N. Hosoe, based on the I/O table for 2005

    SAMGAP = Parameter(m, name="SAMGAP", domain=[u])
    SAMGAP[u] = Sum(v, SAM[u, v] - SAM[v, u])

    # print(SAMGAP.records)

    # ===============================================================
    # Loading the initial values ------------------------------------
    # ===============================================================

    # Base year values
    Y00 = Parameter(m, name="Y00", domain=[j])
    F00 = Parameter(m, name="F00", domain=[h, j])
    X00 = Parameter(m, name="X00", domain=[i, j])
    Z00 = Parameter(m, name="Z00", domain=[j])
    Xp00 = Parameter(m, name="Xp00", domain=[i])
    Xg00 = Parameter(m, name="Xg00", domain=[i])
    Xv00 = Parameter(m, name="Xv00", domain=[i])
    E00 = Parameter(m, name="E00", domain=[i])
    M00 = Parameter(m, name="M00", domain=[i])
    Q00 = Parameter(m, name="Q00", domain=[i])
    D00 = Parameter(m, name="D00", domain=[i])
    Sp00 = Parameter(m, name="Sp00")
    Td00 = Parameter(m, name="Td00")
    Tz00 = Parameter(m, name="Tz00", domain=[j])
    Tm00 = Parameter(m, name="Tm00", domain=[j])
    III00 = Parameter(m, name="III00")
    II00 = Parameter(m, name="II00", domain=[j])
    KK00 = Parameter(m, name="KK00", domain=[j])
    CC00 = Parameter(m, name="CC00")
    FF00 = Parameter(m, name="FF00", domain=[h])
    Sf00 = Parameter(m, name="Sf00")
    tauz00 = Parameter(m, name="tauz00", domain=[i])
    taum00 = Parameter(m, name="taum00", domain=[i])

    # Base run value
    Y0 = Parameter(m, name="Y0", domain=[j, t])
    F0 = Parameter(m, name="F0", domain=[h, j, t])
    X0 = Parameter(m, name="X0", domain=[i, j, t])
    Z0 = Parameter(m, name="Z0", domain=[j, t])
    Xp0 = Parameter(m, name="Xp0", domain=[i, t])
    Xv0 = Parameter(m, name="Xv0", domain=[i, t])
    E0 = Parameter(m, name="E0", domain=[i, t])
    M0 = Parameter(m, name="M0", domain=[i, t])
    Q0 = Parameter(m, name="Q0", domain=[i, t])
    D0 = Parameter(m, name="D0", domain=[i, t])
    Sp0 = Parameter(m, name="Sp0", domain=[t])
    Td0 = Parameter(m, name="Td0", domain=[t])
    Tz0 = Parameter(m, name="Tz0", domain=[j, t])
    Tm0 = Parameter(m, name="Tm0", domain=[j, t])
    III0 = Parameter(m, name="III0", domain=[t])
    II0 = Parameter(m, name="II0", domain=[j, t])
    KK0 = Parameter(m, name="KK0", domain=[j, t])
    CC0 = Parameter(m, name="CC0", domain=[t])
    FF0 = Parameter(m, name="FF0", domain=[h, t])
    pf0 = Parameter(m, name="pf0", domain=[h, j, t])
    py0 = Parameter(m, name="py0", domain=[j, t])
    pz0 = Parameter(m, name="pz0", domain=[j, t])
    pq0 = Parameter(m, name="pq0", domain=[i, t])
    pe0 = Parameter(m, name="pe0", domain=[i, t])
    pm0 = Parameter(m, name="pm0", domain=[i, t])
    pd0 = Parameter(m, name="pd0", domain=[i, t])
    pk0 = Parameter(m, name="pk0", domain=[t])
    epsilon0 = Parameter(m, name="epsilon0", domain=[t])
    PRICE0 = Parameter(m, name="PRICE0", domain=[t])

    # Exogenous variables
    Xg0 = Parameter(m, name="Xg0", domain=[i, t])
    Sf0 = Parameter(m, name="Sf0", domain=[t])
    pWe = Parameter(m, name="pWe", domain=[i])
    pWm = Parameter(m, name="pWm", domain=[i])
    tauz = Parameter(m, name="tauz", domain=[i])
    taum = Parameter(m, name="taum", domain=[i])

    # for result reporting
    Y1 = Parameter(m, name="Y1", domain=[j, t])
    F1 = Parameter(m, name="F1", domain=[h, j, t])
    X1 = Parameter(m, name="X1", domain=[i, j, t])
    Z1 = Parameter(m, name="Z1", domain=[j, t])
    Xp1 = Parameter(m, name="Xp1", domain=[i, t])
    Xv1 = Parameter(m, name="Xv1", domain=[i, t])
    E1 = Parameter(m, name="E1", domain=[i, t])
    M1 = Parameter(m, name="M1", domain=[i, t])
    Q1 = Parameter(m, name="Q1", domain=[i, t])
    D1 = Parameter(m, name="D1", domain=[i, t])
    Sp1 = Parameter(m, name="Sp1", domain=[t])
    Td1 = Parameter(m, name="Td1", domain=[t])
    Tz1 = Parameter(m, name="Tz1", domain=[j, t])
    Tm1 = Parameter(m, name="Tm1", domain=[i, t])
    FF1 = Parameter(m, name="FF1", domain=[h, t])
    II1 = Parameter(m, name="II1", domain=[j, t])
    III1 = Parameter(m, name="III1", domain=[t])
    KK1 = Parameter(m, name="KK1", domain=[j, t])
    CC1 = Parameter(m, name="CC1", domain=[t])
    tauz1 = Parameter(m, name="tauz1", domain=[i, t])
    taum1 = Parameter(m, name="taum1", domain=[i, t])
    pz1 = Parameter(m, name="pz1", domain=[j, t])
    pd1 = Parameter(m, name="pd1", domain=[j, t])
    pm1 = Parameter(m, name="pm1", domain=[j, t])
    pe1 = Parameter(m, name="pe1", domain=[j, t])
    pq1 = Parameter(m, name="pq1", domain=[j, t])
    pf1 = Parameter(m, name="pf1", domain=[h, j, t])
    py1 = Parameter(m, name="py1", domain=[j, t])
    epsilon1 = Parameter(m, name="epsilon1", domain=[t])
    pk1 = Parameter(m, name="pk1", domain=[t])
    PRICE1 = Parameter(m, name="PRICE1", domain=[t])

    Td00.assign = SAM["GOV", "HOH"]
    Tz00[j] = SAM["IDT", j]
    Tm00[j] = SAM["TRF", j]
    F00[h, j] = SAM[h, j]
    Y00[j] = Sum(h, F00[h, j])
    X00[i, j] = SAM[i, j]
    Z00[j] = Y00[j] + Sum(i, X00[i, j])
    M00[i] = SAM["EXT", i]
    tauz00[j] = Tz00[j] / Z00[j]
    taum00[j] = Tm00[j] / M00[j]
    Xp00[i] = SAM[i, "HOH"]
    CC00.assign = Sum(i, Xp00[i])
    FF00[h] = SAM["HOH", h]
    E00[i] = SAM[i, "EXT"]
    D00[i] = (1 + tauz00[i]) * Z00[i] - E00[i]
    Q00[i] = (1 + taum00[i]) * M00[i] + D00[i]
    Sf00.assign = SAM["INV", "EXT"]

    # ===============================================================
    # Adjusting Investment in the SAM for the Assumed BAU Growth Path
    # ===============================================================
    # Scalars
    III_ASS = Parameter(m, name="III_ASS")
    III_SAM = Parameter(m, name="III_SAM")
    adj = Parameter(m, name="adj")

    III_ASS.assign = (pop + dep) / ror * FF00["CAP"]
    III_SAM.assign = Sum(i, SAM[i, "INV"])
    adj.assign = III_ASS / III_SAM

    # Adjusting investment level
    Xv00[i] = SAM[i, "INV"] * adj

    # Reallocating the gap made by the inv. adjustment to gov. cons.
    Xg00[i] = SAM[i, "GOV"] - (Xv00[i] - SAM[i, "INV"])

    # Computing the direct tax revenue that balances the gov. budget
    Td00.assign = Sum(i, Xg00[i]) - Sum(i, Tz00[i] + Tm00[i])

    # Computing the household sav. that balances the household budget
    Sp00.assign = Sum(h, FF00[h]) - (Sum(i, Xp00[i]) + Td00)
    III00.assign = Sum(i, Xv00[i])
    II00[j] = (Sp00 + Sf00) * F00["CAP", j] / Sum(i, F00["CAP", i])
    KK00[j] = F00["CAP", j] / ror

    # ===============================================================
    # Computing the BAU path
    # ===============================================================
    Y0[j, t] = Y00[j] * (1 + pop) ** (Ord(t) - 1)
    F0[h, j, t] = F00[h, j] * (1 + pop) ** (Ord(t) - 1)
    X0[i, j, t] = X00[i, j] * (1 + pop) ** (Ord(t) - 1)
    Z0[j, t] = Z00[j] * (1 + pop) ** (Ord(t) - 1)
    Xp0[i, t] = Xp00[i] * (1 + pop) ** (Ord(t) - 1)
    Xv0[i, t] = Xv00[i] * (1 + pop) ** (Ord(t) - 1)
    E0[i, t] = E00[i] * (1 + pop) ** (Ord(t) - 1)
    M0[i, t] = M00[i] * (1 + pop) ** (Ord(t) - 1)
    Q0[i, t] = Q00[i] * (1 + pop) ** (Ord(t) - 1)
    D0[i, t] = D00[i] * (1 + pop) ** (Ord(t) - 1)
    FF0[h, t] = FF00[h] * (1 + pop) ** (Ord(t) - 1)
    III0[t] = III00 * (1 + pop) ** (Ord(t) - 1)
    II0[j, t] = II00[j] * (1 + pop) ** (Ord(t) - 1)
    KK0[j, t] = KK00[j] * (1 + pop) ** (Ord(t) - 1)
    CC0[t] = CC00 * (1 + pop) ** (Ord(t) - 1)
    Sp0[t] = Sp00 * (1 + pop) ** (Ord(t) - 1)
    Td0[t] = Td00 * (1 + pop) ** (Ord(t) - 1)
    Tz0[j, t] = Tz00[j] * (1 + pop) ** (Ord(t) - 1)
    Tm0[i, t] = Tm00[i] * (1 + pop) ** (Ord(t) - 1)
    pf0[h, j, t] = 1
    py0[j, t] = 1
    pz0[j, t] = 1
    pq0[i, t] = 1
    pe0[i, t] = 1
    pm0[i, t] = 1
    pd0[i, t] = 1
    pk0[t] = 1
    epsilon0[t] = 1
    PRICE0[t] = 1

    # Setting exogenous variables
    Xg0[i, t] = Xg00[i] * (1 + pop) ** (Ord(t) - 1)
    Sf0[t] = Sf00 * (1 + pop) ** (Ord(t) - 1)
    pWe[i] = 1
    pWm[i] = 1
    tauz[i] = tauz00[i]
    taum[i] = taum00[i]

    # ===============================================================
    # Calibration ---------------------------------------------------
    # ===============================================================
    # Parameters
    sigma = Parameter(m, name="sigma", domain=[i])
    psi = Parameter(m, name="psi", domain=[i])
    eta = Parameter(m, name="eta", domain=[i])
    phi = Parameter(m, name="phi", domain=[i])

    sigma[i] = 2
    psi[i] = 2
    eta[i] = (sigma[i] - 1) / sigma[i]
    phi[i] = (psi[i] + 1) / psi[i]

    # Parameters
    alpha = Parameter(m, name="alpha", domain=[i])
    a = Parameter(m, name="a")
    beta = Parameter(m, name="beta", domain=[h, j])
    b = Parameter(m, name="b", domain=[j])
    ax = Parameter(m, name="ax", domain=[i, j])
    ay = Parameter(m, name="ay", domain=[j])
    lamda = Parameter(m, name="lamda", domain=[i])
    iota = Parameter(m, name="iota")
    deltam = Parameter(m, name="deltam", domain=[i])
    deltad = Parameter(m, name="deltad", domain=[i])
    gamma = Parameter(m, name="gamma", domain=[i])
    xid = Parameter(m, name="xid", domain=[i])
    xie = Parameter(m, name="xie", domain=[i])
    theta = Parameter(m, name="theta", domain=[i])
    ssp = Parameter(m, name="ssp")

    alpha[i] = Xp00[i] / Sum(j, Xp00[j])
    a.assign = CC00 / Product(j, Xp00[j] ** alpha[j])
    beta[h, j] = F00[h, j] / Sum(k, F00[k, j])
    b[j] = Y00[j] / Product(h, F00[h, j] ** beta[h, j])
    ax[i, j] = X00[i, j] / Z00[j]
    ay[j] = Y00[j] / Z00[j]
    lamda[i] = Xv00[i] / Sum(j, Xv00[j])
    iota.assign = III00 / Product(i, Xv00[i] ** lamda[i])
    deltam[i] = (
        (1 + taum00[i])
        * M00[i] ** (1 - eta[i])
        / ((1 + taum00[i]) * M00[i] ** (1 - eta[i]) + D00[i] ** (1 - eta[i]))
    )
    deltad[i] = D00[i] ** (1 - eta[i]) / (
        (1 + taum00[i]) * M00[i] ** (1 - eta[i]) + D00[i] ** (1 - eta[i])
    )
    gamma[i] = Q00[i] / (
        deltam[i] * M00[i] ** eta[i] + deltad[i] * D00[i] ** eta[i]
    ) ** (1 / eta[i])
    xie[i] = E00[i] ** (1 - phi[i]) / (
        E00[i] ** (1 - phi[i]) + D00[i] ** (1 - phi[i])
    )
    xid[i] = D00[i] ** (1 - phi[i]) / (
        E00[i] ** (1 - phi[i]) + D00[i] ** (1 - phi[i])
    )
    theta[i] = Z00[i] / (
        xie[i] * E00[i] ** phi[i] + xid[i] * D00[i] ** phi[i]
    ) ** (1 / phi[i])
    ssp.assign = Sp00 / (Sum([h, j], F00[h, j]) - Td00)

    # ===============================================================
    # Defining model system -----------------------------------------
    # ===============================================================
    # Variables
    Y = Variable(m, name="Y", type="free", domain=[j])
    F = Variable(m, name="F", type="free", domain=[h, j])
    X = Variable(m, name="X", type="free", domain=[i, j])
    Z = Variable(m, name="Z", type="free", domain=[j])
    Xp = Variable(m, name="Xp", type="free", domain=[i])
    Xg = Variable(m, name="Xg", type="free", domain=[i])
    Xv = Variable(m, name="Xv", type="free", domain=[i])
    E = Variable(m, name="E", type="free", domain=[i])
    M = Variable(m, name="M", type="free", domain=[i])
    Q = Variable(m, name="Q", type="free", domain=[i])
    D = Variable(m, name="D", type="free", domain=[i])
    FF = Variable(m, name="FF", type="free", domain=[h])
    pf = Variable(m, name="pf", type="free", domain=[h, j])
    py = Variable(m, name="py", type="free", domain=[j])
    pz = Variable(m, name="pz", type="free", domain=[j])
    pq = Variable(m, name="pq", type="free", domain=[i])
    pe = Variable(m, name="pe", type="free", domain=[i])
    pm = Variable(m, name="pm", type="free", domain=[i])
    pd = Variable(m, name="pd", type="free", domain=[i])
    pk = Variable(m, name="pk", type="free")
    epsilon = Variable(m, name="epsilon", type="free")
    Sp = Variable(m, name="Sp", type="free")
    Sf = Variable(m, name="Sf", type="free")
    Td = Variable(m, name="Td", type="free")
    Tz = Variable(m, name="Tz", type="free", domain=[j])
    Tm = Variable(m, name="Tm", type="free", domain=[i])
    KK = Variable(m, name="KK", type="free", domain=[j])
    II = Variable(m, name="II", type="free", domain=[j])
    III = Variable(m, name="III", type="free")
    PRICE = Variable(m, name="PRICE", type="free")
    CC = Variable(m, name="CC", type="free")

    # Equations
    eqpy = Equation(m, name="eqpy", domain=[j])
    eqF = Equation(m, name="eqF", domain=[h, j])
    eqX = Equation(m, name="eqX", domain=[i, j])
    eqY = Equation(m, name="eqY", domain=[j])
    eqpzs = Equation(m, name="eqpzs", domain=[j])
    eqTd = Equation(m, name="eqTd")
    eqTz = Equation(m, name="eqTz", domain=[j])
    eqTm = Equation(m, name="eqTm", domain=[i])
    eqXv = Equation(m, name="eqXv", domain=[i])
    eqSp = Equation(m, name="eqSp")
    eqXp = Equation(m, name="eqXp", domain=[i])
    eqpe = Equation(m, name="eqpe", domain=[i])
    eqpm = Equation(m, name="eqpm", domain=[i])
    eqepsilon = Equation(m, name="eqepsilon")
    eqpqs = Equation(m, name="eqpqs", domain=[i])
    eqM = Equation(m, name="eqM", domain=[i])
    eqD = Equation(m, name="eqD", domain=[i])
    eqpzd = Equation(m, name="eqpzd", domain=[i])
    eqE = Equation(m, name="eqE", domain=[i])
    eqDs = Equation(m, name="eqDs", domain=[i])
    eqpqd = Equation(m, name="eqpqd", domain=[i])
    eqpf1 = Equation(m, name="eqpf1", domain=[h_mob])
    eqpf2 = Equation(m, name="eqpf2", domain=[h_mob, i, j])
    eqpf3 = Equation(m, name="eqpf3", domain=[j])
    eqpk = Equation(m, name="eqpk")
    eqIII = Equation(m, name="eqIII")
    eqII = Equation(m, name="eqII", domain=[j])
    eqCC = Equation(m, name="eqCC")
    eqPRICE = Equation(m, name="eqPRICE")

    # ===============================================================
    # Model equations
    # ===============================================================
    # [domestic production] -
    # composite factor production func.                  (Cobb-Douglas)
    eqpy[j] = Y[j] == b[j] * Product(h, F[h, j] ** beta[h, j])

    # factor demand function                             (Cobb-Douglas)
    eqF[h, j] = F[h, j] == beta[h, j] * py[j] * Y[j] / pf[h, j]

    # intermediate input demand function                     (Leontief)
    eqX[i, j] = X[i, j] == ax[i, j] * Z[j]

    # composite factor demand function                       (Leontief)
    eqY[j] = Y[j] == ay[j] * Z[j]

    # unit price of gross output                             (Leontief)
    eqpzs[j] = pz[j] == ay[j] * py[j] + Sum(i, ax[i, j] * pq[i])

    # [government behavior] -
    # lump Sum direct tax revenue
    eqTd.expr = Td == Sum(i, pq[i] * Xg[i]) - Sum(i, Tm[i] + Tz[i])

    # production tax revenue
    eqTz[j] = Tz[j] == tauz[j] * pz[j] * Z[j]

    # import tariff revenue
    eqTm[i] = Tm[i] == taum[i] * pm[i] * M[i]

    # [investment behavior] -
    # composite investment production function
    eqXv[i] = Xv[i] == lamda[i] * pk * Sum(j, II[j]) / pq[i]

    # [savings] ----------
    # savings function
    eqSp.expr = Sp == ssp * (Sum([h, j], pf[h, j] * F[h, j]) - Td)

    # [household consumption] --                          (Cobb-Douglas)
    eqXp[i] = (
        Xp[i] == alpha[i] * (Sum([h, j], pf[h, j] * F[h, j]) - Sp - Td) / pq[i]
    )

    # [international trade] --
    eqpe[i] = pe[i] == epsilon * pWe[i]

    eqpm[i] = pm[i] == epsilon * pWm[i]

    # BOP constraint
    eqepsilon.expr = Sum(i, pWe[i] * E[i]) + Sf == Sum(i, pWm[i] * M[i])

    # [Armington function] --
    # Armington's composite good production function              (CES)
    eqpqs[i] = Q[i] == gamma[i] * (
        deltam[i] * M[i] ** eta[i] + deltad[i] * D[i] ** eta[i]
    ) ** (1 / eta[i])

    # import demand function                                      (CES)
    eqM[i] = (
        M[i]
        == (gamma[i] ** eta[i] * deltam[i] * pq[i] / ((1 + taum[i]) * pm[i]))
        ** (1 / (1 - eta[i]))
        * Q[i]
    )

    # domestic good demand function                               (CES)
    eqD[i] = (
        D[i]
        == (gamma[i] ** eta[i] * deltad[i] * pq[i] / pd[i])
        ** (1 / (1 - eta[i]))
        * Q[i]
    )

    # [transformation function] --
    # gross domestic output disaggregation function               (CET)
    eqpzd[i] = Z[i] == theta[i] * (
        xie[i] * E[i] ** phi[i] + xid[i] * D[i] ** phi[i]
    ) ** (1 / phi[i])

    # export supply function                                       (CET)
    eqE[i] = (
        E[i]
        == (theta[i] ** phi[i] * xie[i] * (1 + tauz[i]) * pz[i] / pe[i])
        ** (1 / (1 - phi[i]))
        * Z[i]
    )

    # domestic good supply function                                (CET)
    eqDs[i] = (
        D[i]
        == (theta[i] ** phi[i] * xid[i] * (1 + tauz[i]) * pz[i] / pd[i])
        ** (1 / (1 - phi[i]))
        * Z[i]
    )

    # [market clearing condition]
    # Arminton's composite good market
    eqpqd[i] = Q[i] == Xp[i] + Xg[i] + Xv[i] + Sum(j, X[i, j])

    # labor market: quantity
    eqpf1[h_mob] = Sum(j, F[h_mob, j]) == FF[h_mob]

    # labor market: price
    eqpf2[h_mob, i, j] = pf[h_mob, j] == pf[h_mob, i]

    # capital market
    eqpf3[j] = F["CAP", j] == ror * KK[j]

    # investment goods market
    eqpk.expr = Sum(j, II[j]) == III

    # [dynamic equations]
    # composite investment good market clearing condition
    eqIII.expr = III == iota * Product(i, Xv[i] ** lamda[i])

    # sectoral investment allocation
    eqII[j] = pk * II[j] == pf["CAP", j] ** zeta * F["CAP", j] / Sum(
        i, pf["CAP", i] ** zeta * F["CAP", i]
    ) * (Sp + epsilon * Sf)

    # felicity function
    eqCC.expr = CC == a * Product(i, Xp[i] ** alpha[i])

    # Price level [numeraire]
    eqPRICE.expr = PRICE == Sum(j, pq[j] * Q00[j] / Sum(i, Q00[i]))

    # ===============================================================
    # Initializing variables ----------------------------------------
    # ===============================================================
    Y.l[j] = Y00[j]
    F.l[h, j] = F00[h, j]
    X.l[i, j] = X00[i, j]
    Z.l[j] = Z00[j]
    Xp.l[i] = Xp00[i]
    Xv.l[i] = Xv00[i]
    E.l[i] = E00[i]
    M.l[i] = M00[i]
    Q.l[i] = Q00[i]
    D.l[i] = D00[i]
    pf.l[h, j] = 1
    py.l[j] = 1
    pz.l[j] = 1
    pq.l[i] = 1
    pe.l[i] = 1
    pm.l[i] = 1
    pd.l[i] = 1
    pk.l.assign = 1
    epsilon.l.assign = 1
    Sp.l.assign = Sp00
    Td.l.assign = Td00
    Tz.l[j] = Tz00[j]
    Tm.l[i] = Tm00[i]
    FF.l[h] = FF00[h]
    III.l.assign = III00
    II.l[j] = II00[j]

    # ---------------------------------------------------------------
    # Numeraire
    PRICE.fx.assign = 1

    # Initial factor endowments and exogenous variables
    FF.fx[h_mob] = FF00[h_mob]
    KK.fx[j] = KK00[j]
    Xg.fx[i] = Xg00[i]
    Sf.fx.assign = Sf00

    # ===============================================================
    # Defining and solving the model --------------------------------
    # ===============================================================
    dyncge = Model(
        m,
        name="dyncge",
        equations=m.getEquations(),
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=CC,
    )

    dyncge.solve()
    m.addOptions(
        {
            "limRow": 0,
            "limCol": 0,
            "solPrint": "off",
            "solveLink": "%solveLink.loadlibrary%",
        }
    )

    # ===============================================================
    # Simulation Runs: Abolition of Import Tariffs
    # ===============================================================

    # Scenario:
    taum[i] = taum00[i] * 0

    for iteration, _ in t.records.itertuples(index=False):
        dyncge.solve()

        #  storing results -------------------------
        Y1[j, iteration] = Y.l[j]
        F1[h, j, iteration] = F.l[h, j]
        X1[i, j, iteration] = X.l[i, j]
        Z1[j, iteration] = Z.l[j]
        Xp1[i, iteration] = Xp.l[i]
        Xv1[i, iteration] = Xv.l[i]
        E1[i, iteration] = E.l[i]
        M1[i, iteration] = M.l[i]
        Q1[i, iteration] = Q.l[i]
        D1[i, iteration] = D.l[i]
        Sp1[iteration] = Sp.l
        Td1[iteration] = Td.l
        Tz1[j, iteration] = Tz.l[j]
        Tm1[i, iteration] = Tm.l[i]
        FF1[h, iteration] = FF.l[h]
        II1[j, iteration] = II.l[j]
        III1[iteration] = III.l
        KK1[j, iteration] = KK.l[j]
        CC1[iteration] = CC.l
        tauz1[i, iteration] = tauz[i]
        taum1[i, iteration] = taum[i]
        pf1[h, j, iteration] = pf.l[h, j]
        py1[j, iteration] = py.l[j]
        pz1[j, iteration] = pz.l[j]
        pd1[j, iteration] = pd.l[j]
        pe1[j, iteration] = pe.l[j]
        pm1[j, iteration] = pm.l[j]
        pq1[j, iteration] = pq.l[j]
        pk1[iteration] = pk.l
        epsilon1[iteration] = epsilon.l
        PRICE1[iteration] = PRICE.l

        #  updating the state variables --------------
        FF.fx[h_mob] = FF.l[h_mob] * (1 + pop)
        KK.fx[j] = (1 - dep) * KK.l[j] + II.l[j]
        if int(iteration) < 30:
            Xg.fx[i] = Xg0[i, str(int(iteration) + 1)]
            Sf.fx.assign = Sf0[str(int(iteration) + 1)]

    # ===============================================================
    # Aftermath Computation
    # ===============================================================
    # Display of changes --------------------------------------------

    # Parameters
    # changes
    dY = Parameter(
        m,
        name="dY",
        domain=[j, t],
        description="change of composite factor             [%]",
    )
    dF = Parameter(
        m,
        name="dF",
        domain=[h, j, t],
        description="change of factor input                 [%]",
    )
    dX = Parameter(
        m,
        name="dX",
        domain=[i, j, t],
        description="change of intermediate input           [%]",
    )
    dZ = Parameter(
        m,
        name="dZ",
        domain=[j, t],
        description="change of gross output                 [%]",
    )
    dXp = Parameter(
        m,
        name="dXp",
        domain=[i, t],
        description="change of household consumption        [%]",
    )
    dXv = Parameter(
        m,
        name="dXv",
        domain=[i, t],
        description="change of investment demand            [%]",
    )
    dE = Parameter(
        m,
        name="dE",
        domain=[i, t],
        description="change of exports                      [%]",
    )
    dM = Parameter(
        m,
        name="dM",
        domain=[i, t],
        description="change of imports                      [%]",
    )
    dQ = Parameter(
        m,
        name="dQ",
        domain=[i, t],
        description="change of Armington's composite good   [%]",
    )
    dD = Parameter(
        m,
        name="dD",
        domain=[i, t],
        description="change of domestic good                [%]",
    )
    dSp = Parameter(
        m,
        name="dSp",
        domain=[t],
        description="change of private saving               [%]",
    )
    dTd = Parameter(
        m,
        name="dTd",
        domain=[t],
        description="change of direct tax                   [%]",
    )
    dTz = Parameter(
        m,
        name="dTz",
        domain=[j, t],
        description="change of production tax               [%]",
    )
    dTm = Parameter(
        m,
        name="dTm",
        domain=[i, t],
        description="change of import tariff                [%]",
    )
    dFF = Parameter(
        m,
        name="dFF",
        domain=[h, t],
        description="change of initial sectoral factor uses [%]",
    )
    dKK = Parameter(
        m,
        name="dKK",
        domain=[j, t],
        description="change of sectoral capital stock       [%]",
    )
    dII = Parameter(
        m,
        name="dII",
        domain=[j, t],
        description="change of sectoral investment          [%]",
    )
    dIII = Parameter(
        m,
        name="dIII",
        domain=[t],
        description="change of composite investment         [%]",
    )
    dCC = Parameter(
        m,
        name="dCC",
        domain=[t],
        description="change of utility                      [%]",
    )
    dpz = Parameter(
        m,
        name="dpz",
        domain=[j, t],
        description="change of gross output price           [%]",
    )
    dpd = Parameter(
        m,
        name="dpd",
        domain=[j, t],
        description="change of domestic good price          [%]",
    )
    dpm = Parameter(
        m,
        name="dpm",
        domain=[j, t],
        description="change of import price                 [%]",
    )
    dpe = Parameter(
        m,
        name="dpe",
        domain=[j, t],
        description="change of export price                 [%]",
    )
    dpq = Parameter(
        m,
        name="dpq",
        domain=[j, t],
        description="change of Armington's comp. good price [%]",
    )
    dpf = Parameter(
        m,
        name="dpf",
        domain=[h, j, t],
        description="change of factor price                 [%]",
    )
    dpy = Parameter(
        m,
        name="dpy",
        domain=[j, t],
        description="change of composite factor price       [%]",
    )
    depsilon = Parameter(
        m,
        name="depsilon",
        domain=[t],
        description="change of foreign exchange rate        [%]",
    )
    dpk = Parameter(
        m,
        name="dpk",
        domain=[t],
        description="change of capital good price           [%]",
    )

    # BAU growth rate
    gY0 = Parameter(
        m,
        name="gY0",
        domain=[j, t],
        description="growth of composite factor             [%]",
    )
    gF0 = Parameter(
        m,
        name="gF0",
        domain=[h, j, t],
        description="growth of factor input                 [%]",
    )
    gX0 = Parameter(
        m,
        name="gX0",
        domain=[i, j, t],
        description="growth of intermediate input           [%]",
    )
    gZ0 = Parameter(
        m,
        name="gZ0",
        domain=[j, t],
        description="growth of gross output                 [%]",
    )
    gXp0 = Parameter(
        m,
        name="gXp0",
        domain=[i, t],
        description="growth of household consumption        [%]",
    )
    gXv0 = Parameter(
        m,
        name="gXv0",
        domain=[i, t],
        description="growth of investment demand            [%]",
    )
    gE0 = Parameter(
        m,
        name="gE0",
        domain=[i, t],
        description="growth of exports                      [%]",
    )
    gM0 = Parameter(
        m,
        name="gM0",
        domain=[i, t],
        description="growth of imports                      [%]",
    )
    gQ0 = Parameter(
        m,
        name="gQ0",
        domain=[i, t],
        description="growth of Armington's composite good   [%]",
    )
    gD0 = Parameter(
        m,
        name="gD0",
        domain=[i, t],
        description="growth of domestic good                [%]",
    )
    gSp0 = Parameter(
        m,
        name="gSp0",
        domain=[t],
        description="growth of private saving               [%]",
    )
    gTd0 = Parameter(
        m,
        name="gTd0",
        domain=[t],
        description="growth of direct tax                   [%]",
    )
    gTz0 = Parameter(
        m,
        name="gTz0",
        domain=[j, t],
        description="growth of production tax               [%]",
    )
    gTm0 = Parameter(
        m,
        name="gTm0",
        domain=[i, t],
        description="growth of import tariff                [%]",
    )
    gFF0 = Parameter(
        m,
        name="gFF0",
        domain=[h, t],
        description="growth of initial sectoral factor uses [%]",
    )
    gKK0 = Parameter(
        m,
        name="gKK0",
        domain=[j, t],
        description="growth of sectoral capital stock       [%]",
    )
    gII0 = Parameter(
        m,
        name="gII0",
        domain=[j, t],
        description="growth of sectoral investment          [%]",
    )
    gIII0 = Parameter(
        m,
        name="gIII0",
        domain=[t],
        description="growth of composite investment         [%]",
    )
    gCC0 = Parameter(
        m,
        name="gCC0",
        domain=[t],
        description="growth of growth rate of CC            [%]",
    )

    # C/F growth rate
    gY1 = Parameter(
        m,
        name="gY1",
        domain=[j, t],
        description="growth of composite factor             [%]",
    )
    gF1 = Parameter(
        m,
        name="gF1",
        domain=[h, j, t],
        description="growth of factor input                 [%]",
    )
    gX1 = Parameter(
        m,
        name="gX1",
        domain=[i, j, t],
        description="growth of intermediate input           [%]",
    )
    gZ1 = Parameter(
        m,
        name="gZ1",
        domain=[j, t],
        description="growth of gross output                 [%]",
    )
    gXp1 = Parameter(
        m,
        name="gXp1",
        domain=[i, t],
        description="growth of household consumption        [%]",
    )
    gXv1 = Parameter(
        m,
        name="gXv1",
        domain=[i, t],
        description="growth of investment demand            [%]",
    )
    gE1 = Parameter(
        m,
        name="gE1",
        domain=[i, t],
        description="growth of exports                      [%]",
    )
    gM1 = Parameter(
        m,
        name="gM1",
        domain=[i, t],
        description="growth of imports                      [%]",
    )
    gQ1 = Parameter(
        m,
        name="gQ1",
        domain=[i, t],
        description="growth of Armington's composite good   [%]",
    )
    gD1 = Parameter(
        m,
        name="gD1",
        domain=[i, t],
        description="growth of domestic good                [%]",
    )
    gSp1 = Parameter(
        m,
        name="gSp1",
        domain=[t],
        description="growth of private saving               [%]",
    )
    gTd1 = Parameter(
        m,
        name="gTd1",
        domain=[t],
        description="growth of direct tax                   [%]",
    )
    gTz1 = Parameter(
        m,
        name="gTz1",
        domain=[j, t],
        description="growth of production tax               [%]",
    )
    gTm1 = Parameter(
        m,
        name="gTm1",
        domain=[i, t],
        description="growth of import tariff                [%]",
    )
    gFF1 = Parameter(
        m,
        name="gFF1",
        domain=[h, t],
        description="growth of initial sectoral factor uses [%]",
    )
    gKK1 = Parameter(
        m,
        name="gKK1",
        domain=[j, t],
        description="growth of sectoral capital stock       [%]",
    )
    gII1 = Parameter(
        m,
        name="gII1",
        domain=[j, t],
        description="growth of sectoral investment          [%]",
    )
    gIII1 = Parameter(
        m,
        name="gIII1",
        domain=[t],
        description="growth of composite investment         [%]",
    )
    gCC1 = Parameter(
        m,
        name="gCC1",
        domain=[t],
        description="growth of growth rate of CC            [%]",
    )

    # welfare
    EV = Parameter(
        m, name="EV", domain=[t], description="equivalent variations [current]"
    )
    EV_TTL = Parameter(
        m, name="EV_TTL", description="total EV [discounted Sum]"
    )

    dY[j, t].where[Y0[j, t]] = (Y1[j, t] / Y0[j, t] - 1) * 100
    dF[h, j, t].where[F0[h, j, t]] = (F1[h, j, t] / F0[h, j, t] - 1) * 100
    dX[i, j, t].where[X0[i, j, t]] = (X1[i, j, t] / X0[i, j, t] - 1) * 100
    dZ[j, t].where[Z0[j, t]] = (Z1[j, t] / Z0[j, t] - 1) * 100
    dXp[i, t].where[Xp0[i, t]] = (Xp1[i, t] / Xp0[i, t] - 1) * 100
    dXv[i, t].where[Xv0[i, t]] = (Xv1[i, t] / Xv0[i, t] - 1) * 100
    dE[i, t].where[E0[i, t]] = (E1[i, t] / E0[i, t] - 1) * 100
    dM[i, t].where[M0[i, t]] = (M1[i, t] / M0[i, t] - 1) * 100
    dQ[i, t].where[Q0[i, t]] = (Q1[i, t] / Q0[i, t] - 1) * 100
    dD[i, t].where[D0[i, t]] = (D1[i, t] / D0[i, t] - 1) * 100
    dSp[t].where[Sp0[t]] = (Sp1[t] / Sp0[t] - 1) * 100
    dTd[t].where[Td0[t]] = (Td1[t] / Td0[t] - 1) * 100
    dTz[j, t].where[Tz0[j, t]] = (Tz1[j, t] / Tz0[j, t] - 1) * 100
    dTm[i, t].where[Tm0[i, t]] = (Tm1[i, t] / Tm0[i, t] - 1) * 100
    dFF[h, t].where[FF0[h, t]] = (FF1[h, t] / FF0[h, t] - 1) * 100
    dII[j, t].where[II0[j, t]] = (II1[j, t] / II0[j, t] - 1) * 100
    dIII[t].where[III0[t]] = (III1[t] / III0[t] - 1) * 100
    dKK[j, t].where[KK0[j, t]] = (KK1[j, t] / KK0[j, t] - 1) * 100
    dCC[t].where[CC0[t]] = (CC1[t] / CC0[t] - 1) * 100
    dpz[j, t].where[pz0[j, t]] = (pz1[j, t] / pz0[j, t] - 1) * 100
    dpd[j, t].where[pd0[j, t]] = (pd1[j, t] / pd0[j, t] - 1) * 100
    dpm[j, t].where[pm0[j, t]] = (pm1[j, t] / pm0[j, t] - 1) * 100
    dpe[j, t].where[pe0[j, t]] = (pe1[j, t] / pe0[j, t] - 1) * 100
    dpq[j, t].where[pq0[j, t]] = (pq1[j, t] / pq0[j, t] - 1) * 100
    dpf[h, j, t].where[pf0[h, j, t]] = (pf1[h, j, t] / pf0[h, j, t] - 1) * 100
    dpy[j, t].where[py0[j, t]] = (py1[j, t] / py0[j, t] - 1) * 100
    depsilon[t].where[epsilon0[t]] = (epsilon1[t] / epsilon0[t] - 1) * 100
    dpk[t].where[pk0[t]] = (pk1[t] / pk0[t] - 1) * 100
    gY0[j, t.lead(1)].where[Y0[j, t]] = (Y0[j, t.lead(1)] / Y0[j, t] - 1) * 100
    gF0[h, j, t.lead(1)].where[F0[h, j, t]] = (
        F0[h, j, t.lead(1)] / F0[h, j, t] - 1
    ) * 100
    gX0[i, j, t.lead(1)].where[X0[i, j, t]] = (
        X0[i, j, t.lead(1)] / X0[i, j, t] - 1
    ) * 100
    gZ0[j, t.lead(1)].where[Z0[j, t]] = (Z0[j, t.lead(1)] / Z0[j, t] - 1) * 100
    gXp0[i, t.lead(1)].where[Xp0[i, t]] = (
        Xp0[i, t.lead(1)] / Xp0[i, t] - 1
    ) * 100
    gXv0[i, t.lead(1)].where[Xv0[i, t]] = (
        Xv0[i, t.lead(1)] / Xv0[i, t] - 1
    ) * 100
    gE0[i, t.lead(1)].where[E0[i, t]] = (E0[i, t.lead(1)] / E0[i, t] - 1) * 100
    gM0[i, t.lead(1)].where[M0[i, t]] = (M0[i, t.lead(1)] / M0[i, t] - 1) * 100
    gQ0[i, t.lead(1)].where[Q0[i, t]] = (Q0[i, t.lead(1)] / Q0[i, t] - 1) * 100
    gD0[i, t.lead(1)].where[D0[i, t]] = (D0[i, t.lead(1)] / D0[i, t] - 1) * 100
    gSp0[t.lead(1)].where[Sp0[t]] = (Sp0[t.lead(1)] / Sp0[t] - 1) * 100
    gTd0[t.lead(1)].where[Td0[t]] = (Td0[t.lead(1)] / Td0[t] - 1) * 100
    gTz0[j, t.lead(1)].where[Tz0[j, t]] = (
        Tz0[j, t.lead(1)] / Tz0[j, t] - 1
    ) * 100
    gTm0[i, t.lead(1)].where[Tm0[i, t]] = (
        Tm0[i, t.lead(1)] / Tm0[i, t] - 1
    ) * 100
    gFF0[h, t.lead(1)].where[FF0[h, t]] = (
        FF0[h, t.lead(1)] / FF0[h, t] - 1
    ) * 100
    gII0[j, t.lead(1)].where[II0[j, t]] = (
        II0[j, t.lead(1)] / II0[j, t] - 1
    ) * 100
    gIII0[t.lead(1)].where[III0[t]] = (III0[t.lead(1)] / III0[t] - 1) * 100
    gKK0[j, t.lead(1)].where[KK0[j, t]] = (
        KK0[j, t.lead(1)] / KK0[j, t] - 1
    ) * 100
    gCC0[t.lead(1)].where[CC0[t]] = (CC0[t.lead(1)] / CC0[t] - 1) * 100
    gY1[j, t.lead(1)].where[Y1[j, t]] = (Y1[j, t.lead(1)] / Y1[j, t] - 1) * 100
    gF1[h, j, t.lead(1)].where[F1[h, j, t]] = (
        F1[h, j, t.lead(1)] / F1[h, j, t] - 1
    ) * 100
    gX1[i, j, t.lead(1)].where[X1[i, j, t]] = (
        X1[i, j, t.lead(1)] / X1[i, j, t] - 1
    ) * 100
    gZ1[j, t.lead(1)].where[Z1[j, t]] = (Z1[j, t.lead(1)] / Z1[j, t] - 1) * 100
    gXp1[i, t.lead(1)].where[Xp1[i, t]] = (
        Xp1[i, t.lead(1)] / Xp1[i, t] - 1
    ) * 100
    gXv1[i, t.lead(1)].where[Xv1[i, t]] = (
        Xv1[i, t.lead(1)] / Xv1[i, t] - 1
    ) * 100
    gE1[i, t.lead(1)].where[E1[i, t]] = (E1[i, t.lead(1)] / E1[i, t] - 1) * 100
    gM1[i, t.lead(1)].where[M1[i, t]] = (M1[i, t.lead(1)] / M1[i, t] - 1) * 100
    gQ1[i, t.lead(1)].where[Q1[i, t]] = (Q1[i, t.lead(1)] / Q1[i, t] - 1) * 100
    gD1[i, t.lead(1)].where[D1[i, t]] = (D1[i, t.lead(1)] / D1[i, t] - 1) * 100
    gSp1[t.lead(1)].where[Sp1[t]] = (Sp1[t.lead(1)] / Sp1[t] - 1) * 100
    gTd1[t.lead(1)].where[Td1[t]] = (Td1[t.lead(1)] / Td1[t] - 1) * 100
    gTz1[j, t.lead(1)].where[Tz1[j, t]] = (
        Tz1[j, t.lead(1)] / Tz1[j, t] - 1
    ) * 100
    gTm1[i, t.lead(1)].where[Tm1[i, t]] = (
        Tm1[i, t.lead(1)] / Tm1[i, t] - 1
    ) * 100
    gFF1[h, t.lead(1)].where[FF1[h, t]] = (
        FF1[h, t.lead(1)] / FF1[h, t] - 1
    ) * 100
    gII1[j, t.lead(1)].where[II1[j, t]] = (
        II1[j, t.lead(1)] / II1[j, t] - 1
    ) * 100
    gIII1[t.lead(1)].where[III1[t]] = (III1[t.lead(1)] / III1[t] - 1) * 100
    gKK1[j, t.lead(1)].where[KK1[j, t]] = (
        KK1[j, t.lead(1)] / KK1[j, t] - 1
    ) * 100
    gCC1[t.lead(1)].where[CC1[t]] = (CC1[t.lead(1)] / CC1[t] - 1) * 100

    # Welfare measure: Hicksian equivalent variations ---------------
    EV[t] = (CC1[t] - CC0[t]) / a / Product(i, (alpha[i] / 1) ** alpha[i])
    EV_TTL.assign = Sum(t, EV[t] / (1 + ror) ** (Ord(t) - 1))

    print("EV_TTL: ", round(EV_TTL.records.value[0], 3))


if __name__ == "__main__":
    main()
