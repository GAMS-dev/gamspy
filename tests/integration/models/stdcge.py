"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_stdcge.html
## LICENSETYPE: Demo
## MODELTYPE: NLP
## KEYWORDS: nonlinear programming, general equilibrium model, social accounting, matrix, utility maximization problem

Hosoe, N, Gasawa, K, and Hashimoto, H
Handbook of Computible General Equilibrium Modeling
University of Tokyo Press, Tokyo, Japan, 2004
"""

import math
from pathlib import Path

from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Options,
    Parameter,
    Product,
    Set,
    Sum,
    Variable,
)

m = Container()

u = Set(m, "u", domain=["*"], description="SAM entry")
i = Set(m, "i", domain=u, description="goods")
h = Set(m, "h", domain=u, description="factor")
v = Alias(m, "v", alias_with=u)
j = Alias(m, "j", alias_with=i)
k = Alias(m, "k", alias_with=h)
SAM = Parameter(m, "SAM", domain=[u, v], description="social accounting matrix")
Y0 = Parameter(m, "Y0", domain=j, description="composite factor")
F0 = Parameter(
    m,
    "F0",
    domain=[h, j],
    description="the h-th factor input by the j-th firm",
)
X0 = Parameter(m, "X0", domain=[i, j], description="intermediate input")
Z0 = Parameter(m, "Z0", domain=j, description="output of the j-th good")
Xp0 = Parameter(
    m,
    "Xp0",
    domain=i,
    description="household consumption of the i-th good ",
)
Xg0 = Parameter(m, "Xg0", domain=i, description="government consumption")
Xv0 = Parameter(m, "Xv0", domain=i, description="investment demand")
E0 = Parameter(m, "E0", domain=i, description="exports")
M0 = Parameter(m, "M0", domain=i, description="imports")
Q0 = Parameter(m, "Q0", domain=i, description="Armington's composite good")
D0 = Parameter(m, "D0", domain=i, description="domestic good")
Sp0 = Parameter(m, "Sp0", description="private saving")
Sg0 = Parameter(m, "Sg0", description="government saving")
Td0 = Parameter(m, "Td0", description="direct tax")
Tz0 = Parameter(m, "Tz0", domain=j, description="production tax")
Tm0 = Parameter(m, "Tm0", domain=j, description="import tariff")
FF = Parameter(m, "FF", domain=h, description="factor endowment of the h-th factor")
Sf = Parameter(m, "Sf", description="foreign saving in US dollars")
pWe = Parameter(m, "pWe", domain=i, description="export price in US dollars")
pWm = Parameter(m, "pWm", domain=i, description="import price in US dollars")
tauz = Parameter(m, "tauz", domain=i, description="production tax rate")
taum = Parameter(m, "taum", domain=i, description="import tariff rate")
sigma = Parameter(m, "sigma", domain=i, description="elasticity of substitution")
psi = Parameter(m, "psi", domain=i, description="elasticity of transformation")
eta = Parameter(m, "eta", domain=i, description="substitution elasticity parameter")
phi = Parameter(m, "phi", domain=i, description="transformation elasticity parameter")
alpha = Parameter(m, "alpha", domain=i, description="share parameter in utility func.")
beta = Parameter(
    m,
    "beta",
    domain=[h, j],
    description="share parameter in production func.",
)
b = Parameter(m, "b", domain=j, description="scale parameter in production func.")
ax = Parameter(
    m,
    "ax",
    domain=[i, j],
    description="intermediate input requirement coeff.",
)
ay = Parameter(m, "ay", domain=j, description="composite fact. input req. coeff.")
mu = Parameter(m, "mu", domain=i, description="government consumption share")
lambda_renamed = Parameter(m, "lambda", domain=i, description="investment demand share")
deltam = Parameter(m, "deltam", domain=i, description="share par. in Armington func.")
deltad = Parameter(m, "deltad", domain=i, description="share par. in Armington func.")
gamma = Parameter(m, "gamma", domain=i, description="scale par. in Armington func.")
xid = Parameter(m, "xid", domain=i, description="share par. in transformation func.")
xie = Parameter(m, "xie", domain=i, description="share par. in transformation func.")
theta = Parameter(
    m,
    "theta",
    domain=i,
    description="scale par. in transformation func.",
)
ssp = Parameter(
    m,
    "ssp",
    domain=[],
    description="average propensity for private saving",
)
ssg = Parameter(m, "ssg", description="average propensity for gov. saving")
taud = Parameter(m, "taud", description="direct tax rate")
Y = Variable(m, "Y", domain=j, description="composite factor", type="free")
F = Variable(
    m,
    "F",
    domain=[h, j],
    description="the h-th factor input by the j-th firm",
    type="free",
)
X = Variable(m, "X", domain=[i, j], description="intermediate input", type="free")
Z = Variable(m, "Z", domain=j, description="output of the j-th good", type="free")
Xp = Variable(
    m,
    "Xp",
    domain=i,
    description="household consumption of the i-th good",
    type="free",
)
Xg = Variable(m, "Xg", domain=i, description="government consumption", type="free")
Xv = Variable(m, "Xv", domain=i, description="investment demand", type="free")
E = Variable(m, "E", domain=i, description="exports", type="free")
M = Variable(m, "M", domain=i, description="imports", type="free")
Q = Variable(
    m,
    "Q",
    domain=i,
    description="Armington's composite good",
    type="free",
)
D = Variable(m, "D", domain=i, description="domestic good", type="free")
pf = Variable(m, "pf", domain=h, description="the h-th factor price", type="free")
py = Variable(m, "py", domain=j, description="composite factor price", type="free")
pz = Variable(
    m,
    "pz",
    domain=j,
    description="supply price of the i-th good",
    type="free",
)
pq = Variable(
    m,
    "pq",
    domain=i,
    description="Armington's composite good price",
    type="free",
)
pe = Variable(
    m,
    "pe",
    domain=i,
    description="export price in local currency",
    type="free",
)
pm = Variable(
    m,
    "pm",
    domain=i,
    description="import price in local currency",
    type="free",
)
pd = Variable(
    m,
    "pd",
    domain=i,
    description="the i-th domestic good price",
    type="free",
)
epsilon = Variable(m, "epsilon", description="exchange rate", type="free")
Sp = Variable(m, "Sp", description="private saving", type="free")
Sg = Variable(m, "Sg", description="government saving", type="free")
Td = Variable(m, "Td", description="direct tax", type="free")
Tz = Variable(m, "Tz", domain=j, description="production tax", type="free")
Tm = Variable(m, "Tm", domain=i, description="import tariff", type="free")
UU = Variable(m, "UU", description="utility [fictitious]", type="free")
eqpy = Equation(m, "eqpy", domain=j, description="composite factor agg. func.")
eqF = Equation(m, "eqF", domain=[h, j], description="factor demand function")
eqX = Equation(m, "eqX", domain=[i, j], description="intermediate demand function")
eqY = Equation(m, "eqY", domain=j, description="composite factor demand function")
eqpzs = Equation(m, "eqpzs", domain=j, description="unit cost function")
eqTd = Equation(m, "eqTd", description="direct tax revenue function")
eqTz = Equation(m, "eqTz", domain=j, description="production tax revenue function")
eqTm = Equation(m, "eqTm", domain=i, description="import tariff revenue function")
eqXg = Equation(m, "eqXg", domain=i, description="government demand function")
eqXv = Equation(m, "eqXv", domain=i, description="investment demand function")
eqSp = Equation(m, "eqSp", description="private saving function")
eqSg = Equation(m, "eqSg", description="government saving function")
eqXp = Equation(m, "eqXp", domain=i, description="household demand function")
eqpe = Equation(m, "eqpe", domain=i, description="world export price equation")
eqpm = Equation(m, "eqpm", domain=i, description="world import price equation")
eqepsilon = Equation(m, "eqepsilon", description="balance of payments")
eqpqs = Equation(m, "eqpqs", domain=i, description="Armington function")
eqM = Equation(m, "eqM", domain=i, description="import demand function")
eqD = Equation(m, "eqD", domain=i, description="domestic good demand function")
eqpzd = Equation(m, "eqpzd", domain=i, description="transformation function")
eqDs = Equation(m, "eqDs", domain=i, description="domestic good supply function")
eqE = Equation(m, "eqE", domain=i, description="export supply function")
eqpqd = Equation(
    m,
    "eqpqd",
    domain=i,
    description="market clearing cond. for comp. good",
)
eqpf = Equation(m, "eqpf", domain=h, description="factor market clearing condition")
obj = Equation(m, "obj", description="utility function [fictitious]")
m.loadRecordsFromGdx(str(Path(__file__).parent.absolute()) + "/stdcge.gdx")

Td0[...] = SAM["GOV", "HOH"]
Tz0[j] = SAM["IDT", j]
Tm0[j] = SAM["TRF", j]
F0[h, j] = SAM[h, j]
Y0[j] = Sum(h, F0[h, j])
X0[i, j] = SAM[i, j]
Z0[j] = Y0[j] + Sum(i, X0[i, j])
M0[i] = SAM["EXT", i]
tauz[j] = Tz0[j] / Z0[j]
taum[j] = Tm0[j] / M0[j]
Xp0[i] = SAM[i, "HOH"]
FF[h] = SAM["HOH", h]
Xg0[i] = SAM[i, "GOV"]
Xv0[i] = SAM[i, "INV"]
E0[i] = SAM[i, "EXT"]
Q0[i] = ((Xp0[i] + Xg0[i]) + Xv0[i]) + Sum(j, X0[i, j])
D0[i] = ((1 + tauz[i]) * Z0[i]) - E0[i]
Sp0[...] = SAM["INV", "HOH"]
Sg0[...] = SAM["INV", "GOV"]
Sf[...] = SAM["INV", "EXT"]
pWe[i] = 1
pWm[i] = 1
print(Y0.records)
print(F0.records)
print(X0.records)
print(Z0.records)
print(Xp0.records)
print(Xg0.records)
print(Xv0.records)
print(E0.records)
print(M0.records)
print(Q0.records)
print(D0.records)
print(Sp0.records)
print(Sg0.records)
print(Td0.records)
print(Tz0.records)
print(Tm0.records)
print(FF.records)
print(Sf.records)
print(tauz.records)
print(taum.records)
sigma[i] = 2
psi[i] = 2
eta[i] = (sigma[i] - 1) / sigma[i]
phi[i] = (psi[i] + 1) / psi[i]
alpha[i] = Xp0[i] / Sum(j, Xp0[j])
beta[h, j] = F0[h, j] / Sum(k, F0[k, j])
b[j] = Y0[j] / Product(h, (F0[h, j] ** beta[h, j]))
ax[i, j] = X0[i, j] / Z0[j]
ay[j] = Y0[j] / Z0[j]
mu[i] = Xg0[i] / Sum(j, Xg0[j])
lambda_renamed[i] = Xv0[i] / ((Sp0[...] + Sg0[...]) + Sf[...])
deltam[i] = ((1 + taum[i]) * (M0[i] ** (1 - eta[i]))) / (
    ((1 + taum[i]) * (M0[i] ** (1 - eta[i]))) + (D0[i] ** (1 - eta[i]))
)
deltad[i] = (D0[i] ** (1 - eta[i])) / (
    ((1 + taum[i]) * (M0[i] ** (1 - eta[i]))) + (D0[i] ** (1 - eta[i]))
)
gamma[i] = Q0[i] / (
    ((deltam[i] * (M0[i] ** eta[i])) + (deltad[i] * (D0[i] ** eta[i]))) ** (1 / eta[i])
)
xie[i] = (E0[i] ** (1 - phi[i])) / ((E0[i] ** (1 - phi[i])) + (D0[i] ** (1 - phi[i])))
xid[i] = (D0[i] ** (1 - phi[i])) / ((E0[i] ** (1 - phi[i])) + (D0[i] ** (1 - phi[i])))
theta[i] = Z0[i] / (
    ((xie[i] * (E0[i] ** phi[i])) + (xid[i] * (D0[i] ** phi[i]))) ** (1 / phi[i])
)
ssp[...] = Sp0[...] / Sum(h, FF[h])
ssg[...] = Sg0[...] / ((Td0[...] + Sum(j, Tz0[j])) + Sum(j, Tm0[j]))
taud[...] = Td0[...] / Sum(h, FF[h])
print(alpha.records)
print(beta.records)
print(b.records)
print(ax.records)
print(ay.records)
print(mu.records)
print(lambda_renamed.records)
print(deltam.records)
print(deltad.records)
print(gamma.records)
print(xie.records)
print(xid.records)
print(theta.records)
print(ssp.records)
print(ssg.records)
print(taud.records)
eqpy[j] = Y[j] == (b[j] * Product(h, (F[h, j] ** beta[h, j])))
eqF[h, j] = F[h, j] == (((beta[h, j] * py[j]) * Y[j]) / pf[h])
eqX[i, j] = X[i, j] == (ax[i, j] * Z[j])
eqY[j] = Y[j] == (ay[j] * Z[j])
eqpzs[j] = pz[j] == ((ay[j] * py[j]) + Sum(i, (ax[i, j] * pq[i])))
eqTd[...] = Td[...] == (taud[...] * Sum(h, (pf[h] * FF[h])))
eqTz[j] = Tz[j] == ((tauz[j] * pz[j]) * Z[j])
eqTm[i] = Tm[i] == ((taum[i] * pm[i]) * M[i])
eqXg[i] = Xg[i] == (
    (mu[i] * (((Td[...] + Sum(j, Tz[j])) + Sum(j, Tm[j])) - Sg[...])) / pq[i]
)
eqXv[i] = Xv[i] == (
    (lambda_renamed[i] * ((Sp[...] + Sg[...]) + (epsilon[...] * Sf[...]))) / pq[i]
)
eqSp[...] = Sp[...] == (ssp[...] * Sum(h, (pf[h] * FF[h])))
eqSg[...] = Sg[...] == (ssg[...] * ((Td[...] + Sum(j, Tz[j])) + Sum(j, Tm[j])))
eqXp[i] = Xp[i] == (
    (alpha[i] * ((Sum(h, (pf[h] * FF[h])) - Sp[...]) - Td[...])) / pq[i]
)
eqpe[i] = pe[i] == (epsilon[...] * pWe[i])
eqpm[i] = pm[i] == (epsilon[...] * pWm[i])
eqepsilon[...] = (Sum(i, (pWe[i] * E[i])) + Sf[...]) == Sum(i, (pWm[i] * M[i]))
eqpqs[i] = Q[i] == (
    gamma[i]
    * (
        ((deltam[i] * (M[i] ** eta[i])) + (deltad[i] * (D[i] ** eta[i])))
        ** (1 / eta[i])
    )
)
eqM[i] = M[i] == (
    (
        ((((gamma[i] ** eta[i]) * deltam[i]) * pq[i]) / ((1 + taum[i]) * pm[i]))
        ** (1 / (1 - eta[i]))
    )
    * Q[i]
)
eqD[i] = D[i] == (
    (((((gamma[i] ** eta[i]) * deltad[i]) * pq[i]) / pd[i]) ** (1 / (1 - eta[i])))
    * Q[i]
)
eqpzd[i] = Z[i] == (
    theta[i]
    * (((xie[i] * (E[i] ** phi[i])) + (xid[i] * (D[i] ** phi[i]))) ** (1 / phi[i]))
)
eqE[i] = E[i] == (
    (
        (((((theta[i] ** phi[i]) * xie[i]) * (1 + tauz[i])) * pz[i]) / pe[i])
        ** (1 / (1 - phi[i]))
    )
    * Z[i]
)
eqDs[i] = D[i] == (
    (
        (((((theta[i] ** phi[i]) * xid[i]) * (1 + tauz[i])) * pz[i]) / pd[i])
        ** (1 / (1 - phi[i]))
    )
    * Z[i]
)
eqpqd[i] = Q[i] == (((Xp[i] + Xg[i]) + Xv[i]) + Sum(j, X[i, j]))
eqpf[h] = Sum(j, F[h, j]) == FF[h]
obj[...] = UU[...] == Product(i, (Xp[i] ** alpha[i]))
Y.l[j] = Y0[j]
F.l[h, j] = F0[h, j]
X.l[i, j] = X0[i, j]
Z.l[j] = Z0[j]
Xp.l[i] = Xp0[i]
Xg.l[i] = Xg0[i]
Xv.l[i] = Xv0[i]
E.l[i] = E0[i]
M.l[i] = M0[i]
Q.l[i] = Q0[i]
D.l[i] = D0[i]
pf.l[h] = 1
py.l[j] = 1
pz.l[j] = 1
pq.l[i] = 1
pe.l[i] = 1
pm.l[i] = 1
pd.l[i] = 1
epsilon.l[...] = 1
Sp.l[...] = Sp0[...]
Sg.l[...] = Sg0[...]
Td.l[...] = Td0[...]
Tz.l[j] = Tz0[j]
Tm.l[i] = Tm0[i]
Y.lo[j] = 1e-05
F.lo[h, j] = 1e-05
X.lo[i, j] = 1e-05
Z.lo[j] = 1e-05
Xp.lo[i] = 1e-05
Xg.lo[i] = 1e-05
Xv.lo[i] = 1e-05
E.lo[i] = 1e-05
M.lo[i] = 1e-05
Q.lo[i] = 1e-05
D.lo[i] = 1e-05
pf.lo[h] = 1e-05
py.lo[j] = 1e-05
pz.lo[j] = 1e-05
pq.lo[i] = 1e-05
pe.lo[i] = 1e-05
pm.lo[i] = 1e-05
pd.lo[i] = 1e-05
epsilon.lo[...] = 1e-05
Sp.lo[...] = 1e-05
Sg.lo[...] = 1e-05
Td.lo[...] = 1e-05
Tz.lo[j] = 0
Tm.lo[i] = 0
pf.fx["LAB"] = 1
stdcge = Model(
    m,
    name="stdcge",
    equations=[
        eqpy,
        eqF,
        eqX,
        eqY,
        eqpzs,
        eqTd,
        eqTz,
        eqTm,
        eqXg,
        eqXv,
        eqSp,
        eqSg,
        eqXp,
        eqpe,
        eqpm,
        eqepsilon,
        eqpqs,
        eqM,
        eqD,
        eqpzd,
        eqDs,
        eqE,
        eqpqd,
        eqpf,
        obj,
    ],
    problem="NLP",
    sense="MAX",
    objective=UU,
)
stdcge.solve(solver="conopt")
assert math.isclose(stdcge.objective_value, 25.508490012515818), stdcge.objective_value

taum[i] = 0

stdcge = Model(
    m,
    name="stdcge",
    equations=[
        eqpy,
        eqF,
        eqX,
        eqY,
        eqpzs,
        eqTd,
        eqTz,
        eqTm,
        eqXg,
        eqXv,
        eqSp,
        eqSg,
        eqXp,
        eqpe,
        eqpm,
        eqepsilon,
        eqpqs,
        eqM,
        eqD,
        eqpzd,
        eqDs,
        eqE,
        eqpqd,
        eqpf,
        obj,
    ],
    problem="NLP",
    sense="MAX",
    objective=UU,
)
stdcge.solve(solver="conopt", options=Options(basis_detection_threshold=1))
assert math.isclose(stdcge.objective_value, 26.092634381288686), stdcge.objective_value
