"""
## LICENSETYPE: Demo
## MODELTYPE: EMP
## KEYWORDS: Extended Mathematical Programming, EMP, RESHOP

Simple example to illustrate how to model the CVaR as a support function.
Author: Olivier Huber <oli.huber@gmail.com>
"""

import sys

import numpy as np

import gamspy as gp
import gamspy.math as gpm

# START PROBLEM DATA DEFINITION
nSamples = 950

m = gp.Container(output=sys.stdout)
i = gp.Set(m, name="i", records=range(nSamples), description="realizations")
j = gp.Set(m, name="j", records=range(2), description="dimension of x")
k = gp.Set(m, name="k", records=range(4), description="distributions")

xi_data = []
means = [250, 125, 2500, 40000]
stddevs = [75, 62.5125, 2500, 40000]
for k_, mean in enumerate(means):
    xi_data.extend(
        zip(
            [k_] * len(i),
            range(nSamples),
            np.random.normal(loc=mean, scale=stddevs[k_], size=len(i)),
        )
    )
xi = gp.Parameter(m, name="xi", domain=[k, i], records=xi_data)
tail = gp.Parameter(m, name="tail", records=1 - 0.05)
# END PROBLEM DATA DEFINITION

# START OPTIMIZATION PROBLEM DEFINITION
phi = gp.Variable(m, name="phi", domain=i)
x = gp.Variable(m, name="x", domain=j)

x.lo["0"] = 0.1
x.up["0"] = 0.2
x.lo["1"] = 0.1
x.up["1"] = 0.6
x.l[j] = x.lo[j]

defphi = gp.Equation(m, name="defphi", domain=i)
defphi[i] = phi[i] == 4 * xi["0", i] / (x["0"] * x["1"] * xi["3", i]) + 4 * xi[
    "1", i
] * x["1"] / (gpm.sqr(x["0"]) * xi["3", i]) + gpm.sqr(xi["2", i]) / (
    gpm.sqr(x["0"]) * gpm.sqr(xi["3", i])
)
# END OPTIMIZATION PROBLEM DEFINITION


def reset(sym):
    sym.l[sym.domain] = 0
    sym.m[sym.domain] = 0


def ReSHOPAnnotation(m, s):
    return m.addGamsCode("EmbeddedCode ReSHOP:\n" + s + "\nendEmbeddedCode")


superquantile = gp.Model(
    m, name="superquantile", equations=[defphi], problem="emp"
)

# Default solve
ReSHOPAnnotation(
    m,
    """
deffn phi(i) defphi(i)
cvar: MP("cvarup", phi(i), tail=tail)
main: min cvar.valfn x(j)
""",
)
reset(x)
superquantile.solve(output=sys.stdout, solver="reshop")

# Solve via Fenchel dual
ReSHOPAnnotation(
    m,
    """
deffn phi(i) defphi(i)
cvar: MP("cvarup", phi(i), tail=tail)
main: min cvar.dual().valfn x(j)
""",
)
reset(x)
superquantile.solve(output=sys.stdout, solver="reshop")

# Solve as kkt conditions
ReSHOPAnnotation(
    m,
    """
deffn phi(i) defphi(i)
cvar: MP("cvarup", phi(i), tail=tail)
main: min cvar.objfn x(j)
nash_vi_kkt: vi main.kkt() cvar.kkt()
""",
)
reset(x)
df = superquantile.solve(output=sys.stdout, solver="reshop")
