"""
Antalya Forestry Model - Steady State (TFORSS)

This model finds the best management plan for new forests in a steady state
condition.


Bergendorff, H, Glenshaw, P, and Meeraus, A, The Planning of Investment
Programs in the Paper Industry. Tech. rep., The World Bank, 1980.

Keywords: linear programming, forestry, scenario analysis, investment planning,
          forest management planning
"""

from pathlib import Path
from gamspy import (
    Container,
    Model,
    Sum,
    Equation,
    Variable,
    Set,
    Parameter,
    Ord,
)
from gamspy.math import power, Round
import numpy as np


def main():
    cont = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/tforss.gdx",
    )

    # Sets
    c, cf, cl, s, k, at, p, m = cont.getSymbols(
        ["c", "cf", "cl", "s", "k", "at", "p", "m"]
    )

    # Parameters
    scd, land, ymf, a, b, pc, pd, nu, age = cont.getSymbols(
        [
            "scd",
            "land",
            "ymf",
            "a",
            "b",
            "pc",
            "pd",
            "nu",
            "age",
        ]
    )

    # Scalar
    mup, muc, life, rho = cont.getSymbols(
        [
            "mup",
            "muc",
            "life",
            "rho",
        ]
    )

    age[at] = 10 * Ord(at)

    # Model Definition #

    # Equation
    lbal = Equation(
        cont, name="lbal", type="eq", domain=[cl], description="log balances"
    )
    bal = Equation(
        cont,
        name="bal",
        type="eq",
        domain=[c],
        description="material balances of wood processing",
    )
    cap = Equation(
        cont,
        name="cap",
        type="eq",
        domain=[m],
        description="wood processing capacities",
    )
    landc = Equation(
        cont,
        name="landc",
        type="eq",
        domain=[s, k],
        description="land availability constraint",
    )
    ainvc = Equation(
        cont, name="ainvc", type="eq", description="investment cost"
    )
    aproc = Equation(cont, name="aproc", type="eq", description="process cost")
    asales = Equation(
        cont, name="asales", type="eq", description="sales revenue"
    )
    acutc = Equation(cont, name="acutc", type="eq", description="cutting cost")
    aplnt = Equation(
        cont, name="aplnt", type="eq", description="planting cost"
    )
    benefit = Equation(cont, name="benefit", type="eq")

    # Variable
    v = Variable(
        cont,
        name="v",
        type="positive",
        domain=[s, k, at],
        description="management of new forest   (1000ha per year)",
    )
    r = Variable(
        cont,
        name="r",
        domain=[c],
        description="supply of logs to industry (1000m3 per year)",
    )
    z = Variable(
        cont,
        name="z",
        type="positive",
        domain=[p],
        description="process level        (1000m3 input per year)",
    )
    h = Variable(
        cont,
        name="h",
        domain=[m],
        description="capacity             (1000m3 input per year)",
    )
    x = Variable(
        cont,
        name="x",
        type="positive",
        domain=[c],
        description="final shipments        (1000 units per year)",
    )
    phik = Variable(
        cont,
        name="phik",
        description="investment cost           (1000us$ per year)",
    )
    phir = Variable(
        cont,
        name="phir",
        description="process cost              (1000us$ per year)",
    )
    phix = Variable(
        cont,
        name="phix",
        description="sales revenue             (1000us$ per year)",
    )
    phil = Variable(
        cont,
        name="phil",
        description="cutting cost              (1000us$ per year)",
    )
    phip = Variable(
        cont,
        name="phip",
        description="planting cost             (1000us$ per year)",
    )
    phi = Variable(
        cont,
        name="phi",
        description="total benefits             (discounted cost)",
    )

    lbal[cl] = r[cl] == Sum([s, k, at], ymf[at, k, s, cl] * v[s, k, at])

    bal[c] = Sum(p, a[c, p] * z[p]) + r[c].where[cl[c]] >= x[c].where[cf[c]]

    cap[m] = Sum(p, b[m, p] * z[p]) == h[m]

    landc[s, k] = Sum(at, v[s, k, at] * age[at]) <= land[s] * scd[k]

    ainvc.definition = phik == rho / (1 - power((1 + rho), (-life))) * Sum(
        m, nu[m] * h[m]
    )

    aproc.definition = phir == Sum(p, pc[p] * z[p])

    asales.definition = phix == Sum(cf, pd[cf] * x[cf])

    acutc.definition = phil == muc * Sum(cl, r[cl])

    aplnt.definition = phip == mup * Sum(
        [s, k, at], v[s, k, at] * (1 + rho) ** age[at]
    )

    benefit.definition = phi == phix - phik - phir - phil - phip

    # Model definition
    forest = Model(cont, name="forest", equations="all")

    # Case Selection and Report Definitions
    rhoset = Set(
        cont, name="rhoset", records=["rho-03", "rho-05", "rho-07", "rho-10"]
    )

    landcl = Parameter(
        cont, name="landcl", domain=[s, k], description="clean level of landc"
    )
    rep = Parameter(
        cont,
        name="rep",
        domain=[cl, rhoset],
        description="summary report on log supply      (1000m3 per year)",
    )
    reprp = Parameter(
        cont,
        name="reprp",
        domain=[s, k, rhoset],
        description="summary report on rotation period           (years)",
    )
    repsp = Parameter(
        cont,
        name="repsp",
        domain=[s, k, rhoset],
        description="summary report on shadow price of land (us$ per ha)",
    )
    rhoval = Parameter(
        cont,
        name="rhoval",
        domain=[rhoset],
        records=np.array([0.03, 0.05, 0.07, 0.1]),
    )

    for case, _ in rhoset.records.itertuples(index=False):
        rho.assign = rhoval[case]
        cont.solve(forest, problem="LP", sense="max", objective_variable=phi)
        landcl[s, k] = Round(landc.l[s, k], 3)
        rep[cl, case] = r.l[cl]
        reprp[s, k, case] = (landcl[s, k] / Sum(at, v.l[s, k, at])).where[
            landcl[s, k]
        ]
        repsp[s, k, case] = landc.m[s, k]

    print("Summary report on log supply (1000m3 per year):")
    print(rep.pivot().round(3))
    print()

    print("Summary report on rotation period (years)  :")
    print(reprp.pivot().round(3))
    print()

    print("Summary report on shadow price of land (us$ per ha)  :")
    print(repsp.pivot().round(3))


if __name__ == "__main__":
    main()