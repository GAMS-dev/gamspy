"""
Immunization models

Immunization.gms: Immunization models.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 4.4
Last modified: Apr 2008.
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
    Card,
    Number,
    Sense,
)
import gamspy.math as gams_math
from gamspy.math import sqr
import pandas as pd
import numpy as np


def main():
    # Define container
    m = Container()

    # Bond data. Prices, coupons and maturities from the Danish market
    bond_data_recs = pd.DataFrame(
        np.array(
            [
                [112.35, 2006, 8],
                [105.33, 2003, 8],
                [111.25, 2007, 7],
                [107.30, 2004, 7],
                [107.62, 2011, 6],
                [106.68, 2009, 6],
                [101.93, 2002, 6],
                [101.30, 2005, 5],
                [101.61, 2003, 5],
                [100.06, 2002, 4],
            ]
        ),
        columns=["Price", "Maturity", "Coupon"],
        index=[
            "DS-8-06",
            "DS-8-03",
            "DS-7-07",
            "DS-7-04",
            "DS-6-11",
            "DS-6-09",
            "DS-6-02",
            "DS-5-05",
            "DS-5-03",
            "DS-4-02",
        ],
    )

    bond_data_recs = bond_data_recs.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )

    # SETS #
    Time = Set(
        m,
        name="Time",
        records=[str(i) for i in range(2001, 2012)],
        description="Time periods",
    )

    Bonds = Set(
        m,
        name="Bonds",
        records=[
            "DS-8-06",
            "DS-8-03",
            "DS-7-07",
            "DS-7-04",
            "DS-6-11",
            "DS-6-09",
            "DS-6-02",
            "DS-5-05",
            "DS-5-03",
            "DS-4-02",
        ],
        description="Bonds universe",
    )

    # ALIASES #
    t = Alias(m, name="t", alias_with=Time)
    i = Alias(m, name="i", alias_with=Bonds)

    # ALIAS (Time, t, t1, t2)

    # SCALARS #
    Now = Parameter(m, name="Now", description="Current year")
    Horizon = Parameter(m, name="Horizon", description="End of the Horizon")

    Now.assign = 2001
    Horizon.assign = Card(t) - 1

    # PARAMETER #
    tau = Parameter(m, name="tau", domain=[t], description="Time in years")

    # Note: time starts from 0
    tau[t] = Ord(t) - 1

    Coupon = Parameter(m, name="Coupon", domain=[i], description="Coupons")
    Maturity = Parameter(
        m, name="Maturity", domain=[i], description="Maturities"
    )
    F = Parameter(m, name="F", domain=[t, i], description="Cashflows")
    BondData = Parameter(
        m, name="BondData", domain=[i, "*"], records=bond_data_recs
    )
    Liability = Parameter(
        m, name="Liability", domain=[t], description="Stream of liabilities"
    )

    # Copy/transform data. Note division by 100 to get unit data, and
    # subtraction of "Now" from Maturity date (so consistent with tau):

    Coupon[i] = BondData[i, "Coupon"] / 100
    Maturity[i] = BondData[i, "Maturity"] - Now

    # Calculate the ex-coupon cashflow of Bond i in year t:

    F[t, i] = (
        Number(1).where[tau[t] == Maturity[i]]
        + Coupon[i].where[(tau[t] <= Maturity[i]) & (tau[t] > 0)]
    )

    Liability.setRecords(
        np.array(
            [
                0,
                80000,
                100000,
                110000,
                120000,
                140000,
                120000,
                90000,
                50000,
                75000,
                150000,
            ]
        )
    )

    r = Parameter(
        m,
        name="r",
        domain=[t],
        records=np.array(
            [
                0,
                0.0422,
                0.0440,
                0.0450,
                0.0466,
                0.0480,
                0.0482,
                0.0485,
                0.0488,
                0.0491,
                0.0493,
            ]
        ),
        description="spot rates",
    )
    y = Parameter(
        m,
        name="y",
        domain=[i],
        records=np.array(
            [
                0.0501,
                0.0500,
                0.0469,
                0.0426,
                0.0489,
                0.0485,
                0.0392,
                0.0453,
                0.0406,
                0.0386,
            ]
        ),
        description="yield rates",
    )

    # The following are the Present value, Fischer-Weil duration (D^FW)
    # and Convexity (Q_i), for both the bonds and the liabilities:
    # Present value, Fisher & Weil duration, and convexity for
    # the bonds.
    PV = Parameter(
        m, name="PV", domain=[i], description="Present value of assets"
    )
    Dur = Parameter(
        m, name="Dur", domain=[i], description="Duration of assets"
    )
    Conv = Parameter(
        m, name="Conv", domain=[i], description="Convexity of assets"
    )

    # Present value, Fisher & Weil duration, and convexity for
    # the liability.
    PV_Liab = Parameter(
        m, name="PV_Liab", description="Present value of liability"
    )
    Dur_Liab = Parameter(
        m, name="Dur_Liab", description="Duration of liability"
    )
    Conv_Liab = Parameter(
        m, name="Conv_Liab", description="Convexity of liability"
    )

    PV[i] = Sum(t, F[t, i] * gams_math.exp(-r[t] * tau[t]))

    Dur[i] = (1.0 / PV[i]) * Sum(
        t, tau[t] * F[t, i] * gams_math.exp(-r[t] * tau[t])
    )

    Conv[i] = (1.0 / PV[i]) * Sum(
        t, sqr(tau[t]) * F[t, i] * gams_math.exp(-r[t] * tau[t])
    )

    print("PV: \n", PV.records)
    print("Dur: \n", Dur.records)
    print("Conv: \n", Conv.records)

    # Calculate the corresponding amounts for Liabilities. Use its PV as its
    # "price".

    PV_Liab.assign = Sum(t, Liability[t] * gams_math.exp(-r[t] * tau[t]))

    Dur_Liab.assign = (1.0 / PV_Liab) * Sum(
        t, tau[t] * Liability[t] * gams_math.exp(-r[t] * tau[t])
    )

    Conv_Liab.assign = (1.0 / PV_Liab) * Sum(
        t, sqr(tau[t]) * Liability[t] * gams_math.exp(-r[t] * tau[t])
    )

    print("PV_Liab: ", PV_Liab.records.value[0])
    print("Dur_Liab: ", Dur_Liab.records.value[0])
    print("Conv_Liab: ", Conv_Liab.records.value[0])

    # Build a sequence of increasingly sophisticated immunuzation models.

    # VARIABLES #
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i],
        description="Holdings of bonds (amount of face value)",
    )
    z = Variable(m, name="z", description="Objective function value")

    # EQUATIONS #
    PresentValueMatch = Equation(
        m,
        name="PresentValueMatch",
        description=(
            "Equation matching the present value of asset and liability"
        ),
    )
    DurationMatch = Equation(
        m,
        name="DurationMatch",
        description="Equation matching the duration of asset and liability",
    )
    ConvexityMatch = Equation(
        m,
        name="ConvexityMatch",
        description="Equation matching the convexity of asset and liability",
    )
    ObjDef = Equation(
        m,
        name="ObjDef",
        description="Objective function definition",
    )

    ObjDef.expr = z == Sum(i, Dur[i] * PV[i] * y[i] * x[i]) / (
        PV_Liab * Dur_Liab
    )

    PresentValueMatch.expr = Sum(i, PV[i] * x[i]) == PV_Liab

    DurationMatch.expr = Sum(i, Dur[i] * PV[i] * x[i]) == PV_Liab * Dur_Liab

    ConvexityMatch.expr = Sum(i, Conv[i] * PV[i] * x[i]) >= PV_Liab * Conv_Liab

    ImmunizationOne = Model(
        m,
        name="ImmunizationOne",
        equations=[ObjDef, PresentValueMatch, DurationMatch],
        problem="LP",
        sense=Sense.MAX,
        objective=z,
    )
    ImmunizationOne.solve()

    Convexity = Parameter(m, name="Convexity")
    Convexity.assign = (1.0 / PV_Liab) * Sum(i, Conv[i] * PV[i] * x.l[i])
    x_results = []

    x_results.append(x.records.level.tolist())
    print("Convexity: ", Convexity.records.value[0])
    print("Conv_Liab: ", Conv_Liab.records.value[0])

    ImmunizationTwo = Model(
        m,
        name="ImmunizationTwo",
        equations=[ObjDef, PresentValueMatch, DurationMatch, ConvexityMatch],
        problem="LP",
        sense=Sense.MAX,
        objective=z,
    )
    ImmunizationTwo.solve()

    DurationMatch.l.assign = DurationMatch.l / PV_Liab

    ConvexityMatch.l.assign = ConvexityMatch.l / PV_Liab

    x_results.append(x.records.level.tolist())
    print("PresentValueMatch: ", PresentValueMatch.records.level[0])
    print("DurationMatch: ", DurationMatch.records.level[0])
    print("ConvexityMatch: ", ConvexityMatch.records.level[0])

    ConvexityObj = Equation(m, name="ConvexityObj")

    ConvexityObj.expr = z == (1.0 / PV_Liab) * Sum(i, Conv[i] * PV[i] * x[i])

    ImmunizationThree = Model(
        m,
        name="ImmunizationThree",
        equations=[ConvexityObj, PresentValueMatch, DurationMatch],
        problem="LP",
        sense=Sense.MIN,
        objective=z,
    )
    ImmunizationThree.solve()

    x_results.append(x.records.level.tolist())
    x_results = pd.DataFrame(
        np.array(x_results).T,
        columns=["Model1", "Model2", "Model3"],
        index=Bonds.records.uni,
    )
    print("Objective Function Value: ", z.records.level[0])

    print(x_results)


if __name__ == "__main__":
    main()
