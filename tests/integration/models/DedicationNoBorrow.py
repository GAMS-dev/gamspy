"""
Dedication model without borrowing

Dedication.gms:  Dedication model without borrowing.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 2.4
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
import pandas as pd
import numpy as np


def main():
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

    # Define container
    m = Container()

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

    # SCALARS #
    Now = Parameter(m, name="Now", description="Current year")
    Horizon = Parameter(m, name="Horizon", description="End of the Horizon")

    Now.assign = 2001
    Horizon.assign = Card(t) - 1

    # PARAMETER #
    tau = Parameter(m, name="tau", domain=[t], description="Time in years")

    # Note: time starts from 0
    tau[t] = Ord(t) - 1

    Price = Parameter(m, name="Price", domain=[i], description="Bond prices")
    Coupon = Parameter(m, name="Coupon", domain=[i], description="Coupons")
    Maturity = Parameter(
        m, name="Maturity", domain=[i], description="Maturities"
    )
    rf = Parameter(m, name="rf", domain=[t], description="Reinvestment rates")
    F = Parameter(m, name="F", domain=[t, i], description="Cashflows")
    BondData = Parameter(
        m, name="BondData", domain=[i, "*"], records=bond_data_recs
    )
    Liability = Parameter(
        m, name="Liability", domain=[t], description="Stream of liabilities"
    )

    # Copy/transform data. Note division by 100 to get unit data, and
    # subtraction of "Now" from Maturity date (so consistent with tau):
    Price[i] = BondData[i, "Price"] / 100
    Coupon[i] = BondData[i, "Coupon"] / 100
    Maturity[i] = BondData[i, "Maturity"] - Now

    # Calculate the ex-coupon cashflow of Bond i in year t:
    F[t, i] = (
        Number(1).where[tau[t] == Maturity[i]]
        + Coupon[i].where[(tau[t] <= Maturity[i]) & (tau[t] > 0)]
    )

    # For simplicity, we set the short term rate to be 0.03 in each period
    rf[t] = 0.04

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

    # VARIABLES #
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i],
        description="Face value purchased",
    )
    surplus = Variable(
        m,
        name="surplus",
        type="positive",
        domain=[t],
        description="Amount of money reinvested",
    )
    v0 = Variable(m, name="v0", type="free", description="Upfront investment")

    # EQUATION #
    CashFlowCon = Equation(
        m,
        name="CashFlowCon",
        domain=[t],
        description="Equations defining the cashflow balance",
    )

    CashFlowCon[t] = (
        Sum(i, F[t, i] * x[i])
        + (v0 - Sum(i, Price[i] * x[i])).where[tau[t] == 0]
        + ((1 + rf[t.lag(1)]) * surplus[t.lag(1)]).where[tau[t] > 0]
        == surplus[t] + Liability[t].where[tau[t] > 0]
    )

    Dedication = Model(
        m,
        name="Dedication",
        equations=[CashFlowCon],
        problem="LP",
        sense=Sense.MIN,
        objective=v0,
    )
    Dedication.solve()

    print("Objective Function Variable: ", round(v0.records.level[0], 3))


if __name__ == "__main__":
    main()
