"""
Portfolio horizon returns model

Horizon.gms:  Portfolio horizon returns model.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 4.3.1
Last modified: Apr 2008.
"""
import numpy as np
import pandas as pd

from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def BondDataTable():
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
    return bond_data_recs


def main():
    m = Container()

    Time = Set(
        m,
        name="Time",
        records=[str(time) for time in range(2001, 2012)],
        description="Time periods",
    )

    t = Alias(m, name="t", alias_with=Time)

    # SCALARS #
    Now = Parameter(m, name="Now", description="Current year")
    Horizon = Parameter(m, name="Horizon", description="End of the Horizon")

    Now.assign = 2001
    Horizon.assign = Card(t) - 1

    # PARAMETER #

    tau = Parameter(m, name="tau", domain=[t], description="Time in years")

    # Note: time starts from 0

    tau[t] = Ord(t) - 1

    # SET
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

    i = Alias(m, name="i", alias_with=Bonds)

    # SCALARS #
    spread = Parameter(
        m,
        name="spread",
        description="Borrowing spread over the reinvestment rate",
    )
    Budget = Parameter(m, name="Budget", description="Initial budget")

    # PARAMETERS #
    Price = Parameter(m, name="Price", domain=[i], description="Bond prices")
    Coupon = Parameter(m, name="Coupon", domain=[i], description="Coupons")
    Maturity = Parameter(
        m, name="Maturity", domain=[i], description="Maturities"
    )
    Liability = Parameter(
        m, name="Liability", domain=[t], description="Stream of liabilities"
    )
    rf = Parameter(m, name="rf", domain=[t], description="Reinvestment rates")
    F = Parameter(m, name="F", domain=[t, i], description="Cashflows")

    # Bond data. Prices, coupons and maturities from the Danish market

    BondData = Parameter(
        m, name="BondData", domain=[i, "*"], records=BondDataTable()
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
    spread.assign = 0.02

    # Initial available budget to buy the matching portfolio

    Budget.assign = 803021.814
    # 803021.814
    # 850000

    # PARAMETER #
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
    borrow = Variable(
        m,
        name="borrow",
        type="positive",
        domain=[t],
        description="Amount of money borrowed",
    )
    HorizonRet = Variable(m, name="HorizonRet", description="Horizon Return")

    # EQUATION #
    CashFlowCon = Equation(
        m,
        name="CashFlowCon",
        type="regular",
        domain=[t],
        description="Equations defining the cashflow balance",
    )

    CashFlowCon[t] = (
        Sum(i, F[t, i] * x[i])
        + (Budget - Sum(i, Price[i] * x[i])).where[tau[t] == 0]
        + borrow[t].where[tau[t] < Horizon]
        + ((1 + rf[t.lag(1)]) * surplus[t.lag(1)]).where[tau[t] > 0]
        == Liability[t].where[tau[t] > 0]
        + surplus[t].where[tau[t] < Horizon]
        + HorizonRet.where[tau[t] == Horizon]
        + ((1 + rf[t.lag(1)] + spread) * borrow[t.lag(1)]).where[tau[t] > 0]
    )

    HorizonMod = Model(
        m,
        name="HorizonMod",
        equations=[CashFlowCon],
        problem="LP",
        sense="MAX",
        objective=HorizonRet,
    )

    HorizonMod.solve()

    print("HorizonRet: ", round(HorizonRet.toValue(), 3))
    print("borrow: ", borrow.toDict())
    print("surplus: ", surplus.toDict())
    print("x: ", x.toDict())

    # Simulation for different values of the initial budget

    HorizonHandle = open("HorizonPortfolios_new.csv", "w", encoding="UTF-8")

    budget = 778985.948
    while budget <= 818985.948:
        Budget.assign = budget
        HorizonMod.solve()

        for ii in i.toList():
            horizon_ret = round(HorizonRet.records.level[0], 2)
            bond_mat = round(BondData.pivot().loc[ii, "Maturity"], 2)
            coupon = Coupon.records[Coupon.records["i"] == ii].value.array[0]
            purchased_price = round(
                x.records[x.records["i"] == ii].level.array[0]
                * Price.records[Price.records["i"] == ii].value.array[0],
                3,
            )

            HorizonHandle.write(
                f'{round(budget,2)},{horizon_ret},"{ii}",{bond_mat},{coupon},{purchased_price}'
            )
            HorizonHandle.write("\n")

        for tt in t.toList():
            borrow_rec = borrow.records[borrow.records["t"] == tt]
            borrow_rec = (
                round(borrow_rec.level.array[0], 3)
                if not borrow_rec.empty
                else 0
            )
            HorizonHandle.write(f'"{tt}",{borrow_rec}')
            HorizonHandle.write("\n")

        budget += 10000

    HorizonHandle.close()


if __name__ == "__main__":
    main()
