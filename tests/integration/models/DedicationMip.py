"""
Dedication model with tradeability constraints

DedicationMIP.gms:  Dedication model with tradeability constraints.
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 4.3.2
Last modified: Apr 2008.

First model - Simple dedication.
"""
import numpy as np
import pandas as pd

import gamspy.math as gams_math
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

    # SET #
    Time = Set(
        m,
        name="Time",
        records=[str(t) for t in range(2001, 2012)],
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
    v0 = Variable(m, name="v0", description="Upfront investment")

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
        + (v0 - Sum(i, Price[i] * x[i])).where[tau[t] == 0]
        + borrow[t].where[tau[t] < Horizon]
        + ((1 + rf[t.lag(1)]) * surplus[t.lag(1)]).where[tau[t] > 0]
        == surplus[t]
        + Liability[t].where[tau[t] > 0]
        + ((1 + rf[t.lag(1)] + spread) * borrow[t.lag(1)]).where[tau[t] > 0]
    )

    m.addOptions({"SOLVEOPT": "REPLACE"})
    Dedication = Model(
        m,
        name="Dedication",
        equations=[CashFlowCon],
        problem="LP",
        sense="MIN",
        objective=v0,
    )

    Dedication.solve()

    print("* First Model Results\n")
    print("v0: ", v0.records.level.round(3).tolist(), "\n")
    print("borrow: ", borrow.records.level.round(3).tolist(), "\n")
    print("surplus: ", surplus.records.level.round(3).tolist(), "\n")
    print("x: ", x.records.level.round(3).tolist(), "\n\n")

    output_csv = "No trading constraints\n"

    purchased_price = (x.records.level * Price.records.value).round(3).tolist()
    for idx, ii, _ in i.records.itertuples():
        output_csv += f'{round(v0.records.level[0],3)},"{ii}",{BondData.pivot().loc[ii,"Maturity"]},{Coupon.records.value[idx]},{purchased_price[idx]}\n'

    for tt, _ in t.records.itertuples(index=False):
        borrow_rec = borrow.records[borrow.records["t"] == tt]
        borrow_rec = (
            round(borrow_rec.level.array[0], 3) if not borrow_rec.empty else 0
        )
        surplus_rec = surplus.records[surplus.records["t"] == tt]
        surplus_rec = (
            round(surplus_rec.level.array[0], 3)
            if not surplus_rec.empty
            else 0
        )
        output_csv += f'"{tt}",{borrow_rec},{surplus_rec}\n'

    # Second model - Dedication plus even-lot constraints.

    # SCALARS #
    LotSize = Parameter(
        m, name="LotSize", records=1000, description="Even-Lot requirement"
    )
    FixedCost = Parameter(
        m, name="FixedCost", records=20, description="Fixed cost per trade"
    )
    VarblCost = Parameter(
        m, name="VarblCost", records=0.01, description="Variable cost"
    )

    # VARIABLE #
    Y = Variable(
        m,
        name="Y",
        type="integer",
        domain=[i],
        description="Variable counting the number of lot purchased",
    )

    # EQUATION #
    EvenLot = Equation(
        m,
        name="EvenLot",
        type="regular",
        domain=[i],
        description="Equation defining the even-lot requirements",
    )

    EvenLot[i] = x[i] == LotSize * Y[i]

    # Some reasonable upper bounds on Y[i]

    Y.up[i] = gams_math.ceil(Sum(t, Liability[t]) / Price[i] / LotSize)

    DedicationMIPEvenLot = Model(
        m,
        name="DedicationMIPEvenLot",
        equations=[CashFlowCon, EvenLot],
        problem="MIP",
        sense="MIN",
        objective=v0,
    )

    m.addOptions({"OPTCR": 0, "ITERLIM": 999999, "RESLIM": 100})
    DedicationMIPEvenLot.solve()
    print("* Second Model Results\n")
    print("x: ", x.records.level.round(3).tolist(), "\n")
    print("Y: ", Y.records.level.round(3).tolist(), "\n")
    print("v0: ", v0.records.level.round(3).tolist(), "\n\n")

    output_csv += "Even-lot constraints\n"
    purchased_price = (x.records.level * Price.records.value).round(3).tolist()

    for idx, ii, _ in i.records.itertuples():
        output_csv += f'{round(v0.records.level[0],3)},"{ii}",{BondData.pivot().loc[ii,"Maturity"]},{Coupon.records.value[idx]},{purchased_price[idx]}\n'

    for tt, _ in t.records.itertuples(index=False):
        borrow_rec = borrow.records[borrow.records["t"] == tt]
        borrow_rec = (
            round(borrow_rec.level.array[0], 3) if not borrow_rec.empty else 0
        )
        surplus_rec = surplus.records[surplus.records["t"] == tt]
        surplus_rec = (
            round(surplus_rec.level.array[0], 3)
            if not surplus_rec.empty
            else 0
        )
        output_csv += f'"{tt}",{borrow_rec},{surplus_rec}\n'

    # Third model - Dedication plus fixed and variable transaction costs

    # VARIABLES #
    TotalCost = Variable(
        m, name="TotalCost", description="Total cost to minimize"
    )
    TransCosts = Variable(
        m,
        name="TransCosts",
        description="Total transaction costs (fixed + variable)",
    )

    # VARIABLES #
    Z = Variable(
        m,
        name="Z",
        type="binary",
        domain=[i],
        description="Indicator variable for assets included in the portfolio",
    )

    # EQUATIONS #
    CostDef = Equation(
        m,
        name="CostDef",
        type="regular",
        description=(
            "Equation definining the total cost including transaction costs"
        ),
    )
    TransDef = Equation(
        m,
        name="TransDef",
        type="regular",
        description="Equation the transaction costs (fixed + variable)",
    )
    UpBounds = Equation(
        m,
        name="UpBounds",
        type="regular",
        domain=[i],
        description="Upper bounds for each variable",
    )

    CostDef.expr = TotalCost == v0 + TransCosts

    TransDef.expr = TransCosts == Sum(i, FixedCost * Z[i] + VarblCost * x[i])

    UpBounds[i] = x[i] <= x.up[i] * Z[i]

    DedicationMIPTrnCosts = Model(
        m,
        name="DedicationMIPTrnCosts",
        equations=[CashFlowCon, CostDef, TransDef, UpBounds],
        problem="MIP",
        sense="MIN",
        objective=TotalCost,
    )

    # Some conservative bounds on investments

    x.up[i] = LotSize * Y.up[i]

    DedicationMIPTrnCosts.solve()
    print("* Third Model Results\n")
    print("x: ", x.records.level.round(3).tolist(), "\n")
    print("v0: ", v0.records.level.round(3).tolist(), "\n")
    print("TotalCost: ", TotalCost.records.level.round(3).tolist(), "\n")
    print("TransCosts: ", TransCosts.records.level.round(3).tolist(), "\n\n")

    output_csv += "Fixed and variable costs\n"
    purchased_price = (x.records.level * Price.records.value).round(3).tolist()

    for idx, ii, _ in i.records.itertuples():
        output_csv += f'{round(v0.records.level[0],3)},"{ii}",{BondData.pivot().loc[ii,"Maturity"]},{Coupon.records.value[idx]},{purchased_price[idx]}\n'

    for tt, _ in t.records.itertuples(index=False):
        borrow_rec = borrow.records[borrow.records["t"] == tt]
        borrow_rec = (
            round(borrow_rec.level.array[0], 3) if not borrow_rec.empty else 0
        )
        surplus_rec = surplus.records[surplus.records["t"] == tt]
        surplus_rec = (
            round(surplus_rec.level.array[0], 3)
            if not surplus_rec.empty
            else 0
        )
        output_csv += f'"{tt}",{borrow_rec},{surplus_rec}\n'

    # Fourth model - Dedication including even-lot restrictions and
    # transaction costs.

    DedicationMIPAll = Model(
        m,
        name="DedicationMIPAll",
        equations=[CashFlowCon, EvenLot, CostDef, TransDef, UpBounds],
        problem="MIP",
        sense="MIN",
        objective=TotalCost,
    )

    DedicationMIPAll.solve()
    print("* Fourth Model Results\n")
    print("x: ", x.records.level.round(3).tolist(), "\n")
    print("v0: ", v0.records.level.round(3).tolist(), "\n")
    print("TotalCost: ", TotalCost.records.level.round(3).tolist(), "\n")
    print("TransCosts: ", TransCosts.records.level.round(3).tolist(), "\n\n")

    output_csv += "Even-lot constraints and transaction costs\n"
    purchased_price = (x.records.level * Price.records.value).round(3).tolist()

    for idx, ii, _ in i.records.itertuples():
        output_csv += f'{round(v0.records.level[0],3)},"{ii}",{BondData.pivot().loc[ii,"Maturity"]},{Coupon.records.value[idx]},{purchased_price[idx]}\n'

    for tt, _ in t.records.itertuples(index=False):
        borrow_rec = borrow.records[borrow.records["t"] == tt]
        borrow_rec = (
            round(borrow_rec.level.array[0], 3) if not borrow_rec.empty else 0
        )
        surplus_rec = surplus.records[surplus.records["t"] == tt]
        surplus_rec = (
            round(surplus_rec.level.array[0], 3)
            if not surplus_rec.empty
            else 0
        )
        output_csv += f'"{tt}",{borrow_rec},{surplus_rec}\n'

    DedicationHandle = open(
        "DedicationMIPPortfolios.csv", "w", encoding="UTF-8"
    )
    DedicationHandle.write(output_csv)


if __name__ == "__main__":
    main()
