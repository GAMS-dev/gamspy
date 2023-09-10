"""
Food Manufacturing Problem - Blending of Oils (FOOD)

The problem is to plan the blending of five kinds of oil, organized in two
categories (two kinds of vegetable oils and three kinds of non vegetable oils)
into batches of blended products over six months.

Some of the oil is already available in storage. There is an initial stock of
oil of 500 tons of each raw type when planning begins. An equal stock should
exist in storage at the end of the plan. Up to 1000 tons of each type of raw
oil can be stored each month for later use. The price for storage of raw oils
is 5 monetary units per ton. Refined oil cannot be stored. The blended product
cannot be stored either.

The rest of the oil (that is, any not available in storage) must be bought in
quantities to meet the blending requirements. The price of each kind of oil
varies over the six-month period.

The two categories of oil cannot be refined on the same production line.
There is a limit on how much oil of each category (vegetable or non vegetable)
can be refined in a given month:
 - Not more than 200 tons of vegetable oil can be refined per month.
 - Not more than 250 tons of non vegetable oil can be refined per month.

There are constraints on the blending of oils:
 - The product cannot blend more than three oils.
 - When a given type of oil is blended into the product, at least 20 tons of
   that type must be used.
 - If either vegetable oil 1 (v1) or vegetable oil 2 (v2) is blended in the
   product, then non vegetable oil 3 (o3) must also be blended in that product.

The final product (refined and blended) sells for a known price:
150 monetary units per ton.

The aim of the six-month plan is to minimize production and storage costs while
maximizing profit.


This example is taken from the Cplex 12 User's Manual
(ILOG, Cplex 12 User's Manual, 2009)

Williams, H P, Model Building in Mathematical Programming. John Wiley
and Sons, 1978.

Keywords: mixed integer linear programming, food manufacturing, blending
problem
"""

import gamspy as gp
import pandas as pd


def main():
    vegetable_oils = ["v1", "v2"]
    non_vegetable_oils = ["o1", "o2", "o3"]

    # Prepare data
    cost_mp = pd.DataFrame(
        [
            ["m1", "v1", 110],
            ["m1", "v2", 120],
            ["m1", "o1", 130],
            ["m1", "o2", 110],
            ["m1", "o3", 115],
            ["m2", "v1", 130],
            ["m2", "v2", 130],
            ["m2", "o1", 110],
            ["m2", "o2", 90],
            ["m2", "o3", 115],
            ["m3", "v1", 110],
            ["m3", "v2", 140],
            ["m3", "o1", 130],
            ["m3", "o2", 100],
            ["m3", "o3", 95],
            ["m4", "v1", 120],
            ["m4", "v2", 110],
            ["m4", "o1", 120],
            ["m4", "o2", 120],
            ["m4", "o3", 125],
            ["m5", "v1", 100],
            ["m5", "v2", 120],
            ["m5", "o1", 150],
            ["m5", "o2", 110],
            ["m5", "o3", 105],
            ["m6", "v1", 90],
            ["m6", "v2", 100],
            ["m6", "o1", 140],
            ["m6", "o2", 80],
            ["m6", "o3", 135],
        ],
        columns=["planning period", "raw oil", "raw oil cost"],
    )

    products = cost_mp["raw oil"].unique()

    stock_p = pd.DataFrame([(pp, 500) for pp in products])

    hardness = pd.DataFrame(
        [["v1", 8.8], ["v2", 6.1], ["o1", 2], ["o2", 4.2], ["o3", 5]],
        columns=["raw oil", "hardness"],
    )

    maxstore = 1000

    c = gp.Container()

    # Sets
    m = gp.Set(
        c,
        name="m",
        description="planing periods",
        records=cost_mp["planning period"].unique(),
    )
    p = gp.Set(c, name="p", description="raw oils", records=products)
    pv = gp.Set(
        c,
        name="pv",
        domain=p,
        description="vegetable oils",
        records=vegetable_oils,
    )
    pnv = gp.Set(
        c,
        name="pnv",
        domain=p,
        description="non-vegetable oils",
        records=non_vegetable_oils,
    )

    # Parameter
    maxusepv = gp.Parameter(
        c,
        name="maxusepv",
        description="maximum use of vegetable oils",
        records=200,
    )
    maxusepnv = gp.Parameter(
        c,
        name="maxusepnv",
        description="maximum use of non-vegetable oils",
        records=250,
    )
    minusep = gp.Parameter(
        c, name="minusep", description="minimum use of raw oil", records=20
    )
    maxnusep = gp.Parameter(
        c,
        name="maxnusep",
        description="maximum number of raw oils in a blend",
        records=3,
    )
    sp = gp.Parameter(
        c,
        name="sp",
        description="sales price of refined and blended oil",
        records=150,
    )
    sc = gp.Parameter(
        c, name="sc", description="storage cost of raw oils", records=5
    )
    stock = gp.Parameter(
        c,
        name="stock",
        description="stock at the beginning and end",
        domain=p,
        records=stock_p,
    )
    hmin = gp.Parameter(
        c,
        name="hmin",
        description="minimum hardness of refined oil",
        records=3,
    )
    hmax = gp.Parameter(
        c,
        name="hmax",
        description="maximum hardness of refined oil",
        records=6,
    )
    h = gp.Parameter(
        c,
        name="h",
        description="hardness of raw oils",
        domain=p,
        records=hardness,
    )
    cost = gp.Parameter(
        c,
        name="cost",
        description="raw oil cost",
        domain=[m, p],
        records=cost_mp,
    )

    # Variables
    produce = gp.Variable(
        c,
        name="produce",
        description="production of blended and refined oil per month",
        domain=m,
        type="positive",
    )
    use = gp.Variable(
        c,
        name="use",
        description="usage of raw oil per month",
        domain=[m, p],
        type="positive",
    )
    induse = gp.Variable(
        c,
        name="induse",
        description="indicator for usage of raw oil per month",
        domain=[m, p],
        type="binary",
    )
    buy = gp.Variable(
        c,
        name="buy",
        description="purchase of raw oil per month",
        domain=[m, p],
        type="positive",
    )
    store = gp.Variable(
        c,
        name="store",
        description="storage of raw oil at end of the month",
        domain=[m, p],
        type="positive",
    )
    profit = gp.Variable(c, name="profit", description="objective variable")

    # Equation
    defObj = gp.Equation(c, name="defObj")
    defUsePv = gp.Equation(c, name="defUsePv", domain=m)
    defUsePnv = gp.Equation(c, name="defUsePnv", domain=m)
    defProduce = gp.Equation(c, name="defProduce", domain=m)
    defHmin = gp.Equation(c, name="defHmin", domain=m)
    defHmax = gp.Equation(c, name="defHmax", domain=m)
    stockbal = gp.Equation(c, name="stockbal", domain=[m, p])
    minUse = gp.Equation(c, name="minUse", domain=[m, p])
    maxUse = gp.Equation(c, name="maxUse", domain=[m, p])
    maxNuse = gp.Equation(c, name="maxNuse", domain=m)
    defLogic1 = gp.Equation(c, name="defLogic1", domain=m)

    defObj.expr = profit == gp.Sum(m, sp * produce[m]) - gp.Sum(
        (m, p), cost[m, p] * buy[m, p]
    ) - gp.Sum((m, p), sc * store[m, p])
    defUsePv[m] = gp.Sum(pv, use[m, pv]) <= maxusepv
    defUsePnv[m] = gp.Sum(pnv, use[m, pnv]) <= maxusepnv
    defProduce[m] = produce[m] == gp.Sum(p, use[m, p])
    defHmin[m] = gp.Sum(p, h[p] * use[m, p]) >= hmin * produce[m]
    defHmax[m] = gp.Sum(p, h[p] * use[m, p]) <= hmax * produce[m]
    stockbal[m, p] = (
        store[m.lag(1, type="circular"), p] + buy[m, p]
        == use[m, p] + store[m, p]
    )
    minUse[m, p] = use[m, p] >= minusep * induse[m, p]
    maxUse[m, p] = (
        use[m, p]
        <= (maxusepv.where[pv[p]] + maxusepnv.where[pnv[p]]) * induse[m, p]
    )
    maxNuse[m] = gp.Sum(p, induse[m, p]) <= maxnusep
    defLogic1[m] = gp.Sum(pv, induse[m, pv]) <= induse[m, "o3"] * gp.Card(pv)

    # set upper bound and fix variable value
    store.up[m, p] = maxstore
    store.fx["m6", p] = stock[p]

    food = gp.Model(
        c,
        name="food",
        equations=c.getEquations(),
        problem="MIP",
        sense=gp.Sense.MAX,
        objective=profit,
    )

    # set optCr to 0
    c.addOptions({"optcr": 0})

    food.solve()


if __name__ == "__main__":
    main()
