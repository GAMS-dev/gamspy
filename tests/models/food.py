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
    defObj = gp.Equation(c, name="defObj", type="eq")
    defUsePv = gp.Equation(c, name="defUsePv", domain=m, type="leq")
    defUsePnv = gp.Equation(c, name="defUsePnv", domain=m, type="leq")
    defProduce = gp.Equation(c, name="defProduce", domain=m, type="eq")
    defHmin = gp.Equation(c, name="defHmin", domain=m, type="geq")
    defHmax = gp.Equation(c, name="defHmax", domain=m, type="leq")
    stockbal = gp.Equation(c, name="stockbal", domain=[m, p], type="eq")
    minUse = gp.Equation(c, name="minUse", domain=[m, p], type="geq")
    maxUse = gp.Equation(c, name="maxUse", domain=[m, p], type="leq")
    maxNuse = gp.Equation(c, name="maxNuse", domain=m, type="leq")
    defLogic1 = gp.Equation(c, name="defLogic1", domain=m, type="leq")

    defObj.definition = profit == gp.Sum(m, sp * produce[m]) - gp.Sum(
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

    food = gp.Model(c, name="food", equations="all")

    # set optCr to 0
    c.addOptions({"optcr": 0})

    c.solve(food, problem="MIP", sense="max", objective_variable=profit)


if __name__ == "__main__":
    main()
