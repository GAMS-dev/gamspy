"""
October 6, 2005
Optimal management of a river system which includes water resources,
reservoirs, consumers and an estuary.

Adapted from:
McKinney, D.C. and Savitsky, A.G., "Basic optimization models
for water and energy management", Revision 6, February 2003.
"""
import pandas as pd

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def reformat_df(dataframe):
    return dataframe.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )


def data_records():
    # Source records table
    cols = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    inds = ["Source_1", "Source_2"]
    data = [
        [98, 115, 244, 390, 641, 754, 807, 512, 367, 210, 181, 128],
        [29, 49, 78, 121, 198, 144, 105, 98, 79, 72, 45, 29],
    ]
    Source_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    # Demand records table
    cols = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    inds = ["User_1", "User_2", "Outlet"]
    data = [
        [0.0, 0.0, 10, 64.5, 189.8, 184.4, 243.7, 200.9, 99.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10, 13.5, 15.0, 22.1, 26.0, 24.9, 13.0, 0.0, 0.0, 0.0],
        [
            500,
            500,
            500,
            100.0,
            100.0,
            100.0,
            100.0,
            500.0,
            500.0,
            500,
            500,
            500,
        ],
    ]
    Demand_recs = reformat_df(pd.DataFrame(data, columns=cols, index=inds))

    return Source_recs, Demand_recs


def main():
    m = Container()

    # SETS #
    n = Set(
        m,
        name="n",
        records=[
            "Source_1",
            "Source_2",
            *[f"Node_{n}" for n in range(1, 6)],
            "User_1",
            "User_2",
            "Res_1",
            "Res_2",
            "Outlet",
        ],
        description="nodes",
    )
    nn = Set(
        m,
        name="nn",
        domain=[n],
        records=[f"Node_{n}" for n in range(1, 6)],
        description="Nodes",
    )
    ns = Set(
        m,
        name="ns",
        domain=[n],
        records=["Source_1", "Source_2"],
        description="Sources nodes",
    )
    nr = Set(
        m,
        name="nr",
        domain=[n],
        records=["User_1", "User_2", "Outlet"],
        description="Users nodes",
    )
    nl = Set(
        m,
        name="nl",
        domain=[n],
        records=["Res_1", "Res_2"],
        description="Reservoir nodes",
    )
    n_from_n = Set(
        m,
        name="n_from_n",
        domain=[n, n],
        records=[
            ("Res_1", "Source_1"),
            ("Res_2", "Source_2"),
            ("Node_1", "Res_1"),
            ("Node_1", "Res_2"),
            ("Node_2", "Node_1"),
            ("User_1", "Node_2"),
            ("Node_3", "Node_2"),
            ("Node_3", "User_1"),
            ("Node_4", "Node_3"),
            ("User_2", "Node_4"),
            ("Node_5", "Node_4"),
            ("Node_5", "User_2"),
            ("outlet", "Node_5"),
        ],
        description="Topology of the network",
    )
    n_to_nr = Set(
        m,
        name="n_to_nr",
        domain=[n, n],
        records=[
            ("Node_2", "User_1"),
            ("Node_4", "User_2"),
            ("Node_5", "Outlet"),
        ],
        description="Topology of the network",
    )
    t = Set(
        m,
        name="t",
        records=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        description="months",
    )

    # ALIAS #
    n1 = Alias(m, name="n1", alias_with=n)

    # PARAMETERS #
    Ini_S = Parameter(
        m,
        name="Ini_S",
        domain=[n],
        records=[("Res_1", 1000), ("Res_2", 300)],
        description="Initial storage in reservoirs. (m3)",
    )
    ret = Parameter(
        m,
        name="ret",
        domain=[n],
        records=[("User_1", 0.5), ("User_2", 0.5), ("Outlet", 0.0)],
        description="Return flow coefficients",
    )
    Source = Parameter(
        m,
        name="Source",
        domain=[n, t],
        records=data_records()[0],
        description="Flow of water (m3 per sec)",
    )
    Demand = Parameter(
        m,
        name="Demand",
        domain=[n, t],
        records=data_records()[1],
        description="Water demands (m3 per sec)",
    )

    # VARIABLES #
    U = Variable(
        m,
        name="U",
        type="positive",
        domain=[n, t],
        description="Diversions of water from node n in period t (m3)",
    )
    q = Variable(
        m,
        name="q",
        type="positive",
        domain=[n, t],
        description="Inflows in nodde n in period t (m3)",
    )
    r = Variable(
        m,
        name="r",
        type="positive",
        domain=[n, t],
        description="Releases from node n in period t (m3)",
    )
    s = Variable(
        m,
        name="s",
        type="positive",
        domain=[n, t],
        description="Storages of water in node n in period t (m3)",
    )

    obj = Variable(m, name="obj")

    # Upper bounds on users.
    U.up[n, t] = Demand[n, t]

    # Upper bounds on reservoirs.
    s.up["Res_1", t] = 1000
    s.up["Res_2", t] = 300

    # Lower storage in reservoirs (December).
    s.lo["Res_1", "Dec"] = 1000
    s.lo["Res_2", "Dec"] = 300

    # EQUATIONS #
    R_no = Equation(
        m, name="R_no", type="regular", domain=[n, t], description="Node"
    )
    R_ns = Equation(
        m, name="R_ns", type="regular", domain=[n, t], description="Source"
    )
    R_nr = Equation(
        m,
        name="R_nr",
        type="regular",
        domain=[n, t],
        description="Irrigation node",
    )
    R_nl = Equation(
        m,
        name="R_nl",
        type="regular",
        domain=[n, t],
        description="Reservoirs node",
    )
    R_nn = Equation(
        m, name="R_nn", type="regular", domain=[n, t], description="Node"
    )
    Objective = Equation(
        m, name="Objective", type="regular", description="Objective function"
    )

    R_no[n, t].where[nn[n]] = r[n, t] == q[n, t]

    R_ns[n, t].where[ns[n]] = r[n, t] == Source[n, t]

    R_nr[n, t].where[nr[n]] = r[n, t] == ret[n] * U[n, t]

    R_nl[n, t].where[nl[n]] = (
        s[n, t]
        == Ini_S[n].where[Ord(t) == 1]
        + s[n, t.lag(1)].where[Ord(t) >= 1]
        + q[n, t]
        - r[n, t]
    )

    R_nn[n, t] = q[n, t] == Sum(n1.where[n_from_n[n, n1]], r[n1, t]) - Sum(
        n1.where[n_to_nr[n, n1]], U[n1, t]
    )

    Objective.expr = obj == Sum(
        t, Sum(n.where[nr[n]], gams_math.power((U[n, t] - Demand[n, t]), 2))
    )

    riversys = Model(
        m,
        name="riversys",
        equations=m.getEquations(),
        problem="nlp",
        sense="min",
        objective=obj,
    )

    riversys.solve()

    print("Objective Function Value:  ", round(obj.toValue(), 4))

    # Show the solution

    rep = Parameter(m, name="rep", domain=[t, "*"])

    rep[t, "User_1"] = U.l["User_1", t]
    rep[t, "Demand_1"] = Demand["User_1", t]
    rep[t, "User_2"] = U.l["User_2", t]
    rep[t, "Demand_2"] = Demand["User_2", t]
    rep[t, "Outlet"] = U.l["Outlet", t]
    rep[t, "Demand"] = Demand["Outlet", t]

    print("Solution:\n", rep.pivot().round(3))

    # End riversys


if __name__ == "__main__":
    main()
