"""
This is a demonstration of using GAMSPy to solve Multi-Commodity Flow problem via
Column Generation. To achieve this, the model is formulated using
paths rather than edges.
"""

import networkx as nx
import pandas as pd

from gamspy import (
    Alias,
    Card,
    Container,
    Equation,
    Model,
    Number,
    Ord,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
)


def process_solution(sol: list, source: str, sink: str) -> list:
    """
    Processes the solution to find all paths in the graph provided by the pricing problem.
    """
    filtered = [
        (item[0], item[1]) for item in sol if item[2] != 0.0
    ]  # Filter out elements with zero values
    graph = nx.DiGraph()
    graph.add_edges_from(filtered)
    all_paths = list(nx.all_simple_paths(graph, source, sink))

    # Convert paths from node lists to edge lists to match GAMSPy format
    formatted_paths = []
    for path in all_paths:
        edge_path = []
        for i in range(len(path) - 1):
            edge_path.append((path[i], path[i + 1], 1))
        formatted_paths.append(edge_path)
    return formatted_paths


def read_solution(df: pd.DataFrame, cost: float) -> None:
    """
    Reads the solution from a DataFrame and prints it in a readable format.
    """
    solution = {}

    # For each commodity (k1, k2, etc.)
    for commodity in df.columns:
        flows = []
        # Get non-zero flows
        non_zero_flows = df[df[commodity] > 0]
        for (source, target), flow in non_zero_flows[commodity].items():
            e_cost = [
                edge[3]
                for edge in cost
                if edge[0] == commodity
                and edge[1] == source
                and edge[2] == target
            ][0]
            flows.append(
                {
                    "from": source,
                    "to": target,
                    "amount": flow,
                    "cost": flow * e_cost,
                }
            )

        if flows:
            solution[commodity] = flows

    # Print readable output
    for commodity, move_data in solution.items():
        print(f"\v{'-'*20}")
        print(f"{commodity}: ${sum([flow['cost'] for flow in move_data])}")
        for flow in move_data:
            print(
                f"  {flow['from']} → {flow['to']}: {flow['amount']} (${flow['cost']})"
            )


def main():
    """
    Main code.
    """

    m = Container()

    # DATA

    nodes = [f"v{i}" for i in range(1, 6)]
    commodities = [f"k{i}" for i in range(1, 5)]
    possible_paths = [f"p{i}" for i in range(1, 51)]
    edges = [
        ("n1", "n2"),
        ("n1", "n3"),
        ("n2", "n3"),
        ("n2", "n4"),
        ("n3", "n4"),
        ("n3", "n5"),
        ("n4", "n5"),
        ("n1", "n4"),
        ("n1", "n5"),
        ("n2", "n5"),
    ]
    sources = [("k1", "n1"), ("k2", "n1"), ("k3", "n2"), ("k4", "n3")]
    targets = [("k1", "n4"), ("k2", "n5"), ("k3", "n5"), ("k4", "n5")]

    initial_paths = [
        ("p1", "n1", "n4", 1),
        ("p2", "n1", "n5", 1),
        ("p3", "n2", "n5", 1),
        ("p4", "n3", "n5", 1),
    ]

    path_k_compatability = [
        ("k1", "p1", 1),
        ("k2", "p2", 1),
        ("k3", "p3", 1),
        ("k4", "p4", 1),
    ]

    edge_cost = [
        ("k1", "n1", "n2", 1),
        ("k1", "n1", "n3", 5),
        ("k1", "n1", "n4", 15),
        ("k1", "n2", "n3", 1),
        ("k1", "n2", "n4", 4),
        ("k1", "n3", "n4", 8),
        ("k1", "n3", "n5", 5),
        ("k1", "n4", "n5", 3),
        ("k1", "n1", "n5", 1000),
        ("k1", "n2", "n5", 1000),
        ("k2", "n1", "n2", 1),
        ("k2", "n1", "n3", 3),
        ("k2", "n1", "n4", 13),
        ("k2", "n2", "n3", 4),
        ("k2", "n2", "n4", 4),
        ("k2", "n3", "n4", 8),
        ("k2", "n3", "n5", 7),
        ("k2", "n4", "n5", 5),
        ("k2", "n1", "n5", 1000),
        ("k2", "n2", "n5", 1000),
        ("k3", "n1", "n2", 1),
        ("k3", "n1", "n3", 1),
        ("k3", "n1", "n4", 12),
        ("k3", "n2", "n3", 3),
        ("k3", "n2", "n4", 4),
        ("k3", "n3", "n4", 9),
        ("k3", "n3", "n5", 4),
        ("k3", "n4", "n5", 2),
        ("k3", "n1", "n5", 1000),
        ("k3", "n2", "n5", 1000),
        ("k4", "n1", "n2", 1),
        ("k4", "n1", "n3", 2),
        ("k4", "n1", "n4", 11),
        ("k4", "n2", "n3", 2),
        ("k4", "n2", "n4", 4),
        ("k4", "n3", "n4", 6),
        ("k4", "n3", "n5", 8),
        ("k4", "n4", "n5", 3),
        ("k4", "n1", "n5", 1000),
        ("k4", "n2", "n5", 1000),
    ]

    d = [
        ("k1", 15),
        ("k2", 25),
        ("k3", 10),
        ("k4", 5),
    ]

    cap = [
        ("n1", "n2", 20),
        ("n1", "n3", 10),
        ("n2", "n3", 10),
        ("n2", "n4", 15),
        ("n3", "n4", 30),
        ("n3", "n5", 15),
        ("n4", "n5", 30),
        ("n1", "n4", 15),
        ("n1", "n5", 25),
        ("n2", "n5", 10),
    ]

    # SETS

    v = Set(m, "v", records=nodes, description="Nodes")
    k = Set(m, "k", records=commodities, description="Commodities")
    p = Set(m, "p", records=possible_paths, description="Possible paths")
    e = Set(m, "e", [v, v], records=edges, description="Edges")
    ks = Set(m, "ks", [k, v], records=sources, description="Commodity Sources")
    kt = Set(m, "kt", [k, v], records=targets, description="Commodity Sinks")

    pp = Set(m, "pp", p, description="Dynamic subset of p")
    pp[p] = Ord(p) <= Card(k)

    u = Alias(m, "u", v)

    # PARAMETERS

    paths = Parameter(
        m, "paths", [p, v, v], initial_paths, description="All paths"
    )
    cost = Parameter(
        m,
        "cost",
        [k, v, v],
        edge_cost,
        description="Cost of transporting one unit of K_i on edge (u, v)",
    )

    demand = Parameter(
        m, "demand", k, d, description="Demand for each commodity"
    )

    capacity = Parameter(
        m, "capacity", [v, v], cap, description="Capacity of edge (u,v)"
    )

    path_commodity = Parameter(
        m,
        "path_commodity",
        [k, p],
        path_k_compatability,
        description="Demand of each path for commodity k",
    )

    # Restricted Master Problem (RMP)
    # VARIABLES

    f = Variable(
        m,
        name="f",
        type="positive",
        domain=[k, p],
        description="Flow of commodity k on path p",
    )
    z = Variable(
        m, name="z", type="free", description="Total transportation cost"
    )

    # EQUATIONS

    rmp_obj = Equation(
        m,
        name="rmp_obj",
        description="Objective function (minimize total cost)",
    )
    cap_constraint = Equation(
        m,
        name="cap_constraint",
        domain=[v, v],
        description="Capacity constraint for each edge",
    )
    flow_conserve = Equation(
        m,
        name="flow_conserve",
        domain=k,
        description="Flow conservation for each commodity",
    )

    rmp_obj[...] = z == Sum(
        [k, pp], Sum(e, paths[pp, e] * cost[k, e]) * f[k, pp]
    )

    # Capacity constraint: Sum of all flows on each edge `e` across all paths does not exceed cap(e)
    cap_constraint[e[u, v]] = (
        -Sum([k, pp], paths[pp, e] * f[k, pp]) >= -capacity[e]
    )

    # Flow conservation: Total flow for each commodity must equal demand
    flow_conserve[k] = Sum(pp, path_commodity[k, pp] * f[k, pp]) >= demand[k]

    # Define model and solve
    rmp = Model(
        m,
        name="rmp",
        problem=Problem.LP,
        sense=Sense.MIN,
        equations=[rmp_obj, cap_constraint, flow_conserve],
        objective=z,
    )

    # Pricing problem - Shortest path model
    # SETS
    s = Set(m, name="s", domain=v, description="Source node")
    t = Set(m, name="t", domain=v, description="Sink   node")

    # PARAMETERS
    sub_cost = Parameter(m, name="sub_cost", domain=[v, v])
    sub_demand = Parameter(m, name="sub_demand")
    alpha = Parameter(m, name="alpha")

    # VARIABLES
    y = Variable(
        m, name="y", type="positive", domain=[u, v], description="New path"
    )

    # EQUATIONS
    pricing_obj = Equation(
        m,
        name="pricing_obj",
        description="Objective function for shortest path",
    )
    pricing_cap = Equation(
        m,
        name="pricing_cap",
        domain=[u, v],
        description="Capacity constraint for edge (u,v)",
    )
    pricing_source = Equation(
        m,
        name="pricing_source",
        description="Flow conservation at source node",
    )
    pricing_target = Equation(
        m,
        name="pricing_target",
        description="Flow conservation at target node",
    )
    pricing_flow = Equation(
        m,
        name="pricing_flow",
        domain=v,
        description="Flow conservation at intermediate nodes",
    )

    pricing_obj[...] = z == -(alpha * sub_demand) + Sum(
        e, (sub_cost[e] + cap_constraint.m[e]) * y[e]
    )
    pricing_cap[e[u, v]] = y[e] <= capacity[e]
    pricing_source[...] = Sum(e[s, v], y[e]) == sub_demand
    pricing_target[...] = Sum(e[u, t], y[e]) == sub_demand
    pricing_flow[v].where[(~s[v]) & (~t[v])] = Sum(e[v, u], y[e]) == Sum(
        e[u, v], y[e]
    )

    pricing = Model(
        m,
        name="pricing",
        problem=Problem.LP,
        equations=[
            pricing_obj,
            pricing_cap,
            pricing_source,
            pricing_target,
            pricing_flow,
        ],
        sense=Sense.MIN,
        objective=z,
    )

    # Initialization
    pi = Set(m, name="pi", domain=p, description="set of the last path")
    pi[p] = Ord(p) == Card(pp) + 1

    has_negative_reduced_cost = True  # A flag to track negative reduced costs
    path_no = len(pp)

    # Run as long as we have negative reduced costs
    while has_negative_reduced_cost:
        rmp.solve()

        for commodity in k.toList():
            s[v] = ks[commodity, v]
            t[v] = kt[commodity, v]
            sub_cost[e] = cost[commodity, e]
            alpha[...] = flow_conserve.m[commodity]
            sub_demand[...] = demand[commodity]

            pricing.solve()

            # path that might improve the master model found
            if pricing.objective_value < -0.0001:
                new_paths = process_solution(
                    y.toList(), s.toList()[0], t.toList()[0]
                )
                for path in new_paths:
                    path = [(pi.toList()[0],) + edge for edge in path]
                    initial_paths.extend(path)
                    paths.setRecords(initial_paths)
                    path_commodity[commodity, pi] = Number(1)
                    pp[pi] = True
                    pi[p] = pi[p.lag(1)]

        # if no new paths are added (lengths are equal), the flag turns to False
        has_negative_reduced_cost = path_no != len(pp)
        path_no = len(pp)

    rmp.solve()

    import math

    assert math.isclose(rmp.objective_value, 580.0000, rel_tol=0.001)

    print("Objective Function Value:", rmp.objective_value)
    print("Total paths generated:", len(pp) - len(k))
    read_solution(
        (
            f.pivot() @ paths.pivot(index=["p_0"], columns=["n_1", "n_2"])
        ).T.sort_index(level=0),
        edge_cost,
    )


if __name__ == "__main__":
    main()
