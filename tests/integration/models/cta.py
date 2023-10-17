"""
Controlled Tabular Adjustments (CTA)

Statistical agencies publish data which contains items that need to be
altered to protect confidentiality. Controlled Tabular Adjustments (CTA)
is a recent method to limit disclosure and can be elegantly expressed
as a Mixed Integer Programming problem. The programming framework then
allows easy expression of other data relationships like multi-dimensional
adding up conditions. The following model uses a 3-dimensional table from
from Cox, Kelly and Patil (2005) to illustrate this method.

The data is stored in an Excel Spreadsheet.


Lawrence H Cox, James P Kelly and Rahul J Patil, Computational Aspects
of Controlled Tabular Adjustments: Algorithms and Analysis, in The Next
Wave in Computing, Optimization, and Decision Technologies, Eds Bruce L Golden,
S Raghavan and Edward A Wasil, Springer, 2005, pp 45-59.

Keywords: mixed integer linear programming, statistical disclosure limitations
"""
import math
import os
import sys

from gams.connect import ConnectDatabase

from gamspy import Container
from gamspy import Domain
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Parameter
from gamspy import Sense
from gamspy import Set
from gamspy import Sum
from gamspy import Variable
from gamspy.math import Round


def main():
    m = Container(delayed_execution=True)

    # Sets
    i = Set(m, name="i", description="rows")
    j = Set(m, name="j", description="columns")
    k = Set(m, name="k", description="planes")
    v = Set(m, name="v", description="non zero cells", domain=[i, j, k])
    s = Set(m, name="s", description="sensitive cells", domain=[i, j, k])

    # Parameters
    dat = Parameter(
        m, name="dat", description="unprotected data table", domain=[k, i, j]
    )
    pro = Parameter(
        m,
        name="pro",
        description="information sensitive cells",
        domain=[k, i, j],
    )

    # extract data from Excel
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cdb = ConnectDatabase(m.system_directory)
    cdb.exec_task(
        {
            "PandasExcelReader": {
                "file": os.path.join(file_dir, "cta.xlsx"),
                "symbols": [
                    {
                        "name": "dat",
                        "range": "Sheet1!A1",
                        "rowDimension": 2,
                        "columnDimension": 1,
                    },
                    {
                        "name": "pro",
                        "range": "Sheet2!A1",
                        "rowDimension": 2,
                        "columnDimension": 1,
                    },
                ],
            }
        }
    )

    dat.domain_forwarding = True  # let dat fill sets i, j, and k
    dat.setRecords(cdb.container["dat"].records)
    pro.setRecords(cdb.container["pro"].records)

    # do some basic data checks
    check = Parameter(m, name="check")
    check[...] = Sum(
        [i, k], Round(Sum(j, dat[k, i, j]) - 2 * dat[k, i, "total"])
    )
    assert math.isclose(check.toList()[0], 0), "row totals are incorrect"
    check[...] = Sum(
        [j, k], Round(Sum(i, dat[k, i, j]) - 2 * dat[k, "total", j])
    )
    assert math.isclose(check.toList()[0], 0), "column totals are incorrect"
    check[...] = Sum(
        [i, j], Round(Sum(k, dat[k, i, j]) - 2 * dat["total", i, j])
    )
    assert math.isclose(check.toList()[0], 0), "plane totals are incorrect"

    # Parameter BigM
    BigM = Parameter(
        m,
        name="BigM",
        description="the famous big M - make it as small as possible",
    )

    # Variables
    t = Variable(
        m, name="t", description="adjusted cell value", domain=[i, j, k]
    )
    adjn = Variable(m, name="adjn", domain=[i, j, k], type="Positive")
    adjp = Variable(m, name="adjp", domain=[i, j, k], type="Positive")
    b = Variable(m, name="b", domain=[i, j, k], type="Binary")

    # Equations
    defadj = Equation(
        m,
        name="defadj",
        description="define new cell values",
        domain=[i, j, k],
    )
    addrow = Equation(
        m, name="addrow", description="add up for rows", domain=[i, k]
    )
    addcol = Equation(
        m, name="addcol", description="add up for columns", domain=[j, k]
    )
    addpla = Equation(
        m, name="addpla", description="add up for plane", domain=[i, j]
    )
    pmin = Equation(
        m,
        name="pmin",
        description="small value for sensitive cells",
        domain=[i, j, k],
    )
    pmax = Equation(
        m,
        name="pmax",
        description="big value for sensitive cells",
        domain=[i, j, k],
    )
    pminx = Equation(m, name="pminx", domain=[i, j, k])
    pmaxx = Equation(m, name="pmaxx", domain=[i, j, k])

    v[i, j, k] = dat[k, i, j]
    s[i, j, k] = pro[k, i, j]

    BigM.setRecords(3)

    defadj[v[i, j, k]] = t[v] == dat[k, i, j] + adjp[v] - adjn[v]
    addrow[i, k] = Sum(v[i, j, k], t[v]) == 2 * t[i, "total", k]
    addcol[j, k] = Sum(v[i, j, k], t[v]) == 2 * t["total", j, k]
    addpla[i, j] = Sum(v[i, j, k], t[v]) == 2 * t[i, j, "total"]
    pmin[s[i, j, k]] = adjn[s] >= pro[k, i, j] * (1 - b[s])
    pmax[s[i, j, k]] = adjp[s] >= pro[k, i, j] * b[s]
    pminx[s[i, j, k]] = adjn[s] <= BigM * pro[k, i, j] * (1 - b[s])
    pmaxx[s[i, j, k]] = adjp[s] <= BigM * pro[k, i, j] * b[s]

    cox3 = Model(
        m,
        name="cox3",
        equations=m.getEquations(),
        problem="MIP",
        sense=Sense.MIN,
        objective=Sum([i, j, k], adjn[i, j, k] + adjp[i, j, k]),
    )

    cmd_params = {"optCa": 0.99, "resLim": 10}
    cox3.solve(options=cmd_params, output=sys.stdout)

    rep = Parameter(
        m, name="rep", description="summary report", domain=[k, i, j]
    )
    adjsum = Parameter(
        m,
        name="adjsum",
        description="adjustment summary",
        domain=[k, i, j, "*"],
    )
    adjrep = Parameter(
        m, name="adjrep", description="adjustment report", domain=[k, i, j]
    )

    rep[k, i, j] = t.l[i, j, k]
    adjsum[k, i, j, "neg"] = adjn.l[i, j, k]
    adjsum[k, i, j, "pos"] = adjp.l[i, j, k]
    adjsum[k, i, j, "min"] = pro[k, i, j]
    adjrep[k, i, j] = -adjn.l[i, j, k] + adjp.l[i, j, k]

    cdb = ConnectDatabase(m.system_directory, m)
    cdb.exec_task(
        {
            "PandasExcelWriter": {
                "file": os.path.join(file_dir, "results.xlsx"),
                "excelWriterArguments": {"mode": "w"},
                "symbols": [
                    {
                        "name": "adjrep",
                        "range": "adjrep!A1",
                    },
                    {
                        "name": "rep",
                        "range": "rep!A1",
                    },
                    {
                        "name": "adjsum",
                        "range": "adjsum!A1",
                    },
                ],
            }
        }
    )

    binrep = Parameter(
        m,
        name="binrep",
        domain=["*", "*", "*", "*"],
    )

    obj = cox3.objective_value
    best = round(obj)
    num_nodes_used = cox3.num_nodes_used
    solve_time = cox3.total_solve_time

    for it in range(5):
        if (obj - best) / best > 0.01:
            break
        b_list = b.toList("level")

        sol_str = f"solution{it+1}"
        binrep[s, sol_str] = Round(b.l[s])
        binrep["", "", "Obj", sol_str] = obj
        binrep["", "", "mSec", sol_str] = solve_time * 1000
        binrep["", "", "nodes", sol_str] = num_nodes_used
        binrep["Comp", "Cells", "Adjusted", sol_str] = Sum(
            Domain(i, j, k).where[~s[i, j, k]],
            Number(1).where[Round(adjn.l[i, j, k] + adjp.l[i, j, k])],
        )

        cutone = Equation(m, name=f"cutone_{it}")
        cuttwo = Equation(m, name=f"cuttwo_{it}")
        cutone[...] = (
            sum(
                [
                    1 - b[rec[:-1]] if rec[3] > 0.5 else b[rec[:-1]]
                    for rec in b_list
                ]
            )
            >= 1
        )
        cuttwo[...] = (
            sum(
                [
                    1 - b[rec[:-1]] if rec[3] < 0.5 else b[rec[:-1]]
                    for rec in b_list
                ]
            )
            >= 1
        )

        cox3c = Model(
            m,
            name=f"cox3c_{it}",
            equations=m.getEquations(),
            problem="MIP",
            sense=Sense.MIN,
            objective=Sum([i, j, k], adjn[i, j, k] + adjp[i, j, k]),
        )

        cox3c.solve(options=cmd_params, output=sys.stdout)
        obj = cox3c.objective_value
        num_nodes_used = cox3c.num_nodes_used
        solve_time = cox3c.total_solve_time

    cdb = ConnectDatabase(m.system_directory, m)
    cdb.exec_task(
        {
            "PandasExcelWriter": {
                "file": os.path.join(file_dir, "results.xlsx"),
                "symbols": [
                    {
                        "name": "binrep",
                        "range": "binrep!A1",
                        "toExcelArguments": {"merge_cells": False},
                    }
                ],
            }
        }
    )


if __name__ == "__main__":
    main()
