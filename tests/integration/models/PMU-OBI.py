"""
Maximizing the network observability using a limited number of PMU for IEEE 14 network without considering zero injection nodes

For more details please refer to Chapter 8 (Gcode8.4), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: MIP
--------------------------------------------------------------------------------
Contributed by
Dr. Alireza Soroudi
IEEE Senior Member
email: alireza.soroudi@gmail.com
We do request that publications derived from the use of the developed GAMS code
explicitly acknowledge that fact by citing
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
DOI: doi.org/10.1007/978-3-319-62350-4
"""
from __future__ import annotations

import os

from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Options
from gamspy import Parameter
from gamspy import Sense
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def main():
    m = Container(delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)))

    # Set
    bus = Set(m, name="bus", records=[str(idx) for idx in range(1, 15)])
    node = Alias(m, name="node", alias_with=bus)
    conex = Set(
        m,
        name="conex",
        records=[
            ("1", "2"),
            ("1", "5"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("3", "4"),
            ("4", "5"),
            ("4", "7"),
            ("4", "9"),
            ("5", "6"),
            ("6", "11"),
            ("6", "12"),
            ("6", "13"),
            ("7", "8"),
            ("7", "9"),
            ("9", "10"),
            ("9", "14"),
            ("10", "11"),
            ("12", "13"),
            ("13", "14"),
        ],
        domain=[bus, node],
    )
    conex[bus, node].where[conex[node, bus]] = 1

    # Data
    NPMU = Parameter(m, name="NPMU", records=10)

    # Variable
    OF = Variable(m, name="OF")
    PMU = Variable(m, name="PMU", domain=[bus], type="Binary")
    alpha = Variable(m, name="alpha", domain=[bus], type="Binary")

    # Equation
    eq1 = Equation(m, name="eq1")
    eq2 = Equation(m, name="eq2")
    eq3 = Equation(m, name="eq3", domain=[bus])

    eq1[...] = Sum(bus, PMU[bus]) <= NPMU
    eq2[...] = OF == Sum(node, alpha[node])
    eq3[bus] = (
        PMU[bus] + Sum(node.where[conex[bus, node]], PMU[node]) >= alpha[bus]
    )

    placement3 = Model(
        m,
        name="placement3",
        equations=m.getEquations(),
        problem="MIP",
        sense=Sense.MAX,
        objective=OF,
    )

    counter = Set(m, "counter", records=[f"c{idx}" for idx in range(1, 5)])
    report = Parameter(m, "report", domain=[bus, counter])
    OBIrep = Parameter(m, "OBIrep", domain=[counter])

    for idx, iter, _ in counter.records.itertuples():
        NPMU[...] = idx + 1
        placement3.solve(options=Options(relative_optimality_gap=0))
        report[bus, iter] = PMU.l[bus]
        OBIrep[iter] = OF.l


if __name__ == "__main__":
    main()
