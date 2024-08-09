"""
## GAMSSOURCE: https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_PMU.html
## LICENSETYPE: Demo
## MODELTYPE: MIP


PMU allocation for IEEE 14 network without considering zero injection nodes

For more details please refer to Chapter 8 (Gcode8.1), of the following book:
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

from gamspy import Alias, Container, Equation, Model, Set, Sum, Variable


def main():
    m = Container()

    # SETS #
    bus = Set(m, name="bus", records=[str(b) for b in range(1, 15)])
    conex = Set(
        m,
        name="conex",
        domain=[bus, bus],
        description="Bus connectivity matrix",
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
    )

    # ALIAS #
    node = Alias(m, name="node", alias_with=bus)

    conex[bus, node].where[conex[node, bus]] = 1

    # VARIABLES #
    PMU = Variable(m, name="PMU", type="binary", domain=bus)

    # EQUATIONS #
    const1 = Sum(bus, PMU[bus])

    const2 = Equation(m, name="const2", type="regular", domain=bus)
    const2[bus] = PMU[bus] + Sum(node.where[conex[bus, node]], PMU[node]) >= 1

    placement = Model(
        m,
        name="placement",
        equations=[const2],
        problem="mip",
        sense="min",
        objective=const1,
    )
    placement.solve()
    print("PMU:  \n", PMU.toDict())


if __name__ == "__main__":
    main()
