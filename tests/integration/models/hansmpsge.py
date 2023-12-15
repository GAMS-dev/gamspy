"""
A Transportation Problem (TRNSPORT)

This problem finds a least cost shipping schedule that meets
requirements at markets and supplies at factories.


Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

This formulation is described in detail in:
Rosenthal, R E, Chapter 2: A GAMS Tutorial. In GAMS: A User's Guide.
The Scientific Press, Redwood City, California, 1988.

The line numbers will not match those in the book because of these
comments.

Keywords: linear programming, transportation problem, scheduling
"""
from __future__ import annotations

from pathlib import Path

from gamspy import Container
from gamspy import Model
from gamspy import Problem


def main():
    hansen_mpsge = """$onText
$MODEL:HANSEN

$SECTORS:
   Y(S)

$COMMODITIES:
   P(C)

$CONSUMERS:
   HH(H)

$PROD:Y(S)
   O:P(C)   Q:DATA("OUTPUT",C,S)
   I:P(C)   Q:DATA("INPUT" ,C,S)

$DEMAND:HH(H) s:ESUB(H)
   D:P(C)   Q:D(C,H)
   E:P(C)   Q:E(C,H)
$offText
$sysInclude mpsgeset HANSEN
$include HANSEN.GEN
"""
    m = Container(
        working_directory=".",
        load_from=str(Path(__file__).parent.absolute()) + "/hansmpsge.gdx",
    )
    m._addGamsCode(hansen_mpsge)
    hansen = Model(m, "hansen", problem=Problem.MCP)
    hansen.solve()


if __name__ == "__main__":
    main()
