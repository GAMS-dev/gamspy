"""
Hansen's Activity Analysis Example - MPSGE.

Scarf, H, and Hansen, T, The Computation of Economic Equilibria.
Yale University Press, 1973.

Keywords: mixed complementarity problem, general equilibrium model, activity analysis,
          social accounting matrix, European regional policy
"""
from __future__ import annotations

import os
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
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        working_directory=".",
        load_from=str(Path(__file__).parent.absolute()) + "/hansmge.gdx",
    )
    m._addGamsCode(hansen_mpsge)
    hansen = Model(m, "hansen", problem=Problem.MCP)
    hansen.solve()


if __name__ == "__main__":
    main()
