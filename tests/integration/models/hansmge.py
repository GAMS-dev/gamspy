"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_hansmge.html
## LICENSETYPE: Demo
## MODELTYPE: MCP
## DATAFILES: hansmge.gdx
## KEYWORDS:  mixed complementarity problem, general equilibrium model, activity analysis, social accounting matrix, European regional policy


Hansen's Activity Analysis Example - MPSGE.

Scarf, H, and Hansen, T, The Computation of Economic Equilibria.
Yale University Press, 1973.
"""

from __future__ import annotations

from math import isclose
from pathlib import Path

from gamspy import Container, Model, Sum


def main():
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/hansmge.gdx",
    )
    c, n, h, s, e, d, esub, data = (
        m[sym] for sym in ["c", "n", "h", "s", "e", "d", "esub", "data"]
    )
    m.addGamsCode("""
$onText
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
* The next line can go when #6341 gets fixed
hh.l(h) = 1; 
""")
    dummy01, y, p, hh = (m[sym] for sym in ["dummy01", "y", "p", "hh"])
    dummy01[:] = Sum(s, y[s]) + Sum(c, p[c]) + Sum(h, hh[h]) == 0
    hansen = Model(m, name="hansen", equations=m.getEquations(), problem="mcp")
    p.fx[c].where[c.ord == 1] = 1
    m.addGamsCode("$include HANSEN.GEN")
    hansen.solve()

    demands = m["HH"].toList()
    assert isclose(demands[0][1], 5.1549, rel_tol=1e-4)
    assert isclose(demands[1][1], 2.8275, rel_tol=1e-4)
    assert isclose(demands[2][1], 0.5876, rel_tol=1e-4)
    assert isclose(demands[3][1], 8.5600, rel_tol=1e-4)


if __name__ == "__main__":
    main()
