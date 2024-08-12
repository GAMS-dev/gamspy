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

from gamspy import Container


def main():
    hansen_mpsge = """Set
   C    'commodities' / AGRIC, FOOD, TEXTILES, HSERV,   ENTERT, HOUSEOP, CAPEOP
                        STEEL, COAL, LUMBER,   HOUSBOP, CAPBOP, LABOR,   EXCHANGE /
   N(C) 'numeraire'   / LABOR /
   H    'consumers'   / AGENT1*AGENT4 /
   S    'sectors'     / DOM1*DOM12, IMP1*IMP7, EXP1*EXP7 /;

Alias (C,CC);

Table E(C,H) 'commodity endowments'
              AGENT1  AGENT2  AGENT3  AGENT4
   HOUSBOP         2     0.4             0.8
   CAPBOP          3       2             7.5
   LABOR         0.6     0.8       1     0.6;

Table D(C,H) 'reference demands'
              AGENT1  AGENT2  AGENT3  AGENT4
   AGRIC         0.1     0.2     0.3     0.1
   FOOD          0.2     0.2     0.2     0.2
   TEXTILES      0.1     0.1     0.3     0.1
   HSERV         0.1     0.1     0.1     0.1
   ENTERT        0.1     0.1     0.1     0.1
   HOUSEOP       0.3     0.1             0.1
   CAPEOP        0.1     0.2             0.3;

Parameter ESUB(H) 'elasticities in demand' / AGENT1 1, AGENT2 1, AGENT3 1, AGENT4 1 /;

Table DATA(*,C,S) 'activity analysis matrix'
                     DOM1   DOM2  DOM3  DOM4   DOM5
   OUTPUT.AGRIC      5.00
   OUTPUT.FOOD              5.00
   OUTPUT.TEXTILES                2.00
   OUTPUT.HSERV                         2.00
   OUTPUT.ENTERT                               4.00
   OUTPUT.HOUSEOP                       0.32
   OUTPUT.CAPEOP     0.40   1.30  1.20
   INPUT .AGRIC             3.50  0.10         0.70
   INPUT .FOOD       0.90         0.10         0.80
   INPUT .TEXTILES   0.20   0.50        0.10   0.10
   INPUT .HSERV      1.00   2.00  2.00         2.00
   INPUT .STEEL      0.20   0.40  0.20  0.10
   INPUT .COAL       1.00   0.10  0.10  1.00
   INPUT .LUMBER     0.50   0.40  0.30  0.30
   INPUT .HOUSBOP                       0.40
   INPUT .CAPBOP     0.50   1.50  1.50  0.10   0.10
   INPUT .LABOR      0.40   0.20  0.20  0.02   0.40

   +                 DOM6   DOM7  DOM8  DOM9  DOM10
   OUTPUT.HOUSEOP    0.80
   OUTPUT.CAPEOP     1.10   6.00  1.80  1.20   0.40
   OUTPUT.STEEL                   2.00
   OUTPUT.COAL                          2.00
   OUTPUT.LUMBER                               1.00
   INPUT .TEXTILES   0.80   0.40  0.10  0.10   0.10
   INPUT .HSERV      0.40   1.80  1.60  0.80   0.20
   INPUT .STEEL      1.00   2.00        0.50   0.20
   INPUT .COAL              0.20  1.00         0.20
   INPUT .LUMBER     3.00   0.20  0.20  0.50
   INPUT .CAPBOP     1.50   2.50  2.50  1.50   0.50
   INPUT .LABOR      0.30   0.10  0.10  0.40   0.40

   +                DOM11  DOM12  IMP1  IMP2   IMP3
   OUTPUT.AGRIC                   1.00
   OUTPUT.FOOD                          1.00
   OUTPUT.TEXTILES                             1.00
   OUTPUT.HOUSEOP           0.36
   OUTPUT.CAPEOP     0.90
   INPUT .HSERV                   0.40  0.20   0.20
   INPUT .HOUSBOP           0.40
   INPUT .CAPBOP     1.00         0.20  0.10   0.10
   INPUT .LABOR                   0.04  0.02   0.02
   INPUT .EXCHANGE                0.50  0.40   0.80

   +                 IMP4   IMP5  IMP6  IMP7   EXP1
   OUTPUT.CAPEOP     1.00
   OUTPUT.STEEL             1.00
   OUTPUT.COAL                    1.00
   OUTPUT.LUMBER                        1.00
   OUTPUT.EXCHANGE                             0.50
   INPUT .AGRIC                                1.00
   INPUT .HSERV      0.40   0.40  0.40  0.40   0.20
   INPUT .CAPBOP     0.20   0.20  0.20  0.20   0.20
   INPUT .LABOR      0.04   0.04  0.04  0.04   0.04
   INPUT .EXCHANGE   1.20   0.60  0.70  0.40

   +                 EXP2   EXP3  EXP4  EXP5   EXP6
   OUTPUT.EXCHANGE   0.40   0.80  1.20  0.60   0.70
   INPUT .FOOD       1.00
   INPUT .TEXTILES          1.00
   INPUT .HSERV      0.20   0.20  0.40  0.40   0.40
   INPUT .CAPEOP                  1.00
   INPUT .STEEL                         1.00
   INPUT .COAL                                 1.00
   INPUT .CAPBOP     0.10   0.10  0.20  0.20   0.20
   INPUT .LABOR      0.02   0.02  0.04  0.04   0.04

   +                 EXP7
   OUTPUT.EXCHANGE   0.40
   INPUT .HSERV      0.40
   INPUT .LUMBER     1.00
   INPUT .CAPBOP     0.20
   INPUT .LABOR      0.04 ;

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

* READ THE HEADER:
$ifThen set MPSGEMT
$  log --- MPSGEMT has been set to %MPSGEMT% and initializes the -mt option
*  local -mt=0|1 would still overwrite the --MPSGEMT=0|1 from the command line,
*  but here we use now the one from the command line
$  sysInclude mpsgeset HANSEN -mt=%MPSGEMT%
$else
$  sysInclude mpsgeset HANSEN
$endIf

P.fx(C)$(ord(C) = 1) = 1;

* GENERATE AND SOLVE THE MODEL:

* Try first to find the GEN file in the scratch directory (-mt=1)
* and if this does not exist try to find in the working directory
$ifThen exist "%GAMS.SCRDIR%HANSEN.GEN"
$  include "%GAMS.SCRDIR%HANSEN.GEN"
$  if not %GAMS.KEEP%==1 $call rm "%GAMS.SCRDIR%HANSEN.GEN"
$else
$  include HANSEN.GEN
$endIf

solve HANSEN using mcp;
"""
    m = Container()
    m.addGamsCode(hansen_mpsge)
    demands = m["HH"].toList()
    assert isclose(demands[0][1], 5.1549, rel_tol=1e-4)
    assert isclose(demands[1][1], 2.8275, rel_tol=1e-4)
    assert isclose(demands[2][1], 0.5876, rel_tol=1e-4)
    assert isclose(demands[3][1], 8.5600, rel_tol=1e-4)


if __name__ == "__main__":
    main()
