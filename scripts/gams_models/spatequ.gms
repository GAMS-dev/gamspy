$title Spatial Equilibrium (SPATEQU,SEQ=354)

$onText
This program is written for the spatial equilibrium model with linear supply
and demand having two products and three regions.

The model contains multiple approaches (LP, NLP, and MCP) for solving this
problem.


Phan, S H, Policy option to promote the wood-processing industry in
northern Vietnam, forth coming. PhD thesis,
The University of Queensland, Australia, 2011.

Phan, S H, and Harrison, S, A Review of the Formulation and
Application of the Spatial Equilibrium Models to Analyze
Policy. Journal of Forestry Research 22, 4 (2011).

The numerical example has been taken from:
Takayama, T, and Judge, G G, Spatial Equilibrium and Quadratic
Programming. Journal of Farm Economics 46, 1 (1964), 67-93

Contributed by: Phan Sy Hieu, November 2010

Keywords: linear programming, nonlinear programming, mixed complementarity
          problem, spatial equilibrium model
$offText

Set
   c 'commodities' / Com1, Com2       /
   r 'regions'     / Reg1, Reg2, Reg3 /;

Alias (r,rr), (c,cc);

Table AlphaD(r,c) 'constants of demand functions'
          Com1  Com2
   Reg1    200   300
   Reg2    100   200
   Reg3    160   250;

Table BetaD (r,c,cc) 'price coefficients of demand functions'
               Com1  Com2
   Reg1.Com1    -10     1
   Reg1.Com2      1   -10
   Reg2.Com1     -5     1
   Reg2.Com2      1   -20
   Reg3.Com1     -8     1
   Reg3.Com2      1   -10;

Table BetadSq (r,c,cc) 'price coefficients of demand functions for quadratic functions'
                Com1  Com2
   Reg1.Com1    -5       1
   Reg1.Com2     1      -5
   Reg2.Com1    -2.5     1
   Reg2.Com2     1     -10
   Reg3.Com1    -4       1
   Reg3.Com2     1      -5;

Table AlphaS(r,c) 'constants of supply functions'
           Com1  Com2
   Reg1     -50   -60
   Reg2     -50   -60
   Reg3     -50   -60;

Table BetaS (r,c,cc) 'price coefficients of supply functions'
               Com1  Com2
   Reg1.Com1    10    0.5
   Reg1.Com2    0.5   15
   Reg2.Com1    20    0.5
   Reg2.Com2    0.5   25
   Reg3.Com1    10    0.5
   Reg3.Com2    0.5   15 ;

Table BetasSq (r,c,cc) 'price coefficients of supply functions for quadratic functions'
               Com1  Com2
   Reg1.Com1    5     0.5
   Reg1.Com2    0.5   7.5
   Reg2.Com1   10     0.5
   Reg2.Com2    0.5  12.5
   Reg3.Com1    5     0.5
   Reg3.Com2    0.5   7.5;

Table TCost(r,rr,c) 'transportation cost for commodities'
               Com1  Com2
   Reg1.Reg1      0     0
   Reg1.Reg2      2     3
   Reg1.Reg3      2     3
   Reg2.Reg1      2     3
   Reg2.Reg2      0     0
   Reg2.Reg3      1     2
   Reg3.Reg1      2     3
   Reg3.Reg2      1     2
   Reg3.Reg3      0     0;

Variable
   DINT(r,c)        'integrals of demand functions'
   SINT(r,c)        'integrals of supply functions'
   TC               'total transportation cost'
   Qd(r,c)          'demand quantities'
   Qs(r,c)          'supply quantities'
   X(r,rr,c)        'quantities transported between regions'
   P(r,c)           'price'
   OBJ              'objective value of total economic surplus subtracting total transportation';

Positive Variable X, P;

Equation
   DEM(r,c)         'demand functions'
   DEMLOG(r,c)      'demand functions nonlinear'
   DEMINT(r,c)      'integrals of demand functions'
   SUP(r,c)         'supply functions'
   SUPLOG(r,c)      'supply functions nonlinear'
   SUPINT(r,c)      'integrals of supply functions'
   SDBAL(c)         'supply and demand quantity constraints'
   PDIF(r,rr,c)     'price differences between regions'
   TRANSCOST        'transportation cost equation'
   SX(r,c)          'quantities transported and supply quantity'
   DX(r,c)          'quantities transported and demand quantity'
   OBJECT           'objective equation for NLP'
   IN_OUT(r,c)      'trade flows'
   DOM_TRAD(r,rr,c) 'domestic trade price relationship';

DEM(r,c)..     AlphaD(r,c) + sum(cc, (BetaD(r,c,cc)*P(r,c)))      =e= Qd(r,c);

DEMLOG(r,c)..  AlphaD(r,c) + sum(cc, (BetaD(r,c,cc)*log(P(r,c)))) =e= Qd(r,c);

DEMINT(r,c)..  DINT(r,c) =e= AlphaD(r,c)*P(r,c) + sum(cc, BetadSq(r,c,cc)*P(r,cc))*P(r,c);

SUP(r,c)..     AlphaS(r,c) + sum(cc, (BetaS(r,c,cc)*P(r,c)))      =e= Qs(r,c);

SUPLOG(r,c)..  AlphaS(r,c) + sum(cc, (BetaS(r,c,cc)*log(P(r,c)))) =e= Qs(r,c);

SUPINT(r,c)..  SINT(r,c) =e= AlphaS(r,c)*P(r,c)+ sum(cc, BetasSq(r,c,cc)*P(r,cc))*P(r,c);

SDBAL(c)..     sum(r,Qd(r,c)) =e= sum(r, Qs(r,c));

TRANSCOST..    TC  =e= sum((r,rr,c), X(r,rr,c)*TCost(r,rr,c));

OBJECT..       OBJ =e= sum((r,c), DINT(r,c) - SINT(r,c)) - TC;

PDIF(r,rr,c).. P(r,c) - P(rr,c) =l= TCost(r,rr,c);

SX(R,C)..      sum(RR,X(R,RR,C))  =e= Qs(R,C);

DX(r,c)..      sum(rr, X(rr,r,c)) =e= Qd(r,c);

IN_OUT(r,c)..  Qs(r,c) + sum(rr, X(rr,r,c) - X(r,rr,c)) =e= Qd(r,c);

DOM_TRAD(r,rr,c).. P(r,c) + TCost(r,rr,c) =g= P(rr,c);

Model
   P2R3_Linear     / DEM, SUP, SDBAL, PDIF, TRANSCOST, SX, DX /
   P2R3_LinearLog  / DEMLOG, SUPLOG, SDBAL, PDIF, TRANSCOST, SX, DX /
   P2R3_NonLinear  / P2R3_Linear, DEMINT, SUPINT, OBJECT /
   P2R3_MCP        / DEM, SUP, IN_OUT.P, DOM_TRAD.X /;

solve P2R3_Linear    using  lp minimizing TC;
solve P2R3_LinearLog using nlp minimizing TC;
solve P2R3_NonLinear using nlp maximizing OBJ;

X.fx(r,r,c) = 0;

solve P2R3_MCP using mcp;
