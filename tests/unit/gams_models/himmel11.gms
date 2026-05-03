$title Himmelblau Test Problem Number 11 (HIMMEL11,SEQ=95)

$onText
Popular Series of Nonlinear Test Problems.


Himmelblau, D M, Problem Number 11. In Applied Nonlinear Programming.
Mc Graw Hill, New York, 1972.

Keywords: nonlinear programming, quadratic constraint programming, mathematics
$offText

Variable g2, g3, g4, xl, x1, x2, x3, x4, x5, obj;

Equation e1, e2, e3, e4, e5;

e1.. -x1 - x5 + 5*xl + 7*x3 =g= 0;

e2.. g2  =e= -2*xl + 85.334407 + .0056858*x2*x5 + .0006262*x1*x4 - .0022053*x3*x5;

e3.. g3  =e= 80.51249 + .0071317*x2*x5 + .0029955*x1*x2 + .0021813*sqr(x3);

e4.. g4  =e= -4*xl + 9.300961 + .0047026*x3*x5 + .0012547*x1*x3 + .0019085*x3*x4;

e5.. obj =e= 5000*xl + 5.3578547*sqr(x3) + .8356891*x1*x5 + 37.293239*x1 - 40792.141;

g2.lo =  0;    g2.up = 92;    g3.lo = 90;    g3.up = 110;  g4.lo = 20; g4.up = 25;
xl.lo =  0;    x1.lo = 78;    x1.up = 102;   x2.lo = 33;   x2.up = 45; x3.lo = 27;
x3.up = 45;    x4.lo = 27;    x4.up = 45;    x5.lo = 27;   x5.up = 45; x1.l  = 78.62;
x2.l  = 33.44; x3.l  = 31.07; x4.l  = 44.18; x5.l  = 35.22;

Model himmel11 / all /;

solve himmel11 using qcp minimizing obj;
