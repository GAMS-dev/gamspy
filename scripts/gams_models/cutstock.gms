$title Cutting Stock - A Column Generation Approach (CUTSTOCK,SEQ=294)

$onText
The task is to cut out some paper products of different sizes from a
large raw paper roll, in order to meet a customer's order. The objective
is to minimize the required number of paper rolls.


P. C. Gilmore and R. E. Gomory, A linear programming approach to the
cutting stock problem, Part I, Operations Research 9 (1961), 849-859.

P. C. Gilmore and R. E. Gomory, A linear programming approach to the
cutting stock problem, Part II, Operations Research 11 (1963), 863-888.

Keywords: mixed integer linear programming, cutting stock, column generation,
          paper industry
$offText

Set i 'widths' / w1*w4 /;

Parameter
   r    'raw width' / 100 /
   w(i) 'width'     / w1   45, w2   36, w3   31, w4   14 /
   d(i) 'demand'    / w1   97, w2  610, w3  395, w4  211 /;

* Gilmore-Gomory column generation algorithm
Set
   p     'possible patterns'  / p1*p1000 /
   pp(p) 'dynamic subset of p';

Parameter aip(i,p) 'number of width i in pattern growing in p';

* Master model
Variable
   xp(p) 'patterns used'
   z     'objective variable';

Integer Variable xp;
xp.up(p) = sum(i, d(i));

Equation
   numpat    'number of patterns used'
   demand(i) 'meet demand';

numpat..    z =e= sum(pp, xp(pp));

demand(i).. sum(pp, aip(i,pp)*xp(pp)) =g= d(i);

Model master / numpat, demand /;

* Pricing problem - Knapsack model
Variable y(i) 'new pattern';

Integer Variable y;
y.up(i) = ceil(r/w(i));

Equation
   defobj
   knapsack 'knapsack constraint';

defobj..   z =e= 1 - sum(i, demand.m(i)*y(i));

knapsack.. sum(i, w(i)*y(i)) =l= r;

Model pricing / defobj, knapsack /;

* Initialization - the initial patterns have a single width
pp(p)                = ord(p) <= card(i);
aip(i,pp(p))$(ord(i) = ord(p)) = floor(r/w(i));
*display aip;

Set pi(p) 'set of the last pattern';
pi(p) = ord(p) = card(pp) + 1;

option optCr = 0, limRow = 0, limCol = 0, solPrint = off;

while(card(pp) < card(p),
   solve master  using rmip minimizing z;
   solve pricing using  mip minimizing z;

   break$(z.l >= -0.001);

*  pattern that might improve the master model found
   aip(i,pi) = round(y.l(i));
   pp(pi)    = yes;
   pi(p)     = pi(p-1);
);
display 'lower bound for number of rolls', master.objVal;

option solPrint = on;

solve master using mip minimizing z;

Parameter
   patrep 'solution pattern report'
   demrep 'solution demand supply report';

patrep('# produced',p) = round(xp.l(p));
patrep(i,p)$patrep('# produced',p) = aip(i,p);
patrep(i,'total') = sum(p, patrep(i,p));
patrep('# produced','total') = sum(p, patrep('# produced',p));

demrep(i,'produced') = sum(p, patrep(i,p)*patrep('# produced',p));
demrep(i,'demand')   = d(i);
demrep(i,'over')     = demrep(i,'produced') - demrep(i,'demand');

display patrep, demrep;
