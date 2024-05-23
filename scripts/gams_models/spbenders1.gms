$title Stochastic Benders - Sequential GAMS Loop (SPBENDERS1,SEQ=418)

$onText
This example demonstrates a stochastic Benders implementation for the
simple transport example.

This is the first example of a sequence of stochastic Benders
implementations using various methods to solve the master and
subproblem.

This first example implements the stochastic Benders algorithm using
sequential solves of the master and subproblems in a GAMS loop.

Keywords: linear programming, stochastic Benders algorithm, transportation
          problem
$offText

Set
   i 'factories'            / f1*f3 /
   j 'distribution centers' / d1*d5 /;

Parameter
   capacity(i) 'unit capacity at factories'
               / f1 500, f2 450, f3 650 /
   demand(j)   'unit demand at distribution centers'
               / d1 160, d2 120, d3 270, d4 325, d5 700 /
   prodcost    'unit production cost'                    / 14 /
   price       'sales price'                             / 24 /
   wastecost   'cost of removal of overstocked products' /  4 /;

Table transcost(i,j) 'unit transportation cost'
          d1    d2    d3    d4    d5
   f1   2.49  5.21  3.76  4.85  2.07
   f2   1.46  2.54  1.83  1.86  4.76
   f3   3.26  3.08  2.60  3.76  4.45;

$ifThen not set useBig
   Set s 'scenarios' / lo, mid, hi /;

   Table ScenarioData(s,*) 'possible outcomes for demand plus probabilities'
             d1   d2   d3   d4   d5  prob
      lo    150  100  250  300  600  0.25
      mid   160  120  270  325  700  0.50
      hi    170  135  300  350  800  0.25;
$else
$  if not set nrScen $set nrScen 10
   Set s 'scenarios' / s1*s%nrScen% /;
   Parameter ScenarioData(s,*) 'possible outcomes for demand plus probabilities';
   option seed = 1234;
   ScenarioData(s,'prob') = 1/card(s);
   ScenarioData(s,j)      = demand(j)*uniform(0.6,1.4);
$endIf

* Benders master problem
$if not set maxiter $set maxiter 25
Set
   iter             'max Benders iterations' / 1*%maxiter% /
   itActive(iter)   'active Benders cuts';

Parameter
   cutconst(iter)   'constants in optimality cuts'    / #iter    0 /
   cutcoeff(iter,j) 'coefficients in optimality cuts' / #iter.#j 0 /;

Variable
   ship(i,j)        'shipments'
   product(i)       'production'
   received(j)      'quantity sent to market'
   zmaster          'objective variable of master problem'
   theta            'future profit';

Positive Variable ship;

Equation
   masterobj        'master objective function'
   production(i)    'calculate production in each factory'
   receive(j)       'calculate quantity to be send to markets'
   optcut(iter)     'Benders optimality cuts';

masterobj..
   zmaster =e= theta - sum((i,j), transcost(i,j)*ship(i,j))
                     - sum(i, prodcost*product(i));

receive(j)..    received(j) =e= sum(i, ship(i,j));

production(i).. product(i)  =e= sum(j, ship(i,j));

optcut(itActive)..
   theta =l= cutconst(itActive) + sum(j, cutcoeff(itActive,j)*received(j));

product.up(i) = capacity(i);

Model masterproblem / all /;

* Benders' subproblem
Variable
   sales(j)   'sales (actually sold)'
   waste(j)   'overstocked products'
   zsub       'objective variable of sub problem';

Positive Variable sales, waste;

Equation
   subobj     'subproblem objective function'
   selling(j) 'part of received is sold'
   market(j)  'upperbound on sales';

subobj..     zsub =e= sum(j, price*sales(j)) - sum(j, wastecost*waste(j));

selling(j).. sales(j) + waste(j) =e= received.l(j);

market(j)..  sales(j) =l= demand(j);

Model subproblem / subobj, selling, market /;

* Benders loop
Scalar
   rgap       'relative gap'       /    0 /
   lowerBound 'global lower bound' / -inf /
   upperBound 'global upper bound' / +inf /
   objMaster                       /    0 /
   objSub                          /    0 /;

option limRow = 0, limCol = 0, solPrint = silent, solver = cplex, solveLink = %solveLink.loadLibrary%;

received.l(j) = 0;
objMaster     = 0;

$if not set rtol $set rtol 0.001
loop(iter,
   objSub = 0;
   loop(s,
      demand(j) = scenarioData(s,j);
      solve subproblem maximizing zsub using lp;
      objSub = objSub + ScenarioData(s,'prob')*zsub.l;
      cutconst(iter)   = cutconst(iter)   + ScenarioData(s,'prob')*sum(j, market.m(j)*demand(j));
      cutcoeff(iter,j) = cutcoeff(iter,j) + ScenarioData(s,'prob')*selling.m(j);
   );
   itActive(iter) = yes;
   if(lowerBound < objMaster + objSub, lowerBound = objMaster + objSub;);
   rgap = (upperBound - lowerBound)/(1 + abs(upperBound));
   break$(rgap < %rtol%);
   solve masterproblem maximizing zmaster using lp;
   upperBound = zmaster.l;
   objMaster  = zmaster.l - theta.l;
);
abort$(rgap >= %rtol%) 'need more iterations', lowerbound, upperbound;
display 'optimal solution', lowerbound, upperbound;
