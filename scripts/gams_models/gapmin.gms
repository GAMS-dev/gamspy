$title Lagrangian Relaxation for Generalized Assignment (GAPMIN,SEQ=182)

$onText
A general assignment problem is solved via Lagrangian Relaxation
by dualizing the multiple choice constraints and solving
the remaining knapsack subproblems.

The data for this problem are taken from Martello.
The optimal value is 223 and the optimal solution is:
    1 1 4 2 3 5 1 4 3 5, where
in columns 1 and 2, the variable in the first row is equal to 1,
in column 3, the variable in the fourth row is equal to 1, etc...


Martello, S, and Toth, P, Knapsack Problems: Algorithms and Computer
Implementations. John Wiley and Sons, Chichester, 1990.

Guignard, M, and Rosenwein, M, An Improved Dual-Based Algorithm for
the Generalized Assignment Problem. Operations Research 37 (1989), 658-663.


 --- original model definition

Keywords: mixed integer linear programming, relaxed mixed integer linear
          programming, general assignment problem, lagrangian relaxation, knapsack
$offText

$eolCom //

$sTitle Original Model Definition
Set
   i 'resources'
   j 'items';

Variable
   x(i,j) 'assignment of i to j'
   z      'total cost of assignment';

Binary Variable x;

Equation
   capacity(i) 'resource availability'
   choice(j)   'assignment constraint.. one resource per item'
   defz        'definition of total cost';

Parameter
   a(i,j) 'utilization of resource i by item j'
   f(i,j) 'cost of assigning item j to resource i'
   b(i)   'available resources';

capacity(i).. sum(j, a(i,j)*x(i,j)) =l= b(i);

choice(j)..   sum(i, x(i,j)) =e= 1;

defz..        z =e= sum((i,j), f(i,j)*x(i,j));

Model assign 'original assignment model' / capacity, choice, defz /;

* data for Martello model
Set
   i         'resources'          / r1*r5  /
   j         'items'              / i1*i10 /
   xopt(i,j) 'optimal assignment' / r1.(i1,i2,i7),r2.i4, r3.(i5,i9), r4.(i3,i8), r5.(i6,i10) /;

Table a(i,j) 'utilization of resource i by item j'
        i1  i2  i3  i4  i5  i6  i7  i8  i9 i10
   r1   12   8  25  17  19  22   6  22  20  25
   r2    5  15  15  14   7  11  14  16  17  15
   r3   21  24  13  24  12  16  23  20  15   5
   r4   23  17  10   6  24  20  15  10  19   9
   r5   17  20  15  16   5  13   7  16   8   5;

Table f(i,j) 'cost of assigning item j to resource i'
        i1  i2  i3  i4  i5  i6  i7  i8  i9 i10
   r1   16  26  30  47  18  19  33  37  42  31
   r2   38  42  15  21  26  11  11  50  24  19
   r3   48  17  14  22  14  18  47  32  17  42
   r4   22  32  28  39  37  23  25  12  44  17
   r5   31  42  31  40  16  15  29  31  44  41;

Parameter b(i) 'available resources' / r1 28, r2 20, r3 27, r4 24, r5 19 /;

$onText
* if one wants to check the data, one can
* solve the MIP problem, this is just a check

assign.optCr = 0;

solve assign minimizing z using mip;

if(sum(xopt, x.l(xopt) <> 1),
   abort '*** Something wrong with this solution', x.l, xopt);

$offText

$sTitle Relaxed Problem Definition and Subgradient Optimization
* Lagrangian subproblem definition
* uses dynamic set to define WHICH knapsack to solve
Set
   id(i) 'dynamic version of i used to define a subset of i'
   iter  'subgradient iteration index' / iter1*iter20 /;

Alias (i,ii);

Parameter
   w(j)   'Lagrangian multipliers'
   improv 'has the Lagrangian bound improved over the previous iterations';

Variable zlrx 'relaxed objective';

Equation
   knapsack(i) 'capacity with dynamic sets'
   defzlrx     'definition of zlrx';

knapsack(id).. sum(j, a(id,j)*x(id,j)) =l= b(id);

defzlrx..      zlrx =e= sum((id,j), (f(id,j) - w(j))*x(id,j));

Model pknap / knapsack, defzlrx /;

Scalar
   target 'target objective function value'
   alpha  'step adjuster'             /  1 /
   norm   'norm of slacks'
   step   'step size for subgradient' / na /
   zfeas  'value for best known solution or valid upper bound'
   zlr    'Lagrangian objective value'
   zl     'Lagrangian objective value'
   zlbest 'current best Lagrangian lower bound'
   count  'count of iterations without improvement'
   reset  'reset count counter'   / 5    /
   tol    'termination tolerance' / 1e-5 /
   status 'outer loop status'     / 0    /;

Parameter
   s(j)           'slack variable'
   report(iter,*) 'iteration log'
   xrep(j,i,*)    'x iteration report'
   srep(iter,j)   'slack report'
   wrep(iter,j)   'w iteration report';

* calculate initial Lagrangian multipliers
* There are many possible ways to find initial multipliers.
* The choice of initial multipliers is very important for the
* overall performance. The marginals of the relaxed problem
* are often used to initialize the multipliers. Another choice
* is simply to start with zero multipliers.

* replace 'default' with solver of your choice.
option mip = default, rmip = default;

File results 'writes iteration report' / solution /;
put  results 'solvers used: RMIP = ' system.rmip /
             '               MIP = ' system.mip  /;

* solve relaxed problem to get initial multipliers
* Note that different solvers get different dual solutions
* which are not as good as a zero set of initial multipliers.

solve assign minimizing z using rmip;
put / 'RMIP objective value = ', z.l:12:6 /;

if(assign.modelStat = %modelStat.optimal%,
   status = %modelStat.optimal%                        // everything ok
else
   abort '*** relaxed MIP not optimal',
         '    no subgradient iterations', x.l;
);
xrep(j,i,'initial') = x.l(i,j);
xrep(j,i,'optimal') = 1$xopt(i,j);

Parameter wopt(j) 'an optimal set of multipliers'
                  / i1 35, i2 40, i3 60, i4 69, i5  21
                    i6 49, i7 42, i8 47, i9 64, i10 46 /;

zlbest = z.l;

* use RMIP duals
w(j) = choice.m(j);

* use optimal duals
* w(j) = wopt(j);

* use zero starting point
* w(j)   = 0;
* zlbest = 0;

put / / 'zlbest                    objective value  = ', zlbest:12:6;
put / / "Dual values on assignment constraint"/ ;
loop(j, put / "w('",j.tl,"') =  ", w(j):16:6 ";";);

* one needs a value for zfeas
* one can compute a valid upper bound as follows:

zfeas = sum(j, smax(i, f(i,j)));
put / / 'zfeas quick and dirty bound obj value      = ', zfeas:12:6;
display 'a priori upper bound', zfeas;

$onText
another alternative to compute a value for zfeas is
to solve gapmin by B-B and stop
at first 0-1 feasible solution found
using gapmin.optCr = 1, as follows

assign.optCr = 1;
assign.solPrint = %solPrint.quiet%;

!!!

solve assign minimizing z using mip;
zfeas = min(zfeas,z.l);
display 'final zfeas', zfeas;
display 'heuristic solution by B-B ', x.l, z.l;
put / 'zfeas IP solution bound objective value    = ', zfeas.l:12:6;
$offText

put / / / 'Iteration         New Bound   Previous Bound            norm      abs(zl-zf)'/;

* then keep the smaller of the two values as zfeas
pknap.optCr = 0;                    // ask for global solution
pknap.solPrint = %solPrint.quiet%;  // turn off all solution output

*============================================================================*
*                                                                            *
*  beginning of subgradient loop                                             *
*                                                                            *
*============================================================================*
id(i)  = no;  // initially empty
count  = 1;
alpha  = 1;

display status;

loop(iter$(status = 1),    // i.e., repeat while status is 1
*  solve Lagrangian subproblems by solving nonoverlapping knapsack
*  problems. Note the use of the dynamic set id(i) which will
*  contain the current knapsack descriptor.
   zlr = 0;
   loop(ii,
      id(ii) = yes;                          // assume id was empty
      solve pknap using mip minimizing zlrx;
      zlr    = zlr + zlrx.l;
      id(ii) = no;                           // make set empty again
   );
   improv = 0;
   zl     = zlr + sum(j, w(j));
   improv$(zl > zlbest) = 1;                 // is zl better than zlbest?
   zlbest = max(zlbest,zl);
   s(j)   = 1 - sum(i, x.l(i,j));            // subgradient
   norm   = sum(j, sqr(s(j)));

   status$(norm < tol)                             = 2;
   status$(abs(zlbest - zfeas) < 1e-4)             = 3;
   status$(pknap.modelStat <> %modelStat.optimal%) = 4;
   put results / iter.tl, zl:16:6, zlbest:16:6, norm:16:6, abs(zlbest - zfeas):16:6;
   if((status = 2),
      put / /"subgr. method has converged, status = ",status:5:0/ /;
      put / /"last solution found is optimal for IP problem"/ /;
   );    // end if
   if((status = 3),
      put / /"subgr. method has converged, status = ",status:5:0/ /;
      put / /"no duality gap, best sol. found is optimal "/ /;
   );    // end if
   if((status = 4),
      put / /"something wrong with last Lag. subproblem"/ /;
      put / /"status = ",status:5:0/ /;
   );    // end if

   report(iter,'zlr')    = zlr;
   report(iter,'zl')     = zl;
   report(iter,'zlbest') = zlbest;
   report(iter,'norm')   = norm;
   report(iter,'step')   = step;

   wrep(iter,j)   = w(j);
   srep(iter,j)   = s(j);
   xrep(j,i,iter) = x.l(i,j);

   if(status = 1,
      target = (zlbest + zfeas)/2;
      step   = (alpha*(target - zl)/norm)$(norm > tol);
      w(j)   = w(j) + step*s(j);
      if(count > reset,         // too many iterations w/o improvement
         alpha = alpha/2;
         count = 1;
      else
         if(improv,             // reset count if improvement
            count = 1;
         else
            count = count + 1;  // update count if no improvement
         );
      );
   );
);                              // end loop iter

display report, wrep, srep, xrep;
put results / / "Dual values on assignment constraint" /;
loop(j, put /  "w('",j.tl,"') =  ", w(j):16:6  ";";);
put / /"best Lagrangian bound   =   ", zlbest:10:5;
