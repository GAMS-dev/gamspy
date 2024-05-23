$title Traveling Salesman Problem - Four (TSP4,SEQ=180)

$onText
This is the fourth problem in a series of traveling salesman
problems. Here we revisit TSP1 and generate smarter cuts.
The first relaxation is the same as in TSP1.


Kalvelagen, E, Model Building with GAMS. forthcoming

de Wetering, A V, private communication.

Keywords: mixed integer linear programming, traveling salesman problem, iterative
          subtour elimination
$offText

$eolCom //

$include br17.inc

* For this algorithm we can try a larger subset of 12 cities.
Set i(ii) / i1*i12 /;

* options. Make sure MIP solver finds global optima.
option optCr = 0;

Model assign / objective, rowsum, colsum /;

solve assign using mip minimizing z;

* find and display tours
Set t 'tours' / t1*t17 /;
abort$(card(t) < card(i)) "Set t is possibly too small";

Set
   tour(i,j,t) 'subtours'
   visited(i)  'flag whether a city is already visited';

Singleton Set
   from(i) 'contains always one element: the from city'
   next(j) 'contains always one element: the to city'
   tt(t)   'contains always one element: the current subtour';

Alias (i,ix);

* initialize
from(i)$(ord(i) = 1) = yes;    // turn first element on
tt(t)$  (ord(t) = 1) = yes;    // turn first element on

* subtour elimination by adding cuts
Set cc / c1*c1000 /;

Alias(cc,ccc); // we allow up to 1000 cuts

Set
   curcut(cc)  'current cut always one element'
   allcuts(cc) 'total cuts';

Parameter
   cutcoeff(cc, i, j)
   rhs(cc)
   nosubtours 'number of subtours';

Equation cut(cc) 'dynamic cuts';

cut(allcuts).. sum((i,j), cutcoeff(allcuts,i,j)*x(i,j)) =l= rhs(allcuts);

Model tspcut / objective, rowsum, colsum, cut /;

curcut(cc)$(ord(cc) = 1) = yes;

Scalar ok;

loop(ccc,
*  initialize
   from(i)$(ord(i) = 1) = yes;    // turn first element on
   tt(t)$(  ord(t) = 1) = yes;    // turn first element on
   tour(i,j,t) = no;
   visited(i)  = no;
   loop(i,
      next(j)$(x.l(from,j) > 0.5) = yes;  // check x.l(from,j) = 1 would be dangerous
      tour(from,next,tt) = yes;           // store in table
      visited(from) = yes;                // mark city 'from' as visited
      from(j) = next(j);
      if(sum(visited(next),1) > 0,         // if already visited...
         tt(t) = tt(t-1);
         loop(ix$(not visited(ix)),       // find starting point of new subtour
            from(ix) = yes;
         );
      );
   );
   display tour;
   nosubtours = sum(t, max(0,smax(tour(i,j,t),1)));
   display nosubtours;

   break$(nosubtours = 1); // done: no subtours

   // introduce cut
   loop(t$(ord(t) <= nosubtours),
      rhs(curcut) = -1;
      loop(tour(i,j,t),
         cutcoeff(curcut, i, j)$(x.l(i,j) > 0.5) = 1;
* not needed due to nature of assignment constraints
*        cutcoeff(curcut, i, j)$(x.l(i,j) < 0.5) = -1;
         rhs(curcut) = rhs(curcut) + 1;
      );
      allcuts(curcut) = yes;   // include this cut in set
      curcut(cc) = curcut(cc-1);
   );
   solve tspcut using mip minimizing z;
   tspcut.solPrint = %solPrint.quiet%;
   tspcut.limRow   = 0;
   tspcut.limCol   = 0;
);

display x.l;
abort$(nosubtours <> 1) "Too many cuts needed";
