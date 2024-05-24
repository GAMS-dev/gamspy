$eolCom //


* TSP data and incomplete TSP model. The data is problem br17 from TSPLIB.
* (http://www.iwr.uni-heidelberg.de/iwr/comopt/soft/TSPLIB95/TSPLIB.html)

set ii    cities / i1*i17 /
    i(ii) subset of cities
alias (ii,jj),(i,j,k);

table c(ii,jj) cost coefficients (br17 from TSPLIB)
     i1  i2  i3  i4  i5  i6  i7  i8  i9  i10 i11 i12 i13 i14 i15 i16 i17
i1        3   5  48  48   8   8   5   5   3   3   0   3   5   8   8   5
i2    3       3  48  48   8   8   5   5   0   0   3   0   3   8   8   5
i3    5   3      72  72  48  48  24  24   3   3   5   3   0  48  48  24
i4   48  48  74       0   6   6  12  12  48  48  48  48  74   6   6  12
i5   48  48  74   0       6   6  12  12  48  48  48  48  74   6   6  12
i6    8   8  50   6   6       0   8   8   8   8   8   8  50   0   0   8
i7    8   8  50   6   6   0       8   8   8   8   8   8  50   0   0   8
i8    5   5  26  12  12   8   8       0   5   5   5   5  26   8   8   0
i9    5   5  26  12  12   8   8   0       5   5   5   5  26   8   8   0
i10   3   0   3  48  48   8   8   5   5       0   3   0   3   8   8   5
i11   3   0   3  48  48   8   8   5   5   0       3   0   3   8   8   5
i12   0   3   5  48  48   8   8   5   5   3   3       3   5   8   8   5
i13   3   0   3  48  48   8   8   5   5   0   0   3       3   8   8   5
i14   5   3   0  72  72  48  48  24  24   3   3   5   3      48  48  24
i15   8   8  50   6   6   0   0   8   8   8   8   8   8  50       0   8
i16   8   8  50   6   6   0   0   8   8   8   8   8   8  50   0       8
i17   5   5  26  12  12   8   8   0   0   5   5   5   5  26   8   8
*
* for computational work with simple minded
* algorithm we can restrict size of problem
* and define the model over a subset of all cities.
*
*
variables x(ii,jj)  decision variables - leg of trip
          z         objective variable;
binary variable x;

equations objective   total cost
          rowsum(ii)  leave each city only once
          colsum(jj)  arrive at each city only once;
*
*
* the assignment problem is a relaxation of the TSP
*
objective.. z =e= sum((i,j), c(i,j)*x(i,j));

rowsum(i).. sum(j, x(i,j)) =e= 1;
colsum(j).. sum(i, x(i,j)) =e= 1;

* exclude diagonal
*
x.fx(ii,ii) = 0;



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
