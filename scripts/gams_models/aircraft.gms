$title Aircraft Allocation under uncertain Demand (AIRCRAF,SEQ=8)

$onText
The objective of this model is to allocate aircrafts to routes to maximize
the expected profit when traffic demand is uncertain. Two different
formulations are used, the delta and the lambda formulation.


Dantzig, G B, Chapter 28. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

Keywords: linear programming, aircraft managing, allocation problem
$offText

Set
   i 'aircraft types and unassigned passengers' / a*d /
   j 'assigned and unassigned routes'           / route-1*route-5 /
   h 'demand states'                            / 1*5 /;

Alias (h,hp);

Table dd(j,h) 'demand distribution on route j'
               1    2    3    4    5
   route-1   200  220  250  270  300
   route-2    50  150
   route-3   140  160  180  200  220
   route-4    10   50   80  100  340
   route-5   580  600  620          ;

Table lambda(j,h) 'probability of demand state h on route j'
              1    2    3   4   5
   route-1   .2  .05  .35  .2  .2
   route-2   .3  .7
   route-3   .1  .2   .4   .2  .1
   route-4   .2  .2   .3   .2  .1
   route-5   .1  .8   .1         ;

Table c(i,j) 'costs per aircraft (1000s)'
       route-1  route-2  route-3  route-4  route-5
   a        18       21       18       16       10
   b                 15       16       14        9
   c                 10                 9        6
   d        17       16       17       15       10;

Table p(i,j) 'passenger capacity of aircraft i on route j'
       route-1  route-2  route-3  route-4  route-5
   a        16       15       28       23       81
   b                 10       14       15       57
   c                  5                 7       29
   d         9       11       22       17       55;

Parameter
   aa(i)      'aircraft availability' / a 10, b 19, c 25, d 15 /
   k(j)       'revenue lost (1000 per 100 bumped)' / (route-1,route-2) 13
                                                     (route-3,route-4)  7
                                                      route-5           1 /
   ed(j)      'expected demand'
   gamma(j,h) 'probability of exceeding demand increment h on route j'
   deltb(j,h) 'incremental passenger load in demand states';

ed(j)      = sum(h, lambda(j,h)*dd(j,h));
gamma(j,h) = sum(hp$(ord(hp) >= ord(h)), lambda(j,hp));
deltb(j,h) = (dd(j,h) - dd(j,h-1))$dd(j,h);

display ed, gamma, deltb;

Positive Variable
   x(i,j) 'number of aircraft type i assigned to route j'
   y(j,h) 'passengers actually carried'
   b(j,h) 'passengers bumped'
   oc     'operating cost'
   bc     'bumping cost';

Free Variable phi 'total expected costs';

Equation
   ab(i)   'aircraft balance'
   db(j)   'demand balance'
   yd(j,h) 'definition of boarded passengers'
   bd(j,h) 'definition of bumped passengers'
   ocd     'operating cost definition'
   bcd1    'bumping cost definition: version 1'
   bcd2    'bumping cost definition: version 2'
   obj     'objective function';

ab(i)..   sum(j, x(i,j)) =l= aa(i);

db(j)..   sum(i, p(i,j)*x(i,j)) =g= sum(h$deltb(j,h), y(j,h));

yd(j,h).. y(j,h) =l= sum(i, p(i,j)*x(i,j));

bd(j,h).. b(j,h) =e= dd(j,h) - y(j,h);

ocd..     oc  =e= sum((i,j), c(i,j)*x(i,j));

bcd1..    bc  =e= sum(j, k(j)*(ed(j)-sum(h, gamma(j,h)*y(j,h))));

bcd2..    bc  =e= sum((j,h), k(j)*lambda(j,h)*b(j,h));

obj..     phi =e= oc + bc;

Model
   alloc1 'aircraft allocation version 1' / ab, db,     ocd, bcd1, obj /
   alloc2 'aircraft allocation version 2' / ab, yd, bd, ocd, bcd2, obj /;

y.up(j,h) = deltb(j,h);

solve alloc1 minimizing phi using lp;

display y.l;

y.up(j,h) = +inf;

solve alloc2 minimizing phi using lp;

display y.l;
