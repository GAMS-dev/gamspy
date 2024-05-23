$title A Transportation Problem with discretized Economies of Scale (TRNSPWL,SEQ=351)

$onText
This problem finds a least cost shipping schedule that meets
requirements at markets and supplies at factories. This instance
applies economies of scale which results in a non-convex
objective. This is an extension of the trnsport model in the GAMS
Model Library.

The original nonlinear term is "sum((i,j), c(i,j)*sqrt(x(i,j)))".
We use the following discretization f(x) of sqrt(x)

  For x<=50:  f(x) = 1/sqrt(50)*x,
  for x>=400: f(x) = (sqrt(600)-sqrt(400))/200*(x-400) + sqrt(400)
  in between we discretize with linear interpolation between points

This discretization has some good properties:
  0) f(x) is a continuous function
  1) f(0)=0, otherwise we would pick up a fixed cost even for unused connections
  2) a fine representation in the reasonable range of shipments (between 50 and 400)
  3) f(x) underestimates sqrt in the area of x=0 to 600. Past that is overestimates sqrt.

The model is organized as follows:
  1) We set a starting point for the NLP solver so it will get stuck
     in local optimum that is not the global optimum.

  2) We use three formulations for representing piecewise linear
     functions all based on the same discretization.

     a) a formulation with SOS2 variables. This formulation mainly is
        based on the convex combination of neighboring
        points. Moreover, the domain of the discretization can be
        unbounded: we can assign a slope in the (potentially
        unbounded) first and last segment.

     b) a formulation with SOS2 variables based on convex combinations
        of neighboring points. This formulation requires a bounded
        region for the discretization. Here we discretize between 0 and
        600.

     c) a formuation with binary variables. This also requires the
        domain to be bounded, but it does not rely on the convex
        combination of neighboring points. There are examples, where
        this formulation solves much faster than the formulation b).

     In this example x is clearly bounded by 0 from below and
     min(smax(i,a(i),smax(j,b(j)) from above, so formulation b and c
     are sufficient and perform better on this particular model and
     instance. We added the formulation a to demonstrate how to model
     an unbounded discretization, in case there are no derived
     bounds. The formulation a can be easily adjusted to accommodate
     problems where only one end of the discretization is unbounded.

  3) We restart the non-convex NLP from the solution of the discretized
     model and hope that the NLP solver finds the global solution.


Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

Keywords: non linear programming, mixed integer linear programming,
          transportation problem, scheduling, economies of scale, non-convex
          objective, special ordered sets
$offText

Set
   i 'canning plants' / seattle,  san-diego /
   j 'markets'        / new-york, chicago, topeka /;

Parameter
   a(i) 'capacity of plant i in cases'
        / seattle    350
          san-diego  600 /

   b(j) 'demand at market j in cases'
        / new-york   325
          chicago    300
          topeka     275 /;

Table d(i,j) 'distance in thousands of miles'
              new-york  chicago  topeka
   seattle         2.5      1.7     1.8
   san-diego       2.5      1.8     1.4;

Scalar f 'freight in dollars per case per thousand miles' / 90 /;

Parameter c(i,j) 'transport cost in thousands of dollars per case';
c(i,j) = f*d(i,j)/1000;

Variable
   x(i,j) 'shipment quantities in cases'
   z      'total transportation costs in thousands of dollars';

Positive Variable x;

Equation
   cost      'define objective function'
   supply(i) 'observe supply limit at plant i'
   demand(j) 'satisfy demand at market j';

cost..      z =e= sum((i,j), c(i,j)*sqrt(x(i,j)));

supply(i).. sum(j, x(i,j)) =l= a(i);

demand(j).. sum(i, x(i,j)) =g= b(j);

Model transport / all /;

* Start the local NLP solver in a local solution that is not globally optimal
x.l('seattle  ','chicago ') =  25;
x.l('seattle  ','topeka  ') = 275;
x.l('san-diego','new-york') = 325;
x.l('san-diego','chicago ') = 275;

Scalar localopt 'objective of local optimum that is not globally optimal';

option nlp = conopt;

solve transport using nlp minimizing z;

localopt = z.l;

* The first model (formulation a) implements a piecewise linear
* approximation based on the convex combination of neighboring points
* using SOS2 variables with unbounded segments at the beginning and
* end of the discretization
Set
   s     'SOS2 elements' / slope0, s1*s6, slopeN /
   ss(s) 'sample points' /         s1*s6         /;

Parameter
   p(s)     'x coordinate of sample point'
   sqrtp(s) 'y coordinate of sample point'
   xlow     /  50 /
   xhigh    / 400 /
   xmax;

xmax = smax(i, a(i));

abort$(xmax < xhigh) 'xhigh too big', xhigh, xmax;
abort$(xlow < 0)     'xlow less than 0', xlow;

* Equidistant sampling of the sqrt function with slopes at the beginning and end
p('slope0') = -1;
p(ss)       = xlow + (xhigh-xlow)/(card(ss)-1)*ss.off;
p('slopeN') = 1;

sqrtp('slope0') = -1/sqrt(xlow);
sqrtp(ss)       = sqrt(p(ss));
sqrtp('slopeN') = (sqrt(xmax)-sqrt(xhigh))/(xmax-xhigh);

SOS2     Variable xs(i,j,s);
Positive Variable sqrtx(i,j);

Equation defsos1(i,j), defsos2(i,j), defsos3(i,j), defobjdisc;

defsos1(i,j).. x(i,j)     =e= sum(s, p(s)*xs(i,j,s));

defsos2(i,j).. sqrtx(i,j) =e= sum(s, sqrtp(s)*xs(i,j,s));

defsos3(i,j).. sum(ss, xs(i,j,ss)) =e= 1;

defobjdisc..   z =e= sum((i,j), c(i,j)*sqrtx(i,j));

Model trnsdiscA / supply, demand, defsos1, defsos2, defsos3, defobjdisc /;

option optCr = 0;

solve trnsdiscA min z using mip;

* The next model (formulation b) uses the convex combinations of
* neighboring points but requires the discretization to be bounded
* (here we go from 0 to xmax).
p('slope0') = 0;
p(ss)       = xlow + (xhigh - xlow)/(card(ss) - 1)*ss.off;
p('slopeN') = xmax;
sqrtp(s)    = sqrt(p(s));

* We can just use model trnsdiscA but need to include the first and
* last segment into the set ss that builds the convex combinations.
ss(s) = yes;

solve trnsdiscA min z using mip;

* The next model (formulation c) implements another formulation for a
* piecewise linear function. We need to assume that the domain region
* is bounded. We use the same discretization as in the previous formulation.
Set g(s)   'Segments' / slope0, s1*s6 /;

Parameter
   nseg(s) 'relative increase of x in segment'
   ninc(s) 'relative increase of sqrtx in segment';

nseg(g(s)) = p(s+1) - p(s);
ninc(g(s)) = (sqrtp(s+1) - sqrtp(s));

Variable
   seg(i,j,s) 'shipment in segment'
   gs(i,j,s)  'indicator for shipment in segment';

Binary   Variable gs;
Positive Variable seg;

Equation
   defx(i,j)     'definition of x'
   defsqrt(i,j)  'definition of sqrt'
   defseg(i,j,s) 'segment can only have shipment if indicator is on'
   defgs(i,j)    'select at most one segment';

defx(i,j)..     x(i,j)     =e= sum(g,     p(g)*gs(i,j,g) + nseg(g)*seg(i,j,g));

defsqrt(i,j)..  sqrtx(i,j) =e= sum(g, sqrtp(g)*gs(i,j,g) + ninc(g)*seg(i,j,g));

defseg(i,j,g).. seg(i,j,g) =l= gs(i,j,g);

defgs(i,j)..    sum(g, gs(i,j,g)) =l= 1;

Model trnsdiscB / supply, demand, defx, defsqrt, defseg, defgs, defobjdisc /;

solve trnsdiscB min z using mip;

* Now restart the local solver from this approximate point
solve transport min z using nlp;

* Ensure that we are better off than before
abort$(z.l - localopt > 1e-6) 'we should get an improved solution';
