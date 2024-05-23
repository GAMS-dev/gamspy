$title Robust linear programming as an SOCP (ROBUSTLP,SEQ=416)

$onText
Consider a linear optimization problem of the form
min_x c^Tx s.t. a_i^Tx <= b_i, i=1,..,m.

In practice, the coefficient vectors a_i may not be known perfectly,
as they are subject to noise. Assume that we only know that a_i in E_i,
where E_i are given ellipsoids. In robust optimization, we seek to minimize
the original objective, but we insist that each constraint be satisfied,
irrespective of the choice of the corresponding vector a_i in E_i.
We obtain the second-order cone optimization problem
min_x c^Tx s.t. a'_i^Tx + ||R_i^Tx|| <= b_i, i=1,..,m,
where E_i = { a'_i + R_iu | ||u|| <= 1}. In the above, we observe that
the feasible set is smaller than the original one, due to the terms involving
the l_2-norms.

The figure above illustrates the kind of feasible set one obtains in a particular
instance of the above problem, with spherical uncertainties (that is, all the
ellipsoids are spheres, R_i = rho I for some rho >0). We observe that the robust
feasible set is indeed contained in the original polyhedron.

In this particular example we allow coefficients A(i,*) to vary in an ellipsoid.
The robust LP is reformulated as a SOCP.

Contributed by Michael Ferris, University of Wisconsin, Madison


Lobo, M S, Vandenberghe, L, Boyd, S, and Lebret, H, Applications of
Second Order Cone Programming. Linear Algebra and its Applications,
Special Issue on Linear Algebra in Control, Signals and Image
Processing. 284 (November, 1998).

Keywords: linear programming, quadratic constraint programming, robust optimization,
          second order cone programming
$offText

$if not set mu $set mu 1.0e-2

Set
   i / 1*7 /
   j / 1*4 /;

Parameter b(i), c(j), A(i,j);
b(i) =  1;
c(j) = -1;

option seed = 0;
A(i,j) = uniform(0,1);

Variable obj, x(j);

Equation defobj, cons(i);

defobj..  obj =e= sum(j, c(j)*x(j));

cons(i).. sum(j, A(i,j)*x(j)) =l= b(i);

Model lpmod / defobj, cons /;

solve lpmod using lp min obj;

Parameter results(*,*);
results('lp',j)     = x.l(j);
results('lp','obj') = obj.l;

Scalar mu / %mu% /;

Positive Variable lambda(j), gamma(j);

Equation lpcons(i), defdual(j);

* A(i,*) \in A(i,*) + [-mu(i) 1, mu(i) 1] (infty norm ball)
* constraint is mu(i) * norm(x)_1  + Ax <= b  (just use one mu here)
* just implement one norm (dual of inf norm) using lambda and gamma
lpcons(i)..  mu*sum(j, lambda(j) + gamma(j)) + sum(j, A(i,j)*x(j)) =l= b(i);

defdual(j).. lambda(j) - gamma(j) =e= x(j);

Model lproblp / defobj, lpcons, defdual /;

solve lproblp using lp min obj;

results('roblp',j)     = x.l(j);
results('roblp','obj') = obj.l;

Alias (j,k);

Parameter P(i,j,k);
P(i,j,j) = %mu%;

Variable y(i), v(i,k);

Equation defrhs(i), defv(i,k), socpcons(i);

defrhs(i).. y(i) =e= b(i) - sum(j, A(i,j)*x(j));

defv(i,k).. v(i,k) =e= sum(j, P(i,j,k)*x(j));

Equation socpqcpcons(i);

socpqcpcons(i).. sqr(y(i)) =g= sum(k, sqr(v(i,k)));

Model roblpqcp / defobj, socpqcpcons, defrhs, defv /;

y.lo(i) = 0;

option qcp = cplex;

solve roblpqcp using qcp min obj;

results('qcp',j)     = x.l(j);
results('qcp','obj') = obj.l;

display results;
