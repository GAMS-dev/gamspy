$title Truss Toplogy Design with Multiple Loads (TRUSSM,SEQ=432)

$onText
A structure of n linear elastic bars connects a set of m nodes.
The task is to size the bars, i.e. determine t(i), the volume
of the bars, that yield the stiffest truss subject to constraints 
such as total weight limit and k different (nonsimultaneous) loading 
scenarios to be satisfied. For example, the different load scenarios 
for a bridge could include rush hour traffic, night traffic, earthquake 
and side wind.

The model is given as a conic program. The cone implementation comes
from Ben-Tal and Nemirovski.

Suppose we have a truss of n bars and m nodes. Now consider a set of 
k fixed externally applied nodal forces f(k)=[f1, .., fn].

Let d_i denote the small node displacement resulting from the force on
each node i. The objective is to maximize the stiffness of the truss,
which is equivalent to minimizing the elastic stored energy 0.5*f^T*d,
subject to some maximum volume restriction on the truss. 

Using the formulation given in Ben-Tal and Nemirovski (2001), we can
model this as the second order cone problem:

           minimize      tau
           subject to
                         sum(i, t(i)) <= maxvolume

                         s(i,k)^2 <= 2*t(i)*sigma(i,k) 
                         sum(i, sigma(i,k)) <= tau 
                         sum(i,k) s(i,k)*b(i)) <= f(k)

The first constraint is the material volume limitation. The latter 3
constraints and the objective are the compliance constraints, which are
equivalent to minimization of the elastic potential energy under a given
load.


A. Ben-Tal and A. Nemirovski, Lectures on Modern Convex Optimization: 
Analysis, Algorithms, and Engineering Applications, MPS/SIAM Series 
on Optimization, SIAM Press, 2001. 

M.S. Lobo, L. Vandenberghe, S. Boyd, and H. Lebret, "Applications of
Second-order Cone Programming", Linear Algebra and its Applications, 
Special Issue on Linear Algebra in Control, Signals and Image Processing. 
284 (1998) 193-228.
$offText

Set i  "bars"           /i1*i5/
    j  "nodes"          /j1*j4/
    k  "load scenarios" /k1*k3/;

Table f(j,k) "nodal force for scenario k on node j" 
          k1        k2        k3
j1    0.0008    1.0668    0.2944
j2    0.0003    0.0593   -1.3362
j3   -0.0006   -0.0956    0.7143
j4   -1.0003   -0.8323    1.6236; 

Table b(j,i) "stiffness parameter for bar i" 
      i1    i2    i3    i4   i5
j1    1.0   0     0.5   0    0
j2    0     0    -0.5  -1.0  0
j3    0     0.5   0     0    1.0
j4    0     0.5   0     1.0  0;

Scalar
    maxvolume "maximum truss volume"  /10/; 

Variables
    tau        "objective"
    s(i,k)     "stress on bar i under load scenario k, which is elongation times cross-sectional area of bar";
Positive variables
    tk(i,k)    "volume of truss bar i under load scenario k" 
    t(i)       "volume of truss bar i" 
    sigma(i,k) "required cross-sectional area of bar i under load k"; 

Equations
    volumeeq(i,k)  "compute volume t"
    deftk(i,k)     "assignment of tk to keep cones disjoint"
    reseq(k)       "resource restriction on truss"
    trusscomp      "compliance of truss"
    stiffness(j,k) "stiffness requirement for bar j under load k";

* Note that conic variables can occur only once, i.e. the cones are disjoint
volumeeq(i,k)..  2*tk(i,k)*sigma(i,k)  =G= sqr(s(i,k));

deftk(i,k)..     tk(i,k)               =E= t(i);

reseq(k)..       sum(i, sigma(i,k))    =L= tau;

trusscomp..      sum(i, t(i))          =L= maxvolume;

stiffness(j,k).. sum(i, s(i,k)*b(j,i)) =E= f(j,k);  


Model truss  / all /;

* Solve using all loads
sigma.l(i,k) = uniform(0.1,1);
Solve truss using qcp minimizing tau;

* Resolve with only a single load (k1)
f(j,"k2") = 0;
f(j,"k3") = 0;

Solve truss using qcp minimizing tau;
