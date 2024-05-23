$title Equilibrium of System with Piecewise Linear Springs (SPRINGCHAIN,SEQ=431)

$onText
 This model finds the shape of a hanging chain consisting of
 N springs and N-1 nodes. Each spring buckles under compression and each
 node has a weight hanging from it. The springs are assumed 
 weightless. The goal is to minimize the potential energy of the
 system.

 We use rotated quadratic cone constraints to model the extension 
 of each spring.


 M. Lobo, L. Vandenberghe, S. Boyd, and H. Lebret,  
 Applications of second-order cone programming, Linear Algebra and its 
 Applications, 284:193-228, November 1998, Special Issue on Linear Algebra 
 in Control, Signals and Image Processing. 
$offText

* Number of chainlinks
$if not set N $set N 10
$eval NM1 %N%-1

Set n "spring index"  /n0*n%N%/;

Scalars
    a_x   "x coordinate of beginning node" /  0/
    a_y   "y coordinate of beginning node" /  0/
    b_x   "x coordinate of end node"       /  2/
    b_y   "y coordinate of end node"       / -1/
    L0    "rest length of each spring"     /  [2*sqrt(sqr(a_x-b_x) + sqr(a_y-b_y))/%N%]/
    g     "acceleration due to gravity"    /  9.8/
    k     "stiffness of springs"           /100/;

Parameters
    m(n)  "mass of each hanging node"      /n1*n%NM1% 1/;

Variables
    obj
    x(n)       "x-coordinates of nodes"
    y(n)       "y-coordinates of nodes"
    delta_x(n)
    delta_y(n) 
    unit;
    
Positive variable 
    t_L0(n)
    t(n)    "extension of each spring"
    v;

Equations
    pot_energy
    delta_x_eq(n)
    delta_y_eq(n)
    link_L0(n)
    link_up(n)
    cone_eq;

pot_energy..       obj =E=    sum(n$[ord(n)>1 and ord(n)<card(n)], m[n]*g*y[n]) + k*v;

delta_x_eq(n)..    delta_x(n)     =E= x[n] - x[n-1];
delta_y_eq(n)..    delta_y(n)     =E= y[n] - y[n-1];

link_L0(n)..       t_L0[n]        =E= L0 + t[n];
link_up(n)$[ord(n)>1]..       
                   sqr(t_L0[n])   =G= sqr(delta_x[n]) + sqr(delta_y[n]);

cone_eq..          2*v*unit       =G= sum(n$[ord(n)>1], sqr(t[n]));

Model spring /all/;

x.L(n) = ( (ord(n)-1)/%N% )*b_x + (ord(n)/%N%)*a_x;
y.L(n) = ( (ord(n)-1)/%N% )*b_y + (ord(n)/%N%)*a_y;

x.FX['n0']   = a_x;
y.FX['n0']   = a_y;
x.FX['n%N%'] = b_x;
y.FX['n%N%'] = b_y;

unit.fx = 1;

Solve spring using qcp minimizing obj;

