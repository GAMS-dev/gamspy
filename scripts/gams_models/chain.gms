$title Hanging Chain COPS 2.0 #3 (CHAIN,SEQ=231)

$onText
Find the chain (of uniform density) of length L suspended between two
points with minimal potential energy.

This model is from the COPS benchmarking suite.
See http://www-unix.mcs.anl.gov/~more/cops/.

The number of intervals for the discretization can be specified using
the command line parameter --nh. COPS performance tests have been
reported for nh = 50, 100, 200, 400


Dolan, E D, and More, J J, Benchmarking Optimization
Software with COPS. Tech. rep., Mathematics and Computer
Science Division, 2000.

Cesari, L, Optimization - Theory and Applications. Springer
Verlag, 1983.

Keywords: nonlinear programming, engineering, hanging chain problem, catenary
$offText

$if not set nh $set nh 50

Set nh / i0*i%nh% /;

Alias (nh,i);

Scalar
   L  'length of the suspended chain'      / 4 /
   a  'height of the chain at t=0 (left)'  / 1 /
   b  'height of the chain at t=1 (left)'  / 3 /
   tf 'ODEs defined in [0 tf]'             / 1 /
   h  'uniform interval length'
   n  'number of subintervals'
   tmin;

if(b > a, tmin = 0.25; else tmin = 0.75;);

n = card(nh) - 1;
h = tf/n;

Variable
   x(i)   'height of the chain'
   u(i)   'derivative of x'
   energy 'potential energy';

Equation
   obj
   x_eqn(i)
   length_eqn;

obj..
   energy =e= 0.5*h*sum(nh(i+1), x(i)*sqrt(1 + sqr(u(i))) + x(i+1)*sqrt(1 + sqr(u(i+1))));

x_eqn(i+1)..
   x(i+1) =e= x(i) + 0.5*h*(u(i) + u(i+1));

length_eqn..
   0.5*h*sum(nh(i+1), sqrt(1+sqr(u(i))) + sqrt(1+sqr(u(i+1)))) =e= L;

x.fx('i0')    = a;
x.fx('i%nh%') = b;
x.l(i)        = 4*abs(b-a)*((ord(i)  - 1)/n)*(0.5*((ord(i) - 1)/n) - tmin) + a;
u.l(i)        = 4*abs(b-a)*(((ord(i) - 1)/n) - tmin);

Model chain / all /;

$if set workSpace chain.workSpace = %workSpace%

solve chain using nlp minimizing energy;
