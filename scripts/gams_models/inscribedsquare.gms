$title Inscribed Square Problem (INSCRIBEDSQUARE,SEQ=425)

$onText
The inscribed square problem, also known as the square peg problem or
the Toeplitz' conjecture, is an unsolved question in geometry:

  Does every plane simple closed curve contain all four vertices of
  some square?


This is true if the curve is convex or piecewise smooth and in other
special cases. The problem was proposed by Otto Toeplitz in 1911.
See also https://en.wikipedia.org/wiki/Inscribed_square_problem

This model computes a square of maximal area for a given curve.

Use options --fx and --fy to specify the x and y coordinates of a closed
curve as function in variable t, where t ranges from -pi to pi.
Use option --gnuplot 1 to enable plotting the curve and computed square
with gnuplot (if available and a feasible solution has been found).

Contributor: Benjamin Mueller and Felipe Serrano
$offText

*$set fx sin(t)*cos(t)
*$set fy sin(t)*t

*$set fx cos(t-t*t)*sin(t)-t*t*sin(2*t+3*abs(t))
*$set fy sin(t)*t+0.5*t*t*cos(t)

$if not set fx $set fx sin(t) * cos(t-t*t)
$if not set fy $set fy t * sin(t)

$macro fx(t) %fx%
$macro fy(t) %fy%

$if not set gnuplot $set gnuplot 0

Set i   "corner points of square" / 1*4 /;
Variables
  z     "area of square to be maximized",
  t(i)  "position of square corner points on curve",
  x     "x-coordinate of lower-left corner of square (=fx(t('1')))",
  y     "y-coordinate of lower-left corner of square (=fy(t('1')))";
Positive Variables
  a     "horizontal distance between lower-left and lower-right corner of square",
  b     "vertical distance between lower-left and lower-right corner of square";

t.lo(i) = -pi;
t.up(i) =  pi;

Equation
  obj   "area of square, squared"
  e1x   "define x-coordinate of lower-left corner",
  e1y   "define y-coordinate of lower-left corner",
  e2x   "define x-coordinate of lower-right corner",
  e2y   "define y-coordinate of lower-right corner",
  e3x   "define x-coordinate of upper-left corner",
  e3y   "define y-coordinate of upper-left corner",
  e4x   "define x-coordinate of upper-right corner",
  e4y   "define y-coordinate of upper-right corner";

obj.. z =E= sqr(a) + sqr(b);
e1x.. fx(t('1')) =E= x;
e1y.. fy(t('1')) =E= y;
e2x.. fx(t('2')) =E= x + a;
e2y.. fy(t('2')) =E= y + b;
e3x.. fx(t('3')) =E= x - b;
e3y.. fy(t('3')) =E= y + a;
e4x.. fx(t('4')) =E= x + a - b;
e4y.. fy(t('4')) =E= y + a + b;

Model m / all /;

* some starting point to get out of (a,b)=zero solution
t.l(i) = -pi + (ord(i)-1) * 2*pi/card(i);
x.l = fx(t.l('1'));
y.l = fy(t.l('1'));
a.l = 1;
b.l = 1;

Solve m max z using DNLP;

file f / '%gams.scrdir%gnuplot.in' /;
if( %gnuplot% and
 (m.modelstat = %modelStat.optimal% or
  m.modelstat = %modelStat.locallyOptimal% or
  m.modelstat = %modelStat.feasibleSolution%),
  f.nd=6;
  f.nw=0;
  put f;
  put 'set size ratio -1' /;
  put 'set samples 1000' /;
  put 'set object 1 polygon from ' x.l ',' y.l
  put ' to ' (x.l+a.l) ',' (y.l+b.l)
  put ' to ' (x.l+a.l-b.l) ',' (y.l+a.l+b.l)
  put ' to ' (x.l-b.l) ',' (y.l+a.l)
  put ' to ' x.l ',' y.l /;
  put 'set object 1 fc rgb "blue" fillstyle solid 0.1 border lt -1' /;
  put 'set parametric' /;
  put 'set trange [-pi:pi]' /;
  put 'plot %fx%, %fy%' /;
  put 'pause -1' /;
  putclose f;
  execute 'gnuplot %gams.scrdir%gnuplot.in'
  execute 'rm %gams.scrdir%gnuplot.in'
);
