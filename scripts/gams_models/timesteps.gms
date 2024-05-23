$title Accessing previous (or next) Time Steps in an Equation fast (TIMESTEPS,SEQ=413)

$onText
In dynamic models one often needs access to previous or next time steps. Access
to single time steps can be easly implemented via the lag and lead operator.
It gets more difficult if one needs access to a larger set of time steps.
the expression sum(tt$(ord(tt)<=ord(t) and ord(tt)>=ord(t)-n), ...) where t is
the current time step controlled from the outside can be very slow.

The following example model shows how to do this fast in GAMS using an example
from power generation modeling. We have a set of time steps and a number of
generators. A generator can only start once in a given time slice. We implement
the equation that enforces this in three different ways:

1) naive GAMS syntax via ord() calculation
2) calculate a set of time slices for any given active time step
3) fast implementation directly in the equation using the same idas to create
   the set in 2 fast

Solution 2 is actually the fastest, but it consumes a lot of memory. We will
eventually require this much memory in the model generation (we have many
non-zero entires in the equation) but we can safe the extra amount inside GAMS
data by using method 3.

Keywords: mixed integer linear programming, GAMS language features, dynamic
          modelling, time steps, power generation
$offText

$if not set mt    $set mt   2016
$if not set mg    $set mg     17
$if not set mindt $set mindt  10
$if not set maxdt $set maxdt  40
$ifE %mindt%>%maxdt% $abort minimum downtime is larger than maximum downtime

Set
   t 'hours'      / t1*t%mt% /
   g 'generators' / g1*g%mg% /;

Parameter pMinDown(g,t) 'minimum downtime';
pMinDown(g,t) = uniformInt(%mindt%,%maxdt%);

Alias (t,t1,t2);

Set
   sMinDown(g,t1,t2)     'hours t2 g cannot start if we start g in t1'
   sMinDownFast(g,t1,t2) 'hours t2 g cannot start if we start g in t1'
   tt(t)                 'max downtime hours' / t1*t%maxdt% /;

* Slow and fast calculation for the set of time slices t2 for a given time step t1
* Output from profile=1
*----     50 Assignment sMinDown      5.819     5.819 SECS     26 MB  850713
*----     51 Assignment sMinDownFast  0.187     6.006 SECS     48 MB  850713

sMinDown(g,t1,t2) = ord(t1) >= ord(t2) and ord(t2) > ord(t1) - pMinDown(g,t1);
sMinDownFast(g,t1,t + (ord(t1) - pMinDown(g,t1)))$(tt(t) and ord(t) <= pMinDown(g,t1)) = yes;

Set diff(g,t1,t2);
diff(g,t1,t2) = sMinDown(g,t1,t2) xor sMinDownFast(g,t1,t2);
abort$card(diff) 'sets are different', diff;

Binary Variable vStart(g,t);

Variable z;

* Slow, fast, and fastest (but memory intensive way because we need to store sMinDownFast) way to write the equation
* Output from profile = 1
*----     67 Equation   eStartNaive   6.099    12.215 SECS    106 MB  34272
*----     68 Equation   eStartFast    0.593    12.808 SECS    144 MB  34272
*----     69 Equation   eStartFaster  0.468    13.276 SECS    180 MB  34272

Equation eStartNaive(g,t), eStartFast(g,t), eStartFaster(g,t), defobj;

eStartNaive(g,t1)..
   sum(t2$(ord(t1) >= ord(t2) and ord(t2) > ord(t1) - pMinDown(g,t1)), vStart(g,t2)) =l= 1;

eStartFast(g,t1)..
   sum(tt(t)$(ord(t) <= pMinDown(g,t1)), vStart(g,t + (ord(t1) - pMinDown(g,t1)))) =l= 1;

eStartFaster(g,t1)..
   sum(sMinDownFast(g,t1,t2), vStart(g,t2)) =l= 1;

defobj..
   z =e= sum((g,t), vStart(g,t));

Model maxStarts / all /;

solve maxStarts max z using mip;
