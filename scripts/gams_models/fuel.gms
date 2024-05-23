$title Fuel Scheduling and Unit Commitment Problem (FUEL,SEQ=168)

$onText
Fuel scheduling and unit commitment addresses the problem of
fuel supply to plants and determining on/off status of units
simultaneously to minimize total operating cost.
The present problem: there are two generating units to
meet a total load over a 6-hour period. One of the unit is oil-based
and has to simultaneously meet the storage requirements, flow rates
etc. There are limits on the generation levels for both the units.


Wood, A J, and Wollenberg, B F, Example Problem 4e. In Power Generation,
Operation and Control. John Wiley and Sons, 1984, pp. 85-88.

Keywords: mixed integer nonlinear programming, scheduling, engineering, power
          generation, unit commitment problem
$offText

Set
   t 'scheduling periods (2hrs)' / period-1*period-3 /
   u 'generating units'          / oil, others /;

Parameter
   load(t)    'system load' / period-1 400, period-2 900, period-3 700 /
   initlev(t) 'initial level of the oil storage tank'  / period-1 3000 /;

Variable
   status(t) 'on or off status of the oil based generating unit'
   poil(t)   'generation level of oil based unit'
   others(t) 'other generation'
   oil(t)    'oil consumption'
   volume(t) 'the volume of oil in the storage tank'
   cost      'total operating cost';

Binary   Variable status;
Positive Variable volume, oil;

volume.up(t) = 4000;
volume.lo(t)$(ord(t) = card(t)) = 2000;

others.lo(t) =  50;
others.up(t) = 700;

Equation
   costfn     'total operating cost of unit 2 -- the objective fn'
   lowoil(t)  'lower limit on oil generating unit'
   maxoil(t)  'upper limit on oil generating unit'
   floweq(t)  'the oil flow balance in the storage tank'
   demcons(t) 'total generation must meet the load'
   oileq(t)   'calculation of oil consumption';

costfn..     cost =e= sum(t, 300 + 6*others(t) + 0.0025*sqr(others(t)));

lowoil(t)..  poil(t) =g= 100*status(t);

maxoil(t)..  poil(t) =l= 500*status(t);

floweq(t)..  volume(t) =e= volume(t - 1) + 500 - oil(t) + initlev(t);

oileq(t)..   oil(t) =e= 50*status(t) + poil(t) + 0.005*sqr(poil(t));

demcons(t).. poil(t) + others(t) =g= load(t);

Model ucom / all /;

poil.l(t) = 100;

solve ucom using minlp minimizing cost;
