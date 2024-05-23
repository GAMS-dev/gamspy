$title Weapons Assignment (WEAPONS,SEQ=18)

$onText
This model determines an assignment of weapons to targets in order
to inflict maximum damage at minimal cost. This is a classic
NLP test problem.


Bracken, J, and McCormick, G P, Chapter 2. In Selected Applications of
Nonlinear Programming. John Wiley and Sons, New York, 1968, pp. 22-27.

Keywords: nonlinear programming, assignment problem, military application,
          nlp test problem
$offText

Set
   w 'weapons' / ICBM      'Intercontinental Ballistic Missiles'
                 MRBM-1    'Medium-Range Ballistic Missiles from first area'
                 LR-Bomber 'Longe-Range Bomber'
                 F-Bomber  'Fighter Bomber'
                 MRBM-2    'Medium-Range Ballistic Missiles from second area' /
   t 'targets' / 1*20 /;

Table td 'target data'
            1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  avail
icbm          .05             .15 .10 .15 .20                                 .05            200
mrbm-1    .16 .17 .15 .16 .15 .19 .19 .18 .20 .14     .02     .12 .13 .12 .15 .16 .15 .15    100
lr-bomber .04 .05 .04 .04 .04 .10 .08 .09 .08 .05 .01 .02 .01 .02 .03 .02 .05 .08 .07 .08    300
f-bomber                                      .04 .09 .08 .09 .08 .02 .07                    150
mrbm-2    .08 .06 .08 .05 .05 .02 .02         .10 .05 .04 .09 .02 .01 .01                    250
damage     60  50  50  75  40  60  35  30  25 150  30  45 125 200 200 130 100 100 100 150
target     30                 100              40              50  70  35              10       ;

Parameter
   wa(w) 'weapons availability'
   tm(t) 'minimum number of weapons per target'
   mv(t) 'military value of target';

wa(w) = td(w,"avail");
tm(t) = td("target",t);
mv(t) = td("damage",t);

display wa, tm, mv;

Variable
   x(w,t)   'weapons assignment'
   prob(t)  'probability for each target'
   tetd     'total expected damage';

Positive Variable x;

Equation
   maxw(w)  'weapons balance'
   minw(t)  'minimum number of weapons required per target'
   probe(t) 'probability definition'
   etdp     'total expected damage alternate formulation'
   etd      'total expected damage';

maxw(w)..       sum(t$td(w,t), x(w,t)) =l= wa(w);

minw(t)$tm(t).. sum(w$td(w,t), x(w,t)) =g= tm(t);

probe(t)..      prob(t) =e= 1 - prod(w$td(w,t), (1-td(w,t))**x(w,t));

etdp..          tetd =e= sum(t, mv(t)*prob(t));

etd..           tetd =e= sum(t, mv(t)*(1-prod(w$td(w,t), (1-td(w,t))**x(w,t))));

Model
   war  'traditional formulation' / maxw, minw, etd         /
   warp 'extended formulation'    / maxw, minw, probe, etdp /;

x.l(w,t)$td(w,t) = wa(w)/card(t);

solve war maximizing tetd using nlp;

Parameter report 'summary report';

* option report:0;
report(w,t)       = x.l(w,t);
report('total',t) = sum(w, x.l(w,t));
report(w,'total') = sum(t, x.l(w,t));

display report;
