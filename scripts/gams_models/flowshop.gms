$title Flow Shop Scheduling - (FLOWSHOP,SEQ=376)

$onText
A workshop that produces metal pipes on demand for automotive industry
has three machines for bending the pipes, soldering the fastenings,
and assembling the links. The workshop has to produce six items, for
which the durations of the processing steps are given below. Once
started, jobs must be carried out to completion, but the
workpieces(items) may wait between the machines.

Every machine only processes one item at a time. A workpiece(item) may
not overtake any other.

What is the sequence that minimizes the total time for completing all
items (makespan)?


Gueret, C, Prins, C, and Sevaux, M, Applications of Optimization with
Xpress-MP, Translated and revised by Susanne Heipcke. Dash
Optimization, 2002.

Keywords: mixed integer linear programming, relaxed mixed integer programming,
          scenario analysis, GUSS, flow shop scheduling, production planning
$offText

Set
   i 'item'    / i1*i6 /
   m 'machine' / bending, soldering, assembly /;

$set lastrank    i6
$set lastmachine assembly

Table proctime(m,i)
              i1   i2   i3   i4   i5   i6
   bending     3    6    3    5    5    7
   soldering   5    4    2    4    4    5
   assembly    5    2    4    6    3    6;

Alias (i,k);

Variable
   rank(i,k)  'item i has position k'
   start(m,k) 'start time for job in position k on m'
   comp(m,k)  'completion time for job in postion k on m'
   totwait    'time before first job + times between jobs on last machine';

Binary   Variable rank;
Positive Variable start, comp;

Equation
   oneInPosition(k) 'every position gets a jobs'
   oneRankPer(i)    'every job is assigned a rank'
   onmachrel(m,k)   'relations between the end of job rank k on machine m and start of job rank k on machine m+1'
   permachrel(m,k)  'relations between the end of job rank k on machine m and start of job rank k+1 on machine m'
   defcomp(m,k)     'calculation of completetion time based on start time and proctime'
   defobj           'completion time of job rank last';

oneInPosition(k)..  sum(i, rank(i,k)) =e= 1;

oneRankPer(i)..     sum(k, rank(i,k)) =e= 1;

onmachrel(m,k+1)..  start(m,k+1) =g= comp(m,k);

permachrel(m+1,k).. start(m+1,k) =g= comp(m,k);

defcomp(m,k)..      comp(m,k) =e= start(m,k) + sum(i, proctime(m,i)*rank(i,k));

defobj..            totwait =g= comp('%lastmachine%','%lastrank%');

Model sequence / all /;

option optCr = 0;

solve sequence using mip min totwait;

* Maybe the following is better info to output
Parameter startjob(m,i);
startjob(m,i) = sum(k$(rank.l(i,k)>0.5), start.l(m,k));

option  startjob:0:1:1;
display startjob;

* For small data we can just enumerate all permutations and evaluate the schedule
$ifE card(i)>8 $exit

* Only test with some LP solvers
$ifI %gams.rmip% == cplex  $goto continue
$ifI %gams.rmip% == xpress $goto continue
$ifI %gams.rmip% == gurobi $goto continue
$ifI %gams.rmip% == soplex $goto continue
$exit

$label continue
$eval pmax fact(card(i))
Set
   p / p1*p%pmax% /
   rankall(p,i,i);

option rankall > i;

Parameter
   prankall(p,i,i) 'parameter version of rankall'
   ptotwait(p)     'scenario objective';

prankall(rankall) = 1;

* Use GUSS to evaluate all scenarios
Set dict / p.      scenario. ''
           rank.   fixed.    prankall
           totwait.level.    ptotwait /;

solve sequence using rmip min totwait scenario dict;

Scalar besttotwait / +inf /;

loop(p,
   if(besttotwait > ptotwait(p),
      besttotwait = ptotwait(p);
      rank.l(i,k) = 1$(rankall(p,i,k));
   );
);
display besttotwait, rank.l;
