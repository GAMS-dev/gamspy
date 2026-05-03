$title A P E X - Production Scheduling Model (PRODSCH,SEQ=9)

$onText
A company specializing in the manufacture of outboard motors faces
highly seasonal demands and wants to minimize production cost. The
three main cost categories are:
  1. Direct production cost (nonlinear production relations and shift
     operations are possible)
  2. Inventory cost (rent or lease option)
  3. Workforce fluctuation cost.


CDC, APEX-III Reference Manual Version 1.2, Control Data Corporation,
Minneapolis, 1980. MIP Sample Problem

Keywords: mixed integer linear programming, scheduling, production planning
$offText

Set
   q    'quarters'                   / summer, fall, winter, spring /
   s    'shifts'                     / first, second /
   l    'production levels'          / 1*4 /
   i(l) 'production level intervals' / 1*3 /;

Parameter
   d(q)    'demand        (motors per season)' / spring 24000 /
   lc(q)   'leasing cost (dollars per season)' / summer 15000 /
   ei(q)   'initial employment'                / summer    84 /
   delt(q) 'discount factor';

Scalar
   mc     'material cost  (dollars per motor)' / 100 /
   sr     'space rental   (dollars per motor)' /   2 /
   invmax 'upper bound on inventory  (motors)'
   hc     'hiring cost (dollars per employee)' / 900 /
   fc     'firing cost (dollars per employee)' / 150 /;

delt(q) = 1/1.03**(ord(q) - 1);
invmax  = sum(q, d(q));

Table pr(*,l) 'production relationship'
              1     2     3     4
   labor     20    40    50    60
   motor   1000  3000  4500  5800;

Table sc(*,s) 'shift cost (dollars per shift)'
            first  second
   fixed    10000   16000
   labor     3500    4100;

Variable
   cost       'total discounted cost per year         (1000 $)'
   dpc(q)     'direct production cost      (1000 $ per season)'
   isc(q)     'inventory storage cost      (1000 $ per season)'
   wfc(q)     'workforce fluctuation cost  (1000 $ per season)'
   src(q)     'space rental cost           (1000 $ per season)'
   p(q)       'production                  (motors per season)'
   ss(l,q,s)  'production segments                 (sos2 type)'
   ssb(l,q,s) '0-1 needed for ss sos2 formulation'
   inv(q)     'inventory                   (motors per season)'
   lease      'lease-rent option'
   e(q)       'total employment                    (employees)'
   se(q,s)    'shift employment          (employees per shift)'
   shift(q,s) 'shift use indicator                    (binary)'
   h(q)       'hirings in quarter                  (employees)'
   f(q)       'firings in quarter                  (employees)';

Positive Variable p, ss, inv, src, h, f;
Binary   Variable bpl, lease, shift, ssb;

Equation
   acost       'total cost definition                 (1000 $)'
   ddpc(q)     'direct production cost definition     (1000 $)'
   disc(q)     'inventory storage cost definition     (1000 $)'
   dwfc(q)     'workforce fluctuation cost definition (1000 $)'
   sbp(q)      'sos product balance                   (motors)'
   sbse(q,s)   'sos shift employment balance       (employees)'
   scc(q,s)    'sos shift link'
   invb(q)     'inventory balance                     (motors)'
   dsrc(q)     'definition: space rental'
   ed(q)       'total employment definition        (employees)'
   eb1(q)      'employment balance type 1          (employees)'
   eb2(q)      'employment balance type 2          (employees)'
   messb(q,s)  'mutual exclusivity for ssb'
   lssb(l,q,s) 'ss - ssb linkage';

acost..       cost   =e= sum(q, delt(q)*( dpc(q) + isc(q) + wfc(q)));

ddpc(q)..     dpc(q) =e= (mc*p(q) + sum(s, sc("fixed",s)*shift(q,s) + sc("labor",s)*se(q,s)))/1000;

sbp(q)..      p(q)   =e= sum((s,l), pr("motor",l)*ss(l,q,s));

sbse(q,s)..   se(q,s) =e= sum(l, pr("labor",l)*ss(l,q,s));

scc(q,s)..    sum(l, ss(l,q,s)) =e= shift(q,s);

invb(q)..     inv(q) =e= inv(q-1) + p(q) - d(q);

disc(q)..     isc(q) =e= (lc(q)*lease + src(q))/1000;

dsrc(q)..     src(q) =g= sr*( inv(q) - invmax*lease );

dwfc(q)..     wfc(q) =e= (hc*h(q) + fc*f(q))/1000;

ed(q)..       e(q)   =e= sum(s, se(q,s));

eb1(q)..      e(q)   =e= e(q-1) + h(q) - f(q) + ei(q);

eb2(q)..      e(q)   =e= e(q--1) + h(q) - f(q);

messb(q,s)..  sum(l, ssb(l,q,s)) =e= 1;

lssb(l,q,s).. ss(l-1,q,s) + ss(l,q,s) =l= ssb(l-2,q,s) + ssb(l-1,q,s) + ssb(l,q,s);

p.up("spring") = .8*card(s)*smax(l, pr("motor",l));

Model
   prod1 'initial employment' / all - eb2 /
   prod2 'steady state'       / all - eb1 /;

solve prod1 minimizing cost using mip;

display se.l, p.l;

Parameter
   rep1 'cost summary'
   rep2 'production summary';

rep1("direct",q)    = dpc.l(q);
rep1("storage",q)   = isc.l(q);
rep1("hire-fire",q) = wfc.l(q);
rep1("*total*",q)   = dpc.l(q) + isc.l(q) + wfc.l(q);

rep2("output",q)     = p.l(q);
rep2("inventory",q)  = inv.l(q);
rep2("sales",q)      = d(q);
rep2("employment",q) = e.l(q);

display rep1, rep2;
