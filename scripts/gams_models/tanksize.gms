$title Tank Size Design Problem - (TANKSIZE,SEQ=350)

$onText
We discuss a tank design problem for a multi product plant, in which the
optimal cycle time and the optimal campaign size are unknown. A mixed in-
teger nonlinear programming formulation is presented, where non-convexities
are due to the tank investment cost, storage cost, campaign setup cost and
variable production rates. The objective of the optimization model is to
minimize the sum of the production cost per ton per product produced. A
continuous-time mathematical programming formulation for the problem is
implemented with a fixed number of event points.


Rebennack, S, Kallrath, J, and Pardalos, P M, Optimal Storage Design
for a Multi-Product Plant: A Non-Convex MINLP Formulation. Tech. rep.,
University of Florida, 2009. Submitted to Computers and Chemical
Engineering

Keywords: mixed integer nonlinear programming, storage design, global optimization
          continuous-time model, chemical engineering
$offText

$eolCom //

$sTitle Define the Model Size and Data
Set
   p 'products'     / P1*P3 /
   n 'event points' / N1*N3 /;

Parameter
   PRMIN(p)  'volume flow of products in m^3 per day'
   PRMAX(p)  'volume flow of products in m^3 per day'
   SLB(p)    'lower bound on inventory in m^3'
   SUB(p)    'upper bound on inventory in m^3'
   SI(p)     'initial inventory in m^3 (10% of the lower bound)'
   DLB(p)    'lower bound on PRODUCTION length d(n)'
   DUB(p)    'upper bound on PRODUCTION length d(n)'
   DEMAND(p) 'volume flow of products in m^3 per year!!'
   TS(p)     'campaign setup times in days'
   CSTI(p)   'tank variable cost per ton'
   CSTC(p)   'campaign setup cost'
   B         'variable part of the tank investment cost'   / 0.3271 /;

Table pdata(p,*)
      prmin  prmax    slb      sub     si  dlb  dub  demand   ts     csti  cstc
   P1  15.0   50.0  643.0  4018.36  707.0    1   40    4190  0.4  18.8304    10
   P2  15.0   50.0  536.0  3348.63  589.0    1   40    3492  0.2  19.2934    20
   P3   7.0   50.0  214.0  1339.45  235.0    1   40    1397  0.1  19.7563    30;

$onEchoV > assignpar.gms
$label start
%1(p) = pdata(p,'%1');
$shift
$if not x%1 == x $goto start
$offEcho
$batInclude assignpar prmin prmax slb sub si dlb dub demand ts csti cstc

* Derived data
Parameter
   DPD(p)  'compute the demand per day per product [tons per day]'
   L       'compute the demand per day             [tons per day]'
   CAL     'longest campain'
   PRL     'maximum production length'
   CSTCMin 'minimum setup cost'
   CSTCMax 'maximum setup cost';

DPD(p)  = DEMAND(p)/365;
L       = sum(p, DPD(p));
CSTI(p) = CSTI(p)/365;   // scale the storage cost
CAL     = max(0, smax(p, DUB(p) + TS(p)));
PRL     = max(0, smax(p, DUB(p)));
CSTCMin = smin(p, CSTC(p));
CSTCMax = smax(p, CSTC(p));

$sTitle Model Formulation
Alias (p,pp);

Positive Variable
   d(n)         'duration of the campaigns'
   pC(p,n)      'amount of product p produced in campaign n'
   s(p,n)       'amount of product p stored at the beginning of campaign n'
   sM(p)        'size of the product tanks in tons'
   sH(p,n)      'auxiliary variables'
   cI           'investment costs'
   cC           'campaign setup costs'
   cS           'variable storage costs'
   T            'cycle time';

Binary Variable
   omega(p,n)   'binary variable indicating product in campaign';

Variable
   cPT          'cost per ton: the objective variable to minimize';

Equation
   TIMECAP      'time capacity'
   UNIQUE(n)    'at most one product per campaign'
   MATBAL(p,n)  'material balance constraint'
   TANKCAP(p,n) 'tank capacity constraint'
   PPN1(p,n)    'compute the nonlinear products pR(rp)*d(n)*omega'
   PPN2(p,n)    'compute the nonlinear products pR(rp)*d(n)*omega'
   SCCam1(n)    'semi-continuous bound on campaigns'
   SCCam2(n)    'semi-continuous bound on campaigns'
   DEFcC        'campaign setup costs'
   DEFcI        'investment cost'
   DEFcS        'variable storage costs'
   DefsH(p,n)   'define the auxiliary variables'
   DEFcPT       'total costs per ton produced'
   NONIDLE(n)   'force not to be idle';

* time balance constraint with unknown cycle time T
TIMECAP..        sum(n, d(n) + sum (p, TS(p)*omega(p,n))) =e= T;

* at most one product per campaign
UNIQUE(n)..      sum(p, omega(p,n)) =l= 1;

* no idle states are allowed
NONIDLE(n)..     sum(p, DUB(p)*omega(p,n)) =g= d(n);

* material balance equation (steady state):
* storage at end of n for product p = storage at start of n+1 for product p
* storage at end of n for product p = storage at start of n
*                                   + total production of product p in n
*                                   - total demand in period n
MATBAL(p,n)..
   s(p,n++1) =e= s(p,n) + pC(p,n) - DPD(p)*(d(n) + sum(pp, TS(pp)*omega(pp,n)));

* tank capacity constraint:
* this connects the tank desing capacity variable with the storage level
TANKCAP(p,n).. s(p,n) =l= sM(p);

* compute the nonlinear products: pR(p,n)*d(n)*omega
* connects the production of product p in period n with
*  -> the omega variables
*  -> the lenght of the PRODUCTION period
*  -> the production rate
*  PPN(p,n)..    pC(p,nbl(n)) =e= pR(p,n)*d(n)*omega(p,n);
PPN1(p,n)..      pC(p,n) =l= PRMAX(p)*d(n)*omega(p,n);
PPN2(p,n)..      pC(p,n) =g= PRMIN(p)*d(n)*omega(p,n);

* semi-continuous lower and upper bound on campaigns
SCCam2(n)..      d(n) =g= sum(p, DLB(p)*omega(p,n));
SCCam1(n)..      d(n) =l= sum(p, DUB(p)*omega(p,n));

* define the total costs per ton: cPT
DEFcPT..         (cPT*L - cI )*T =e= cC + cS;

* define the campaign setup costs
DEFcC..          cC =e= sum((p,n), CSTC(p)*omega(p,n));

* define the tank investment costs
DEFcI..          cI =e= B*sum(p, sqrt(sM(p)));

* define the variable tank costs
DEFcS..          cS =e= sum((p,n), CSTI(p)*sH(p,n)
                                   *(d(n) + sum(pp, TS(pp)*omega(pp,n))));

* auxiliary variables for the objective
DefsH(p,n)..     sH(p,n) =e= 0.5*(s(p,n++1) + s(p,n)) - SLB(p);

* additional constraints to break symmetry
Equation
   SEQUENCE(p,n) 'redundant consteraint on the omega'
   SYMMETRY(n)   'break the symmetry of active campaigns';

* if a product is produced during period n, then it cannot be produced during
* period n+1
SEQUENCE(p,n)..  1 - omega(p,n) =g= omega(p,n+1);

* break symmetry buy shift empty periods to the end
SYMMETRY(n)..    sum(p, omega(p,n)) =g= sum(p, omega(p,n+1));

* lower und upper bound on inventory
s.lo(p,n) =  SLB(p);
s.up(p,n) =  SUB(p);

* initial storage
s.fx('P1','N1') = SLB('P1');

* the initial storage has some implications
omega.fx(p,'N1')    = 0;
omega.fx('P1','N1') = 1;
omega.fx('P1','N2') = 0;

* lower and upper bound on tank size
sM.lo(p) = SLB(p);
sM.up(p) = SUB(p);

Model Sequenz / all /;

* Get out of the poor starting point
omega.l(p,n) = uniform(0,1);

solve Sequenz using minlp minimizing cPT;
