$title Ramsey Model of Optimal Economic Growth (RAMSEY,SEQ=63)

$onText
This formulation is described in 'GAMS/MINOS: Three examples'
by Alan S. Manne, Department of Operations Research, Stanford
University, May 1986.


Ramsey, F P, A Mathematical Theory of Saving. Economics Journal (1928).

Murtagh, B, and Saunders, M A, A Projected Lagrangian Algorithm and
its Implementation for Sparse Nonlinear Constraints. Mathematical
Programming Study 16 (1982), 84-117.

The optimal objective value is 2.4875

Keywords: nonlinear programming, economic development, economic growth,
          theory of saving, investment planning
$offText

*---------------------------------------------------------------------
* The planning horizon covers the years from 1990 (TFIRST) to 2000
* (TLAST). The intervening asterisk indicates that this set includes
* all the integers between these two values. This first statement is
* the only one that needs to be changed if one wishes to examine a
* different planning horizon.
*---------------------------------------------------------------------
Set
   t         'time periods' / 1990*2000 /
   tfirst(t) 'first period'
   tlast(t)  'last period';

*---------------------------------------------------------------------
* Data may also be entered in the form of SCALAR(S), as illustrated
* below.
*---------------------------------------------------------------------
Scalar
   bet "discount factor"          /  .95 /
   b   "capital's value share"    /  .25 /
   g   "labor growth rate"        /  .03 /
   ac  "absorptive capacity rate" /  .15 /
   k0  "initial capital"          / 3.00 /
   i0  "initial investment"       /  .05 /
   c0  "initial consumption"      /  .95 /
   a   "output scaling factor";

Parameter
   beta(t) 'discount factor'
   al(t)   'output-labor scaling vector';

*-----------------------------------------------------------------------
* The following statements show how we may avoid entering information
* about the planning horizon in more than one place. Here the symbol
* "$" means "such that"; "ORD" defines the ordinal position in a set;
* "CARD" defines the cardinality of the set. Thus, TFIRST is
* determined by the first member included in the set; and TLAST by the
* cardinality (the last member) of the set.
* This seems like a roundabout way to do things, but is useful if we
* want to be able to change the length of the planning horizon by
* altering a single entry in the input data. The same programming style
* is employed when we calculate the present-value factor BETA(T) and the
* output-labor vector AL(T).
*-----------------------------------------------------------------------
tfirst(t) = yes$(ord(t) = 1);
tlast(t)  = yes$(ord(t) = card(t));
display tfirst, tlast;

beta(t)     = bet**ord(t);
beta(tlast) = beta(tlast)/(1 - bet);

*-----------------------------------------------------------------------
* BETA(TLAST), the last period's utility discount factor, is calculated
* by summing the infinite geometric series from the horizon date onward.
* Because of the logarithmic form of the utility function, the
* post-horizon consumption growth term may be dropped from the maximand.
*-----------------------------------------------------------------------
a     = (c0 + i0)/k0**b;
al(t) = a*(1 + g)**((1 - b)*(ord(t) - 1));
display beta, al;

Variable
   k(t) 'capital stock        (trillion rupees)'
   c(t) 'consumption (trillion rupees per year)'
   i(t) 'investment  (trillion rupees per year)'
   utility;

*---------------------------------------------------------------------*
* Note that variables and equations cannot be identified by the same
* name. That is why the capital stock variables are called K(T), and
* the capital balance equations are KK(T).
*---------------------------------------------------------------------*
Equation
   cc(t) 'capacity constraint         (trillion rupees per year)'
   kk(t) 'capital balance                      (trillion rupees)'
   tc(t) 'terminal condition (provides for post-terminal growth)'
   util  'discounted log of consumption: objective function';

*---------------------------------------------------------------------*
cc(t).. al(t)*k(t)**b  =e=  c(t) + i(t);

kk(t+1)..       k(t+1) =e=  k(t) + i(t);

tc(tlast).. g*k(tlast) =l=  i(tlast);

util..         utility =e=  sum(t, beta(t)*log(c(t)));

*-----------------------------------------------------------------------
* Instead of requiring that "ALL" of these constraints are to be
* included, we specify that the RAMSEY model consists of each of the
* four individual constraint types. If, for example, we omit TC, we can
* check the sensitivity of the solution to this terminal condition.
*-----------------------------------------------------------------------
Model ramsey / all /;

*-----------------------------------------------------------------------
* The following statements represent lower bounds on the individual
* variables K(T), C(T) and I(T); a fixed value for the initial period's
* capital stock, K(TFIRST); and upper bounds (absorptive capacity
* constraints) on I(T). Bounds are required for K and C because
* LOG(C(T)) and K(T)**B are defined only for positive values of C and K
*-----------------------------------------------------------------------
k.lo(t)      = k0;
c.lo(t)      = c0;
i.lo(t)      = i0;
k.fx(tfirst) = k.lo(tfirst);
i.up(t)      = i0*((1 + ac)**(ord(t) - 1));

*-----------------------------------------------------------------------
solve ramsey maximizing utility using nlp;
