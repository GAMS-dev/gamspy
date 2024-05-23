$title Parts Supply Problem w/ 10 Types w/ Random p(i) (PS10_S_MN,SEQ=369)

$onText
Hideo Hashimoto, Kojun Hamada, and Nobuhiro Hosoe, "A Numerical Approach
to the Contract Theory: the Case of Adverse Selection", GRIPS Discussion
Paper 11-27, National Graduate Institute for Policy Studies, Tokyo, Japan,
March 2012.

Keywords: nonlinear programming, contract theory, principal-agent problem,
          adverse selection, parts supply problem
$offText

option limCol = 0, limRow = 0, solPrint = off;

Set
   i 'type of supplier'         / 0*9    /
   t 'no. of Monte-Carlo draws' / 1*1000 /;

Alias (i,j);

Parameter
   theta(i) 'efficiency'
   pt(i,t)  'probability of type'
   p(i)     'probability of type';

theta(i) = ord(i)/card(i);

* Generating probability
loop(t, pt(i,t) = uniform(0,1););
pt(i,t) = pt(i,t)/sum(j,pt(j,t));
* pt(i,"1") = 1/card(i);

Parameter
   F(i,t)       'cumulative probability (Itho p. 42)'
   noMHRC0(i,t) 'no MHRC combination between i and i-1'
* (MHRC: monotone hazard rate condition)
   noMHRC(t)    '>=1: no MHRC case';

F(i,t) = sum(j$(ord(j) <=  ord(i)), pt(j,t));
noMHRC0(i,t)  $(ord(i) <  card(i))    = 1$(F(i,t)/pt(i+1,t) < F(i-1,t)/pt(i,t));
noMHRC(t)$(sum(i, noMHRC0(i,t)) >= 1) = 1;

Scalar ru 'reservation utility' / 0 /;

* Definition of Primal/Dual Variables
Positive Variable
   x(i) "quality"
   b(i) "maker's revenue"
   w(i) "price";

Variable Util "maker's utility";

Equation
   obj     "maker's utility function"
   rev(i)  "maker's revenue function"
   pc(i)   "participation constraint"
   licd(i) "incentive compatibility constraint"
   licu(i) "incentive compatibility constraint"
   ic(i,j) "global incentive compatibility constraint"
   mn(i)   "monotonicity constraint";

obj..     Util =e= sum(i, p(i)*(b(i) - w(i)));

rev(i)..  b(i) =e= x(i)**(0.5);

pc(i)..   w(i) - theta(i)*x(i) =g= ru;

licd(i).. w(i) - theta(i)*x(i) =g= w(i+1) - theta(i)*x(i+1);

licu(i).. w(i) - theta(i)*x(i) =g= w(i-1) - theta(i)*x(i-1);

ic(i,j).. w(i) - theta(i)*x(i) =g= w(j)   - theta(i)*x(j);

mn(i)..   x(i) =g= x(i+1);

* Setting Lower Bounds on Variables to Avoid Division by Zero
x.lo(i) = 0.0001;

Model
   SB_lic  / obj, rev, pc, licd     /
   SB_lic2 / obj, rev, pc, licd, mn /;

* Options to solve models quickly
SB_lic.solveLink  = 5;
SB_lic2.solveLink = 5;

Parameter
   Util_lic(t)  'util solved w/o MN'
   Util_lic2(t) 'util solved w/ MN'
   Util_gap(t)  'gap between these two util'
   x_lic(i,t)   'x solved in w/o MN'
   x_lic2(i,t)  'x solved in w/ MN'
   MN_lic(t)    'monotonicity of x solved w/o MN'
   MN_lic2(t)   'monotonicity of x solved w/ MN';

loop(t,
   p(i) = pt(i,t);

*  Solving the model w/o MN
   solve SB_lic maximizing Util using nlp;
   Util_lic(t) = util.l;
   x_lic(i,t)  = x.l(i);
   MN_lic(t)   = sum(i, 1$(round(x.l(i),10) < round(x.l(i+1),10)));

*  Solving the model w/ MN
   solve SB_lic2 maximizing Util using nlp;
   Util_lic2(t) = util.l;
   x_lic2(i,t)  = x.l(i);
   MN_lic2(t)   = sum(i, 1$(round(x.l(i),10) < round(x.l(i+1),10)));
);

Util_gap(t) = 1$(round(Util_lic(t),10) <> round(Util_Lic2(t),10));

* Computing probability that MHRC and MN holds.
Parameter
   p_noMHRC   'no MHRC case          [%]'
   p_noMN_lic 'no MN case            [%]'
   p_Util_gap 'no util-equality case [%]';

p_noMHRC   = sum(t$(noMHRC(t)   > 0), 1)/card(t)*100;
p_noMN_lic = sum(t$(MN_lic(t)   > 0), 1)/card(t)*100;
p_Util_gap = sum(t$(Util_gap(t) > 0), 1)/card(t)*100;

display p_noMHRC, p_noMN_LIC, p_Util_gap;

* Generating CSV file for summary
File sol / solution_lic.csv /;
put  sol;
sol.pc =     5;
sol.pw = 32767;

put "";
loop(i, put "pt(i,t)";);

put "" "" "" "";
loop(i, put "x: w/o MN";);
loop(i, put  "x: w/ MN";);
put /;

put "";
loop(i, put i.tl;);
put ">=1: no MHRC" "Util: w/o MN" "Util: w/ MN" "Util_gap: =1: not equal";
loop(i, put i.tl;);
loop(i, put i.tl;);
put "MN_lic: >=1: no MN" "MN_lic2: >=1: no MN"/;
loop(t, put t.tl;
   loop(i, put pt(i,t):10:5;);
   put noMHRC(t) Util_lic(t):20:10 Util_Lic2(t):20:10 Util_gap(t);
   loop(i, put X_lic(i,t););
   loop(i, put X_lic2(i,t););
   put MN_lic(t) MN_lic2(t)/;
);
put /;
