$title Standard QP Model - QP4 expressed as MCP (QP6,SEQ=184)

$onText
Formulate the QP as an LCP, ie write down the first order
conditions of QP4 and solve.


Kalvelagen, E, Model Building with GAMS. forthcoming
de Wetering, A V, private communication.

Keywords: mixed complementarity problem, quadratic programming, finance
$offText

$include qpdata.inc

Set
   d(days)   'selected days'
   s(stocks) 'selected stocks';

Alias (s,t);

* select subset of stocks and periods
d(days)   = ord(days) > 1 and ord(days) < 31;
s(stocks) = ord(stocks) < 51;

Parameter
   mean(stocks)     'mean of daily return'
   dev(stocks,days) 'deviations'
   totmean          'total mean return';

mean(s)  = sum(d, return(s,d))/card(d);
dev(s,d) = return(s,d) - mean(s);
totmean  = sum(s, mean(s))/(card(s));

Variable
   x(stocks) 'investments'
   w(days)   'intermediate variables';

Positive Variable x;

Equation
   budget
   retcon 'return constraint'
   wdef(days);

wdef(d).. w(d) =e= sum(s, x(s)*dev(s,d));

budget..  sum(s, x(s)) =e= 1.0;

retcon..  sum(s, mean(s)*x(s)) =g= totmean*1.25;

Equation
   d_x(stocks)
   d_w(days);

Variable
   m_budget
   m_wdef(days);

Positive Variable
   m_retcon;

m_wdef.fx(days)$(not d(days)) = 0;

d_x(s).. sum(d,m_wdef(d)*dev(s,d)) =g= m_retcon*mean(s) + m_budget;

d_w(d).. 2*w(d)/(card(d) - 1) =e= m_wdef(d);

Model qp6 / d_x.x, d_w.w, retcon.m_retcon, budget.m_budget, wdef.m_wdef /;

solve qp6 using mcp;

Parameter z;
z = sum(d, sqr(w.l(d)))/(card(d) - 1);
display x.l, z;
