Set days(*) "100 days from 95-11-27 to 96-04-29";
Set stocks(*) "170 selected stocks";
Set d(days) "selected days";
Set s(stocks) "selected stocks";
Parameter mean(stocks) "mean of daily return";
Parameter dev(stocks,days) "deviations";
Parameter totmean "total mean return";
positive Variable x(stocks) "investments";
free Variable w(days) "intermediate variables";
Equation budget;
Equation retcon "returns constraint";
Equation wdef(days);
Equation d_x(stocks);
Equation d_w(days);
free Variable m_budget;
free Variable m_wdef(days);
positive Variable m_retcon;
Model qp6 / d_x.x,d_w.w,retcon.m_retcon,budget.m_budget,wdef.m_wdef /;
$onMultiR
$gdxLoadAll /home/muhammet/Documents/gams_workspace/gamspy/tmp/to_gams/qp6_data.gdx
$$offMulti
d_x(s) .. sum(d,(m_wdef(d) * dev(s,d))) =g= ((m_retcon * mean(s)) + m_budget);
d_w(d) .. ((2 * w(d)) / (card(d) - 1)) =e= m_wdef(d);
retcon .. sum(s,(mean(s) * x(s))) =g= (totmean * 1.25);
budget .. sum(s,x(s)) =e= 1.0;
wdef(d) .. w(d) =e= sum(s,(x(s) * dev(s,d)));
solve qp6 using MCP;
