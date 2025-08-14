Set w(*) "weapons";
Set t(*) "targets";
Parameter td(w,t) "target data";
Parameter wa(w) "weapons availability";
Parameter tm(t) "minimum number of weapons per target";
Parameter mv(t) "military value of target";
positive Variable x(w,t) "weapons assignment";
Equation maxw(w) "weapons balance";
Equation minw(t) "minimum number of weapons required per target";
free Variable war_objective_variable;
Equation war_objective;
Model war / maxw,minw,war_objective /;
$onMultiR
$gdxLoadAll /home/muhammet/Documents/gams_workspace/gamspy/tmp/to_gams/war_data.gdx
$offMulti
maxw(w) .. sum(t $ (td(w,t)),x(w,t)) =l= wa(w);
minw(t) $ (tm(t)) .. sum(w $ (td(w,t)),x(w,t)) =g= tm(t);
war_objective .. sum(t,mv(t) * (1 - prod(w $ (td(w,t)),rPower(1 - td(w,t),x(w,t))))) =e= war_objective_variable;
solve war using NLP MAX war_objective_variable;
