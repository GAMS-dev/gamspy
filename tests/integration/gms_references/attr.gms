Set s(*);
positive Variable x(s);
Equation eq(s);
free Variable attr_objective_variable;
Equation attr_objective;
Model attr / eq,attr_objective /;
eq(s) $ ( not s.first) .. x(s) =g= 1;
attr_objective .. sum(s,x(s)) =e= attr_objective_variable;
solve attr using LP MIN attr_objective_variable;