Set S(*);
Parameter p(S);
positive Variable x(S) / /;
Equation e(S) / /;
free Variable math_objective_variable / /;
Equation math_objective / /;
Model math / e,math_objective /;
$onMultiR
$gdxLoadAll /home/muhammet/Documents/gams_workspace/gamspy/tmp/to_gams/math_data.gdx
$offMulti
e(S) .. power((p(S) * x(S)),2) =l= 4;
math_objective .. sum(S,x(S)) =e= math_objective_variable;
solve math using QCP MIN math_objective_variable;
