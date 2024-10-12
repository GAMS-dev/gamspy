Set gen(*);
Parameter load;
Parameter data(gen,*);
free Variable P(gen);
Equation eq2;
free Variable ECD_objective_variable;
Equation ECD_objective;
Model ECD / eq2,ECD_objective /;
$gdxLoadAll C:\Users\muhammet\Documents\gams_workspace\gamspy\tmp\to_gams\ECD_data.gdx
eq2 .. sum(gen,P(gen)) =g= load;
ECD_objective .. sum(gen,((((data(gen,"a") * P(gen)) * P(gen)) + (data(gen,"b") * P(gen))) + data(gen,"c"))) =e= ECD_objective_variable;
solve ECD using QCP MIN ECD_objective_variable;