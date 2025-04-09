Set i(*) "canning plants";
Set j(*) "markets";
Parameter a(i) "capacity of plant i in cases";
Parameter b(j) "demand at market j in cases";
Parameter c(i,j) "transport cost in thousands of dollars per case";
positive Variable x(i,j) "shipment quantities in cases" / /;
Equation supply(i) "observe supply limit at plant i" / /;
Equation demand(j) "satisfy demand at market j" / /;
free Variable transport_objective_variable / /;
Equation transport_objective / /;
Model transport / supply,demand,transport_objective /;
$onMultiR
$gdxLoadAll /home/muhammet/Documents/gams_workspace/gamspy/tmp/to_gams/transport_data.gdx
$offMulti
supply(i) .. sum(j,x(i,j)) =l= a(i);
demand(j) .. sum(i,x(i,j)) =g= b(j);
transport_objective .. sum((i,j),(c(i,j) * x(i,j))) =e= transport_objective_variable;
transport.dictfile = 0;
solve transport using LP MIN transport_objective_variable;
