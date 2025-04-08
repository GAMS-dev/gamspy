Set i(*) "corner points of square";
free Variable t(i) "position of square corner points on curve";
free Variable x "x-coordinate of lower-left corner of square (=fx(t('1')))";
free Variable y "y-coordinate of lower-left corner of square (=fy(t('1')))";
positive Variable a "horizontal distance between lower-left and lower-right corner of square";
positive Variable b "vertical distance between lower-left and lower-right corner of square";
Equation e1x "define x-coordinate of lower-left corner";
Equation e1y "define y-coordinate of lower-left corner";
Equation e2x "define x-coordinate of lower-right corner";
Equation e2y "define y-coordinate of lower-right corner";
Equation e3x "define x-coordinate of upper-left corner";
Equation e3y "define y-coordinate of upper-left corner";
Equation e4x "define x-coordinate of upper-right corner";
Equation e4y "define y-coordinate of upper-right corner";
free Variable square_objective_variable;
Equation square_objective;
Model square / e1x,e1y,e2x,e2y,e3x,e3y,e4x,e4y,square_objective /;
$onMultiR
$gdxLoadAll /home/muhammet/Documents/gams_workspace/gamspy/tmp/to_gams/square_data.gdx
$$offMulti
e1x .. (( sin(t("1")) ) * ( cos((t("1") - (t("1") * t("1")))) )) =e= x;
e1y .. (t("1") * ( sin(t("1")) )) =e= y;
e2x .. (( sin(t("2")) ) * ( cos((t("2") - (t("2") * t("2")))) )) =e= (x + a);
e2y .. (t("2") * ( sin(t("2")) )) =e= (y + b);
e3x .. (( sin(t("3")) ) * ( cos((t("3") - (t("3") * t("3")))) )) =e= (x - b);
e3y .. (t("3") * ( sin(t("3")) )) =e= (y + a);
e4x .. (( sin(t("4")) ) * ( cos((t("4") - (t("4") * t("4")))) )) =e= ((x + a) - b);
e4y .. (t("4") * ( sin(t("4")) )) =e= ((y + a) + b);
square_objective .. (( power(a,2) ) + ( power(b,2) )) =e= square_objective_variable;
solve square using DNLP MAX square_objective_variable;
