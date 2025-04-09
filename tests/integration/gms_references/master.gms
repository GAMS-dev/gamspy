Set i(*) "widths";
Set p(*) "possible patterns";
Set pp(p) "dynamic subset of p";
Parameter d(i) "demand";
Parameter aip(i,p) "number of width i in pattern growing in p";
integer Variable xp(p) "patterns used";
free Variable z "objective variable";
Equation numpat "number of patterns used";
Equation demand(i) "meet demand";
Model master / numpat,demand /;
$onMultiR
$gdxLoadAll /home/muhammet/Documents/gams_workspace/gamspy/tmp/to_gams/master_data.gdx
$offMulti
numpat .. z =e= sum(pp,xp(pp));
demand(i) .. sum(pp,(aip(i,pp) * xp(pp))) =g= d(i);
solve master using RMIP MIN z;
