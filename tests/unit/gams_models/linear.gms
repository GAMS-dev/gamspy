$title Linear Regression with Various Criteria (LINEAR,SEQ=23)

$onText
This example solves linear models with differing objective functions.
Absolute deviations cannot be solved in a reliable manner with
most NLP systems and one has to resort to a formulation with
negative and positive deviations (models ending with the letter a).


Bracken, J, and McCormick, G P, Chapter 8.2. In Selected Applications of
Nonlinear Programming. John Wiley and Sons, New York, 1968, pp. 86-88.

Keywords: linear programming, nonlinear programming, discontinuous derivatives,
          linear regression, econometrics
$offText

Set
   i 'observation number'             / 1*20       /
   n 'index of independent variables' / a, b, c, d /;

Table dat(i,*)
         y   a    b    c    d
    1   99   1   85   76   44
    2   93   1   82   78   42
    3   99   1   75   73   42
    4   97   1   74   72   44
    5   90   1   76   73   43
    6   96   1   74   69   46
    7   93   1   73   69   46
    8  130   1   96   80   36
    9  118   1   93   78   36
   10   88   1   70   73   37
   11   89   1   82   71   46
   12   93   1   80   72   45
   13   94   1   77   76   42
   14   75   1   67   76   50
   15   84   1   82   70   48
   16   91   1   76   76   41
   17  100   1   74   78   31
   18   98   1   71   80   29
   19  101   1   70   83   39
   20   80   1   64   79   38;

Variable
   obj     'objective value'
   dev(i)  'total deviation'
   devp(i) 'positive deviation'
   devn(i) 'negative deviation'
   b(n)    'estimates';

Positive Variable devp, devn;

Equation
   ddev    'definition of deviations using total deviations'
   ddeva   'definition of deviations using positive and negative deviations'
   ls1
   ls1a
   ls2
   ls3
   ls4
   ls5
   ls5a
   ls6
   ls7
   ls8;

ddev(i)..  dev(i) =e= dat(i,"y") - sum(n, b(n)*dat(i,n));

ddeva(i).. devp(i) - devn(i) =e= dat(i,"y") - sum(n, b(n)*dat(i,n));

ls1..      obj =e= sum(i, abs(dev(i)));

ls1a..     obj =e= sum(i, devp(i)+devn(i));

ls2..      obj =e= sum(i, sqr(dev(i)));

ls3..      obj =e= sum(i, power(abs(dev(i)),3));

ls4..      obj =e= sum(i, power(dev(i),4));

ls5..      obj =e= sum(i, abs(dev(i)/dat(i,"y")));

ls5a..     obj =e= sum(i, (devp(i)+devn(i))/dat(i,"y"));

ls6..      obj =e= sum(i, sqr(dev(i)/dat(i,"y")));

ls7..      obj =e= sum(i, power(abs(dev(i)/dat(i,"y")),3));

ls8..      obj =e= sum(i, power(dev(i)/dat(i,"y"),4));

Model
   mod1  / ddev, ls1  /
   mod1a / ddeva,ls1a /
   mod2  / ddev, ls2  /
   mod3  / ddev, ls3  /
   mod4  / ddev, ls4  /
   mod5  / ddev, ls5  /
   mod5a / ddeva,ls5a /
   mod6  / ddev, ls6  /
   mod7  / ddev, ls7  /
   mod8  / ddev, ls8  /;

Parameter result 'summary table';

b.l(n)     = 1;
dev.l(i)   = dat(i,"y") - sum(n, b.l(n)*dat(i,n));
dev.up(i)  =  100;
dev.lo(i)  = -100;
devp.up(i) =  100;
devn.up(i) =  100;

option limRow = 0, limCol = 0;

solve mod1  min obj using dnlp; result("mod1" ,n) = b.l(n); result("mod1" ,"obj") = obj.l;
solve mod1a min obj using lp;   result("mod1a",n) = b.l(n); result("mod1a","obj") = obj.l;
solve mod2  min obj using nlp;  result("mod2" ,n) = b.l(n); result("mod2" ,"obj") = obj.l;
solve mod3  min obj using dnlp; result("mod3" ,n) = b.l(n); result("mod3" ,"obj") = obj.l;
solve mod4  min obj using nlp;  result("mod4" ,n) = b.l(n); result("mod4" ,"obj") = obj.l;
solve mod5  min obj using dnlp; result("mod5" ,n) = b.l(n); result("mod5" ,"obj") = obj.l;
solve mod5a min obj using lp;   result("mod5a",n) = b.l(n); result("mod5a","obj") = obj.l;
solve mod6  min obj using nlp;  result("mod6" ,n) = b.l(n); result("mod6" ,"obj") = obj.l;
solve mod7  min obj using dnlp; result("mod7" ,n) = b.l(n); result("mod7" ,"obj") = obj.l;
solve mod8  min obj using nlp;  result("mod8" ,n) = b.l(n); result("mod8" ,"obj") = obj.l;

display result;
