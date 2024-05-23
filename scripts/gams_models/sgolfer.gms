$title Social Golfer Problem (SGOLFER,SEQ=409)

$onText
In a golf club, there are 32 social golfers, each of whom play golf once a week,
and always in groups of 4. The problem is to build a schedule of play for 10
weeks with ''maximum socialisation''; that is, as few repeated meetings as
possible. More generally the problem is to schedule m groups of n golfers over
p weeks, with maximum socialisation.


Warwick, H, The Fully Social Golfer Problem. In Smith, B, and Warwick, H, Eds,
Proceedings of the Third International Workshop on Symmetry in Constraint Satisfaction
Problems (SymCon 2003). 2003, pp. 75-85.

Keywords: mixed integer linear programming, mixed integer nonlinear programming,
          social golfer problem, combinatorial optimization
$offText

* Number of groups, golfers in a group, number of weeks
$if not set gr $set gr 8
$if not set gg $set gg 4
$if not set nw $set nw 10

$eval gf %gr%*%gg%

Set
   gf 'golfers' / 1*%gf% /
   gr 'groups'  / 1*%gr% /
   w  'weeks'   / 1*%nw% /;

Alias (gf,gf1,gf2);

Set mgf(gf1,gf2) 'possible meeting';
mgf(gf1,gf2) = ord(gf2) > ord(gf1);

Variable
   x(w,gr,gf)    'golfer gf is in group gr on week w'
   m(w,gr,gf,gf) 'golfers meet in week w in some group'
   numm(gf,gf)   'number of meetings'
   redm(gf,gf)   'number of redundant meetings'
   obj           'objective';

Binary Variable x;

Equation
   defx(w,gf)       'each golfer is assigned to exactly one group'
   defgr(w,gr)      'each group contains exactly |gg| golfers'
   defm(w,gr,gf,gf) 'meet in group gr on week w'
   defnumm(gf,gf)   'number of meetings'
   defredm(gf,gf)   'number of redundant meetings'
   defobj           'minimize redundant meetings';

$ifThen set MIP
   Binary   Variable m;
   Positive Variable redm;

   Equation
      defm2(w,gr,gf,gf) 'meet in group gr on week w'
      defm3(w,gr,gf,gf) 'meet in group gr on week w';

   defm (w,gr,mgf(gf1,gf2)).. m(w,gr,mgf) =l= x(w,gr,gf1);

   defm2(w,gr,mgf(gf1,gf2)).. m(w,gr,mgf) =l= x(w,gr,gf2);

   defm3(w,gr,mgf(gf1,gf2)).. m(w,gr,mgf) =g= x(w,gr,gf1) + x(w,gr,gf2) - 1;

   defredm(mgf)..             redm(mgf)   =g= numm(mgf) - 1;
$else
   defm(w,gr,mgf(gf1,gf2))..  m(w,gr,mgf) =e= x(w,gr,gf1) and x(w,gr,gf2);

   defredm(mgf)..             redm(mgf)   =e= max(0,numm(mgf) - 1);
$endIf

defx(w,gf)..   sum(gr, x(w,gr,gf)) =e= 1;

defgr(w,gr)..  sum(gf, x(w,gr,gf)) =e= %gg%;

defnumm(mgf).. numm(mgf) =e= sum((w,gr), m(w,gr,mgf));

defobj..       obj =e= sum(mgf, redm(mgf));

x.l(w,gr,gf)$((ord(gr) - 1)*%gg% + 1 <= ord(gf) and ord(gf) <= (ord(gr))*%gg%) = 1;
display x.l;

Model social_golfer / all /;

$ifThen set MIP
   solve social_golfer min obj using mip;
$else
   solve social_golfer min obj using minlp;
$endIf
