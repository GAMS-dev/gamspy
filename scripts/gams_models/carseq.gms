$title Car Sequencing (CARSEQ,SEQ=407)

$onText
A number of cars are to be produced; they are not identical, because
different options are available as variants on the basic model. The
assembly line has different stations which install the various options
(air-conditioning, sun-roof, etc.). These stations have been designed
to handle at most a certain percentage of the cars passing along the
assembly line. Furthermore, the cars requiring a certain option must not
be bunched together, otherwise the station will not be able to cope.
Consequently, the cars must be arranged in a sequence so that the capacity
of each station is never exceeded. For instance, if a particular station
can only cope with at most half of the cars passing along the line, the
sequence must be built so that at most 1 car in any 2 requires that option.
The problem has been shown to be NP-hard (Gent 1999).

This is the example given in Dincbas, et al. More instances can
be downloaded from CSPLib (problem prob001).


Dincbas et al., Dincbas, M., Simonis, H., and Van Hentenryck, P.
Solving the car-sequencing problem in constraint logic programming.
In 8th European Conference on Artificial Intelligence (ECAI 88) ,
Y. Kodratoff, Ed. Pitmann Publishing, London, Munich, Germany, 290-295, 1988

Keywords: mixed integer linear programming, mixed integer nonlinear programming,
          production planning, car manufacturing, line problem
$offText

Set
   p 'position'
   o 'options'
   c 'classes';

Parameter
   maxc(o)        'maximum number of cars with that option in a block'
   bs(o)          'block size to which the maximum number maxc refers'
   classData(c,*) 'class data';

$if not set instance $set instance
$ifThen '%instance%'==''
   Set
      p 'position' / pos1*pos10    /
      o 'options'  / opt1*opt5     /
      c 'classes'  / class1*class6 /;

   Parameter
      maxc(o) 'maximum number of cars with that option in a block'
              / (opt1,opt3,opt5) 1, (opt2,opt4) 2 /
      bs(o)   'block size to which the maximum number maxc refers'
              / opt1 2, (opt2,opt3) 3, (opt4,opt5) 5 /;

   Table classData(c,*) 'class data'
               numCars  opt1  opt2  opt3  opt4  opt5
      class1         1     1     0     1     1     0
      class2         1     0     0     0     1     0
      class3         2     0     1     0     0     1
      class4         2     0     1     0     1     0
      class5         2     1     0     1     0     0
      class6         2     1     1     0     0     0;
$else
$onEcho > cs.awk
BEGIN { nr=1 }
!/^#/ {
   if (nr==1) {
      o = $2;
      printf("set p /1*%d/, o /1*%d/, c /1*%d/\n", $1,o,$3)
   } if (nr==2) {
      printf("Parameter maxc(o) / 1 %d", $1);
      for (j=2; j<=o; j++) printf(", %d %d", j, $j);
      printf("/\n");
   } if (nr==3) {
      printf("Parameter bs(o) / 1 %d", $1);
      for (j=2; j<=o; j++) printf(", %d %d", j, $j);
      printf("/\n");
   } if (nr==4) {
      printf("Table classData(c,*)\n$ondelim\nclass,numCars");
      for (j=1; j<=o; j++) printf(",%d", j);
      printf("\n");
   } if (nr>3) {
      printf("%d,%d", $1+1, $2)
      for (j=1; j<=o; j++) printf(",%d", $(j+2));
      printf("\n");
   }
   nr++;
}
END { printf("$offdelim\n;") }
$offEcho
$call awk -f cs.awk %instance% > "%gams.scrdir%csinst.%gams.scrext%"
$ifE errorLevel<>0 $abort Problems running awk
$include "%gams.scrdir%csinst.%gams.scrext%"
$endIf

abort$(card(p) <> sum(c, classData(c,'numCars'))) 'inconsistent number of cars';

Alias (p,pp);

Set
   blk(o,p)     'blocks of positions to monitor'
   blkc(o,p,pp) 'positions in the blocks';

blkc(o,p,pp)$(ord(p) <= card(p) - bs(o) + 1) = ord(pp) >= ord(p) and ord(pp) < ord(p) + bs(o);
blk(o,p) = sum(pp$blkc(o,p,pp), 1);

Variable
   sumc(o,p)
   cp(c,p)   'class k is scheduled at position p'
   op(o,p)   'option o appears at position p'
   v(o,p)    'violations in a block'
   obj       'sum of violations';

Binary Variable cp;

$ifThen set MIP
   Positive Variable v;
   Binary   Variable op;
$endIf

Equation
   defnumCars(c)  'exactly numCars of class c assigned to positions'
   defoneCar(p)   'one car assigned to each position p'
   defop(o,p)     'option o appears at position p'
   defopLS(o,p)   'option o appears at position p'
   defviol(o,p)   'violations in a block'
   defviolLS(o,p) 'violations in a block'
   defobj         'minimize violations'
   defsumc(o,p);

defnumCars(c)..       sum(p, cp(c,p)) =e= classData(c,'numCars');

defoneCar(p)..        sum(c, cp(c,p)) =e= 1;

defop(o,p)..          sum(c$classData(c,o),cp(c,p)) =l= op(o,p);

defsumc(o,p)..        sumc(o,p) =e= sum(c$classData(c,o),cp(c,p));

defopLS(o,p)..        op(o,p) =e= ifthen(sumc(o,p) >= 0.5, 1, 0);

* defopLS(o,p)..      op(o,p) =e= ifthen(sum(c$classData(c,o),cp(c,p)) > 0.5, 1, 0);

defviol(blk(o,p))..   sum(blkc(blk,pp), op(o,pp)) =l= maxc(o) + v(o,p);

defviolLS(blk(o,p)).. v(o,p) =e= max(sum(blkc(blk,pp), op(o,pp)) - maxc(o), 0);

defobj..              obj =e= sum(blk(o,p), v(o,p));

Model
   carseqMIP / all - defopLS - defviolLS - defsumc /
   carseqLS  / all - defop   - defviol             /;

option optCr = 0;

$ifThen set MIP
   Positive Variable v;
   Binary   Variable op;
   solve carseqMIP min obj using mip;
$else
   solve carseqLS  min obj using minlp;
$endIf

Parameter rep(p,c,o);
rep(p,c,o)$(cp.l(c,p) > 0.5) = classData(c,o);

option  rep:0:2:1;
display rep;
