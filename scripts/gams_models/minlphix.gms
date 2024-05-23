$title Heat Integrated Distillation Sequences - MINLP (MINLPHIX,SEQ=227)

$onText
This is a direct MINLP formulation of the model MINLPHI.


Morari, M, and Grossmann, I E, Eds, Chemical Engineering
Optimization Models with GAMS. Computer Aids for Chemical
Engineering Corporation, 1991.

Floudas, C A, and Paules IV, G E, A Mixed-Integer Nonlinear
Programming Formulation for the Synthesis of Heat Integrated
Distillation Sequence. Computers and Chemical Engineering 12,
6 (1988), 531-546.


This formulation provides the Optimal Heat Integrated
Distillation Sequence with Pressure as a continuous variable
for a three component separation.

           Components:     a == Hexane
                           b == Benzene
                           c == Heptane

total feed to superstructure == 396 kgmol/hr

multicomponent feed composition: a = 0.80
                                 b = 0.10
                                 c = 0.10


A Superstructure of the form ...

                        _______               _______
                       _|_    |              _|_    |
                      /   \  ( )            /   \  ( )
                      |   |___|__ A         |   |___|___ B
                      |   |                 |   |
            |---------| 1 |                 | 3 |
            |         |   |       ----------|   |
            |         |   |       |         |   |
            |         |   |_______|         |   |
            |         \___/  |  BC          \___/_______ C
     F      |           |   ( )               |     |
   -------->|           |____|                |----( )
   (ABC)    |
            |           _______               _______
            |          _|_    |              _|_    |
            |         /   \  ( )            /   \  ( )
            |         |   |___| AB          |   |___|___ A
            |         |   |   |_____________|   |
            |---------| 2 |                 | 4 |
                      |   |                 |   |
                      |   |                 |   |
                      |   |______ C         |   |_______ B
                      \___/  |              \___/   |
                        |   ( )               |    ( )
                        |____|                |_____|


is used with binary variables representing:
   a_  the existence of columns in the sequence.
   b_  the selection of heat exchangers for heat integration.
   c_  the selection of hot and cold utilities.


Associated Reference:

"A Mixed-Integer Nonlinear Programming formulation for the
 synthesis of Heat-Integrated Distillation Sequences"

 C.A. Floudas and G.E. Paules IV,  1988.
 Computers and Chemical Engineering vol. 12 no. 6 pp. 531-546

Keywords: mixed integer nonlinear programming, chemical engineering, distillation
          sequences, heat integrated distillation
$offText

Set
   i       'condensers-columns' / c-1*c-4 /
   j       'reboilers'          / r-1*r-4 /
   hu      'hot utilities'      / lp, ex  /
   cu      'cold utilities'     / cw /
   n       'index'              / a, b /
   m       'intermediates'      / ab, bc /
   pm(i,m) 'products'           / c-1.bc, c-2.ab /
   fm(i,m) 'feeds'              / c-3.bc, c-4.ab /;

Alias (ip,i), (jp,j);

*=====================================================================
* Definition of "z" sets for conditional control of model
* used to map permissible matches between condensers and reboilers
* and the position of columns in the superstructure
*=====================================================================
Set
   zlead(i)   'leading columns in superstructure'       / c-1, c-2 /
   zcrhx(i,j) 'condenser to reboiler allowable matches' / c-1.r-3, c-2.r-4, c-3.r-1, c-4.r-2 /
   zlim(i,j)  'direction of heat integration'
   zcr(i,j)   'reboiler-condenser pairs';

zlim(i,j) = zcrhx(i,j) and (ord(i) < ord(j));
zcr(i,j)  = ord(i) = ord(j);

Parameter
   spltfrc(i,m) 'split fraction of distillation columns'
                / c-1.bc   0.20
                  c-2.ab   0.90 /
   tcmin(i)     'minimum condenser temperatures'
                / c-1    341.92
                  c-2    343.01
                  c-3    353.54
                  c-4    341.92 /
   trmax(j)     'maximum reboiler temperatures';

trmax(j) = 1000;

*====================================================================
* scaled cost coefficients for distillation column fits
* nonlinear fixed-charge cost model
*   cost = fc*y + vc*flow*temp
* scaling factor = 1000
*====================================================================
Parameter
   fc(i)   'fixed charge for distillation columns'
           / c-1    151.125
             c-2    180.003
             c-3    4.2286
             c-4    213.42   /
   vc(i)   'variable charge for distillation columns'
           / c-1    0.003375
             c-2    0.000893
             c-3    0.004458
             c-4    0.003176 /
   thu(hu) 'hot utility temperatures'
           / lp     421.0
             ex     373.0    /;

* hot utility cost coeff - gives cost in thousands of dollars per year
* ucost = q(10e+6 kj/hr)*costhu(hu)

Parameter costhu(hu) 'hot utility cost coefficients' / lp 24.908, ex 9.139 /;

Table kf(i,n) 'coeff. for heat duty temperature fits'
            a        b
   c-1   32.4   0.0225
   c-2   25.0   0.0130
   c-3   3.76   0.0043
   c-4   35.1   0.0156;

Table af(i,n) 'coeff. for column temperature fits'
            a       b
   c-1  9.541   1.028
   c-2  12.24   1.050
   c-3  8.756   1.029
   c-4  9.181   1.005;

Scalar
   totflow 'total flow to superstructure'               /  396      /
   fchx    'fixed charge for heat exchangers scaled'    /    3.392  /
   vchx    'variable charge for heat exchangers scaled' /    0.0893 /
   htc     'overall heat transfer coefficient'          /    0.0028 /
   dtmin   'minimum temperature approach'               /   10.0    /
   tcin    'inlet temperature of cold water'            /  305.0    /
   tcout   'outlet temperature of cold water'           /  325.0    /
   costcw  'cooling water cost coefficient'             /    4.65   /
   beta    'income tax correction factor'               /    0.52   /
   alpha   'one over payout time factor in years'       /    0.40   /
   u       'large number for logical constraints'       / 1500      /
   uint    'upper bound for integer logical'            /   20      /;

Variable zoau 'objective function value';

Positive Variable
   f(i)      'flowrates to columns'
   qr(j)     'reboiler duties for column with reboiler j'
   qc(i)     'condenser duties for column i'
   qcr(i,j)  'heat integration heat transfer'
   qhu(hu,j) 'hot utility heat transfer'
   qcu(i,cu) 'cold utility heat transfer'
   tc(i)     'condenser temperature for column with cond. i'
   tr(j)     'reboiler temperature for column with reb. j'
   lmtd(i)   'lmtd for cooling water exchanges'
   sl1(i)    'artificial slack variable for lmtd equalities'
   sl2(i)    'artificial slack variable for lmtd equalities'
   s1(i)     'artificial slack variable for reb-con equalities'
   s2(i)     'artificial slack variable for reb-con equalities'
   s3(i)     'artificial slack variable for duty equalities'
   s4(i)     'artificial slack variable for duty equalities';

Binary Variable
   yhx(i,j)  'heat integration matches condenser i reboiler j'
   yhu(hu,j) 'hot utility matches hot utility hu reboiler j'
   ycu(i,cu) 'cold utility matches condenser i cold util cu'
   ycol(i)   'columns in superstructure';

Equation
   nlpobj        'nlp subproblems objective'
   tctrlo(i,j)   'prevent division by 0 in the objective'
   lmtdlo(i)     'prevent division by 0 in the objective'
   lmtdsn(i)     'nonlinear form of lmtd definition'
   tempset(i)    'sets temperatures of inactive columns to 0 (milp)'
   artrex1(i)    'relaxes artificial slack variables (nlp)'
   artrex2(i)    'relaxes artificial slack variables (nlp)'
   material(m)   'material balances for each intermediate product'
   feed          'feed to superstructure'
   matlog(i)     'material balance logical constraints'
   duty(i)       'heat duty definition of condenser i'
   rebcon(i,j)   'equates condenser and reboiler duties'
   conheat(i)    'condenser heat balances'
   rebheat(j)    'reboiler heat balances'
   dtminlp(j)    'minimum temp approach for low pressure steam'
   dtminc(i)     'minimum temp allowable for each condenser'
   trtcdef(i,j)  'relates reboiler and condenser temps of columns'
   dtmincr(i,j)  'minimum temp approach for heat integration'
   dtminex(j)    'minimum temp approach for exhaust steam'
   hxclog(i,j)   'logical constraint for heat balances'
   hxhulog(hu,j) 'logical constraint for heat balances'
   hxculog(i,cu) 'logical constraint for heat balances'
   qcqrlog(i)    'logical constraint for con-reb duties'

* these are the pure binary constraints of the minlp
   sequen(m)     'restricts superstructure to a single sequence'
   lead          'sequence control'
   limutil(j)    'limits columns to have a single hot utility'
   hidirect(i,j) 'requires a single direction of heat integration'
   heat(i)       'logical integer constraint';

nlpobj..
   zoau =e= alpha*(sum(i, fc(i)*ycol(i) + vc(i)*(tc(i) - tcmin(i))*f(i))
                 + sum(zcrhx(i,j), fchx*yhx(i,j) + (vchx/htc)*(qcr(i,j)/(tc(i) - tr(j) + 1 - ycol(i))))
                 + sum((i,cu), fchx*ycu(i,cu) + (vchx/htc)*(qcu(i,cu)/(lmtd(i) + 1 - ycol(i))))
                 + sum((hu,j), fchx*yhu(hu,j) + (vchx/htc)*(qhu(hu,j)/(thu(hu) - tr(j)))))
         +  beta *(sum((i,cu), costcw*qcu(i,cu))
                 + sum((hu,j), costhu(hu)*qhu(hu,j)));

* limit the denominator in the second line of the objective away from zero
tctrlo(zcrhx(i,j)).. tc(i) - tr(j) + 1 - ycol(i) =g= 1;

* lmtd and ycol from being 0 and 1 at the same time to prevent divding
* by 0 in the objective
lmtdlo(i).. lmtd(i) =g= 2*ycol(i);

lmtdsn(i).. lmtd(i) =e= (2/3)*sqrt((tc(i) - tcin)*(tc(i) - tcout))
                      + (1/6)*((tc(i) - tcin) + (tc(i) - tcout)) + sl1(i) - sl2(i);

artrex1(i).. s1(i) + s2(i) + sl1(i) =l= u*(1 - ycol(i));

artrex2(i).. s3(i) + s4(i) + sl2(i) =l= u*(1 - ycol(i));

material(m).. sum(pm(i,m), spltfrc(i,m)*f(i)) =e= sum(fm(i,m), f(i));

feed..        sum(zlead(i), f(i)) =e= totflow;

duty(i).. qc(i) =e= (kf(i,"a") + kf(i,"b")*(tc(i)-tcmin(i))) + s3(i) - s4(i);

rebcon(zcr(i,j)).. qr(j) =e= qc(i);

conheat(i).. qc(i) =e= sum(zcrhx(i,j), qcr(i,j)) + sum(cu, qcu(i,cu));

rebheat(j).. qr(j) =e= sum(zcrhx(i,j), qcr(i,j)) + sum(hu, qhu(hu,j));

trtcdef(zcr(i,j)).. tr(j) =e= (af(i,"a") + af(i,"b")*(tc(i) - tcmin(i))) + s1(i) - s2(i);

dtminlp(j).. dtmin - (thu("lp") - tr(j)) =l= 0;

dtminex(j).. dtmin - (thu("ex") - tr(j)) - u*(1 - yhu("ex",j)) =l= 0;

tempset(i).. tc(i) + lmtd(i) + sum(zcr(i,j), tr(j)) =l= u*ycol(i);

matlog(i).. f(i) =l= u*ycol(i);

dtminc(i).. tcmin(i) - tc(i) =l= u*(1 - ycol(i));

dtmincr(zcrhx(i,j)).. tr(j) - tc(i) - u*(1 - yhx(i,j)) + dtmin =l= 0;

hxclog(zcrhx(i,j)).. qcr(i,j) =l= u*yhx(i,j);

hxhulog(hu,j).. qhu(hu,j) =l= u*yhu(hu,j);

hxculog(i,cu).. qcu(i,cu) =l= u*ycu(i,cu);

qcqrlog(i).. qc(i) + sum(j$zcr(i,j), qr(j)) =l= u*ycol(i);

sequen(m)..  sum(pm(i,m), ycol(i)) =e= sum(fm(i,m), ycol(i));

lead..       sum(zlead(i), ycol(i)) =e= 1;

limutil(j).. sum(hu, yhu(hu,j)) =l= 1;

* only one of the mutual heat integration binaries can be 1
hidirect(zlim(i,j))..
   yhx(i,j) + sum((ip,jp)$(ord(ip) = ord(j) and ord(jp) = ord(i)), yhx(ip,jp)) =l= 1;

* if a column doesn't exist then all binary variables associated
* with it must also be set to zero
heat(i).. sum(zcrhx(i,j), yhx(i,j)
       +  sum((ip,jp)$((ord(ip) = ord(j)) and (ord(jp) = ord(i))), yhx(ip,jp)))
       +  sum((hu,zcr(i,j)), yhu(hu,j))
       +  sum(cu, ycu(i,cu))
      =l= uint*ycol(i);

tc.lo("c-1") = tcout + 1;
tc.up("c-2") = tcin  - 1;
tc.lo("c-3") = tcout + 1;
tc.up("c-4") = tcin  - 1;
tr.up(j)     = trmax(j);

Model skip / all /;

option domLim = 100;

solve skip using minlp minimizing zoau;
