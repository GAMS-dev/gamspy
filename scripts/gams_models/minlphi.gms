$title Heat Integrated Distillation Sequences (MINLPHI,SEQ=118)

$onText
This problem describes a formulation and algorithmic procedure
for obtaining heat-integrated distillation sequences for the separation
of a given multi component feed stream into its pure components products.


Morari, M, and Grossmann, I E, Eds, Chemical Engineering Optimization
Models with GAMS. Computer Aids for Chemical Engineering Corporation,
1991.

Floudas, C A, and Paules IV, G E, A Mixed-Integer Nonlinear Programming
Formulation for the Synthesis of Heat Integrated Distillation Sequence.
Computers and Chemical Engineering 12, 6 (1988), 531-546.

======================================================================

   A MATHEMATICAL PROGRAMMING FORMULATION FOR PROCESS SYNTHESIS

===================================================================

   copyright    G.E. PAULES IV & C.A. FLOUDAS

            *** Dept. of Chemical Engineering ***
                 *** Princeton University ***
                         May 23, 1987


   Algorithm:  The Outer Approximation with Equality Relaxation
          Full Solution with Starting Point from FIXDT

======================================================================

        This formulation provides the Optimal Heat Integrated
    Distillation Sequence with Pressure as a continuous variable
                for a three component separation.
   The Outer Approximation with Equality Relaxation algorithm is
        used in the automatic solution procedure using GAMS

             Components:     a == Hexane
                             b == Benzene
                             c == Heptane

Total feed to superstructure == 396 kgmol/hr

Multicomponent feed composition:
                        a = 0.80
                        b = 0.10
                        c = 0.10

======================================================================


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
_______________________
 "A Mixed-Integer Nonlinear Programming formulation for the
  synthesis of Heat-Integrated Distillation Sequences"

  C.A. Floudas and G.E. Paules IV,  1988.
  Computers and Chemical Engineering vol. 12 no. 6 pp. 531-546

Keywords: mixed integer linear programming, nonlinear programming, chemical engineering,
          distillation sequences, heat integrated distillation
======================================================================
$offText
$offSymXRef offSymList

*======================================================================
* Set Options
*======================================================================
option limCol = 0, limRow = 0, bRatio = 1, domLim = 1000, optCr = 0;
* reduce default factorization frequency of zoom

*=====================================================================
* Declaration of sets
*=====================================================================
Set
* the set of all columns and their condensers in the superstructure
   i  'condensers-columns' / c-1*c-4 /
* the set of all reboilers in the superstructure
   j  'reboilers' / r-1*r-4 /
* the set of all hot utilities available
   hu 'hot utilities' / lp,ex /
* the set of all cold utilities available
   cu 'cold utilities' / cw /
* an index for linear fit coefficients
   n  'index' / a,b /
* the set of all intermediate products in superstructure
   m  'intermediates' / ab,bc /
* this set maps columns to produced intermediate products
   pm(i,m)  'products' / c-1.bc, c-2.ab /
* this set maps columns to intermediate product feeds
   fm(i,m)  'feeds' / c-3.bc, c-4.ab /
* these sets are for dynamic control of solution algorithm
   km          'static iterations' / k-1*k-100 /
   k(km)       'dynamic iterations'
   kiter(km)   'dynamic counter'
   kdynmax(km) 'dynamic loop control';

* alias sets for condensers and reboilers
Alias (ip,i), (jp,j);
* alias driving loop index - cant appear in equations
Alias (kloop,km);

*=====================================================================
* Definition of "z" parameters for conditional control of model
* used to map permissible matches between condensers and reboilers
* and the position of columns in the superstructure
*=====================================================================

* defines the set of leading columns in the superstructure
Parameter zlead(i) 'leading columns in superstructure' / c-1 1, c-2 1 /;

* defines allowable matches of heat integration for superstructure
* only permits heat integration between columns in the same sequence
Table zcrhx(i,j) 'condenser to reboiler allowable matches'
         r-1  r-2  r-3  r-4
   c-1               1
   c-2                    1
   c-3     1
   c-4          1          ;

* Parameter used in pure integer constraint to permit only one
* direction of heat integration between two columns
* this would yield an infeasible solution but the constraint
* is included explicitly to reduce milp solution time
Parameter zlim(i,j) 'direction of heat integration';
zlim(i,j) = 1$(zcrhx(i,j) and (ord(i) < ord(j)));

* relates appropriate reboiler to the condenser of same column
* (preferably should use an alias rather than a different set)
Parameter zcr(i,j) 'reboiler-condenser pairs';
zcr(i,j) = 1$(ord(i) = ord(j));

*=====================================================================
* Binary variables are divided into 4 classes and variable/parameter
* names starting with "y"
*     ycol - column selection
*     yhx  - heat integration exchanger matches
*     yhu  - hot utility matches
*     ycup - cold utiltiy matches

* These parameters store first guess combination of binary variables
* used to initialize minlp algorithm and parameterize the minlp
* primal problem throughout the rest of the iterations
*=====================================================================
Parameter
   yhxp(i,j)  'current proposal for heat integration matches'    / c-1.r-3  1 /
   yhup(hu,j) 'current binary proposal for hot utility matches'  / lp.r-1   1 /
   ycup(i,cu) 'current binary proposal for cold utility matches' / c-1.cw   1
                                                                   c-3.cw   1 /
   ycolp(i)   'current storage for columns in superstructure'    / c-1      1
                                                                   c-3      1 /;

*=====================================================================
* These parameters store the values of the binary proposals
* for all the iterations performed for use in integer cuts
* and recovering optimal solution
*=====================================================================
Parameter
   yhxk(i,j,km)  'binary storage parameter yhx'
   yhuk(hu,j,km) 'binary storage parameter yhu'
   ycuk(i,cu,km) 'binary storage parameter ycu'
   ycolk(i,km)   'binary storage parameter ycol';

*=====================================================================
* Declaration of parameters for rest of model
*=====================================================================
* mass balances for each sharp separator
Parameter
   spltfrc(i,m) 'split fraction of distillation columns' / c-1.bc 0.20
                                                           c-2.ab 0.90 /
* minimum condenser temperatures obtained from simulation data
   tcmin(i)     'minimum condenser temperatures'         / c-1  341.92
                                                           c-2  343.01
                                                           c-3  353.54
                                                           c-4  341.92 /
* either hottest hot utility-dtmin or for individual separations
* 2*dtmin below critical temperature of bottoms product
   trmax(j)     'maximum reboiler temperatures';

trmax(j) = 1000;

*====================================================================
* scaled cost coefficients for distillation column fits
* nonlinear fixed-charge cost model
* cost = fc*y + vc*flow*temp
* scaling factor = 1000
*====================================================================
Parameter
   fc(i)   'fixed charge for distillation columns'    / c-1  151.125
                                                        c-2  180.003
                                                        c-3  4.2286
                                                        c-4  213.42   /
   vc(i)   'variable charge for distillation columns' / c-1  0.003375
                                                        c-2  0.000893
                                                        c-3  0.004458
                                                        c-4  0.003176 /
   thu(hu) 'hot utility temperatures'                 / lp   421.0
                                                        ex   373.0    /;
* hot utility cost coeff - gives cost in thousands of dollars per year
* ucost = q(10e+6 kj/hr)*costhu(hu)
Parameter costhu(hu) 'hot utility cost coefficients' / lp 24.908, ex 9.139 /;

Table kf(i,n) 'coeff. for heat duty temperature fits'
            a       b
   c-1   32.4  0.0225
   c-2   25.0  0.0130
   c-3   3.76  0.0043
   c-4   35.1  0.0156;

Table af(i,n) 'coeff. for column temperature fits'
             a      b
   c-1   9.541  1.028
   c-2   12.24  1.050
   c-3   8.756  1.029
   c-4   9.181  1.005;

*=====================================================================
* define scalar quantities for rest of model
*=====================================================================
Scalar
   totflow 'total flow to superstructure'               / 396    /
   u       'large number for logical constraints'       / 1500   /
   uint    'upper bound for integer logical'            / 20     /
   fchx    'fixed charge for heat exchangers scaled'    / 3.392  /
   vchx    'variable charge for heat exchangers scaled' / 0.0893 /
   htc     'overall heat transfer coefficient'          / 0.0028 /
   dtmin   'minimum temperature approach'               / 10.0   /
   tcin    'inlet temperature of cold water'            / 305.0  /
   tcout   'outlet temperature of cold water'           / 325.0  /
   costcw  'cooling water cost coefficient'             / 4.65   /
   beta    'income tax correction factor'               / 0.52   /
   alpha   'one over payout time factor in years'       / 0.40   /;

$onText
*=====================================================================
  The parameters declared here are assigned throughout the
  algorithmic procedures.
  They perform the following tasks in the algorithm
    1) transfer of solution data between master and subproblem
    2) storage of solution data
    3) control of upper and lower bounds in milp master
    4) storage of optimal solution
*=====================================================================
$offText

* Storage of variable levels for each iteration
* Identifier derived from name of variable with letter "k" appended
Parameter
   fk(i,km)      'storage of flowrates'
   qrk(j,km)     'storage of reboiler duties'
   qck(i,km)     'storage of condenser duties'
   qcrk(i,j,km)  'storage of heat integrated exchanges'
   qhuk(hu,j,km) 'storage of hot utility usage'
   qcuk(i,cu,km) 'storage of cold utility usage'
   tck(i,km)     'storage of condenser temperatures'
   trk(j,km)     'storage of reboiler temperatures'
   lmtdk(i,km)   'storage of lmtds';

Scalar
   zoaup         'single value storage of upper bound'  /  inf /
   zoalo         'single value storage of lower bounds' / -inf /;

* storage of optimal binary variable combination
* continuous variable levels are not stored separately as they
* can be obtained from the xxxk storage parameters above
Parameter
   yhxopt(i,j)   'optimal heat integration'
   yhuopt(hu,j)  'optimal hot utility match'
   ycuopt(i,cu)  'optimal cold utility match'
   ycolopt(i)    'optimal superstructure';

Scalar kopt      'iteration at which optimal solution was found';

* storage of sign() of Lagrange multiplier from nonlinear equalities
Parameter lmtdmar(i,km) 'direction matrix for nonlinear equalities';

*=====================================================================
* declaration of variables
*=====================================================================
Variable
   zoau         'objective function value of nlp subproblem'
   zoal         'objective function value of milp masters'
   vqcr(km)     'heat integration contribution to milpcon'
   vqhu(km)     'hot utility exchange contribution to milpcon'
   vqcu(km)     'cold utility exchange contribution to milpcon';

Positive Variable
   f(i)         'flowrates to columns'
   qr(j)        'reboiler duties for column with reboiler j'
   qc(i)        'condenser duties for column i'
   qcr(i,j)     'heat integration heat transfer'
   qhu(hu,j)    'hot utility heat transfer'
   qcu(i,cu)    'cold utility heat transfer'
   tc(i)        'condenser temperature for column with cond. i'
   tr(j)        'reboiler temperature for column with reb. j'
   lmtd(i)      'lmtd for cooling water exchanges'
   sl1(i)       'artificial slack variable for lmtd equalities'
   sl2(i)       'artificial slack variable for lmtd equalities'
   s1(i)        'artificial slack variable for reb-con equalities'
   s2(i)        'artificial slack variable for reb-con equalities'
   s3(i)        'artificial slack variable for duty equalities'
   s4(i)        'artificial slack variable for duty equalities';

Binary Variable
   yhx(i,j)     'heat integration matches condenser i reboiler j'
   yhu(hu,j)    'hot utility matches hot utility hu reboiler j'
   ycu(i,cu)    'cold utility matches condenser i cold util cu'
   ycol(i)      'columns in superstructure';

*=====================================================================
* declaration of equations
* for solution of the nlp subproblems:
* early versions of GAMS did not permit binary variables to appear
* in the constraints of a nonlinear programming problem even if
* they appeared in linear constraints and were fixed at a bound
* therefore -
* constraints that contain the binary variables are duplicated:
* one form contains the declared binary variable and the  other
* substitutes a parameter that is assigned the current level of
* the binary variable.  constraints that are duplicated and are to
* appear in the nlp subproblem model have the letter "n" prepended
* to the equation name.
*=====================================================================
Equation
   nlpobj         'nlp subproblems objective'
   milpcon(km)    'nonlinear contribution to milp objective'
   evqcr(km)      'heat integration contribution to milpcon'
   evqhu(km)      'hot utility exchange contribution to milpcon'
   evqcu(km)      'cold utility exchange contribution to milpcon'
   lmtdsn(i)      'nonlinear form of lmtd definition'
   lmtdsm(i,km)   'linearization of lmtdsn(i) in milp masters'
   ntempset(i)    'sets temperatures of inactive columns to 0 (nlp)'
   tempset(i)     'sets temperatures of inactive columns to 0 (milp)'
   nartrex1(i)    'relaxes artificial slack variables (nlp)'
   artrex1(i)     'relaxes artificial slack variables (milp)'
   nartrex2(i)    'relaxes artificial slack variables (nlp)'
   artrex2(i)     'relaxes artificial slack variables (milp)'
   material(m)    'material balances for each intermediate product'
   feed           'feed to superstructure'
   nmatlog(i)     'material balance logical constraints (nlp)'
   matlog(i)      'material balance logical constraints'
   duty(i)        'heat duty definition of condenser i'
   rebcon(i,j)    'equates condenser and reboiler duties'
   conheat(i)     'condenser heat balances'
   rebheat(j)     'reboiler heat balances'
   dtminlp(j)     'minimum temp approach for low pressure steam'
   ndtminc(i)     'minimum temp allowable for each condenser (nlp)'
   dtminc(i)      'minimum temp allowable for each condenser'
   trtcdef(i,j)   'relates reboiler and condenser temps of columns'
   ndtmincr(i,j)  'minimum temp approach for heat integration (nlp)'
   ndtminex(j)    'minimum temp approach for exhaust steam (nlp)'
   nhxclog(i,j)   'logical constraint for heat balances (nlp)'
   nhxhulog(hu,j) 'logical constraint for heat balances (nlp)'
   nhxculog(i,cu) 'logical constraint for heat balances (nlp)'
   nqcqrlog(i)    'logical constraint for con-reb duties (nlp)'
   dtmincr(i,j)   'minimum temp approach for heat integration'
   dtminex(j)     'minimum temp approach for exhaust steam'
   hxclog(i,j)    'logical constraint for heat balances'
   hxhulog(hu,j)  'logical constraint for heat balances'
   hxculog(i,cu)  'logical constraint for heat balances'
   qcqrlog(i)     'logical constraint for con-reb duties'
   boundup        'upper bound on milp objective'
   boundlo        'lower bound on milp objective'

* these are the pure binary constraints of the minlp
   sequen(m)      'restricts superstructure to a single sequence'
   lead           'sequence control'
   limutil(j)     'limits columns to have a single hot utility'
   hidirect(i,j)  'requires a single direction of heat integration'
   heat(i)        'logical integer constraint'
   cuts(km)       'integer cuts for kth iteration';

*=====================================================================
* equations for nlp subproblems
* note that some equations are duplicated in structure but
* given different names in the nlp and milp. these equations
* involve both continuous and binary variables. In older
* versions of GAMS, it was not permissible to pose nonlinear
* models with discrete variables present, even when their values
* were held fixed (rmidnlp). This required two forms of the equation
* two be declared: one with the discrete variables present (milp)
* and one with binary variables replaced by parameters that have
* been assigned the current levels of their associated binary
* variables (nlp). These equations start with the letter "n"
* in the nlp subproblems.
*=====================================================================
*                         capital costs
nlpobj.. zoau =e= alpha*( sum(i,fc(i)*ycolp(i) + vc(i)*(tc(i) - tcmin(i))*f(i))
                        + sum((i,j)$zcrhx(i,j),fchx*yhxp(i,j)
                        + (vchx/htc)*(qcr(i,j)/(tc(i) - tr(j) + 1 - ycolp(i))))
                        + sum((i,cu),fchx*ycup(i,cu)
                        + (vchx/htc)*(qcu(i,cu)/(lmtd(i) + 1 - ycolp(i))))
                        + sum((hu,j),fchx*yhup(hu,j)
                        + (vchx/htc)*(qhu(hu,j)/(thu(hu) - tr(j)))))
*                         operating costs
                        + beta*((costcw*sum((i,cu),qcu(i,cu)))+ sum((hu,j),costhu(hu)*qhu(hu,j)));

lmtdsn(i)..   lmtd(i) - (2/3)*sqrt((tc(i) - tcin)*(tc(i) - tcout))
            - (1/6)*((tc(i) - tcin) + (tc(i) - tcout)) - (sl1(i) - sl2(i)) =e= 0;

nartrex1(i).. s1(i) + s2(i) + sl1(i) - u*(1 - ycolp(i)) =l= 0;

nartrex2(i).. s3(i) + s4(i) + sl2(i) - u*(1 - ycolp(i)) =l= 0;

ntempset(i).. tc(i) + lmtd(i) + sum(j$zcr(i,j),tr(j)) - u*ycolp(i) =l= 0;

material(m).. sum(i$pm(i,m),spltfrc(i,m)*f(i)) - sum(i$fm(i,m),f(i)) =e= 0;

feed..        sum(i$zlead(i),f(i)) =e= totflow;

duty(i).. qc(i) - (kf(i,"a") + kf(i,"b")*(tc(i) - tcmin(i))) - (s3(i) - s4(i)) =e= 0;

rebcon(i,j)$zcr(i,j).. qr(j) - qc(i) =e= 0;

conheat(i).. qc(i) =e= sum(j$zcrhx(i,j),qcr(i,j)) + sum(cu,qcu(i,cu));

rebheat(j).. qr(j) =e= sum(i$zcrhx(i,j),qcr(i,j)) + sum(hu,qhu(hu,j));

trtcdef(i,j)$zcr(i,j)..
   tr(j) - (af(i,"a") + af(i,"b")*(tc(i) - tcmin(i))) - (s1(i) - s2(i)) =e= 0;

nmatlog(i).. f(i) - u*ycolp(i) =l= 0;

ndtminc(i).. (tcmin(i) - tc(i) - u*(1 - ycolp(i))) =l= 0;

dtminlp(j).. dtmin - (thu("lp") - tr(j)) =l= 0;

ndtmincr(i,j)$zcrhx(i,j).. tr(j) - tc(i)  - u*(1 - yhxp(i,j)) + dtmin =l= 0;

ndtminex(j).. dtmin - (thu("ex") - tr(j)) - u*(1 - yhup("ex",j)) =l= 0;

nhxclog(i,j)$zcrhx(i,j).. qcr(i,j) =l= u*yhxp(i,j);

nhxhulog(hu,j).. qhu(hu,j) =l= u*yhup(hu,j);

nhxculog(i,cu).. qcu(i,cu) =l= u*ycup(i,cu);

nqcqrlog(i)..    qc(i) + sum(j$zcr(i,j),qr(j)) - u*ycolp(i) =l= 0;

Model nlpsub  '- collection of equations for nlp subproblems'
              / nlpobj , lmtdsn  , nartrex1, nartrex2, ntempset, material, feed
                nmatlog, duty    , rebcon  , conheat , rebheat , ndtminc , dtminlp
                trtcdef, ndtmincr, ndtminex, nhxclog , nhxhulog, nhxculog, nqcqrlog /;

nlpsub.solPrint=%solPrint.Report%;

*======================================================================
* Define equations for milp master problems
* Note: the nonlinear parts of the objective function related
*       to heat exchanger area have been broken out into separate
*       constraints to perform their linearizations, only a
*       contribution term appears in the linearized objective
*       function milpcon.
*======================================================================
milpcon(k)..
   zoal =g= alpha*(sum(i,fc(i)*ycol(i))
         +  fchx*(sum((i,j)$zcrhx(i,j),yhx(i,j))
         +  sum((hu,j),yhu(hu,j)) + sum((i,cu),ycu(i,cu)))
         +  sum(i, (vc(i)*((tck(i,k) - tcmin(i))*(f(i) - fk(i,k))
                 +  fk(i,k)*(tc(i) - tcmin(i)))))
         +  (vchx/htc)*(vqcr(k) + vqhu(k) + vqcu(k)))
         +  beta*((costcw*sum((i,cu),qcu(i,cu)))
         +  sum((hu,j),costhu(hu)*qhu(hu,j)));

*==========================================================================
* these are the linearized contributions to the objective related
* to heat exchange.  the appearance of the binary variable storage
* parameters in the denominator of some of the expressions is done
* to prevent division by zero during model generation for linearizations
* done at points where the temperatures were set to zero for unused
* columns.  the numerator is zero then also and no error is introduced.
*==========================================================================
evqcr(k)..
   vqcr(k) =e= sum((i,j)$zcrhx(i,j),((qcrk(i,j,k)/(tck(i,k) - trk(j,k) + 1 - ycolk(i,k)))
                    + ((1/(tck(i,k) - trk(j,k) + 1 - ycolk(i,k)))
                    * (qcr(i,j) - qcrk(i,j,k)))*ycolk(i,k)
                    + ((qcrk(i,j,k)/(sqr(tck(i,k) - trk(j,k)) + 1 - ycolk(i,k)))
                    * ((tr(j) - trk(j,k)) - (tc(i) - tck(i,k))))));

evqhu(k)..
   vqhu(k) =e= sum((hu,j),((qhuk(hu,j,k)/(thu(hu) - trk(j,k)))
                    + ((1/(thu(hu) - trk(j,k)))
                    * (qhu(hu,j) - qhuk(hu,j,k)))*sum(i$zcr(i,j), ycolk(i,k))
                    + ((qhuk(hu,j,k)/sqr(thu(hu) - trk(j,k)))*(tr(j) - trk(j,k)))));

evqcu(k)..
   vqcu(k) =e= sum((i,cu),((qcuk(i,cu,k)/(lmtdk(i,k) + 1 - ycolk(i,k)))
                    + ((1/(lmtdk(i,k) + 1 - ycolk(i,k)))
                    * (qcu(i,cu) - qcuk(i,cu,k)))*ycolk(i,k)
                    - ((qcuk(i,cu,k)/(sqr(lmtdk(i,k)) + 1 - ycolk(i,k)))
                    * (lmtd(i) - lmtdk(i,k)))));

lmtdsm(i,k)..
     lmtdmar(i,k)*(lmtd(i) - (2/3)*sqrt((tck(i,k) - tcin)*(tck(i,k) - tcout))
   - (1/6)*((tck(i,k) - tcin) + (tck(i,k) - tcout))
   - ((1/3)*(((2*tck(i,k) - (tcin + tcout))
   / sqrt(sqr(tck(i,k)) - (tcin + tcout)*tck(i,k) + (tcin*tcout))) + 1))
   * (tc(i) - tck(i,k)) - (sl1(i) - sl2(i))) =l= 0;

artrex1(i).. s1(i) + s2(i) + sl1(i) - u*(1 - ycol(i)) =l= 0;

artrex2(i).. s3(i) + s4(i) + sl2(i) - u*(1 - ycol(i)) =l= 0;

tempset(i).. tc(i) + lmtd(i) + sum(j$zcr(i,j), tr(j)) - u*ycol(i) =l= 0;

matlog(i).. f(i) - u*ycol(i) =l= 0;

dtminc(i).. (tcmin(i) - tc(i) - u*(1 - ycol(i))) =l= 0;

dtmincr(i,j)$zcrhx(i,j).. tr(j) - tc(i) - u*(1 - yhx(i,j)) + dtmin =l= 0;

dtminex(j).. dtmin - (thu("ex") - tr(j)) - u*(1 - yhu("ex",j)) =l= 0;

hxclog(i,j)$zcrhx(i,j).. qcr(i,j) =l= u*yhx(i,j);

hxhulog(hu,j).. qhu(hu,j) =l= u*yhu(hu,j);

hxculog(i,cu).. qcu(i,cu) =l= u*ycu(i,cu);

qcqrlog(i).. qc(i) + sum(j$zcr(i,j), qr(j)) - u*ycol(i) =l= 0;

* pure binary constraints
* material balances determine sequence
sequen(m).. sum(i$pm(i,m), ycol(i)) - sum(i$fm(i,m), ycol(i)) =e= 0;

* select 1 sequence
lead.. sum(i$zlead(i),ycol(i)) =e= 1;

* limit choice of hot utility to 1
limutil(j).. sum(hu,yhu(hu,j)) =l= 1;

* only one of the mutual heat integration binaries can be 1
hidirect(i,j)$zlim(i,j).. yhx(i,j) + sum((ip,jp)$(ord(ip) = ord(j) and
                                                  ord(jp) = ord(i)), yhx(ip,jp)) =l= 1;

* if a column doesn't exist then all binary variables associated
* with it must also be set to zero
heat(i)..    sum(j$zcrhx(i,j),yhx(i,j)
          +  sum((ip,jp)$((ord(ip) = ord(j)) and (ord(jp) = ord(i))),yhx(ip,jp)))
          +  sum((hu,j),yhu(hu,j)$zcr(i,j))
          +  sum(cu,ycu(i,cu)) - uint*ycol(i)
         =l= 0;

* integer cuts
cuts(k)..    sum(i,sign(ycolk(i,k) - 0.5)*ycol(i))
          +  sum((i,j)$zcrhx(i,j),sign(yhxk(i,j,k) - 0.5)*yhx(i,j))
          +  sum((hu,j),sign(yhuk(hu,j,k) - 0.5)*yhu(hu,j))
          +  sum((i,cu),sign(ycuk(i,cu,k) - 0.5)*ycu(i,cu))
         =l= sum(i,ycolk(i,k)) + sum((i,j)$zcrhx(i,j),yhxk(i,j,k))
          +  sum((hu,j),yhuk(hu,j,k)) + sum((i,cu),ycuk(i,cu,k)) - 1;

*======================================================================
* declare the milp master problem
*======================================================================
Model master 'milp master problem'
             / milpcon, evqcr  , evqhu   , evqcu   , lmtdsm , artrex1
               artrex2, tempset, material, feed    , matlog , duty
               rebcon , conheat, rebheat , dtminc  , dtminlp, trtcdef
               dtmincr, dtminex, hxclog  , hxhulog , hxculog, qcqrlog
               sequen , lead   , limutil , hidirect, heat   , cuts    /;

master.solPrint=%solPrint.Summary%;

*=====================================================================
* all declarations made, start algorithmic procedures

* initialize the optimal storage parameters to 1st guess
*=====================================================================
yhxopt(i,j)  = yhxp(i,j);
yhuopt(hu,j) = yhup(hu,j);
ycuopt(i,cu) = ycup(i,cu);
ycolopt(i)   = ycolp(i);
kopt         = 1;

*======================================================================
* assign the initial configuration to the binary proposal parameter
*======================================================================
kiter("k-1") = yes;

yhxk(i,j,kiter)  = yhxp(i,j);
yhuk(hu,j,kiter) = yhup(hu,j);
ycuk(i,cu,kiter) = ycup(i,cu);
ycolk(i,kiter)   = ycolp(i);
yhx.l(i,j)       = yhxp(i,j);
yhu.l(hu,j)      = yhup(hu,j);
ycu.l(i,cu)      = ycup(i,cu);
ycol.l(i)        = ycolp(i);

* set an arbitrary initial lower bound
zoal.l = -10e+6;

*======================================================================
* give the continuous variables a starting point for 1st nlp
*======================================================================
tr.l("r-1")   = 410;
tc.l("c-1")   = 390;
tc.l("c-3")   = 360;
tr.l("r-3")   = 380;
tc.l("c-2")   = 0;
tr.l("r-2")   = 0;
tc.l("c-4")   = 0;
tr.l("r-4")   = 0;
f.l("c-1")    = totflow;
lmtd.l("c-1") = 75;
lmtd.l("c-3") = 25;
lmtd.l("c-2") = 0;
lmtd.l("c-4") = 0;
qr.l("r-2")   = 0;
qc.l("c-2")   = 0;
qr.l("r-4")   = 0;
qc.l("c-4")   = 0;

*======================================================================
* add bounds on tc. A sqrt in equation lmtdsn is defined for tc > tcout
* and for tc < tcin. The relevant interval is determined for each
* element of tc based on the initial values given above.
*======================================================================
tc.lo("c-1") = tcout+1;
tc.up("c-2") = tcin-1;
tc.lo("c-3") = tcout+1;
tc.up("c-4") = tcin-1;

*======================================================================
* bound the reboiler temperatures by their maximum allowable
*======================================================================
tr.up(j) = trmax(j);

*======================================================================
* initialize the dynamic sets for algorithm control
*======================================================================
k(km)       =  no;
kiter(km)   =  no;
kdynmax(km) = yes;

*======================================================================
* major driving loop of algorithm
*======================================================================
loop(kloop$kdynmax(kloop),

*  update the dynamic iteration sets
* -set kiter to contain only the current iteration element
* -add to k the current iteration element
   kiter(km) = yes$(ord(km) = ord(kloop));
   k(kiter)  = yes;

*  store the current binary combination
   yhxk(i,j,kiter)  = yhx.l(i,j);
   yhuk(hu,j,kiter) = yhu.l(hu,j);
   ycuk(i,cu,kiter) = ycu.l(i,cu);
   ycolk(i,kiter)   = ycol.l(i);

*  set the current combination parameters that appear in the nlp constraints
   yhxp(i,j)  = yhx.l(i,j);
   yhup(hu,j) = yhu.l(hu,j);
   ycup(i,cu) = ycu.l(i,cu);
   ycolp(i)   = ycol.l(i);
   zoal.lo    = zoal.l;

*======================================================================
* the current levels of the lmtds are moved away from zero
* to prevent evaluation errors in the next nlp subproblem
*======================================================================
   lmtd.l(i) = lmtd.l(i) + 1;

   nlpsub.resLim = 15;
*  solve the nlp subproblem
   solve nlpsub using nlp minimizing zoau;

*  resolve with Conopt to get marginals for lmtdsn, if not provided by used NLP solver
   if(nlpsub.marginals = 0,
      option nlp = conopt;
      solve nlpsub using nlp minimizing zoau;
      option nlp = %system.nlp%;
   );

*======================================================================
* update the optimal solution storage parameters if new nlp
* objective function value is less than the incumbent
*======================================================================
   if((zoau.l < zoaup),
      yhxopt(i,j)  = yhx.l(i,j);
      yhuopt(hu,j) = yhu.l(hu,j);
      ycuopt(i,cu) = ycu.l(i,cu);
      ycolopt(i)   = ycol.l(i);
      kopt         = ord(kloop);
   );

*======================================================================
* assign the solution levels of the variables that appear in the
* nonlinear equations to their corresponding storage parameters
*======================================================================
   fk(i,kiter)      = f.l(i);
   qrk(j,kiter)     = qr.l(j);
   qck(i,kiter)     = qc.l(i);
   qcrk(i,j,kiter)  = qcr.l(i,j);
   qhuk(hu,j,kiter) = qhu.l(hu,j);
   qcuk(i,cu,kiter) = qcu.l(i,cu);
   tck(i,kiter)     = tc.l(i);
   trk(j,kiter)     = tr.l(j);
   lmtdk(i,kiter)   = lmtd.l(i);

*======================================================================
* assign the sign of marginal values of the nonlinear equalities
* to the storage parameter lmtdmar
*======================================================================
   lmtdmar(i,kiter) = -sign(lmtdsn.m(i))$(lmtdsn.m(i) <> eps);

*======================================================================
* store the smallest nlp objective value for upper bound on master
*======================================================================
   zoaup   = min(zoaup,zoau.l);
   zoal.up = zoaup;
*  protect against numerical errors introduced by the solver
   zoal.lo = min(zoal.lo,zoal.up);

*  now solve the milp master problem
   solve master using mip minimizing zoal;

   display "new binary combination", ycol.l, yhx.l, yhu.l, ycu.l;

*======================================================================
* check stopping criterion:
* master problem integer infeasible
*======================================================================
   if((master.modelStat=%modelStat.infeasible% or
       master.modelStat=%modelStat.integerInfeasible% or
       master.modelStat=%modelStat.infeasibleNoSolution%),
      kdynmax(km) = no;
      display "stopping criterion met", zoaup, yhxopt, yhuopt, ycuopt, ycolopt, kopt;
   );
);
