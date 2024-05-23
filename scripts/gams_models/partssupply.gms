$title Parts Supply Problem (PARTSSUPPLY,SEQ=404)

$onText
This model is based on the ps2_f_s.358 .. ps10_s_mn.396 models by
Hideo Hashimoto, Kojun Hamada, and Nobuhiro Hosoe.

Using the following options, these models can be run:

ps2_f              : default
ps2_f_eff          : --nsupplier=1
ps2_f_inf          : --nsupplier=1 --alttheta=1
ps2_f_s            : --useic=1
ps2_s              : --useic=1
ps3_f              : --nsupplier=3
ps3_s              : --nsupplier=3  --uselicd=1
ps3_s_gic          : --nsupplier=3  --useic=1
ps3_s_mn  1st solve: --nsupplier=3  --uselicd=1
          2nd solve: --nsupplier=3  --uselicd=1 --altpi=1
          3rd solve: --nsupplier=3  --uselicd=1 --alttheta=1
ps3_s_scp 1st solve: --nsupplier=3  --alttheta=2 --modweight=1 --useic=1
          2nd solve: --nsupplier=3  --alttheta=2 --modweight=1 --uselicd=1 --uselicu=1
ps5_s_mn           : --nsupplier=5  --uselicd=1 --nsamples=1000
ps10_s             : --nsupplier=10 --uselicd=1
ps10_s_mn          : --nsupplier=10 --uselicd=1 --nsamples=1000

Alternatively, the corresponding original model files can be found in
the GAMS model library.

Keywords: nonlinear programming, contract theory, principal-agent problem,
          adverse selection, parts supply problem
$offText

$if not set nsupplier $set nsupplier 2
$if not set modweight $set modweight 0
$if not set useic     $set useic     0
$if not set uselicd   $set uselicd   0
$if not set uselicu   $set uselicu   0
$if not set usemn     $set usemn     0
$if not set altpi     $set altpi     0
$if not set alttheta  $set alttheta  0
$if not set nsamples  $set nsamples  1

Set
   i 'type of supplier'  / 1*%nsupplier% /
   t 'Monte-Carlo draws' / 1*%nsamples%  /;

Alias (i,j);

Parameter
   theta(i)    'efficiency'
   pt(i,t)     'probability of type'
   p(i)        'probability of type for currently evaluated scenario'
   icweight(i) 'weight in ic constraints';

Scalar ru 'reservation utility' / 0 /;

* Data
$ifThen %nsupplier% == 1
$if %alttheta% == 0 Parameter theta(i) / 1 0.2 /;
$if %alttheta% == 1 Parameter theta(i) / 1 0.3 /;
Parameter p(i) / 1 1 /;

$elseIf %nsupplier% == 2
Parameter
   theta(i) / 1 0.2, 2 0.3 /
   p(i)     / 1 0.2, 2 0.8 /;

$elseIf %nsupplier% == 3
$if %alttheta% == 0  Parameter theta(i) / 1 0.1,  2 0.2,  3 0.3  /;
$if %alttheta% == 1  Parameter theta(i) / 1 0.1,  2 0.3,  3 0.31 /;
$if %alttheta% == 2  Parameter theta(i) / 1 0.1,  2 0.4,  3 0.9  /;
$if %altpi%    == 0  Parameter p(i)     / 1 0.2,  2 0.5,  3 0.3  /;
$if %altpi%    == 1  Parameter p(i)     / 1 0.3,  2 0.1,  3 0.6  /;

$else
theta(i) = ord(i)/card(i);
p(i)     =      1/card(i);
$endIf

loop(t, pt(i,t) = uniform(0,1));
pt(i,t) = pt(i,t)/sum(j, pt(j,t));
$if %nsamples% == 1 pt(i,t) = p(i);

Positive Variable
   x(i) "quality"
   b(i) "maker's revenue"
   w(i) "price";

Variable Util "maker's utility";

Equation
   obj     "maker's utility function"
   rev(i)  "maker's revenue function"
   pc(i)   "participation constraint"
   ic(i,j) "incentive compatibility constraint"
   licd(i) "incentive compatibility constraint"
   licu(i) "incentive compatibility constraint"
   mn(i)   "monotonicity constraint";

obj..    Util =e= sum(i, p(i)*(b(i) - w(i)));

rev(i).. b(i) =e= sqrt(x(i));

pc(i)..
$if %modweight% == 0  w(i) - theta(i)   *x(i) =g= ru;
$if %modweight% == 1  w(i) - icweight(i)*x(i) =g= ru + theta(i);

ic(i,j).. w(i) - icweight(i)*x(i) =g= w(j)   - icweight(i)*x(j);

licd(i)$(ord(i) < card(i))..
   w(i) - icweight(i)*x(i) =g= w(i+1) - icweight(i)*x(i+1);

licu(i)$(ord(i) > 1)..
   w(i) - icweight(i)*x(i) =g= w(i-1) - icweight(i)*x(i-1);

mn(i)$(ord(i) < card(i)).. x(i) =g= x(i+1);

* Setting Lower Bounds on Variables to Avoid Division by Zero
x.lo(i) = 0.0001;

Model m 'parts supply model w/o monotonicity'
/ all - mn
$if %useic%   == 0 -ic
$if %uselicd% == 0 -licd
$if %uselicu% == 0 -licu
/;

Model m_mn 'parts supply model w/ monotonicity' / m + mn /;

* Parameters to store some solution values
Parameter
   Util_lic(t)  'util solved w/o MN'
   Util_lic2(t) 'util solved w/  MN'
   x_lic(i,t)   'x solved    w/o MN'
   x_lic2(i,t)  'x solved    w/  MN';

* Solving the Model
option limRow = 0, limCol = 0;

loop(t,
   p(i) = pt(i,t);
   icweight(i) = theta(i)$(not %modweight%) + (1 - theta(i) + sqr(theta(i)))$(%modweight%);
   solve m maximizing Util using nlp;

   Util_lic(t) = util.l;
   x_lic(i,t)  = x.l(i);
   solve m_mn maximizing Util using nlp;

   Util_lic2(t) = util.l;
   x_lic2(i,t)  = x.l(i);
   option solPrint = off;
);

$if %nsamples% == 1 $exit

* Evaluation and display results as in ps5_s_mn
Parameter
   MN_lic(t)    'monotonicity of x solved w/o MN'
   MN_lic2(t)   'monotonicity of x solved w/  MN'
   Util_gap(t)  'gap between Util_lic and Util_lic2'
   F(i,t)       'cumulative probability (Itho p. 42)'
   noMHRC0(i,t) 'no MHRC combination between i and i-1 (MHRC: monotone hazard rate condition)'
   noMHRC(t)    '>=1: no MHRC case'
   p_noMHRC     'no MHRC case [%]'
   p_noMN_lic   'no MN case [%]'
   p_Util_gap   'no util-equality case [%]';

MN_lic(t)   = sum(i, 1$(round(x_lic (i,t),10) < round(x_lic (i+1,t),10)));
MN_lic2(t)  = sum(i, 1$(round(x_lic2(i,t),10) < round(x_lic2(i+1,t),10)));
Util_gap(t) = 1$(round(Util_lic(t),10) <> round(Util_Lic2(t),10));
F(i,t)      = sum(j$(ord(j) <= ord(i)), pt(j,t));
noMHRC0(i,t)$(ord(i) < card(i)) = 1$(F(i,t)/pt(i+1,t) < F(i-1,t)/pt(i,t));
noMHRC(t)$(sum(i, noMHRC0(i,t)) >= 1) = 1;

* Computing probability that MHRC and MN holds.
p_noMHRC   = sum(t$(noMHRC(t)   > 0), 1)/card(t)*100;
p_noMN_lic = sum(t$(MN_lic(t)   > 0), 1)/card(t)*100;
p_Util_gap = sum(t$(Util_gap(t) > 0), 1)/card(t)*100;

display p_noMHRC, p_noMN_LIC, p_Util_gap;

* Generating CSV file for summary
File sol / solution_lic.csv /;
put  sol;
sol.pc = 5;
sol.pw = 32767;

put "";
loop(i, put "pt(i,t)";);   put "" "" "" "";
loop(i, put "x: w/o MN";);
loop(i, put  "x: w/ MN";); put /;
put "";
loop(i, put i.tl;); put ">=1: no MHRC" "Util: w/o MN" "Util: w/ MN" "Util_gap: =1: not equal";
loop(i, put i.tl;);
loop(i, put i.tl;); put "MN_lic: >=1: no MN" "MN_lic2: >=1: no MN"/;
loop(t,
   put t.tl;
   loop(i, put pt(i,t):10:5;);
   put noMHRC(t) Util_lic(t):20:10 Util_Lic2(t):20:10 Util_gap(t);
   loop(i, put X_lic(i,t););
   loop(i, put X_lic2(i,t););
   put MN_lic(t) MN_lic2(t)/;
);
put /;
