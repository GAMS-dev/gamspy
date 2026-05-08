$title Cameroon General Equilibrium Model (CAMCNS,SEQ=209)

$onText
This general equilibrium model is widely used as a blueprint
for new model developments. It follows closely the style and type
of model pioneered by Devis, De Melo and Robinson in the late 1970.

The original model (CAMCGE,SEQ=81) is formulated as an optimization
model, but it is really a square system of nonlinear equations.
In this version, we formulate the model directly as a square system
using the model type CNS = Constrained Nonlinear System.

An MCP version exist under the name (CAMMCP,SEQ=129) and an MPSGE
version exist under the name (CAMMGE,SEQ=140).


Condon, T, Dahl, H, and Devarajan, S, Implementing a Computable General
Equilibrium Model on GAMS - The Cameroon Model. Tech. rep., The World
Bank, 1987.

Keywords: constraint nonlinear system, general equilibrium model, economic policy,
          Cameroonian economy, market equilibrium
$offText

Set
   i     'sectors'          / ag-subsist 'food crops'
                              ag-exp+ind 'cash crops'
                              sylvicult  'forestry'
                              ind-alim   'food processing'
                              biens-cons 'consumer goods'
                              biens-int  'intermediate goods'
                              cim-int    'construction materials'
                              biens-cap  'capital goods'
                              construct  'construction'
                              services   'private services'
                              publiques  'public services'        /
   lc    'labor categories' / rural, urban-unsk, urban-skil       /
   it(i) 'traded sectors'
   in(i) 'nontraded sectors';

Alias (i,j);

Parameter
   delta(i)    'Armington function share parameter                      (unity)'
   ac(i)       'Armington function shift parameter                      (unity)'
   rhoc(i)     'Armington function exponent                             (unity)'
   rhot(i)     'cet function exponent                                   (unity)'
   at(i)       'cet function shift parameter                            (unity)'
   gamma(i)    'cet function share parameter                            (unity)'
   eta(i)      'export demand elasticity                                (unity)'
   ad(i)       'production function shift parameter                     (unity)'
   cles(i)     'private consumption shares                              (unity)'
   gles(i)     'government consumption shares                           (unity)'
   depr(i)     'depreciation rates                                      (unity)'
   dstr(i)     'ratio of inventory investment to gross output           (unity)'
   kio(i)      'shares of investment by sector of destination           (unity)'
   tm0(i)      'tariff rates                                            (unity)'
   te(i)       'export duty rates                                       (unity)'
   itax(i)     'indirect tax rates                                      (unity)'
   alphl(lc,i) 'labor share parameter in production function            (unity)'

* dummies to hold initial data
   m0(i)      "volume of imports                             ('79-80 bill cfaf)"
   e0(i)      "volume of exports                             ('79-80 bill cfaf)"
   xd0(i)     "volume of domestic output by sector           ('79-80 bill cfaf)"
   k0(i)      "volume of capital stocks by sector            ('79-80 bill cfaf)"
   id0(i)     "volume of investment by sector of origin      ('79-80 bill cfaf)"
   dst0(i)    "volume of inventory investment by sector      ('79-80 bill cfaf)"
   int0(i)    "volume of intermediate input demands          ('79-80 bill cfaf)"
   xxd0(i)    "volume of domestic sales by sector            ('79-80 bill cfaf)"
   x0(i)      "volume of composite good supply               ('79-80 bill cfaf)"
   pwe0(i)    "world market price of exports                            (unity)"
   pwm0(i)    "world market price of imports                            (unity)"
   pd0(i)     "domestic good price                                      (unity)"
   pe0(i)     "domestic price of exports                                (unity)"
   pm0(i)     "domestic price of imports                                (unity)"
   pva0(i)    "value added price by sector                              (unity)"
   qd(i)      "dummy variable for computing ad(i)                       (unity)"
   xllb(i,lc) "dummy variable (l matrix with no zeros)                  (unity)"
   wa0(lc)    "average wage rate by labor category ('79-80 mill cfaf pr worker)"
   ld(lc)     "employment                                        (1000 persons)"
   ls0(lc)    "labor supplies by category                        (1000 persons)";

* base data
wa0("rural")      =  .11;
wa0("urban-unsk") =  .15678;
wa0("urban-skil") = 1.8657;

Scalar
   er     "real exchange rate                   (unity)" /    .21  /
   gr0    "government revenue        ('79-80 bill cfaf)" / 179.00  /
   gdtot0 "government consumption    ('79-80 bill cfaf)" / 135.03  /
   cdtot0 "private consumption       ('79-80 bill cfaf)" / 947.98  /
   fsav0  "foreign saving         ('79-80 bill dollars)" /  36.841 /;

Table io(i,j) 'input-output coefficients (unity)'
               ag-subsist ag-exp+ind sylvicult ind-alim biens-cons biens-int cim-int biens-cap construct services publiques
   ag-subsist      .03046                        .30266     .00206                                         .04120
   ag-exp+ind                 .01518             .02043     .01123    .00669
   sylvicult                                     .00243               .02106
   ind-alim        .00341     .00629             .03241     .01234    .00503                               .00092    .01532
   biens-cons                                    .00105     .05385    .00435                               .00103    .00338
   biens-int       .00676     .12385    .02095   .03794     .08309    .23461  .18289    .01567    .14665   .00929    .08466
   cim-int         .00002     .00025    .00017   .11238     .05095    .05593  .27608    .11722    .18643   .00018
   biens-cap       .00041     .00971    .02427   .00931     .01229    .05259  .02053    .05013    .02622   .00389
   construct       .00472     .00113    .00318   .10456     .01831    .05302  .00172    .00031    .01457   .00385    .00394
   services        .00375     .30649    .26666   .10100     .26072    .23006  .11793    .09922    .13692   .13728    .24145
   publiques       .00022     .00293    .00327   .00536     .00539    .00957  .00486    .00081    .00447   .00219          ;

Table imat(i,j) 'capital composition matrix (unity)'
                 ag-subsist ag-exp+ind sylvicult ind-alim biens-cons biens-int cim-int biens-cap construct services publiques
   ag-subsist        .23637
   biens-cap         .59530     .60608    .63876   .60608     .78723    .63876  .63876    .60608    .71728   .17610    .17610
   construct         .16833     .39392    .36124   .39392     .21277    .36124  .36124    .39392    .28272   .82390    .82390;

Table wdist(i,lc) 'wage proportionality factors (unity)'
                   rural urban-unsk urban-skil
   ag-subsist    1.01890     .71491
   ag-exp+ind     .49556     .34774     .29222
   sylvicult     3.26280    2.28900    1.92320
   ind-alim      1.45710    1.02230     .85902
   biens-cons    1.13350     .79531     .66829
   biens-int     3.10740    2.18060    1.83230
   cim-int       6.32240    4.43640    3.72770
   biens-cap     2.50350    1.75520    1.47580
   construct     2.92040    2.04920    1.72200
   services      1.40390     .98502     .82776
   publiques                1.32630    1.11460;

Table xle(i,lc) 'employment by sector and labor category (1000 persons)'
                   rural urban-unsk urban-skil
   ag-subsist    1654.43  162.89
   ag-exp+ind     399.93   45.50800    5.05700
   sylvicult     7.66200    1.78900     .59700
   ind-alim     12.98900    9.43400    2.35800
   biens-cons   28.34400   37.46200   12.48800
   biens-int    18.33100   16.55300    8.30000
   cim-int       1.45800    1.31700     .66000
   biens-cap     3.11200    2.82000    1.20800
   construct    22.58400   28.46200    7.11600
   services       121.20  125.8       61.96000
   publiques               83.029     32.77100;

Table zz(*,i) 'miscellaneous parameters and initial data'
               ag-subsist  ag-exp+ind sylvicult  ind-alim biens-cons  biens-int  cim-int  biens-cap construct  services  publiques
   m0            2.461       8.039      .023    17.961     37.062    138.57      49.616  134.72                74.439
   e0            4.594     125.07     22.337    23.451      5.864    101.33      10.501    3.838               81.626
   xd0         330.480     131.45     29.503    72.024    118.430    284.38      34.169   10.298     174.12   615.79     163.98
   k           495.730     170.89     73.760      .14e+03 236.870    853.13     102.51    20.600     435.29   769.73     180.36
   depr           .0246       .0472     .0244     .0144      .0212      .0335      .0335    .0111       .0232    .0637      .0637
   rhoc          1.5          .9        .4       1.25       1.25        .5         .75      .4          .4       .4         .4
   rhot          1.5          .9        .4       1.25       1.25        .5         .75      .4          .4       .4         .4
   eta           1.0         1.0       1.0       4.00       4.00       4.0        4.00     4.0         4.0      4.0        4.0
   pd0           1.0         1.0       1.0       1.00       1.00       1.0        1.00     1.0         1.0      1.0        1.0
   tm0            .2205       .2330     .278      .3534      .3826      .1768      .2633    .268
   itax           .0020       .1910     .057      .038       .096       .026       .014     .029        .034     .076
   cles           .2744       .00445              .05599     .14099     .17738                          .004     .31921     .02358
   gles                                                                                                                    1.00
   kio            .11         .09       .06       .01        .04        .14        .02      .01         .08      .34        .100
   dstr           .012203     .026694   .034742   .044291    .059958    .012287             .042047
   dst           4.033       3.509     1.025     3.19        7.101     3.494                .433
   id            6.710                                                                   113.36      138.13                     ;

* computation of parameters and coefficients for calibration
depr(i)    = zz("depr",i);
rhoc(i)    = (1/zz("rhoc",i)) - 1;
rhot(i)    = (1/zz("rhot",i)) + 1;
eta(i)     = zz("eta",i);
tm0(i)     = zz("tm0",i);
te(i)      = 0;
*te(i)     = zz("te",i);
itax(i)    = zz("itax",i);
cles(i)    = zz("cles",i);
gles(i)    = zz("gles",i);
kio(i)     = zz("kio",i);
dstr(i)    = zz("dstr",i);
xllb(i,lc) = xle(i,lc) + (1 - sign(xle(i,lc)));
m0(i)      = zz("m0",i);
it(i)      = yes$m0(i);
in(i)      = not it(i);
e0(i)      = zz("e0",i);
xd0(i)     = zz("xd0",i);
k0(i)      = zz("k",i);
pd0(i)     = zz("pd0",i);
pm0(i)     = pd0(i);
pe0(i)     = pd0(i);
pwm0(i)    = pm0(i)/((1 + tm0(i))*er);
pwe0(i)    = pe0(i)/((1 + te(i))*er);
pva0(i)    = pd0(i) - sum(j, io(j,i)*pd0(j)) - itax(i);
xxd0(i)    = xd0(i) - e0(i);
dst0(i)    = zz("dst",i);
id0(i)     = zz("id",i);
ls0(lc)    = sum(i, xle(i,lc));

* calibration of all shift and share parameters
* get delta from costmin, x0 from absorption , ac from armington

delta(it)$m0(it) = pm0(it)/pd0(it)*(m0(it)/xxd0(it))**(1+rhoc(it));
delta(it)        = delta(it)/(1+delta(it));
x0(i)            = pd0(i)*xxd0(i) + (pm0(i)*m0(i))$it(i);
ac(it)           =  x0(it)/(delta(it)*m0(it)**(-rhoc(it))
                 + (1 - delta(it))*xxd0(it)**(-rhoc(it)))**(-1/rhoc(it));

* get int0 from inteq, gamma from esupply, alphl from profitmax

int0(i)     = sum(j, io(i,j)*xd0(j) );
gamma(it)   = 1/(1 + pd0(it)/pe0(it)*(e0(it)/xxd0(it))**(rhot(it) - 1));
gamma(in)   = 0;
alphl(lc,i) = (wdist(i,lc)*wa0(lc)*xle(i,lc))/(pva0(i)*xd0(i));

* get ad from output, ld from  profitmax, at from cet

qd(i)  = (xllb(i,"rural")**alphl("rural",i))
       * (xllb(i,"urban-unsk")**alphl("urban-unsk",i))
       * (xllb(i,"urban-skil")**alphl("urban-skil",i))
       * (k0(i)**(1 - sum(lc, alphl(lc,i))));

ad(i)  = xd0(i)/qd(i);
ld(lc) = sum(i, (xd0(i)*pva0(i)*alphl(lc,i)/(wdist(i,lc)*wa0(lc)))$wdist(i,lc));
at(it) = xd0(it)/(gamma(it)*e0(it)**rhot(it) + (1 - gamma(it))
       * xxd0(it)**rhot(it))**(1/rhot(it));

Variable
* prices block
   pd(i)    'domestic prices                                            (unity)'
   pm(i)    'domestic price of imports                                  (unity)'
   pe(i)    'domestic price of exports                                  (unity)'
   pk(i)    'rate of capital rent by sector                             (unity)'
   px(i)    'average output price by sector                             (unity)'
   p(i)     'price of composite goods                                   (unity)'
   pva(i)   'value added price by sector                                (unity)'
   pwm(i)   'world market price of imports                              (unity)'
   pwe(i)   'world market price of exports                              (unity)'
   tm(i)    'xxxxxxxxxxxxxxx                                                   '
*  tm(it)   'tariff rates                                               (unity)'

* production block
   x(i)     "composite goods supply                          ('79-80 bill cfaf)"
   xd(i)    "domestic output by sector                       ('79-80 bill cfaf)"
   xxd(i)   "domestic sales                                  ('79-80 bill cfaf)"
   e(i)     "exports by sector                               ('79-80 bill cfaf)"
   m(i)     "imports                                         ('79-80 bill cfaf)"

* factors block
   k(i)     "capital stock by sector                         ('79-80 bill cfaf)"
   wa(lc)   "average wage rate by labor category    (curr mill. cfaf pr person)"
   ls(lc)   "labor supply by labor category                      (1000 persons)"
   l(i,lc)  "employment by sector and labor category             (1000 persons)"

* demand block
   int(i)   "intermediates uses                              ('79-80 bill cfaf)"
   cd(i)    "final demand for private consumption            ('79-80 bill cfaf)"
   gd(i)    "final demand for government consumption         ('79-80 bill cfaf)"
   id(i)    "final demand for productive investment          ('79-80 bill cfaf)"
   dst(i)   "inventory investment by sector                  ('79-80 bill cfaf)"
   y        "private gdp                                       (curr bill cfaf)"
   gr       "government revenue                                (curr bill cfaf)"
   tariff   "tariff revenue                                    (curr bill cfaf)"
   indtax   "indirect tax revenue                              (curr bill cfaf)"
   duty     "export duty revenue                               (curr bill cfaf)"
   gdtot    "total volume of government consumption          ('79-80 bill cfaf)"
   mps      "marginal propensity to save                                (unity)"
   hhsav    "total household savings                           (curr bill cfaf)"
   govsav   "government savings                                (curr bill cfaf)"
   deprecia "total depreciation expenditure                    (curr bill cfaf)"
   savings  "total savings                                     (curr bill cfaf)"
   fsav     "foreign savings                                (curr bill dollars)"
   dk(i)    "volume of investment by sector of destination   ('79-80 bill cfaf)"

* welfare indicator for objective function
   omega    "objective function variable                     ('79-80 bill cfaf)";

p.lo(i)  = .01; pd.lo(i)   = .01; pm.lo(it) = .01;
pk.lo(i) = .01; px.lo(i)   = .01; x.lo(i)   = .01;
m.lo(it) = .01; xxd.lo(it) = .01; wa.lo(lc) = .01;
e.lo(it) = .01; l.lo(i,lc) = .01; y.lo      = .01;
xd.lo(i) = .01; pwe.lo(it) = .01; int.lo(i) = .01;

Equation
* price block
   pmdef(i)        "definition of domestic import prices                (unity)"
   pedef(i)        "definition of domestic export prices                (unity)"
   absorption(i)   "value of domestic sales                    (curr bill cfaf)"
   sales(i)        "value of domestic output                   (curr bill cfaf)"
   actp(i)         "definition of activity prices                       (unity)"
   pkdef(i)        "definition of capital goods price                   (unity)"

* output block
   activity(i)     "production function                      ('79-80 bill cfaf)"
   profitmax(i,lc) "first order condition for profit maximum     (1000 persons)"
   lmequil(lc)     "labor market equilibrium                     (1000 persons)"
   cet(i)          "cet function                             ('79-80 bill cfaf)"
   edemand(i)      "export demand                                       (unity)"
   esupply(i)      "export supply                                       (unity)"
   armington(i)    "composite good aggregation function      ('79-80 bill cfaf)"
   costmin(i)      "first order condition for cost minimization of composite good (unity)"
   xxdsn(i)        "domestic sales for nontraded sectors               ('79-80 bill cfaf)"
   xsn(i)          "composite good aggregation for nontraded sectors   ('79-80 bill cfaf)"

* demand block
   inteq(j)        "total intermediate uses                  ('79-80 bill cfaf)"
   cdeq(i)         "private consumption behavior               (curr bill cfaf)"
   dsteq(i)        "inventory investment                     ('79-80 bill cfaf)"
   gdp             "private gdp                                (curr bill cfaf)"
   gdeq            "government consumption behavior          ('79-80 bill cfaf)"
   greq            "government revenue                         (curr bill cfaf)"
   tariffdef       "tariff revenue                             (curr bill cfaf)"
   indtaxdef       "indirect taxes on domestic production      (curr bill cfaf)"
   dutydef         "export duties                              (curr bill cfaf)"

* savings-investment block
   hhsaveq         "household savings                          (curr bill cfaf)"
   gruse           "government savings                         (curr bill cfaf)"
   depreq          "depreciation expenditure                   (curr bill cfaf)"
   totsav          "total savings                              (curr bill cfaf)"
   prodinv(i)      "investment by sector of destination        (curr bill cfaf)"
   ieq(i)          "investment by sector of origin           ('79-80 bill cfaf)"

* balance of payments
   caeq            "current account balance                  (curr bill dollar)"

* market clearing
   equil(i)        "goods market equilibrium                 ('79-80 bill cfaf)"

* objective function
   obj             "objective function                       ('79-80 bill cfaf)";

* price block
pmdef(it)..       pm(it) =e= pwm(it)*er*(1 + tm(it));

pedef(it)..       pe(it)*(1 + te(it)) =e= pwe(it)*er;

absorption(i)..   p(i)*x(i)   =e= pd(i)*xxd(i) + (pm(i)*m(i))$it(i);

sales(i)..        px(i)*xd(i) =e= pd(i)*xxd(i) + (pe(i)*e(i))$it(i);

actp(i)..         px(i)*(1-itax(i)) =e= pva(i) + sum(j, io(j,i)*p(j));

pkdef(i)..        pk(i) =e= sum(j, p(j)*imat(j,i) );

* output and factors of production block
activity(i)..     xd(i) =e= ad(i) * prod(lc$wdist(i,lc), l(i,lc)**alphl(lc,i))
                                  * k(i)**(1 - sum(lc, alphl(lc,i)));

profitmax(i,lc)$wdist(i,lc).. wa(lc)*wdist(i,lc)*l(i,lc) =e= xd(i)*pva(i)*alphl(lc,i);

lmequil(lc)..     sum(i, l(i,lc)) =e= ls(lc);

cet(it)..         xd(it) =e= at(it)*(gamma(it)*e(it)**rhot(it) + ( 1 - gamma(it))
                                   * xxd(it)**rhot(it))**(1/rhot(it));

edemand(it)..     e(it)/e0(it)  =e= (pwe0(it)/pwe(it))**eta(it);

esupply(it)..     e(it)/xxd(it) =e=   (pe(it)/pd(it)*(1 - gamma(it))/gamma(it))
                                    **(1/(rhot(it) - 1));

armington(it)..   x(it) =e= ac(it)*(delta(it)*m(it)**(-rhoc(it)) + (1 - delta(it))
                                  * xxd(it)**(-rhoc(it)))**(-1/rhoc(it));

costmin(it)..     m(it)/xxd(it) =e=   (pd(it)/pm(it)*delta(it)/(1 - delta(it)))
                                    **(1/(1 + rhoc(it)));

xxdsn(in)..       xxd(in)     =e= xd(in);

xsn(in)..         x(in)       =e= xxd(in);

* demand block
inteq(j)..        int(j)      =e= sum(i, io(j,i)*xd(i));

dsteq(i)..        dst(i)      =e= dstr(i)*xd(i);

cdeq(i)..         p(i)*cd(i)  =e= cles(i)*(1 - mps)*y;

gdp..             y           =e= sum(i, pva(i)*xd(i)) - deprecia;

hhsaveq..         hhsav       =e= mps*y;

greq..            gr          =e= tariff + duty + indtax;

gruse..           gr          =e= sum(i, p(i)*gd(i)) + govsav;

gdeq(i)..         gd(i)       =e= gles(i)*gdtot;

tariffdef..       tariff      =e= sum(it, tm(it)*m(it)*pwm(it))*er;

indtaxdef..       indtax      =e= sum(i, itax(i)*px(i)*xd(i));

dutydef..         duty        =e= sum(it, te(it)*e(it)*pe(it));

depreq..          deprecia    =e= sum(i, depr(i)*pk(i)*k(i));

totsav..          savings     =e= hhsav + govsav + deprecia + fsav*er;

prodinv(i)..      pk(i)*dk(i) =e= kio(i)*savings - kio(i)*sum(j, dst(j)*p(j));

ieq(i)..          id(i)       =e= sum(j, imat(i,j)*dk(j));

caeq..            sum(it, pwm(it)*m(it)) =e= sum(it, pwe(it)*e(it)) + fsav;

* market clearing
equil(i)..        x(i)        =e= int(i) + cd(i) + gd(i) + id(i) + dst(i);

obj..             omega       =e= prod(i$cles(i), cd(i)**cles(i));

* model setup - initialization
x.l(i)   = x0(i);   xd.l(i)  = xd0(i);  xxd.l(i)  = xxd0(i);   cd.l(i)  = cles(i)*cdtot0;
m.l(i)   = m0(i);   e.l(i)   = e0(i);   id.l(i)   = id0(i);    dst.l(i) = dst0(i);
int.l(i) = int0(i); pd.l(i)  = pd0(i);  pm.l(i)   = pm0(i);    pe.l(i)  = pe0(i);
p.l(i)   = pd0(i);  px.l(i)  = pd0(i);  pk.l(i)   = pd0(i);    pva.l(i) = pva0(i);
pwe.l(i) = pwe0(i); wa.l(lc) = wa0(lc); l.l(i,lc) = xle(i,lc); gr.l     = gr0;

y.l               = sum(i, pva0(i)*xd0(i) - depr(i)*k0(i));
fsav.l            = fsav0 ;
tm.l(it)          = tm0(it);
gd.l("publiques") = 135.03;
tariff.l          = 76.548;
indtax.l          = 102.45;
savings.l         = 280.98;

* closure
k.fx(i)   = k0(i);
pwm.fx(i) = pwm0(i);
ls.fx(lc) = ls0(lc);
tm.fx(it) = tm0(it);
fsav.fx   = fsav0 ;
mps.fx    = .09305;
gdtot.fx  = gdtot0;
m.fx(in)  = 0;
l.fx("publiques","rural")       = 0;
l.fx("ag-subsist","urban-skil") = 0;
e.fx(in)  = 0;

Model camcns 'square base model' / all - caeq /;

solve camcns using cns;
