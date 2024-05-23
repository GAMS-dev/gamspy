$title General Equilibrium Model for Korea (KORCNS,SEQ=212)

$onText
This mini equilibrium model of Korea for the year 1963 is used to
illustrate the basic use of CGE models. This version follows closely
Chapter 11 of the reference.

The original model (KORCGE,SEQ=100) is formulated as an optimization
model, but it is really a square system of nonlinear equations.
In this version, we formulate the model directly as a square system
using the model type CNS = Constrained Nonlinear System.

An MCP version exist under the name (KORMCP,SEQ=130).


Lewis, J, and Robinson, S, Chapter 11. In Chenery, H B, Robinson, S,
and Syrquin, S, Eds, Industrialization and Growth: A Comparative
Study. Oxford University Press, London, 1986.

Keywords: constrained nonlinear system, general equilibrium model, economic growth,
          industrialization, economic policy, Korean economy
$offText

Set
   i     'sectors'          / agricult 'agriculture'
                              industry 'industrial sectors'
                              services 'infra. & services'    /
   hh    'household type'   / lab-hh   'labor households'
                              cap-hh   'capitalist household' /
   lc    'labor categories' / labor1   'agricultural labor'
                              labor2   'industrial labor'
                              labor3   'service labor'        /
   it(i) 'traded sectors'
   in(i) 'nontraded sectors';

Alias (i,j);

Parameter
   delta(i) 'Armington function share parameter'
   ac(i)    'Armington function shift parameter'
   rhoc(i)  'Armington function exponent'
   rhot(i)  'cet function exponent'
   at(i)    'cet function shift parameter'
   gamma(i) 'cet function share parameter'
   ad(i)    'production function shift parameter'
   gles(i)  'government consumption shares'
   depr(i)  'depreciation rates'
   dstr(i)  'ratio of inventory investment to gross output'
   kio(i)   'shares of investment by sector of destination'
   te(i)    'export duty rates'
   itax(i)  'indirect tax rates'
   htax(hh) 'income tax rate by household type'
   pwm(i)   'world market price of imports    (in dollars)'
   pwe(i)   'world market price of exports    (in dollars)'
   tm(i)    'tariff rates on imports'
   pwts(i)  'cpi weights';

htax("lab-hh  ") = 0.08910;
htax("cap-hh  ") = 0.08910;

Table alphl(i,lc) 'labor share parameter in production function'
               labor1   labor2   labor3
   agricult   0.38258  0.06740  0.00000
   industry   0.00000  0.53476  0.00000
   services   0.00000  0.16234  0.42326;

Table io(i,j) 'input-output coefficients'
              agricult  industry  services
   agricult    0.12591   0.19834   0.01407
   industry    0.10353   0.35524   0.18954
   services    0.02358   0.11608   0.08390;

Table imat(i,j) 'capital composition matrix'
              agricult  industry  services
   agricult    0.00000   0.00000   0.00000
   industry    0.93076   0.93774   0.93080
   services    0.06924   0.06226   0.06920;

Table wdist(i,lc) 'wage proportionality factors'
               labor1   labor2   labor3
   agricult   1.00000  0.52780  0.00000
   industry   0.00000  1.21879  0.00000
   services   0.00000  1.11541  1.00000;

Table cles(i,hh) 'private consumption shares'
               lab-hh   cap-hh
   agricult   0.47000  0.47000
   industry   0.31999  0.31999
   services   0.21001  0.21001;

Table zz(*,i) 'miscellaneous parameters'
              agricult  industry  services
   depr        0.00000   0.00000   0.00000
   itax        0.01000   0.03920   0.05000
   gles        0.02000   0.07000   0.91000
   kio         0.13000   0.29000   0.58000
   dstr        0.00000   0.00000   0.00000
   te          0.00000   0.00000   0.00000
   tm          0.10000   0.22751   0.08084
   ad          0.61447   1.60111   0.52019
   pwts        0.33263   0.43486   0.23251
   pwm         0.90909   0.81466   0.92521
   pwe         1.00000   1.00000   1.00000
   sigc        2.00000   0.66000   0.40000
   delta       0.24820   0.05111   0.00001
   ac          1.59539   1.34652   1.01839
   sigt        2.00000   2.00000   2.00000
   gamma       0.86628   0.84602   0.82436
   at          3.85424   3.51886   3.23592;

depr(i)  = zz("depr",i);
itax(i)  = zz("itax",i);
gles(i)  = zz("gles",i);
kio(i)   = zz("kio",i);
dstr(i)  = zz("dstr",i);
te(i)    = zz("te",i);
tm(i)    = zz("tm",i);
ad(i)    = zz("ad",i);
pwts(i)  = zz("pwts",i);
pwm(i)   = zz("pwm",i);
pwe(i)   = zz("pwe",i);
rhoc(i)  = (1/zz("sigc",i)) - 1 ;
delta(i) = zz("delta",i);
ac(i)    = zz("ac",i);
rhot(i)  = (1/zz("sigt",i)) + 1;
gamma(i) = zz("gamma",i);
at(i)    = zz("at",i);

$sTitle Model Definition
Variable
* prices block
   er        'real exchange rate                          (won per dollar)'
   pd(i)     'domestic prices'
   pm(i)     'domestic price of imports'
   pe(i)     'domestic price of exports'
   pk(i)     'rate of capital rent by sector'
   px(i)     'average output price by sector'
   p(i)      'price of composite goods'
   pva(i)    'value added price by sector'
   pr        'import premium'
   pindex    'general price level'

* production block
   x(i)      "composite goods supply                        ('68 bill won)"
   xd(i)     "domestic output by sector                     ('68 bill won)"
   xxd(i)    "domestic sales                                ('68 bill won)"
   e(i)      "exports by sector                             ('68 bill won)"
   m(i)      "imports                                       ('68 bill won)"

* factors block
   k(i)      "capital stock by sector                       ('68 bill won)"
   wa(lc)    "average wage rate by labor category     (mill won pr person)"
   ls(lc)    "labor supply by labor category                (1000 persons)"
   l(i,lc)   "employment by sector and labor category       (1000 persons)"

* demand block
   int(i)    "intermediates uses                            ('68 bill won)"
   cd(i)     "final demand for private consumption          ('68 bill won)"
   gd(i)     "final demand for government consumption       ('68 bill won)"
   id(i)     "final demand for productive investment        ('68 bill won)"
   dst(i)    "inventory investment by sector                ('68 bill won)"
   y         "private gdp                                       (bill won)"
   gr        "government revenue                                (bill won)"
   tariff    "tariff revenue                                    (bill won)"
   indtax    "indirect tax revenue                              (bill won)"
   netsub    "export duty revenue                               (bill won)"
   gdtot     "total volume of government consumption        ('68 bill won)"
   hhsav     "total household savings                           (bill won)"
   govsav    "government savings                                (bill won)"
   deprecia  "total depreciation expenditure                    (bill won)"
   invest    "total investment                                  (bill won)"
   savings   "total savings                                     (bill won)"
   mps(hh)   "marginal propensity to save by household type"
   fsav      "foreign savings                               (bill dollars)"
   dk(i)     "volume of investment by sector of destination ('68 bill won)"
   ypr       "total premium income accruing to capitalists      (bill won)"
   remit     "net remittances from abroad                   (bill dollars)"
   fbor      "net flow of foreign borrowing                 (bill dollars)"
   yh(hh)    "total income by household type                    (bill won)"
   tothhtax  "household tax revenue                             (bill won)"

* welfare indicator for objective function
   omega     "objective function variable                   ('68 bill won)";

er.l       =    1.0000;
pr.l       =    0.0000;
pindex.l   =    1.0000;
gr.l       =  194.0449;
tariff.l   =   28.6572;
indtax.l   =   65.2754;
netsub.l   =    0.0000;
gdtot.l    =  141.1519;
hhsav.l    =   61.4089;
govsav.l   =   52.8930;
deprecia.l =    0.0000;
savings.l  =  159.1419;
invest.l   =  159.1419;
fsav.l     =   39.1744;
fbor.l     =   58.7590;
remit.l    =    0.0000;
tothhtax.l =  100.1122;
y.l        = 1123.5941;

Table labres1(i,lc) 'summary matrix with sectoral employment results'
                labor1   labor2   labor3
   agricult   2515.900  442.643    0.000
   industry      0.000  767.776    0.000
   services      0.000  355.568  948.100;

Table labres2(*,lc) 'summary matrix with aggregate employment results'
          labor1    labor2    labor3
   wa      0.074     0.140     0.152
   ls   2515.900  1565.987  948.100;

Table hhres(*,hh) 'summary matrix with household results'
          lab-hh    cap-hh
   yh   548.7478  574.8463
   mps    0.0600    0.0600;

l.l(i,lc) = labres1(i,lc);
ls.l(lc)  = labres2("ls",lc);
wa.l(lc)  = labres2("wa",lc);
mps.l(hh) = hhres("mps",hh);
yh.l(hh)  = hhres("yh",hh);

Table sectres(*,i) 'summary matrix with sectoral results'
         agricult  industry  services
   pd      1.0000    1.0000    1.0000
   pk      1.0000    1.0000    1.0000
   pva     0.7370    0.2911    0.6625
   x     711.6443  930.3509  497.4428
   xd    657.3677  840.0500  515.4296
   xxd   641.7037  812.2222  492.0307
   e      15.6639   27.8278   23.3988
   m      69.9406  118.1287    5.4120
   k     657.5754  338.7076 1548.5192
   int   256.6450  464.1656  156.2598
   cd    452.1765  307.8561  202.0416
   gd      2.8230    9.8806  128.4482
   id      0.0000  148.4488   10.6931
   dst     0.0000    0.0000    0.0000
   dk     20.6884   46.1511   92.3023
   pm      1.0000    1.0000    1.0000
   pe      1.0000    1.0000    1.0000
   px      1.0000    1.0000    1.0000
   p       1.0000    1.0000    1.0000;

pd.l(i)  = sectres("pd",i);
pm.l(i)  = sectres("pm",i);
pe.l(i)  = sectres("pe",i);
pk.l(i)  = sectres("pk",i);
px.l(i)  = sectres("px",i);
p.l(i)   = sectres("p",i);
pva.l(i) = sectres("pva",i);
x.l(i)   = sectres("x",i);
xd.l(i)  = sectres("xd",i);
xxd.l(i) = sectres("xxd",i);
e.l(i)   = sectres("e",i);
m.l(i)   = sectres("m",i);
k.l(i)   = sectres("k",i);
int.l(i) = sectres("int",i);
cd.l(i)  = sectres("cd",i);
gd.l(i)  = sectres("gd",i);
id.l(i)  = sectres("id",i);
dst.l(i) = sectres("dst",i);
dk.l(i)  = sectres("dk",i);
it(i)    = yes$(e.l(i) or m.l(i));
in(i)    = not it(i);
k.fx(i)  = k.l(i);
m.fx(in) = 0;
e.fx(in) = 0;
l.fx(i,lc)$( l.l(i,lc) = 0) = 0;

p.lo(i)   = .01; pd.lo(i)  = .01; pm.lo(it)  = .01;
pk.lo(i)  = .01; px.lo(i)  = .01; x.lo(i)    = .01;
xd.lo(i)  = .01; m.lo(it)  = .01; xxd.lo(it) = .01;
wa.lo(lc) = .01; int.lo(i) = .01; y.lo       = .01;
e.lo(it)  = .01; l.lo(i,lc)$(l.l(i,lc) <> 0) = .01;

$sTitle Equation Definitions
Equation
* price block
   pmdef(i)        'definition of domestic import prices'
   pedef(i)        'definition of domestic export prices'
   absorption(i)   'value of domestic sales'
   sales(i)        'value of domestic output'
   actp(i)         'definition of activity prices'
   pkdef(i)        'definition of capital goods price'
   pindexdef       'definition of general price level'

* output block
   activity(i)     'production function'
   profitmax(i,lc) 'first order condition for profit maximum'
   lmequil(lc)     'labor market equilibrium'
   cet(i)          'cet function'
   esupply(i)      'export supply'
   armington(i)    'composite good aggregation function'
   costmin(i)      'f.o.c. for cost minimization of composite good'
   xxdsn(i)        'domestic sales for nontraded sectors'
   xsn(i)          'composite good agg. for nontraded sectors'

* demand block
   inteq(i)        'total intermediate uses'
   cdeq(i)         'private consumption behavior'
   dsteq(i)        'inventory investment'
   gdp             'private gdp'
   labory          'total income accruing to labor'
   capitaly        'total income accruing to capital'
   hhtaxdef        'total household taxes collected by govt.'
   gdeq            'government consumption shares'
   greq            'government revenue'
   tariffdef       'tariff revenue'
   premium         'total import premium income'
   indtaxdef       'indirect taxes on domestic production'
   netsubdef       'export duties'

* savings-investment block
   hhsaveq         'household savings'
   gruse           'government savings'
   depreq          'depreciation expenditure'
   totsav          'total savings'
   prodinv(i)      'investment by sector of destination'
   ieq(i)          'investment by sector of origin'

* balance of payments
   caeq            'current account balance (bill dollars)'

* market clearing
   equil(i)        'goods market equilibrium'

* objective function
   obj             'objective function';

* price block
pmdef(it)..       pm(it)            =e= pwm(it)*er*(1 + tm(it) + pr);

pedef(it)..       pe(it)            =e= pwe(it)*(1 + te(it))*er;

absorption(i)..   p(i)*x(i)         =e= pd(i)*xxd(i) + (pm(i)*m(i))$it(i);

sales(i)..        px(i)*xd(i)       =e= pd(i)*xxd(i) + (pe(i)*e(i))$it(i);

actp(i)..         px(i)*(1-itax(i)) =e= pva(i) + sum(j, io(j,i)*p(j));

pkdef(i)..        pk(i)             =e= sum(j, p(j)*imat(j,i));

pindexdef..       pindex            =e= sum(i, pwts(i)*p(i));

* output and factors of production block
activity(i)..     xd(i) =e= ad(i)*prod(lc$wdist(i,lc), l(i,lc)**alphl(i,lc))
                         *  k(i)**(1 - sum(lc, alphl(i,lc)));

profitmax(i,lc)$wdist(i,lc)..
   wa(lc)*wdist(i,lc)*l(i,lc)     =e= xd(i)*pva(i)*alphl(i,lc);

lmequil(lc)..     sum(i, l(i,lc)) =e= ls(lc) ;

cet(it)..         xd(it) =e=  at(it)*(gamma(it)*e(it)**rhot(it)
                          +  (1 - gamma(it))*xxd(it)**rhot(it) )**(1/rhot(it));

esupply(it)..     e(it)/xxd(it) =e= (pe(it)/pd(it)*(1 - gamma(it))/gamma(it))
                                 ** (1/(rhot(it) - 1) );

armington(it)..   x(it) =e=  ac(it)*(delta(it)*m(it)**(-rhoc(it))
                         +  (1 - delta(it))*xxd(it)**(-rhoc(it)))**(-1/rhoc(it));

costmin(it)..     m(it)/xxd(it) =e= (pd(it)/pm(it)*delta(it)/(1 - delta(it)))
                                 ** (1/(1 + rhoc(it)));

xxdsn(in)..       xxd(in)       =e= xd(in);

xsn(in)..         x(in)         =e= xxd(in);

* demand block
inteq(i)..        int(i)        =e= sum(j, io(i,j)*xd(j));

dsteq(i)..        dst(i)        =e= dstr(i)*xd(i);

cdeq(i)..         p(i)*cd(i)    =e= sum(hh, cles(i,hh)*(1 - mps(hh))*yh(hh)*(1 - htax(hh)));

gdp..             y             =e= sum(hh, yh(hh));

labory..          yh("lab-hh")  =e= sum(lc, wa(lc)*ls(lc)) + remit*er;

capitaly..        yh("cap-hh")  =e= sum(i, pva(i)*xd(i)) - deprecia
                                 -  sum(lc, wa(lc)*ls(lc)) + fbor*er + ypr;

hhsaveq..         hhsav         =e= sum(hh, mps(hh)*yh(hh)*(1 - htax(hh)));

greq..            gr            =e= tariff - netsub + indtax +tothhtax;

gruse..           gr            =e= sum(i, p(i)*gd(i)) + govsav;

gdeq(i)..         gd(i)         =e= gles(i)*gdtot;

tariffdef..       tariff        =e= sum(it, tm(it)*m(it)*pwm(it))*er;

indtaxdef..       indtax        =e= sum(i,  itax(i)*px(i)*xd(i));

netsubdef..       netsub        =e= sum(it, te(it)*e(it)*pwe(it))*er;

premium..         ypr           =e= sum(it, pwm(it)*m(it))*er*pr;

hhtaxdef..        tothhtax      =e= sum(hh, htax(hh)*yh(hh));

depreq..          deprecia      =e= sum(i, depr(i)*pk(i)*k(i));

totsav..          savings       =e= hhsav + govsav + deprecia + fsav*er;

prodinv(i)..      pk(i)*dk(i)   =e= kio(i)*invest - kio(i)*sum(j, dst(j)*p(j));

ieq(i)..          id(i)         =e= sum(j, imat(i,j)*dk(j));

* balance of payments
caeq..            sum(it, pwm(it)*m(it)) =e= sum(it, pwe(it)*e(it))
                                          +  fsav + remit + fbor;
* market clearing
equil(i)..        x(i)  =e= int(i) + cd(i) + gd(i) + id(i) + dst(i);

* objective function
obj..             omega =e= prod(i$cles(i,"lab-hh"), cd(i)**cles(i,"lab-hh"));

er.fx      = er.l;
fsav.fx    = fsav.l;
remit.fx   = remit.l;
fbor.fx    = fbor.l;
pindex.fx  = pindex.l;
mps.fx(hh) = mps.l(hh);
gdtot.fx   = gdtot.l;
ls.fx(lc)  = ls.l(lc);

Model model1 'square base model' / all /;

solve model1 using cns;
