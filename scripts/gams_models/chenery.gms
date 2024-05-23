$title Substitution and Structural Change (CHENERY,SEQ=33)

$onText
This model follows conventional input-output formulations for production
with nonlinear demand functions, import and export functions and production
functions for direct factor use.


Chenery, H B, and Raduchel, W J, Substitution and Structural Change.
In Chenery, H B, Ed, Structural Change and Development Policy. Oxford
University Press, New York and Oxford, 1979.

Keywords: nonlinear programming, econometrics, economic development
$offText

Set
   i    'sectors'               / light-ind, food+agr, heavy-ind, services /
   t(i) 'tradables'             / light-ind, food+agr, heavy-ind /
   lmh  'possible elasticities' / low, medium, high /
   sde  'other parameters'      / subst, distr, effic /;

Alias (i,j);

Table aio(i,i) 'input coefficients'
               light-ind  food+agr  heavy-ind  services
   food+agr           .1
   heavy-ind          .2        .1
   services           .2        .3         .1          ;

* In the next 3 tables data is specified for many different possible
* hypotheses about the economy. One particular subset is used for any
* individual model. See assignment statements below.

Table pdat(lmh,*,sde,i) 'production data'
                    light-ind  food+agr  heavy-ind  services
   low.a.subst
   low.a.distr           .915      .944      2.60     .80
   low.a.effic          3.83      3.24       4.0     1.8
   low.b.subst
   low.b.distr           .276     1.034      2.60     .77
   low.b.effic          2.551     3.39       4.0     1.77
   medium.a.subst        .11       .29        .2      .05
   medium.a.distr        .326      .443       .991    .00798
   medium.a.effic       3.97      3.33       1.67    1.84
   medium.b.subst        .22       .58        .4      .1
   medium.b.distr        .41       .47        .92     .08
   medium.b.effic       3.99      3.33       1.8     1.89
   high.a.subst          .45      1.15        .4      .2
   high.a.distr          .456      .483       .917    .23
   high.a.effic         4.0       3.33       1.8     1.92
   high.b.subst          .93      1.15        .8      .4
   high.b.distr          .484      .483       .769    .344
   high.b.effic         4.0       3.33       1.96    1.96   ;

Table ddat(lmh,*,i) 'demand parameters'
                            light-ind  food+agr  heavy-ind  services
   (low,medium,high).ynot        100      230         220       450
   medium.p-elas                -.674     -.246      -.587     -.352
   high  .p-elas               -1        -1         -1        -1    ;

Table tdat(lmh,*,t) 'trade parameters'
                     light-ind  food+agr  heavy-ind
   medium.alp            .005      .001      .01
   high  .alp            .0025     .0005     .00178
   (medium,high).gam    1.0       1.1       1.0
   (medium,high).xsi     .005      .0157     .00178;

Parameter
   mew(t)  'intercept on import cost function'
   xsi(t)  'slope of import cost function'
   gam(t)  'intercept on export revenue function'
   alp(t)  'slope of export revenue function'
   ynot(i) 'demands in base year'
   sig(i)  'elasticity of substitution capital to labor'
   thet(i) 'price elasticity of demand'
   rho(i)  'substitution parameter in production functions'
   del(i)  'distribution parameter in production functions'
   efy(i)  'efficiency parameter in production functions';

Scalar
   lbar    'total supply of labor' / 750 /
   plab    'price of labor'        /   1 /
   kbar    'capital stock'         / 500 /
   dbar    'trade deficit'         /   0 /;

Variable
   x(i)    'quantity of output'
   v(i)    'value added per unit output at current prices'
   y(i)    'final consumption'
   p(i)    'prices'
   l(i)    'labor use per unit of output'
   k(i)    'capital use per unit of output'
   e(i)    'quantity of exports'
   m(i)    'quantity of imports'
   g(t)    'foreign exchange cost of imports'
   h(t)    'foreign exchange value of exports'
   pk      'nominal market price of capital'
   pi      'factor price ratio'
   pd      'price deflator'
   td      'total demand'
   vv(i)   'intermediate result';

Positive Variable x, y, e, m, g, h, p, k, l, v;

Equation
   dty     'total demand: definition'
   mb(i)   'material balance'
   tb      'trade balance'
   dg(t)   'definition of imports'
   dh(t)   'definition of exports'
   dem(i)  'demand equations'
   lc      'labor constraint'
   kc      'capital constraint'
   sup(i)  'supply equations'
   fpr     'factor price ratio definition'
   dvv(i)  'definition of vv'
   dl(i)   'definition of labor coefficient'
   dk(i)   'definition of capital coefficient'
   dv(i)   'value added';

* The naming convention followed below is -
* endogenous variables have 1 or 2 character names
* exogenous parameters have 3 or more characters

dty..    td   =e= sum(i, y(i));

mb(i)..  x(i) =g= y(i) + sum(j, aio(i,j)*x(j)) + (e(i) - m(i))$t(i);

tb..     sum(t, g(t)*m(t) - h(t)*e(t)) =l= dbar;

dg(t)..  g(t) =e= mew(t) + xsi(t)*m(t);

dh(t)..  h(t) =e= gam(t) - alp(t)*e(t);

dem(i).. y(i) =e= ynot(i)*(pd*p(i))**thet(i);

lc..     sum(i, l(i)*x(i)) =l= lbar;

kc..     sum(i, k(i)*x(i)) =e= kbar;

sup(i).. p(i) =e= sum(j, aio(j,i)*p(j)) + v(i);

fpr..    pi   =e= pk/plab;

dvv(i)$(sig(i) <> 0).. vv(i) =e= (pi*(1 - del(i))/del(i))**(-rho(i)/(1 + rho(i)));

dl(i)..  l(i)*efy(i) =e= ((del(i)/vv(i) + (1 - del(i)))**(1/rho(i)))$(sig(i) <> 0) + 1$(sig(i) = 0);

dk(i)..  k(i)*efy(i) =e= ((del(i) + (1 - del(i))*vv(i))**(1/rho(i)))$(sig(i) <> 0) + del(i)$(sig(i) = 0);

dv(i)..  v(i) =e= pk*k(i) + plab*l(i);

Model chenrad 'chenery raduchel model' / all /;

* bounds for variables
y.up(i)  = 2000;
x.up(i)  = 2000;
e.up(t)  =  400;
m.up(t)  =  400;
g.up(t)  =    4;
h.up(t)  =    4;
p.up(i)  =  100;
p.lo(i)  =    0.1;
l.up(i)  =    1;
k.up(i)  =    1;
pk.lo    =    0.25;
pk.up    =    4;
pi.lo    =    0.25;
pi.up    =    4;
v.up(i)  =  100;
vv.lo(i) =    0.001;

* select coefficient values for this run
mew(t)  = 1.0;
xsi(t)  = tdat("medium","xsi",t);
gam(t)  = tdat("medium","gam",t);
alp(t)  = tdat("medium","alp",t);
ynot(i) = ddat("medium","ynot",i);
thet(i) = ddat("medium","p-elas",i);
sig(i)  = pdat("medium","a","subst",i);
del(i)  = pdat("medium","a","distr",i);
efy(i)  = pdat("medium","a","effic",i);
rho(i)$(sig(i) <> 0) = 1./sig(i) - 1.;

* initial values for variables
y.l(i) = 250;
x.l(i) = 200;
e.l(t) =   0;
m.l(t) =   0;
g.l(t) = mew(t) + xsi(t)*m.l(t);
h.l(t) = gam(t) - alp(t)*e.l(t);
pd.l   =   0.3;
p.l(i) =   3;
pk.l   =   3.5;
pi.l   = pk.l/plab;

vv.l(i)$sig(i) = (pi.l*(1 - del(i))/del(i))**(-rho(i)/(1 + rho(i)));

l.l(i) = (((del(i)/vv.l(i) + (1 - del(i)))**(1/rho(i)))$(sig(i) <> 0)
         + 1$(sig(i) = 0))/efy(i);

k.l(i) = (((del(i) + (1 - del(i))*vv.l(i))**(1/rho(i)))$(sig(i) <> 0)
         + del(i)$(sig(i) = 0))/efy(i);

v.l(i) = pk.l*k.l(i) + plab*l.l(i);

* add bounds to avoid function evaluation errors
pd.lo   = 0.01;
p.lo(i) = 0.1;

solve chenrad using nlp maximizing td;

Scalar
   cva 'total value added at current prices'
   rva 'real value added'
   fve 'foreign exchange value of exports'
   emp 'total employment'
   cli 'cost of living index';

cva = sum(i, v.l(i)*x.l(i));
fve = sum(t, e.l(t)*h.l(t));
emp = sum(i, l.l(i)*x.l(i));
cli = sum(i, p.l(i)*ynot(i))/sum(i, ynot(i));
rva = cva/cli;

display cli, cva, rva, fve, emp;
