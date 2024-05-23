$title Nonlinear Simple Agricultural Sector Model (QDEMO7,SEQ=284)

$onText
This is a QCP version of the gamslib model DEMO7. The original NLP
formulation was concerned with good starting points. QCPs do not
need starting a point.

This is the last in a series of agricultural farm level and sector
models, this model simulates the market behavior of the sector
using a partial equilibrium framework. The technique is
the maximization of consumers and producers surplus.


Kutcher, G P, Meeraus, A, and O'Mara, G T, Agriculture Sector and
Policy Models. The World Bank, 1988.

Keywords: quadratic constraint programming, farming, agricultural economics,
          partial equilibrium, market behavior
$offText

Set
   c       'crop'                / wheat, clover, beans, onions, cotton, maize, tomato /
   cl      'livestock feed'      / clover, straw /
   t       'month'               / jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec  /
   r       'feeding recipes'     / rec-1,  rec-2 /
   s       'seasons'             / summer, winter /
   sc(s,c) 'season crop mapping' / summer.(cotton,maize,tomato), winter.(wheat,clover,beans,onions) /
   cn(c)   'crops sold in national market'
   ce(c)   'export commodities'
   cm(c)   'import commodities';

Table a(t,c) 'months of land occupation by crop    (hectares)'
         wheat  clover  beans  onions  cotton  maize  tomato
   jan     1.     1.     1.      1.
   feb     1.     1.     1.      1.
   mar     1.      .5    1.      1.       .5
   apr     1.            1.      1.      1.
   may     1.                     .25    1.      .25
   jun                                   1.     1.
   jul                                   1.     1.       .75
   aug                                   1.     1.      1.
   sep                                   1.     1.      1.
   oct                                   1.      .5     1.
   nov      .5     .25    .25     .5      .75            .75
   dec     1.     1.     1.      1.                         ;

Table lc(t,c) 'crop labor requirements (man-days per hectare)'
         wheat  clover beans  onions  cotton  maize  tomato
   jan   1.72      4.5   .75    5.16
   feb   .5        1.    .75    5.
   mar   1.        8.    .75    5.       5.
   apr   1.            16.    19.58      5.
   may   17.16                 2.42      9.    4.3
   jun   2.34                            2.    5.04
   jul                                   1.5   7.16     17.
   aug                                   2.    7.97     15.
   sep                                   1.    4.41     12.
   oct                                  26.    1.12      7.
   nov   2.43      2.5  7.5    11.16    12.              6.
   dec   1.35      7.5   .75    4.68                       ;

Table lio(cl,r) 'livestock input output matrix'
            rec-1  rec-2
   clover     1.3    2.0
   straw      1.6     .8;

Table demdat(c,*) 'demand data'
            ref-p    ref-q   elas  exp-p  imp-p
*              ($)  (1000t)           ($)    ($)
   wheat      100     2700    -.8           140
   beans      200      900    -.4           270
   onions     125      700   -1.      40    inf
   cotton     350     2100   -1.     300    inf
   maize       70     3800    -.5            85
   tomato     120      500   -1.2     60    inf;

Scalar
   fnum   'number  of  farms in sector'                   / 1000    /
   land   'farmsize                           (hectares)' /    4    /
   famlab 'family labor available       (days per month)' /   25    /
   dpm    'work days per month'                           /   25    /
   rwage  'reservation wage rate       (dollars per day)' /    3    /
   twage  'temporary labor wage        (dollars per day)' /    4    /
   llab   'livestock labor requirements (days per month)' /    2    /
   trent  'tractor rental cost      (dollar per hectare)' /   40    /
   hpa    'land plowed by animals  (hectares per animal)' /    2    /
   straw  'straw yield from wheat'                        /    1.75 /;

Parameter
   yield(c)   'crop yield         (tons per hectare)'
              / wheat  1.5, clover 6, beans  1, onions 3
                cotton 1.5, maize  2, tomato 3           /
   miscost(c) 'misc cash costs (dollars per hectare)'
              / wheat  10, beans 5, onions 50
                cotton 80, maize 5, tomato 50 /
   price(c)   'reference (observed) price  (dollars)'
   pe(c)      'commodity export prices     (dollars)'
   pm(c)      'commodity import prices     (dollars)'
   alpha(c)   'demand curve intercept'
   beta(c)    'demand curve gradient';

cn(c) = yes$demdat(c,"ref-p");
ce(c) = yes$demdat(c,"exp-p");
cm(c) = yes$(demdat(c,"imp-p") < inf );
cm("clover") = no;
price(c) = demdat(c,"ref-p");
pe(ce)   = demdat(ce,"exp-p");
pm(cm)   = demdat(cm,"imp-p");

beta(cn)$demdat(cn,"ref-q") = demdat(cn,"ref-p")/demdat(cn,"ref-q")/demdat(cn,"elas");
alpha(cn) = demdat(cn,"ref-p") - beta(cn)*demdat(cn,"ref-q");
demdat(cn,"dem-a") = alpha(cn);
demdat(cn,"dem-b") = beta(cn);

display cn, cm, ce, price, pe, beta, alpha, demdat;

Variable
   xcrop(c)   'cropping activity                 (hectares)'
   yfarm      'farm income                        (dollars)'
   revenue    'value of production                (dollars)'
   mcost      'misc cash cost                     (dollars)'
   pcost      'tractor plowing cost'
   labcost    'labor cost                         (dollars)'
   rescost    'family labor reservation wage cost (dollars)'
   tcost      'total farm cost including rescost'
   flab(t)    'family labor use                      (days)'
   tlab(t)    'temporary labor                       (days)'
   xlive(r)   'livestock activity                   (units)'
   natprod(c) 'net production                        (tons)'
   thire(s)   'tractor rental             (hectares plowes)'
   natcon(c)  'domestic consumption             (1000 tons)'
   exports(c) 'national exports                 (1000 tons)'
   imports(c) 'national imports                 (1000 tons)'
   cps        'consumers and producers surplus'
   valpro     'value of net production at ref prices'
   employ     'employment generated             (man-years)'
   tradebal   'net exports                         (1000 $)';

Positive Variable xcrop, xlive, thire, flab, tlab, natcon, natprod, exports, imports;

Equation
   landbal(t)  'land balance             (hectares)'
   laborbal(t) 'labor balance                (days)'
   flabor(t)   'family labor balance         (days)'
   plow(s)     'land plowed   (hectares per season)'
   arev        'revenue accounting        (dollars)'
   ares        'reservation labor cost    (dollars)'
   acost       'total cost accounting     (dollars)'
   amisc       'misc cost accounting'
   aplow
   alab        'labor cost accounting     (dollars)'
   lclover     'clover balance'
   lstraw      'straw balance'
   income      'income definition         (dollars)'
   proc(c)     'net production definition    (tons)'
   dem(c)      'national demand balance (1000 tons)'
   objn        'objective function';

landbal(t)..  sum(c, xcrop(c)*a(t,c))  =l= land*fnum;

laborbal(t).. sum(c, xcrop(c)*lc(t,c)) + sum(r, xlive(r))*llab  =l= flab(t) + tlab(t);

amisc..       mcost   =e= sum(c, xcrop(c)*miscost(c));

alab..        labcost =e= sum(t, tlab(t)*twage);

ares..        rescost =e= sum(t, flab(t)*rwage);

aplow..       pcost   =e= sum(s, thire(s)*trent);

acost..       tcost   =e= mcost + labcost + rescost + pcost;

lclover..     xcrop("clover")*yield("clover") =g= sum(r, xlive(r)*lio("clover",r));

lstraw..      xcrop("wheat")*straw =g= sum(r, xlive(r)*lio("straw",r));

plow(s)..     sum(c$sc(s,c), xcrop(c)) =l= sum(r, xlive(r))*hpa + thire(s);

proc(c)..     natprod(c) =e= xcrop(c)*yield(c);

dem(cn)..     natcon(cn) =e= natprod(cn) + imports(cn)$cm(cn) - exports(cn)$ce(cn);

objn.. cps =e= sum(cn, alpha(cn)*natcon(cn) + .5*beta(cn)*sqr(natcon(cn)))
            +  sum(ce, exports(ce)*pe(ce))
            -  sum(cm, imports(cm)*pm(cm))
            -  tcost;

flab.up(t) = famlab*fnum;

Model demo7n 'qcp version' / landbal, laborbal, plow,   ares, alab
                             acost,   dem,      proc,   amisc
                             aplow,   lclover,  lstraw, objn       /;

option limCol = 0, limRow = 0;

solve demo7n maximizing cps using qcp;
