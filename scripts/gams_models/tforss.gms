$title Antalya Forestry Model - Steady State (TFORSS,SEQ=61)

$onText
This model finds the best management plan for new forests in a steady state
condition.


Bergendorff, H, Glenshaw, P, and Meeraus, A, The Planning of Investment
Programs in the Paper Industry. Tech. rep., The World Bank, 1980.

Keywords: linear programming, forestry, scenario analysis, investment planning,
          forest management planning
$offText

Set
   c     'commodities'      / pulplogs, sawlogs, residuals, pulp, sawnwood   /
   cf(c) 'final products'   / pulp,     sawnwood /
   cl(c) 'log types'        / pulplogs, sawlogs  /
   s     'species'          / nigra,    brutia   /
   k     'site classes'     / good, medium, poor /
   at    'tree age'         / a-10, a-20, a-30, a-40, a-50, a-60, a-70, a-80 /
   p     'processes'        / pulp-pl, pulp-sl, pulp-rs, sawing /
   m     'productive units' / pulp-mill, saw-mill /;

Parameter
   scd(k)  'site class distribution' / good .25, medium .50 , poor .25 /
   land(s) 'land available (1000ha)' / nigra 143.679, brutia 227.58    /;

Table ymf(at,k,s,cl) 'yield of managed forest (m3 per ha)'
                 nigra.pulplogs  nigra.sawlogs  brutia.pulplogs  brutia.sawlogs
   a-10.good                                               17.5
   a-10.medium
   a-10.poor
   a-20.good              120.0                            66.8
   a-20.medium             95.0                            51.1
   a-20.poor               80.0                            37.8
   a-30.good              132.6           37.4             91.3            25.7
   a-30.medium            120.2           14.8             81.4            10.0
   a-30.poor              115.0                            71.3
   a-40.good              121.0           99.0             91.3            74.7
   a-40.medium            115.5           59.5             86.5            44.5
   a-40.poor              119.0           21.0             90.1            15.9
   a-50.good              108.0          162.0             76.0           114.0
   a-50.medium            112.0          108.0             77.0            74.0
   a-50.poor              112.2           57.8             92.0            47.6
   a-60.good              104.0          221.0             76.0           116.0
   a-60.medium            106.0          159.0             76.0           113.0
   a-60.poor              110.0           90.0             95.2            77.8
   a-70.good              105.0          270.0             78.0           200.0
   a-70.medium             98.0          207.0             72.0           153.0
   a-70.poor               97.0          128.0             88.0           116.0
   a-80.good              102.0          323.0             76.0           240.0
   a-80.medium            105.0          235.0             80.0           177.0
   a-80.poor               92.0          163.0             84.0           148.0;

Table  a(c,p) 'input output matrix'
               pulp-pl  pulp-sl  pulp-rs  sawing
   pulplogs     -1.0
   sawlogs               -1.0               -1.0
   residuals                      -1.0       0.4
   pulp           .207     .207     .207
   sawnwood                                  0.6;

Table  b(m,p) 'capacity utilization'
               pulp-pl  pulp-sl  pulp-rs  sawing
   pulp-mill         1        1        1
   saw-mill                                    1;

Parameter
   pc(p)   'process cost     (us$ per m3 input)' /(pulp-pl,pulp-sl,pulp-rs) 5.96, sawing 6 /
   pd(cf)  'sales price          (us$ per unit)' / pulp      147.0, sawnwood 70.0 /
   nu(m)   'investment costs (us$ per m3 input)' / pulp-mill  37.8, saw-mill 61.5 /
   age(at) 'age of trees                (years)';

Scalar
   mup     'planting cost          (us$ per ha)' / 150.0 /
   muc     'cutting cost           (us$ per m3)' /   7.0 /
   life    'plant life                  (years)' /  30   /
   rho     'discount rate'                       /    na /;

age(at) = 10*ord(at);

$sTitle Model Definition
Equation
   lbal(cl)   'log balances'
   bal(c)     'material balances of wood processing'
   cap(m)     'wood processing capacities'
   landc(s,k) 'land availability constraint'
   ainvc      'investment cost'
   aproc      'process cost'
   asales     'sales revenue'
   acutc      'cutting cost'
   aplnt      'planting cost'
   benefit;

Variable
   v(s,k,at)  'management of new forest   (1000ha per year)'
   r(c)       'supply of logs to industry (1000m3 per year)'
   z(p)       'process level        (1000m3 input per year)'
   h(m)       'capacity             (1000m3 input per year)'
   x(c)       'final shipments        (1000 units per year)'
   phik       'investment cost           (1000us$ per year)'
   phir       'process cost              (1000us$ per year)'
   phix       'sales revenue             (1000us$ per year)'
   phil       'cutting cost              (1000us$ per year)'
   phip       'planting cost             (1000us$ per year)'
   phi        'total benefits             (discounted cost)';

Positive Variable v, z, x;

lbal(cl)..   r(cl) =e= sum((s,k,at), ymf(at,k,s,cl)*v(s,k,at));

bal(c)..     sum(p, a(c,p)*z(p)) + r(c)$cl(c) =g= x(c)$cf(c);

cap(m)..     sum(p, b(m,p)*z(p)) =e= h(m);

landc(s,k).. sum(at, v(s,k,at)*age(at)) =l= land(s)*scd(k);

ainvc..      phik =e= rho/(1 - (1 + rho)**(-life))*sum(m, nu(m)*h(m));

aproc..      phir =e= sum(p, pc(p)*z(p));

asales..     phix =e= sum(cf, pd(cf)*x(cf));

acutc..      phil =e= muc*sum(cl, r(cl));

aplnt..      phip =e= mup*sum((s,k,at), v(s,k,at)*(1 + rho)**age(at));

benefit..    phi  =e= phix - phik - phir - phil - phip;

Model forest / all /;

$sTitle Case Selection and Report Definitions
Set rhoset / rho-03, rho-05, rho-07, rho-10 /;

Parameter
   landcl(s,k)       'clean level of landc'
   repr(cl,rhoset)   'summary report on log supply      (1000m3 per year)'
   reprp(s,k,rhoset) 'summary report on rotation period           (years)'
   repsp(s,k,rhoset) 'summary report on shadow price of land (us$ per ha)'
   rhoval(rhoset)    / rho-03 .03, rho-05 .05, rho-07 .07, rho-10 .1 /;

loop(rhoset,
   rho = rhoval(rhoset);
   solve forest maximizing phi using lp;
   landcl(s,k)       = round(landc.l(s,k),3);
   repr(cl  ,rhoset) = r.l(cl);
   reprp(s,k,rhoset) = (landcl(s,k)/sum(at, v.l(s,k,at)))$landcl(s,k);
   repsp(s,k,rhoset) = landc.m(s,k)
);

display repr, reprp, repsp;
