$title Mexico Steel - Small Static (MEXSS,SEQ=15)

$onText
A simplified representation of the Mexican steel sector is used
to introduce a process industry production and distribution
scheduling problem.


Kendrick, D, Meeraus, A, and Alatorre, J, The Planning of Investment
Programs in the Steel Industry. The Johns Hopkins University Press,
Baltimore and London, 1984.

A scanned version of this out-of-print book is accessible at
http://www.gams.com/docs/pdf/steel_investment.pdf

Keywords: linear programming, production problem, distribution problem, scheduling,
          micro economics, steel industry
$offText

$sTitle Set Definitions
Set
   i     'steel plants'          / ahmsa      'altos hornos - monclova'
                                   fundidora  'monterrey'
                                   sicartsa   'lazaro cardenas'
                                   hylsa      'monterrey'
                                   hylsap     'puebla'                  /
   j     'markets'               / mexico-df, monterrey, guadalaja /
   c     'commodities'           / pellets    'iron ore pellets - tons'
                                   coke       'tons'
                                   nat-gas    'natural gas - 1000 n cubic meters'
                                   electric   'electricity - mwh'
                                   scrap      'tons'
                                   pig-iron   'molten pig iron - tons'
                                   sponge     'sponge iron - tons'
                                   steel      'tons'                   /
   cf(c) 'final products'        / steel /
   ci(c) 'intermediate products' / sponge, pig-iron /
   cr(c) 'raw materials'         / pellets, coke, nat-gas, electric, scrap /
   p     'processes'             / pig-iron   'pig iron production from pellets'
                                   sponge     'sponge iron production'
                                   steel-oh   'steel production: open hearth'
                                   steel-el   'steel production: electric furnace'
                                   steel-bof  'steel production: bof'   /
   m     'productive units'      / blast-furn 'blast furnaces'
                                   openhearth 'open hearth furnaces'
                                   bof        'basic oxygen converters'
                                   direct-red 'direct reduction units'
                                   elec-arc   'electric arc furnaces'   /;

$sTitle Model Parameters
Table a(c,p) 'input-output coefficients'
            pig-iron    sponge  steel-oh  steel-el  steel-bof
   pellets     -1.58     -1.38
   coke         -.63
   nat-gas                -.57
   electric                                   -.58
   scrap                            -.33                 -.12
   pig-iron     1.0                 -.77                 -.95
   sponge                 1.0                -1.09
   steel                            1.0       1.0         1.0;

Table b(m,p) 'capacity utilization'
              pig-iron  sponge  steel-oh  steel-el  steel-bof
   blast-furn      1.0
   openhearth                        1.0
   bof                                                    1.0
   direct-red              1.0
   elec-arc                                    1.0           ;

Table k(m,i) 'capacities of productive units (mill tpy)'
               ahmsa  fundidora  sicartsa  hylsa  hylsap
   blast-furn   3.25       1.40      1.10
   openhearth   1.50        .85
   bof          2.07       1.50      1.30
   direct-red                                .98    1.00
   elec-arc                                 1.13     .56;

Scalar
   dt  'total demand for final goods in 1979 (million tons)' /  5.209 /
   rse 'raw steel equivalence                     (percent)' / 40     /;

Parameter
   d(c,j) 'demand for steel in 1979 (mill tpy)'
   dd(j)  'distribution of demand' / mexico-df 55, monterrey 30, guadalaja 15 /;

d("steel",j) = dt * (1 + rse/100) * dd(j)/100;

Table rd(*,*) 'rail distances from plants to markets (km)'
             mexico-df  monterrey  guadalaja    export
   ahmsa          1204        218       1125       739
   fundidora      1017                  1030       521
   sicartsa        819       1305        704
   hylsa          1017                  1030       521
   hylsap          185       1085        760       315
   import          428        521        300          ;

Parameter
   muf(i,j) 'transport rate: final products(us$ per ton)'
   muv(j)   'transport rate: imports       (us$ per ton)'
   mue(i)   'transport rate: exports       (us$ per ton)';

muf(i,j) = ( 2.48 + .0084*rd(i,j))       $rd(i,j);
muv(j)   = ( 2.48 + .0084*rd("import",j))$rd("import",j);
mue(i)   = ( 2.48 + .0084*rd(i,"export"))$rd(i,"export");

Table prices(c,*) 'product prices (us$ per unit)'
              domestic  import  export
   pellets        18.7
   coke           52.17
   nat-gas        14.0
   electric       24.0
   scrap         105.0
   steel                   150.   140.;

Parameter
   pd(c) 'domestic prices(us$ per unit)'
   pv(c) 'import prices  (us$ per unit)'
   pe(c) 'export prices  (us$ per unit)'
   eb    'export bound       (mill tpy)';

pd(c) = prices(c,"domestic");
pv(c) = prices(c,"import");
pe(c) = prices(c,"export");
eb    = 1.0;

$sTitle Model Definition
Variable
   z(p,i)   'process level                             (mill tpy)'
   x(c,i,j) 'shipment of final products                (mill tpy)'
   u(c,i)   'purchase of domestic materials (mill units per year)'
   v(c,j)   'imports                                   (mill tpy)'
   e(c,i)   'exports                                   (mill tpy)'
   phi      'total cost                                (mill us$)'
   phipsi   'raw material cost                         (mill us$)'
   philam   'transport cost                            (mill us$)'
   phipi    'import cost                               (mill us$)'
   phieps   'export revenue                            (mill us$)';

Positive Variable z, x, u, v, e;

Equation
   mbf(c,i) 'material balances: final products (mill tpy)'
   mbi(c,i) 'material balances: intermediates  (mill tpy)'
   mbr(c,i) 'material balances: raw materials  (mill tpy)'
   cc(m,i)  'capacity constraint               (mill tpy)'
   mr(c,j)  'market requirements               (mill tpy)'
   me(c)    'maximum export                    (mill tpy)'
   obj      'accounting: total cost            (mill us$)'
   apsi     'accounting: raw material cost     (mill us$)'
   alam     'accounting: transport cost        (mill us$)'
   api      'accounting: import cost           (mill us$)'
   aeps     'accounting: export cost           (mill us$)';

mbf(cf,i)..  sum(p, a(cf,p)*z(p,i)) =g= sum(j, x(cf,i,j)) + e(cf,i);

mbi(ci,i)..  sum(p, a(ci,p)*z(p,i)) =g= 0;

mbr(cr,i)..  sum(p, a(cr,p)*z(p,i)) + u(cr,i) =g= 0;

cc(m,i)..    sum(p, b(m,p)*z(p,i)) =l= k(m,i);

mr(cf,j)..   sum(i, x(cf,i,j)) + v(cf,j) =g= d(cf,j);

me(cf)..     sum(i, e(cf,i)) =l= eb;

obj..        phi      =e= phipsi + philam + phipi - phieps;

apsi..       phipsi   =e= sum((cr,i), pd(cr)*u(cr,i));

alam..       philam   =e=   sum((cf,i,j), muf(i,j)*x(cf,i,j))
                          + sum((cf,j),   muv(j)*v(cf,j))
                          + sum((cf,i),   mue(i)*e(cf,i));

api..        phipi    =e= sum((cf,j), pv(cf)*v(cf,j));

aeps..       phieps   =e= sum((cf,i), pe(cf)*e(cf,i));

Model mexss 'small static problem' / all /;

solve mexss using lp minimizing phi;
