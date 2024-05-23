$title Food Manufacturing Problem - Blending of Oils (FOOD,SEQ=352)

$onText
The problem is to plan the blending of five kinds of oil, organized in two
categories (two kinds of vegetable oils and three kinds of non vegetable oils)
into batches of blended products over six months.

Some of the oil is already available in storage. There is an initial stock of
oil of 500 tons of each raw type when planning begins. An equal stock should
exist in storage at the end of the plan. Up to 1000 tons of each type of raw oil
can be stored each month for later use. The price for storage of raw oils is
5 monetary units per ton. Refined oil cannot be stored. The blended product
cannot be stored either.

The rest of the oil (that is, any not available in storage) must be bought in
quantities to meet the blending requirements. The price of each kind of oil
varies over the six-month period.

The two categories of oil cannot be refined on the same production line.
There is a limit on how much oil of each category (vegetable or non vegetable)
can be refined in a given month:
 - Not more than 200 tons of vegetable oil can be refined per month.
 - Not more than 250 tons of non vegetable oil can be refined per month.

There are constraints on the blending of oils:
 - The product cannot blend more than three oils.
 - When a given type of oil is blended into the product, at least 20 tons of
   that type must be used.
 - If either vegetable oil 1 (v1) or vegetable oil 2 (v2) is blended in the
   product, then non vegetable oil 3 (o3) must also be blended in that product.

The final product (refined and blended) sells for a known price:
150 monetary units per ton.

The aim of the six-month plan is to minimize production and storage costs while
maximizing profit.


This example is taken from the Cplex 12 User's Manual
(ILOG, Cplex 12 User's Manual, 2009)

Williams, H P, Model Building in Mathematical Programming. John Wiley
and Sons, 1978.

Keywords: mixed integer linear programming, food manufacturing, blending problem
$offText

Set
   m         'planing period'     / m1*m6 /
   p         'raw oils'           / v1*v2, o1*o3 /
   pv(p)     'vegetable oils'     / v1*v2 /
   pnv(p)    'non-vegetable oils' / o1*o3 /;

Parameter
   maxstore  'maximum storage of each type of raw oil' /   1000 /
   maxusepv  'maximum use of vegetable oils'           /    200 /
   maxusepnv 'maximum use of non-vegetable oils'       /    250 /
   minusep   'minimum use of raw oil'                  /     20 /
   maxnusep  'maximum number of raw oils in a blend'   /      3 /
   sp        'sales price of refined and blended oil'  /    150 /
   sc        'storage cost of raw oils'                /      5 /
   stock(p)  'stock at the beginning and end'          / #p 500 /
   hmin      'minimum hardness of refined oil'         /      3 /
   hmax      'maximum hardness of refined oil'         /      6 /
   h(p)      'hardness of raw oils'                    / v1 8.8, v2 6.1, o1 2
                                                         o2 4.2, o3 5.0       /;

Table cost(m,p) 'raw oil cost'
         v1  v2  o1  o2  o3
   m1   110 120 130 110 115
   m2   130 130 110  90 115
   m3   110 140 130 100  95
   m4   120 110 120 120 125
   m5   100 120 150 110 105
   m6    90 100 140  80 135;

Variable
   produce(m)    'production of blended and refined oil per month'
   use(m,p)      'usage of raw oil per month'
   induse(m,p)   'indicator for usage of raw oil per month'
   buy(m,p)      'purchase of raw oil per month'
   store(m,p)    'storage of raw oil at end of the month'
   profit        'objective variable';

Positive Variable produce, buy, store, use;
Binary   Variable induse;

Equation
   defobj        'objective'
   defusepv(m)   'maximum use of vegetable oils'
   defusepnv(m)  'maximum use of non-vegetable oils'
   defproduce(m) 'production of refined oil'
   defhmin(m)    'minmum hardness requirement'
   defhmax(m)    'maximum hardness requirement'
   stockbal(m,p) 'stock balance constraint'
   minuse(m,p)   'minimum usage of raw oil'
   maxuse(m,p)   'usage of raw oil is 0 if induse is 0'
   maxnuse(m)    'maximum number of raw oils used in a blend'
   deflogic1(m)  'if some vegetable raw oil is use we also need to use o1';

defobj..        profit =e= sum(m, sp*produce(m))
                        -  sum((m,p), cost(m,p)*buy(m,p))
                        -  sum((m,p), sc*store(m,p));

defusepv(m)..   sum(pv, use(m,pv))   =l= maxusepv;

defusepnv(m)..  sum(pnv, use(m,pnv)) =l= maxusepnv;

defproduce(m).. produce(m) =e= sum(p, use(m,p));

defhmin(m)..    sum(p, h(p)*use(m,p)) =g= hmin*produce(m);

defhmax(m)..    sum(p, h(p)*use(m,p)) =l= hmax*produce(m);

* steady-state stock
stockbal(m,p).. store(m--1,p) + buy(m,p) =e= use(m,p) + store(m,p);

* Now come the logical constraints
minuse(m,p)..   use(m,p) =g= minusep*induse(m,p);

maxuse(m,p)..   use(m,p) =l= (maxusepv$pv(p) + maxusepnv$pnv(p))*induse(m,p);

maxnuse(m)..    sum(p, induse(m,p)) =l= maxnusep;

* sum(pv, induse(m,pv))>=1 => induse(m,'o3')=1
* turn around induse(m,'o3')=0 => sum(pv, induse(m,pw))=0
deflogic1(m)..  sum(pv, induse(m,pv)) =l= induse(m,'o3')*card(pv);

store.up(m,p)    = maxstore;
store.fx('m6',p) = stock(p);

Model food / all /;

option optCr = 0;

solve food max profit using mip;
