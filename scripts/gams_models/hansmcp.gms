$title Hansen's Activity Analysis Example (HANSMCP,SEQ=135)

$onText
Hansen's Activity Analysis Example.


Scarf, H, and Hansen, T, The Computation of Economic Equilibria.
Yale University Press, 1973.

Keywords: mixed complementarity problem, activity analysis, general equilibrium model,
          social accounting matrix, european regional policy, impact analysis
$offText

Set
   c 'commodities' / agric, food, textiles, hserv,   entert, houseop, capeop
                     steel, coal, lumber,   housbop, capbop, labor,   exchange /
   h 'consumers'   / agent1*agent4 /
   s 'sectors'     / dom1*dom12, imp1*imp7, exp1*exp7 /;

Alias (c,cc);

Table e(c,h) 'commodity endowments'
              agent1  agent2  agent3  agent4
   housbop       2       0.4             0.8
   capbop        3       2               7.5
   labor         0.6     0.8       1     0.6;

Table d(c,h) 'reference demands'
              agent1  agent2  agent3  agent4
   agric         0.1     0.2     0.3     0.1
   food          0.2     0.2     0.2     0.2
   textiles      0.1     0.1     0.3     0.1
   hserv         0.1     0.1     0.1     0.1
   entert        0.1     0.1     0.1     0.1
   houseop       0.3     0.1             0.1
   capeop        0.1     0.2             0.3;

Parameter esub(h) 'elasticities in demand' / agent1  1, agent2  1
                                             agent3  1, agent4  1 /;

Table data(*,c,s) 'activity analysis matrix'
                      dom1  dom2  dom3  dom4  dom5
   output.agric       5.00
   output.food              5.00
   output.textiles                2.00
   output.hserv                         2.00
   output.entert                              4.00
   output.houseop                       0.32
   output.capeop      0.40  1.30  1.20
   input .agric             3.50  0.10        0.70
   input .food        0.90        0.10        0.80
   input .textiles    0.20  0.50        0.10  0.10
   input .hserv       1.00  2.00  2.00        2.00
   input .steel       0.20  0.40  0.20  0.10
   input .coal        1.00  0.10  0.10  1.00
   input .lumber      0.50  0.40  0.30  0.30
   input .housbop                       0.40
   input .capbop      0.50  1.50  1.50  0.10  0.10
   input .labor       0.40  0.20  0.20  0.02  0.40

   +                  dom6  dom7  dom8  dom9  dom10
   output.houseop     0.80
   output.capeop      1.10  6.00  1.80  1.20   0.40
   output.steel                   2.00
   output.coal                          2.00
   output.lumber                               1.00
   input .textiles    0.80  0.40  0.10  0.10   0.10
   input .hserv       0.40  1.80  1.60  0.80   0.20
   input .steel       1.00  2.00        0.50   0.20
   input .coal              0.20  1.00         0.20
   input .lumber      3.00  0.20  0.20  0.50
   input .capbop      1.50  2.50  2.50  1.50   0.50
   input .labor       0.30  0.10  0.10  0.40   0.40

   +                 dom11  dom12  imp1  imp2  imp3
   output.agric                    1.00
   output.food                           1.00
   output.textiles                             1.00
   output.houseop            0.36
   output.capeop      0.90
   input .hserv                    0.40  0.20  0.20
   input .housbop            0.40
   input .capbop      1.00         0.20  0.10  0.10
   input .labor                    0.04  0.02  0.02
   input .exchange                 0.50  0.40  0.80

   +                  imp4  imp5  imp6  imp7  exp1
   output.capeop      1.00
   output.steel             1.00
   output.coal                    1.00
   output.lumber                        1.00
   output.exchange                            0.50
   input .agric                               1.00
   input .hserv       0.40  0.40  0.40  0.40  0.20
   input .capbop      0.20  0.20  0.20  0.20  0.20
   input .labor       0.04  0.04  0.04  0.04  0.04
   input .exchange    1.20  0.60  0.70  0.40

   +                  exp2  exp3  exp4  exp5  exp6
   output.exchange    0.40  0.80  1.20  0.60  0.70
   input .food        1.00
   input .textiles          1.00
   input .hserv       0.20  0.20  0.40  0.40  0.40
   input .capeop                  1.00
   input .steel                         1.00
   input .coal                                1.00
   input .capbop      0.10  0.10  0.20  0.20  0.20
   input .labor       0.02  0.02  0.04  0.04  0.04

   +                  exp7
   output.exchange    0.40
   input .hserv       0.40
   input .lumber      1.00
   input .capbop      0.20
   input .labor       0.04;

Parameter
   alpha(c,h) 'demand function share parameter'
   a(c,s)     'activity analysis matrix';

alpha(c,h) = d(c,h)/sum(cc, d(cc,h));
a(c,s)     = data("output",c,s) - data("input",c,s);

Positive Variable
   p(c) 'commodity price'
   y(s) 'production'
   i(h) 'income';

Equation
   mkt(c)    'commodity market'
   profit(s) 'zero profit'
   income(h) 'income index';

* distinguish ces and cobb-douglas demand functions:
mkt(c)..       sum(s, a(c,s)*y(s)) + sum(h, e(c,h))
           =g= sum(h$(esub(h) <> 1), (i(h)/sum(cc, alpha(cc,h)*p(cc)**(1 - esub(h))))
                                   *  alpha(c,h)*(1/p(c))**esub(h))
            +  sum(h$(esub(h) = 1), i(h)*alpha(c,h)/p(c));

profit(s)..   -sum(c, a(c,s)*p(c)) =g= 0;

income(h)..    i(h) =g= sum(c, p(c)*e(c,h));

Model hansen / mkt.p, profit.y, income.i /;

p.l(c)  = 1;
y.l(s)  = 1;
i.l(h)  = 1;
p.lo(c) = 0.00001$(smax(h, alpha(c,h)) > eps);

* fix the price of numeraire commodity:
p.fx(c)$(ord(c) = 1) = 1;

solve hansen using mcp;
