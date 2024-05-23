$title A Production Mix Problem (PRODMIX,SEQ=3)

$onText
A furniture company wants to maximize its profits from the
manufacture of different types of desks.


Dantzig, G B, Chapter 3.5. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

Keywords: linear programming, production planning, manufacturing, furniture production
$offText

Set
   desk / d1*d4 /
   shop / carpentry, finishing /;

Table labor(shop,desk) 'labor requirements (man-hours)'
                d1    d2    d3    d4
   carpentry     4     9     7    10
   finishing     1     1     3    40;

Parameter
   caplim(shop) 'capacity (man hours)' / carpentry 6000, finishing 4000 /
   price(desk)  'per unit sold ($)'    / d1 12, d2 20, d3 18, d4 40     /;

Variable
   mix(desk) 'mix of desks produced (number of desks)'
   profit    'total profit                        ($)';

Positive Variable mix;

Equation
   cap(shop) 'capacity constraint (man-hours)'
   ap        'accounting: total profit    ($)';

cap(shop).. sum(desk, labor(shop,desk)*mix(desk)) =l= caplim(shop);

ap..        profit =e= sum(desk, price(desk)*mix(desk));

Model pmp 'product mix problem' / all /;

solve pmp maximizing profit using lp;
