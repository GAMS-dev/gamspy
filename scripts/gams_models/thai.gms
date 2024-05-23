$title Thai Navy Problem (THAI,SEQ=98)

$onText
This model is used to allocate ships to transport personnel from
different port to a training center.


Choypeng, P, Puakpong, P, and Rosenthal, R E, Optimal Ship Routing
and Personnel Assignment for Naval Recruitment in Thailand.
Interfaces 16, 4 (1986), 356-366.

Keywords: mixed integer linear programming, routing, naval recruitment,
          scheduling
$offText

Set
   i       'ports'           / chumphon, surat, nakon, songkhla /
   j       'voyages'         / v-01*v-15 /
   k       'ship classes'    / small, medium, large /
   sc(i,k) 'ship capability' / chumphon.(small,medium,large)
                              (surat,nakon).(medium,large)
                               songkhla.large                /
   vc(j,k) 'voyage capability';

Parameter
   d(i)       'number of men at port p needing transport' / chumphon  475
                                                            surat     659
                                                            nakon     672
                                                            songkhla 1123 /
   shipcap(k) 'ship capacity in men'                      / small     100
                                                            medium    200
                                                            large     600 /
   n(k)       'number of ships available'                 / small       2
                                                            medium      3
                                                            large       4 /;

Table a(j,*) 'assignment of ports to voyages'
              dist  chumphon  surat  nakon  songkhla
   v-01        370         1
   v-02        460                1
   v-03        600                       1
   v-04        750                                 1
   v-05        515         1      1
   v-06        640         1             1
   v-07        810         1                       1
   v-08        665                1      1
   v-09        665                1                1
   v-10        800                       1         1
   v-11        720         1      1      1
   v-12        860         1      1                1
   v-13        840         1             1         1
   v-14        865                1      1         1
   v-15        920         1      1      1         1;

Scalar
   w1 'ship assignment weight'           / 1.00   /
   w2 'ship distance traveled weight'    /  .01   /
   w3 'personnel distance travel weight' /  .0001 /;

vc(j,k) = prod(i$a(j,i), sc(i,k));
display vc;

Variable
   z(j,k)   'number of times voyage jk is used'
   y(j,k,i) 'number of men transported from port i via voyage jk'
   obj;

Integer  Variable z;
Positive Variable y;

Equation
   objdef
   demand(i)   'pick up all the men at port i'
   voycap(j,k) 'observe variable capacity of voyage jk'
   shiplim(k)  'observe limit of class k';

demand(i)..           sum((j,k)$(a(j,i)$vc(j,k)), y(j,k,i)) =g= d(i);

voycap(j,k)$vc(j,k).. sum(i$a(j,i), y(j,k,i)) =l= shipcap(k)*z(j,k);

shiplim(k)..          sum(j$vc(j,k), z(j,k))  =l= n(k);

objdef.. obj =e= w1*sum((j,k)$vc(j,k), z(j,k))
              +  w2*sum((j,k)$vc(j,k), a(j,"dist")*z(j,k))
              +  w3*sum((j,k,i)$(a(j,i)$vc(j,k)), a(j,"dist")*y(j,k,i));

Model thainavy / all /;

z.up(j,k)$vc(j,k) = n(k);

solve thainavy minimizing obj using mip;

display y.l, z.l;
