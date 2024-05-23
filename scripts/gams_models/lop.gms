$title Line Optimization (LOP,SEQ=221)

$onText
The problem finds line plans for a given rail network and origin
destination demand data. Models for minimum cost and direct traveler
objectives are given. The set of possible lines is defined by the
shortest paths in the rail network.


Bussieck, M R, Optimal Lines in Public Rail Transport. PhD thesis,
TU Braunschweig, 1998.

Bussieck, M R, Kreuzer, P, and Zimmermann, U T, Optimal Lines for
Railway Systems. European Journal of Operation Research 96, 1 (1996),
54-63.

Claessens, M T, van Dijk, N M, and Zwaneveld, P J, Cost Optimal
Allocation of Rail Passenger Lines. European Journal Operation
Research 110, 3 (1998), 474-489.

Keywords: linear programming, mixed integer linear programming, passenger railway optimization,
          shortest path, dutch railway, public rail transport, network optimization
$offText

$eolCom //

Set s 'stations'
      / Ah   'Arnhem',             Apd  'Apeldoorn',        Asd 'Amsterdam CS'
        Asdz 'Amsterdam Zuid WTC', Asn  'Assen',            Bd  'Breda'
        Ehv  'Eindhoven',          Gn   'Groningen',        Gv  'Den Haag HS'
        Gvc  'Den Haag CS',        Hgl  'Hengelo',          Hr  'Heerenveen'
        Lls  'Lelystad Centrum',   Lw   'Leeuwarden',       Mt  'Maastricht'
        Odzg 'Oldenzaal Grens',    Rsdg 'Roosendaal Grens', Rtd 'Rotterdam CS'
        Shl  'Schiphol',           Std  'Sittard',          Ut  'Utrecht CS'
        Zl   'Zwolle',             Zvg  'Zevenaar Grens'                      /;

Table rt(s,s) 'running time (defines edges of rail network)'
        Hr Asn Zl Hgl Ut Shl Asdz Asd Gv Gvc Rtd Bd Ehv Std Mt Lls Rsdg Zvg Odzg
     Lw 29
     Gn     28
     Hr        66
    Asn        78
     Zl               85                                        50
    Apd            69 64           89
    Hgl                                                                       18
     Ah               58                                                 19
     Ut                        34  39     61  57 92  81
    Shl                         9  19 43  43
   Asdz                    9                                    56
    Asd                                                         54
     Gv                                    1  23
    Rtd                                          49                  18
    Ehv                                                  78
    Std                                                     21                 ;

Parameter tt(s) 'turn-a-round time'
                / (Ehv,Gn,Gvc,Hgl,Lls,Rsdg,Std)       5.0
                  (Apd,Asn,Gv,Hr,Mt,Shl,Zvg)         13.8
                  (Ah,Asd,Asdz,Bd,Lw,Odzg,Rtd,Ut,Zl) 14.1 /;

Table lfr(s,s) 'line frequency requirement'
        Hr Asn Zl Hgl Ut Shl Asdz Asd Gv Gvc Rtd Bd Ehv Std Mt Lls Rsdg Zvg Odzg
     Lw  1
     Gn      1
     Hr         1
    Asn         1
     Zl                1                                         1
    Apd             1  1            1
    Hgl                                                                        1
     Ah                2                                                  1
     Ut                         2   2      2   1  1   2
    Shl                         1   3  2   1
   Asdz                    1                                     1
    Asd                                                          1
     Gv                                    3   1
    Rtd                                           2                   1
    Ehv                                                   1
    Std                                                      1                 ;

Table od(s,s) 'origin destination matrix'
         Hr  Asn  Zl Hgl Ah   Ut Shl Asdz  Asd   Gv  Gvc  Rtd   Bd  Ehv Std  Mt  Lls Rsdg Zvg Odzg
     Lw 478      380     13  145  20   21   90    6   26   36   14    9   9   4   77    7  14
     Gn     1720 720         331  48   88  205   12   73   75   34   28  29  13  200   33  14
     Hr          511     11  209  20   16  115   10   48   58   16   11   8   4   77   10  19
    Asn          854     16  502  32   58  235   13  117  125   42   33  28  14  152   48  19
     Zl                  56 1112  64  171  400   33  163  182   79   47  46  21  390  100  32
    Apd              468    1160  32   76  917   21  202  143   57   62  10   5        47  83   71
    Hgl                      422  11   24  287   20   81   52   39   28  20  12        24       75
     Ah                     4244  60  721  726  109  741  180  136  101            8  320 602
     Ut                          278 5826 4919  225 3138 2260 1165 3109 720 359   89  325 996   21
    Shl                              1456 6469 1339 1503  509    7   99  44  29  103  164
   Asdz                                         461  207  369  138  542 203 149  819    6 155
    Asd                                         730 2540 1756  154  437 155  37 2783 2258 489   22
     Gv                                              785 4586  531   35  22   8   29  890
    Gvc                                                  2829  228  335 104  41   31    3 229    7
    Rtd                                                       1829  569 179  73   46 1077 157   11
     Bd                                                             950 157  79    6  329  14    5
    Ehv                                                                 936 404    8   75  11    3
    Std                                                                     863    2   19
     Mt                                                                            1   22
    Lls                                                                                15         ;

Set
   lf 'line frequencies'          / 1 'every 60 minutes', 2 'every 30 minutes' /
   ac 'additional cars per train' / 0*9 /;

Scalar
   mincars 'minimum number of cars per train' /      3 /
   ccap    'passenger capacity per car'       /    467 /
   cfx     'fix cost per car'                 / 353100 /
   crm     'cost per ride minute per car'     /   5803 /
   trm     'cost per ride minute per train'   /  44959 /
   cmp     'additional cost multiplier'       /     90 /
   maxtcap 'maximum train capacity';

maxtcap = (mincars + card(ac) - 1)*ccap;

Alias (s,s1,s2,s3);

* Some data checks
Set error01(s,s);
error01(s1,s2) = rt(s1,s2) and not lfr(s1,s2) or not rt(s1,s2) and lfr(s1,s2);
abort$card(error01) 'Inconsistent Edge data', error01;

$onText
Generate Shortest Path from all nodes to all nodes. Instead of solving s
network problems we solve all shortest path problems simultaneously.
$offText

Set d(s,s) 'directed version of the network';
d(s1,s2) = rt(s1,s2) or rt(s2,s1);

Variable
   f(s,s1,s2) 'flow from s on edge s1s2'
   spobj      'objective variable';

Positive Variable f;

Equation
   balance(s,s1) 'flow balance constraint for flow from s in s1'
   defspobj      'definition of the combined min cost flow objective';

balance(s,s1)..
   sum(d(s1,s2), f(s,d)) =e= sum(d(s2,s1), f(s,d)) + sameas(s,s1)*card(s) - 1;

defspobj..
   spobj =e= sum((s,d(s1,s2)), f(s,d)*max(rt(s1,s2),rt(s2,s1)));

Model sp 'shortest path model' / balance, defspobj /;

solve sp minimzing spobj using lp;

Set tree(s,s1,s2) 'shortest path tree from s';
tree(s,s1,s2) = f.l(s,s1,s2);
abort$(card(tree) <> card(s)*(card(s) - 1)) 'wrong tree computation';

* From the trees we generate all shortest path lines. We perfom a BFS.
Set
   r             'rank' / 1*100 /
   k(s,s)        'arcs from root to a node'
   v(s,r)        'nodes with rank from root to a node'
   unvisit(s)    'unvisited nodes'
   visit(s)      'visited nodes'
   from(s)       'from nodes'
   to(s)         'to nodes'
   l(s,s1,s2,s3) 'line from s to s1 with edge s2s3'
   lr(s,s1,s2,r) 'rank of s2 in line from s to s1';

Alias (s,root), (r,r1);

l(root,s,s1,s2) = no;
lr(root,s,s1,r) = no;

loop(root,
   from(root) = yes;  // We start with the root node
   unvisit(s) = yes;  // All nodes are unvisited
   visit(s)   =  no;  // No node is visited
   loop(r$(ord(r) > 1 and card(unvisit)),
      unvisit(from) =  no;
      visit(from)   = yes;
      // nodes that can be reach from a node in from
      to(unvisit) = sum(tree(root,from,unvisit), yes);
      loop(from,
         k(s2,s3)$l(root,from,s2,s3)  = yes;  // arcs of line root-from
         v(s2,r1)$lr(root,from,s2,r1) = yes;  // nodes of line root-from
         v(from,"1")$(card(k) = 0)    = yes;  // this is the root node

         // the line root-to has the arcs/nodes of line root-from
         l(root,to,k)$tree(root,from,to)       = yes;
         lr(root,to,v)$tree(root,from,to)      = yes;
         // plus the arc from-to
         l(root,to,from,to)$tree(root,from,to) = yes;
         lr(root,to,to,r)$tree(root,from,to)   = yes;

         k(s2,s3) = no;
         v(s2,r1) = no;
      );
      from(s)  =  no;
      from(to) = yes;  // move one layer down
      to(s)    =  no;
   );
   from(s) = no;
);

Set error02(s1,s2) 'arcs not covered by shortest path lines';
error02(s1,s2) = lfr(s1,s2) and sum(l(root,s,s1,s2), 1) = 0;
abort$card(error02) error02;

* Lines are symetric, so delete one half of them
Set ll(s,s) 'station pair represening a line';
ll(s1,s2) = ord(s1) < ord(s2);

l(root,s,s1,s2)$(not ll(root,s)) = no;
lr(root,s,s1,r)$(not ll(root,s)) = no;

* and order the edges in the lines in the way we stored them
l(root,s,s1,s2)$(l(root,s,s2,s1) and rt(s1,s2)) = yes;
l(root,s,s1,s2)$(not rt(s1,s2)) = no;

Parameter
   rp(s,s,s)   'rank of node'
   lastrp(s,s) 'rank of the last node in line';

rp(ll,s)   = sum(r$lr(ll,s,r), ord(r));
lastrp(ll) = smax(s,rp(ll,s));

Parameter load(s1,s2) 'passenger load of an edge';
load(s1,s2)$rt(s1,s2) = sum(l(root,s,s1,s2)$od(root,s), od(root,s));

$onText
Model dtlop:
   Determines a line plan with a maximizing the number of direct
   travelers. The number of direct traveler represents an upper
   bound because the capcity constraint is relaxed.
$offText

Variable
   dt(s1,s2)   'direct traveler between s1 and s2'
   freq(s1,s2) 'frequency on arc s1s2'
   phi(s1,s2)  'frequency of line between s1 and s2'
   obj         'objective variable';

Integer Variable phi;

Equation
   deffreqlop(s1,s2) 'definition of the frequency for each edge'
   dtlimit(s1,s2)    'limit the direct travelers'
   defobjdtlop       'objective function';

deffreqlop(s1,s2)$rt(s1,s2)..
   freq(s1,s2) =e= sum(l(ll,s1,s2), phi(ll));

dtlimit(s1,s2)$od(s1,s2)..
   dt(s1,s2) =l= min(od(s1,s2),maxtcap)*sum(ll$(rp(ll,s1) and rp(ll,s2)), phi(ll));

defobjdtlop..
   obj =e= sum((s1,s2)$od(s1,s2), dt(s1,s2));

Model lopdt / deffreqlop, dtlimit, defobjdtlop /;

freq.lo(s1,s2)$rt(s1,s2) = max(lfr(s1,s2),ceil(load(s1,s2)/maxtcap));
freq.up(s1,s2)$rt(s1,s2) = freq.lo(s1,s2);
dt.up(s1,s2)$od(s1,s2)   = od(s1,s2);

solve lopdt maximizing obj using mip;

* Store the solution for further reporting
Parameter solrep, solsum;
solrep('DT',ll,'freq') = phi.l(ll);
solrep('DT',ll,'cars')$phi.l(ll) = mincars + card(ac) - 1;

$onText
Model ILP:
   Determines a line plan of minimum cost.
$offText

Parameter
   xcost(root,s,lf) 'operating and capcital cost for line with mincars cars'
   ycost(root,s,lf) 'operating and capcital cost for additional cars'
   len(s,s)         'length of line'
   sigma(s,s)       'line circulation factor';

len(ll)      =  sum(l(ll,s1,s2), rt(s1,s2));
sigma(ll)    = (len(ll) + sum(s$lr(ll,s,"1"), tt(s)) + sum(s$(rp(ll,s) = lastrp(ll)), tt(s)))/60;
xcost(ll,lf) =  ord(lf)*len(ll)*(trm + mincars*crm) + mincars*ceil(sigma(ll)*ord(lf))*cfx;
ycost(ll,lf) =  ord(lf)*len(ll)*crm + ceil(sigma(ll)*ord(lf))*cfx;

Variable
   x(s1,s2,lf) 'line frequency indicator of line s1-s2'
   y(s1,s2,lf) 'additional cars on line s1-s2 with frequency lf';

Integer Variable y;
Binary  Variable x;

Equation
   deffreqilp(s,s)  'definition of the frequency for each edge'
   defloadilp(s,s)  'capacity of lines fulfill the demand'
   oneilp(s,s)      'only one frequency per line'
   couplexy(s,s,lf) 'coupling constraints'
   defobjilp        'definition of the objective';

deffreqilp(s1,s2)$rt(s1,s2)..
   freq(s1,s2) =e= sum((l(ll,s1,s2),lf), ord(lf)*x(ll,lf));

defloadilp(s1,s2)$rt(s1,s2).. ceil(load(s1,s2)/ccap) =l=
   sum((l(ll,s1,s2),lf), ord(lf)*(mincars*x(ll,lf) + y(ll,lf)));

oneilp(ll)..
   sum(lf, x(ll,lf)) =l= 1;

couplexy(ll,lf)..
   y(ll,lf) =l= y.up(ll,lf)*x(ll,lf);

defobjilp..
   obj =e= sum((ll,lf), xcost(ll,lf)*x(ll,lf) + ycost(ll,lf)*y(ll,lf));

Model ilp / defobjilp, deffreqilp, defloadilp, oneilp, couplexy /;

y.up(ll,lf) = card(ac) - 1;
freq.up(s1,s2)$rt(s1,s2) = 100;

ilp.optCr  = 0;
ilp.resLim = 100;

solve ilp minimizing obj using mip;

solrep('ILP',ll,'freq') = sum(lf$x.l(ll,lf), ord(lf));
solrep('ILP',ll,'cars') = sum(lf$x.l(ll,lf), mincars + y.l(ll,lf));
solsum('ILP','cost')    = obj.l;

* We have now two line plans. Lets make a comparison: Cost and Direct
* Travelers. The Direct Traveler Evaluation requires another more
* detailed model. Model lopdt gave just an upper bound.

$onText
Model EvalDT:
   Determines for a given line plan (routes, frequency, capacity) the
   maximum number of direct travelers.
$offText

Parameter cap(s,s) 'the capacity of a line';
Set       sol(s,s) 'the actual lines in a line plan';

Positive Variable dtr(s,s,s,s) 'direct travelers of OD pair u v in line on route s s';

Equation
   dtllimit(s,s1,s2,s3) 'limit direct travelers in line s-s1 on edge s2-s3'
   sumbound(s,s)        'sum of direct travels <= total number of travelers';

dtllimit(l(sol,s,s1))..
   sum((s2,s3)$(od(s2,s3) and rp(sol,s2) and rp(sol,s3) and
       (min(rp(sol,s),rp(sol,s1)) >= rp(sol,s2) and // s and s1 must be
        max(rp(sol,s),rp(sol,s1)) <= rp(sol,s3)  or // between the nodes of the
        min(rp(sol,s),rp(sol,s1)) >= rp(sol,s3) and // origin destination pair
        max(rp(sol,s),rp(sol,s1)) <= rp(sol,s2))),  // s2-s3 in order to
   dtr(sol,s2,s3)) =l= cap(sol);                    // occupy capacity of s s1

sumbound(s2,s3)$od(s2,s3)..
   sum(sol$(rp(sol,s2) and rp(sol,s3)), dtr(sol,s2,s3)) =e= dt(s2,s3);

Model evaldt / dtllimit, sumbound, defobjdtlop /;

* Evaluate direct travelers for DT line plan
sol(ll)  = solrep('DT',ll,'freq');
cap(sol) = solrep('DT',sol,'freq')*solrep('DT',sol,'cars')*ccap;

solve evaldt maximizing obj using lp;

solsum('DT','dtrav') = obj.l;
solsum('DT','cost')  = sum(sol,  solrep('DT',sol,'freq')*len(sol)*trm
                              + (solrep('DT',sol,'freq')*len(sol)*crm
                              +  ceil(sigma(sol)*solrep('DT',sol,'freq'))*cfx)
                              *  solrep('DT',sol,'cars'));

* Evaluate DT for ILP line plan
sol(ll)  = solrep('ILP',ll,'freq');
cap(sol) = solrep('DT',sol,'freq')*solrep('DT',sol,'cars')*ccap;

solve evaldt maximizing obj using lp;
solsum('ILP','dtrav') = obj.l;

display solrep, solsum;
