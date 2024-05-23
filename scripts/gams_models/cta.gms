$title Controlled Tabular Adjustments (CTA,SEQ=310)

$onText
Statistical agencies publish data which contains items that need to be
altered to protect confidentiality. Controlled Tabular Adjustments (CTA)
is a recent method to limit disclosure and can be elegantly expressed
as a Mixed Integer Programming problem. The programming framework then
allows easy expression of other data relationships like multi-dimensional
adding up conditions. The following model uses a 3-dimensional table from
from Cox, Kelly and Patil (2005) to illustrate this method.

The data is stored in an Excel Spreadsheet.


Lawrence H Cox, James P Kelly and Rahul J Patil, Computational Aspects
of Controlled Tabular Adjustments: Algorithms and Analysis, in The Next
Wave in Computing, Optimization, and Decision Technologies, Eds Bruce L Golden,
S Raghavan and Edward A Wasil, Springer, 2005, pp 45-59.

Keywords: mixed integer linear programming, statistical disclosure limitations
$offText

Set
   i 'rows'
   j 'columns'
   k 'planes';

Parameter
   dat(k<,i<,j<) 'unprotected data table'
   pro(k,i,j)    'information sensitive cells';

* extract data from Excel
$onEmbeddedCode Connect:
- ExcelReader:
    file: cox3.xlsx
    symbols:
      - name: dat
        range: Sheet1!A1
        rowDimension: 2
        columnDimension: 1
      - name: pro
        range: Sheet2!A1
        rowDimension: 2
        columnDimension: 1
- GAMSWriter:
    writeAll: True
$offEmbeddedCode

* do some basic data checks
abort$sum((i,k), round(sum(j, dat(k,i,j)) - 2*dat(k,i,'total'))) 'row totals are incorrect', dat;
abort$sum((j,k), round(sum(i, dat(k,i,j)) - 2*dat(k,'total',j))) 'column totals are incorrect', dat;
abort$sum((i,j), round(sum(k, dat(k,i,j)) - 2*dat('total',i,j))) 'plane totals are incorrect', dat;

Variable
   t(i,j,k) 'adjusted cell value'
   obj;

Positive Variable adjN(i,j,k), adjP(i,j,k);
Binary   Variable b(i,j,k);

Equation
   defadj(i,j,k) 'define new cell values'
   addrow(i,k)   'add up for rows'
   addcol(j,k)   'add up for columns'
   addpla(i,j)   'add up for plane'
   pmin(i,j,k)   'small value for sensitive cells'
   pmax(i,j,k)   'big value for sensitive cells'
   defobj;

Set
   v(i,j,k) 'non zero cells'
   s(i,j,k) 'sensitive cells';

Parameter BigM 'the famous big M - make it as small as possible';

defadj(v(i,j,k)).. t(v) =e= dat(k,i,j) + adjP(v) - adjN(v);

addrow(i,k)..      sum(v(i,j,k), t(v)) =e= 2*t(i,'total',k);

addcol(j,k)..      sum(v(i,j,k), t(v)) =e= 2*t('total',j,k);

addpla(i,j)..      sum(v(i,j,k), t(v)) =e= 2*t(i,j,'total');

pmin(s(i,j,k))..   adjN(s) =g= pro(k,i,j)*(1 - b(s));

pmax(s(i,j,k))..   adjP(s) =g= pro(k,i,j)*b(s);

Equation pminx, pmaxx;

pminx(s(i,j,k))..  adjN(s) =l= BigM*pro(k,i,j)*(1 - b(s));

pmaxx(s(i,j,k))..  adjP(s) =l= BigM*pro(k,i,j)*b(s);

defobj..           obj =e= sum(v, adjN(v) + adjP(v));

Model cox3 / all /;

v(i,j,k) = dat(k,i,j);
s(i,j,k) = pro(k,i,j);

option limCol = 0, limRow = 0, solPrint = off, optCr = 0, optCa = 0.99, resLim = 10;

BigM = 2;

solve cox3 min obj using mip;

Parameter
   rep(k,i,j)      'summary report'
   adjsum(k,i,j,*) 'adjustment summary'
   adjrep(k,i,j)   'adjustment report';

option rep:0:2:1, adjrep:0:2:1, adjsum:3:3:1;

rep(k,i,j)          =  t.l(i,j,k);
adjsum(k,i,j,'neg') =  adjn.l(i,j,k);
adjsum(k,i,j,'pos') =  adjp.l(i,j,k);
adjsum(k,i,j,'min') =  pro(k,i,j);
adjrep(k,i,j)       = -adjN.l(i,j,k) + adjp.l(i,j,k);

embeddedCode Connect:
- GAMSReader:
    symbols:
      - name: adjrep
      - name: rep
      - name: adjsum
- ExcelWriter:
    file: results.xlsx
    clearSheet: True
    symbols:
      - name: adjrep
      - name: rep
      - name: adjsum
endEmbeddedCode

* now we find the next best 5 solutions
Set
   l     'solution labels' / solution1*solution5 /
   ll(l) 'dynamic version of l';

Parameter
   binrep(*,*,*,l) 'binary for protected variables'
   best            'best objective value';

option binrep:0:3:1;

Equation
   cutone(l) 'cuts to exclude previous solutions'
   cuttwo(l) 'cuts to exclude previous solutions';

* there is always a complementary solution by just changing all the signs
* cut(ll)..  sum(s, abs(b(s) - binrep(s,ll)) =g= 1;

cutone(ll).. sum(s$binrep(s,ll), 1 - b(s)) + sum(s$(not binrep(s,ll)), b(s)) =g= 1;

cuttwo(ll).. sum(s$(not binrep(s,ll)), 1 - b(s)) + sum(s$binrep(s,ll), b(s)) =g= 1;

Model cox3c 'includes cuts' / all /;

* find the card(l) best solutions that are within 1% of the global
best         = round(obj.l);
cox3c.resUsd = cox3.resUsd;
cox3c.nodUsd = cox3.nodUsd;

loop(l$((obj.l - best)/best <= 0.01),
   ll(l) = yes;
   binrep(s,l)             = round(b.l(s));
   binrep('','','Obj',l)   = obj.l;
   binrep('','','mSec',l)  = cox3c.resUsd*1000;
   binrep('','','nodes',l) = cox3c.nodUsd;
   binrep('Comp','Cells','Adjusted',l) = sum((i,j,k)$(not s(i,j,k)), 1$round(adjn.l(i,j,k) + adjp.l(i,j,k)));
   solve cox3c min obj using mip;
);

embeddedCode Connect:
- GAMSReader:
    symbols:
      - name: binrep
- ExcelWriter:
    file: results.xlsx
    clearSheet: True
    symbols:
      - name: binrep
endEmbeddedCode
