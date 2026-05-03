$title Financial Optimization: Risk Management (MEANVARX,SEQ=113)

$onText
Minimum and maximum trade constraints are added to the standard
mean-variance model. If it is not profitable to trade within these
ranges, no trade should take place. A turnover constraint is added
to improve stability of the solution to small changes in data. The
resulting model is a nonlinear mixed-integer problem.

Two important modeling tricks are demonstrated: (1) use of only
the triangular part of the Q matrix, and (2) introduction of
the marginal variance to improve computational performance of
large QP problems.


Dahl, H, Meeraus, A, and Zenios, S A, Some Financial Optimization
Models: Risk Management. In Zenios, S A, Ed, Financial Optimization.
Cambridge University Press, New York, NY, 1993.

Keywords: mixed integer nonlinear programming, risk management, finance,
          financial optimization
$offText

$eolCom //

Set i 'securities' / cn, fr, gr, jp, sw, uk, us /;

Alias (i,j);

Parameter mu(i) 'expected return of security' / cn 0.1287
                                                fr 0.1096
                                                gr 0.0501
                                                jp 0.1524
                                                sw 0.0763
                                                uk 0.1854
                                                us 0.0620 /;

Table q(i,j) 'covariance matrix'
           cn     fr     gr     jp     sw     uk     us
   cn   42.18
   fr   20.18  70.89
   gr   10.88  21.58  25.51
   jp    5.30  15.41   9.60  22.33
   sw   12.32  23.24  22.63  10.32  30.01
   uk   23.84  23.80  13.22  10.46  16.36  42.23
   us   17.41  12.62   4.70   1.00   7.20   9.90  16.42;

*  we will continue to use only the lower triangle of the q-matrix
*  and adjust the off diagonal entries to give the correct results.
q(i,j) = 2*q(j,i);
q(i,i) = q(i,i)/2;

Scalar
   tau    'bounding parameter on turnover of current holdings'
   lambda 'return versus variance component tradeoff parameter' ;

Set pd 'portfolio data labels'
       / old  'current holdings fraction of the portfolio'
         umin 'minimum increase of holdings fraction of security i'
         umax 'maximum increase of holdings fraction of security i'
         lmin 'minimum decrease of holdings fraction of security i'
         lmax 'maximum decrease of holdings fraction of security i' /;

Table bdata(i,pd) 'portfolio data and trading restrictions'
*            - increase -  - decrease -
        old  umin    umax  lmin    lmax
   cn   0.2  0.03    0.11  0.02    0.30
   fr   0.2  0.04    0.10  0.02    0.15
   gr   0.0  0.04    0.07  0.04    0.10
   jp   0.0  0.03    0.11  0.04    0.10
   sw   0.2  0.03    0.20  0.04    0.10
   uk   0.2  0.03    0.10  0.04    0.15
   us   0.2  0.03    0.10  0.04    0.30;

bdata(i,'lmax') = min(bdata(i,'old'),bdata(i,'lmax')); // tighten bound

abort$(abs(sum(i, bdata(i,'old')) - 1) >= 1e5) 'error in bdata', bdata;

Variable
    omega   'objective variable definition for minlp'
    x(i)    'fraction of portfolio of current holdings of i'
    xi(i)   'fraction of portfolio increase'
    xd(i)   'fraction of portfolio decrease'
    mvar(i) 'marginal variance'
    y(i)    'binary switch for increasing current holdings of i'
    z(i)    'binary switch for decreasing current holdings of i';

Binary   Variable y, z;
Positive Variable x, xi, xd;

Equation
   budget     'budget constraint'
   turnover   'restrict maximum turnover of portfolio'
   maxinc(i)  'bound of maximum lot increase of fraction of i'
   mininc(i)  'bound of minimum lot increase of fraction of i'
   maxdec(i)  'bound of maximum lot decrease of fraction of i'
   mindec(i)  'bound of minimum lot decrease of fraction of i'
   binsum(i)  'restrict use of binary variables'
   xdef(i)    'final portfolio definition'
   mvardef(i) 'marginal variance definition'
   obj        'objective function'
   objx       'objective function';

budget..     sum(i, x(i)) =e= 1;

xdef(i)..    x(i)  =e= bdata(i,'old') - xd(i) + xi(i);

maxinc(i)..  xi(i) =l= bdata(i,'umax')*y(i);

mininc(i)..  xi(i) =g= bdata(i,'umin')*y(i);

maxdec(i)..  xd(i) =l= bdata(i,'lmax')*z(i);

mindec(i)..  xd(i) =g= bdata(i,'lmin')*z(i);

binsum(i)..  y(i) + z(i) =l= 1;

turnover..   sum(i, xi(i)) =l= tau;

mvardef(i).. mvar(i) =e= sum(j, q(i,j)*x(j));

obj..        omega =e= sum((i,j), x(i)*q(i,j)*x(j)) - lambda*sum(i, mu(i)*x(i));

objx..       omega =e= sum(i, x(i)*mvar(i))         - lambda*sum(i, mu(i)*x(i));

Model
   mean / all - mvardef - objx /
   marg / all - obj /;

lambda = 0.5;
tau    = 0.3;

solve mean minimizing omega using minlp;

solve marg minimizing omega using minlp;

Parameter report 'summary report';
report(i,'old') = bdata(i,'old');
report(i,'inc') = xi.l(i);
report(i,'dec') = xd.l(i);
report(i,'new') = x.l(i);
display report;
