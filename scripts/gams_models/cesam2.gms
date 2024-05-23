$title Cross Entropy SAM Estimation (CESAM2,SEQ=393)

$onText
CESAM2 illustrates a cross entropy technique for estimating the cells of a
consistent SAM assuming that the initial data are inconsistent and measured
with error. The method is applied to estimate the macro SAM used in CESAM.GMS.
Cell elements, some macro control totals, and row and column totals are assumed
to be measured with error. We assume that the user can specify prior estimates
of the values and standard errors of measurement for the cell values, macro
control totals, and row and column sums.

The original version of this code, CESAM.GMS, assumed that the SAM column
coefficients, A(i,j) = SAM(i,j)/SUM(i, SAM(i,j)), are treated as analogous to
probabilities and are included directly in the cross-entropy minimand. In this
version each SAM element, SAM(i,j), is assumed to be measured with error, and
all the errors are treated as probability-weighted sums of error support sets.
Only probabilities are included in the cross-entropy minimand, which is
consistent with the information-theoretic Bayesian approach to estimate
probabilities. The cost of this approach is that in CESAM2 there are many more
probabilities to be estimated. However, new solution algorithms are able to
solve large problems of this type, so size is no longer a serious constraint.

In the estimation procedure, we assume prior information on either:
 (1) values of cells, SAM(i,j), or
 (2) coefficients, A(i,j).

Errors can be treated as either:
 (1) additive       [e.g., SAM(i,j) = sam0(i,j) + err(i,j)], or
 (2) multiplicative [e.g., A(i,j)   = abar0(i,j)*EXP(err(i,j))]

where sam0(i,j) and abar0(i,j) are prior values of the cell value or
coefficient, and err(i,j) is the estimated measurement error.

In the first case, the prior mean of the errors is assumed to be zero. In the
second case, it will be one. In the first case, it is possible for the
posterior estimated cell value to change sign from the prior, while in the
second case the posterior estimated coefficient value can never change sign.

In the code below, we assume a prior on coefficients measured with
multiplicative errors for selected SAM accounts defined by the set acoeff(i).

Note that it is important to scale the SAM. Ideally, the SAM being estimated
should be scaled so that it does not contain values larger than about 1e3.

Note also that by default we use the GAMS intrinsic function centropy() in
the objective definition.  If you define NOCENTROPY (e.g. by running with
--NOCENTROPY = 1 on the command line) the cross-entropy function is written
explicitly using logs, etc.


References:

Robinson, S, Cattaneo, A, and El-Said, M, Updating and Estimating
a Social Accounting Matrix Using Cross Enthropy Methods. Economic
System Research 13, 1 (2001).

Golan, G, Judge, G, and Miller, D, Maximum Enthropy Econometrics.
John Wiley and Sons, 1996.

Judge, George G. and Ron C. Mittelhammer, An Information Theoretic Approach
to Econometrics. Cambridge: Cambridge University Press, 2012.

Programmed by Sherman Robinson, April 2013

Environment and Production Technology Division and
Development Strategy and Governance Division
International Food Policy Research Institute (IFPRI)
2033 K Street, N.W.
Washington, DC 20006 USA
Email: S.Robinson@CGIAR.ORG

Earlier version, CESAM, programmed by Sherman Robinson and Moataz El-Said,
November 2000.
Original version programmed by Sherman Robinson and Andrea Cattaneo.

Keywords: nonlinear programming, micro economics, cross entropy, social accounting matrix
$offText

* by default use centropy(), uncomment the line below to use logs explicitly
* $set NOCENTROPY 1

Set
   i 'SAM accounts' / ACT 'Activities',      COM   'Commodities'
                      FAC 'Factors',         ENT   'Enterprises'
                      HOU 'Households',      GOV   'Govt recurrent expenditures'
                      GIN 'Govt investment', CAP   'Capital account'
                      ROW 'Rest of world',   TOTAL 'Row and column totals'     /
   icoeff(i,i)  'SAM elements whose prior is specified as coefficients'
   ival(i,i)    'SAM elements whose prior is specified as values'
   NONZERO(i,i) 'SAM elements that can be nonzero and hence estimated'
   ii(i)        'all accounts in i except total'
   macro        'macro controls' / gdpfc2, gdp2 /

* The set jwt defines the dimension of the support set for the error
* distribution and the number of weights that must be estimated for each
* error. In this case, we specify an uninformative prior for jwt1,
* a normal prior for jwt2, and a general two-parameter distribution for jwt3.
   jwt       'master set of possible weights'             / 1*7 /
   jwt1(jwt) 'set of weights for errors in column sums'   / 1*7 /
   jwt2(jwt) 'set of weights for errors in macro totals'  / 1*5 /
   jwt3(jwt) 'set of weights for errors in cell elements' / 1*3 /;

Alias (i,j), (ii,jj);
ii(i)       = yes;
ii("Total") =  no;

Parameter
   stderr1  'standard error of measurement for column sums'   / .05  /
   stderr2  'standard error of measurement for macro totals'  / .05  /
   stderr3  'standard error of measurement for cell elements' / .25  /
   scalesam 'scale factor for scaling initial SAM'            / 1e3  /
   delta    'small number for CE objective function'          / 1e-8 /;

* Prior unbalanced proto-SAM
* ########################    SAM DATABASE       ########################
* The SAM is unbalanced by adding new rows with bad data
Table SAM(i,j) 'prior unbalanced social accounting matrix'
                ACT         COM         FAC         ENT
   ACT          0.0  14827.4240         0.0         0.0
   COM    7917.5040         0.0         0.0         0.0
   FAC    9805.4140         0.0         0.0         0.0
   ENT          0.0         0.0   3699.7060         0.0
*  HOU          0.0         0.0   6031.3080   3417.5060
   HOU                            6000.0000   3300.0000
   GOV     733.6000    357.4000     74.4000    165.2000
   GIN          0.0         0.0         0.0         0.0
   CAP          0.0         0.0         0.0    150.0000
   ROW          0.0   5573.8150         0.0         0.0
   Total 18456.5180   20758.639    9805.414    3732.706

   +            HOU         GOV         GIN         CAP
   ACT    2101.0490     -0.3270         0.0         0.0
*  COM    6753.3320   1764.5000   2118.5000   2197.7980
   COM    6953.3320   1564.5000   2518.5000   2597.7980
   FAC          0.0         0.0         0.0         0.0
   ENT          0.0     33.0000         0.0         0.0
   HOU          0.0     29.6000         0.0         0.0
   GOV     139.5000         0.0         0.0         0.0
   GIN          0.0         0.0         0.0         0.0
   CAP     649.1560   -356.6730   -406.2000         0.0
   ROW          0.0         0.0         0.0         0.0
   Total   9643.037      1470.1      1712.3    2197.798

   +            ROW       Total
   ACT    1488.1570   18416.303
   COM          0.0   20751.634
   FAC          0.0    9805.414
   ENT          0.0    3732.706
*  HOU     209.5010    9687.915
   HOU     200.0000    9687.915
   GOV          0.0      1470.1
   GIN    1712.3000      1712.3
   CAP    2163.8570     2200.14
   ROW          0.0    5573.815
   Total   5573.815;

Parameter
   SAM0(i,j)        'unbalance prior or proto-SAM transactions matrix'
   SAMBALCHK(i)     'column sums minus row sums in the SAM'
   Abar0(i,j)       'prior SAM coefficient matrix'
   ColSum0(i)       'targets for macro SAM column totals'
   macrov0(macro)   'target values for macro aggregates'
   vbar1(i,jwt)     'error support set 1 for column sums'
   vbar2(macro,jwt) 'error support set 2 for macro aggregates'
   vbar3(i,j,jwt)   'error support set 3 for SAM elements'
   wbar1(i,jwt)     'weights on error support set 1 for column totals'
   wbar2(macro,jwt) 'weights on error support set 2 for macro aggregates'
   wbar3(i,j,jwt)   'weights on error support set 3 for SAM elements'
   sigmay1(i)       'prior standard error of column sums'
   sigmay2(macro)   'prior standard error of macro aggregates'
   sigmay3(i,j)     'prior standard error of SAM elements'

* macro control totals
   gdp0             'base GDP'
   gdpfc0           'base GDP at factor cost'
   gdp00            'GDP from final SAM'
   gdpfc00          'GDP at factor cost from final SAM';

* ################# Initializing Parameters #################
SAM("TOTAL",jj) = sum(ii, SAM(ii,jj));
SAM(ii,"TOTAL") = sum(jj, SAM(ii,jj));

* Divide SAM entries by scalesam for better scaling.
* The SAM is scaled to enhance solver efficiency. Nonlinear solvers are
* more efficient if variables are scaled to be around 1.
SAM(i,j)                     = SAM(i,j)/scalesam;
abar0(ii,jj)$SAM("TOTAL",jj) = SAM(ii,jj)/SAM("TOTAL",jj);
SAM0(ii,jj)      = SAM(ii,jj);
SAM0("TOTAL",jj) = sum(ii, SAM(ii,jj));
SAM0(ii,"TOTAL") = sum(jj, SAM(ii,jj));
SAMBALCHK(jj)    = SAM0('TOTAL',jj) - SAM0(jj,'TOTAL');

display abar0, sam0, sambalchk;

* ########################  CROSS ENTROPY  ##############################
Parameter
   NegSam(i,j) 'negative SAM values'
   chkset(i,j) 'check coefficient and value sets';

* identify negative SAM entries for information
NegSam(i,j)$(SAM0(i,j) < 0)  = SAM(i,j);

* Define set of elements of SAM that can be nonzero. In this case, only
* elements which are nonzero in initial SAM.
NONZERO(ii,jj)$(Abar0(ii,jj)) = yes;

* SAM cells with priors on coefficients. We will also assume they have
* multiplicative errors.
Set acoeff(i) 'accounts with prior on column coefficients' / act, fac, ent, hou /;

icoeff(ii,acoeff)$NONZERO(ii,acoeff)              = yes;
ival(ii,jj)$(SAM0(ii,jj) and (not icoeff(ii,jj))) = yes;
chkset(ii,jj) = 1$ival(ii,jj) + 1$icoeff(ii,jj) - 1$NONZERO(ii,jj);

display icoeff, ival, chkset;

* Note that target column sums are being set to average of initial
* row and column sums. Initial column sums or other values
* could have been used instead, depending on knowledge of data quality
* and any other prior information.
ColSum0(ii) = (sam(ii,"total") + sam("total",ii))/2;
gdpfc0      =  sam("fac","act");
gdp0        =  sam("fac","act") + sam("gov","act") - sam("act","gov") + sam("gov","com");

macrov0("gdp2")   = gdp0;
macrov0("gdpfc2") = gdpfc0;

display negsam, ColSum0, gdpfc0, gdp0, macrov0;

$onText
*############### Define prior distributions of errors #####################
Start from assumed prior knowledge of the means and standard errors of
measurement of the cell elements, macro aggregates, and column sums. Below,
we assume that all column sums and macro aggregates have standard errors set
in stderr1, stderr2, and stderr3. These are Bayesian priors, not a maintained
hypothesis.

The estimated error is weighted sum of elements in an error support
set:
       ERR(ii)    = sum(jwt, W(ii,jwt)*VBAR(ii,jwt))
where the W's are estimated in the CE procedure and the support set, VBAR,
is specified to span the possible domain of the errors.

The prior mean (zero) and variance of these errors is given by:
          0       = sum(jwt, WBAR(ii,jwt)*VBAR(ii,jwt)) and
  (sigmay(ii))**2 = sum(jwt, WBAR(ii,jwt)*(VBAR(ii,jwt))**2)
where the WBAR's are the priors on the probability weights.

The VBARs are chosen to define a domain for the support set of +/- 3
standard errors. The prior on the weights, WBAR, are then calculated
to yield the specified prior on the standard error, sigmay.

In Robinson, Cattaneo, and El-Said (2001), we specify prior weights
(WBAR) that are "uninformative", given by a uniform distribution,
and set the prior standard errors by the choice of support set, VBAR.
In that paper, we use a three-weight specification (jwt /1*3/) with uniform
prior weights. Our current practice to specify an uniformative prior is to use
a seven-element support set with uniform prior weights, WBAR. To illustrate,
we specify an uninformative prior for col/row sums below

We assume two possible "informative" priors:
(1) A general two-parameter distribution with priors on the mean (zero) and
    variance (sigmay**2). This prior requires a three-element support set.
    We specify this prior for SAM elements with either additive or
    multiplicative errors.
(2) A normal distribution with priors on the mean (zero), variance (sigmay**2),
    skewness (zero), and kurtosis (3*sigmay**4). This prior requires a
    five-element support set. We specify this prior for macro totals.

We solve for the prior weights (wbar) given the specification of the prior
distribution and choice of the support set values (vbar)

For example, for the prior of a normal distribution, we assumes a prior mean
of zero, skewness of zero, and a prior value of kurtosis consistent with a
prior normal distribution with mean zero and variance of sigmay**2. In this
case, kurtosis equals 3*sigmay**4. To specify a four-parameter distribution
(mean, variance, skewness, kurtosis) requires a five-weight support set.

The prior weights (wbar) are specified so that:
sum(jwt, wbar(ii,jwt)*vbar(ii,jwt)**4) = 3*sigmay(ii,jwt)**4
as well as defining the variance as above, a mean of zero, and skewness (third
moment) of zero. These equations suffice to determine the values of the prior
weights (wbar).

The choice of +/- 1.5 standard error for vbar(ii,"2") and vbar(ii,"4") is
arbitrary, but it is convenient to have equal spacing of the error support set.

These are priors, not maintained hypotheses. The actual moments are estimated
as part of the estimation procedure.
$offText

* Set standard deviation for errors on column/row totals
sigmay1(ii) = stderr1*ColSum0(ii);

* Set constants for 7-weight error distribution (uninformative uniform prior)
vbar1(ii,"1")  = -3*sigmay1(ii);
vbar1(ii,"2")  = -2*sigmay1(ii);
vbar1(ii,"3")  = -1*sigmay1(ii);
vbar1(ii,"4")  =  0;
vbar1(ii,"5")  =  1*sigmay1(ii);
vbar1(ii,"6")  =  2*sigmay1(ii);
vbar1(ii,"7")  =  3*sigmay1(ii);
wbar1(ii,jwt1) =  1/7;

* Set standard deviation for errors on macro aggregates
sigmay2(macro) = stderr2*macrov0(macro);

* Set constants for 5-weight error distribution (normal prior)
vbar2(macro,"1") = -3  *sigmay2(macro);
vbar2(macro,"2") = -1.5*sigmay2(macro);
vbar2(macro,"3") =  0;
vbar2(macro,"4") =  1.5*sigmay2(macro);
vbar2(macro,"5") =  3  *sigmay2(macro);
wbar2(macro,"1") =  1/162;
wbar2(macro,"2") = 16/81;
wbar2(macro,"3") = 48/81;
wbar2(macro,"4") = 16/81;
wbar2(macro,"5") =  1/162;

* Set constants for 3-weight error distribution (2-parameter prior)
loop((ii,jj)$NONZERO(ii,jj),
*  Set standard deviation for errors on cell values or coefficients
*  Additive errors
   sigmay3(ii,jj)$ival(ii,jj)   = stderr3*ABS(sam0(ii,jj));
*  Multiplicative errors
   sigmay3(ii,jj)$icoeff(ii,jj) = stderr3;
   vbar3(ii,jj,"1") = -3*sigmay3(ii,jj);
   vbar3(ii,jj,"2") =  0;
   vbar3(ii,jj,"3") =  3*sigmay3(ii,jj);
   wbar3(ii,jj,"1") =  1/18;
   wbar3(ii,jj,"2") = 16/18;
   wbar3(ii,jj,"3") =  1/18;
*  end ii,jj loop
);
display vbar1, vbar2, vbar3, sigmay1, sigmay2, sigmay3;

Variable
   A(i,j)        'posterior SAM coefficient matrix'
   TSAM(i,j)     'posterior matrix of SAM transactions'
   MACROV(macro) 'macro aggregates'
   Y(i)          'row sum of SAM'
   ERR1(i)       'error value on column sums'
   ERR2(macro)   'error value for macro aggregates'
   ERR3(i,j)     'error value for SAM elements'
   W1(i,jwt)     'error weights for column sums'
   W2(macro,jwt) 'error weights for macro aggregates'
   W3(i,j,jwt)   'error weights for cell elements'
   DENTROPY      'entropy difference (objective)';

* ########################## INITIALIZE VARIABLES ##################
A.L(ii,jj)      = Abar0(ii,jj);
TSAM.L(ii,jj)   = sam0(ii,jj);
Y.L(ii)         = ColSum0(ii);
MACROV.L(macro) = macrov0(macro);
ERR1.L(ii)      = 0.0;
ERR2.L(macro)   = 0.0;
ERR3.L(ii,jj)$NONZERO(ii,jj)   = 0.0;
W1.L(ii,jwt)    = wbar1(ii,jwt);
W2.L(macro,jwt) = wbar2(macro,jwt);
W3.L(ii,jj,jwt)$NONZERO(ii,jj) = wbar3(ii,jj,jwt);
DENTROPY.L      = 0.0;

* ############ CORE EQUATIONS ############
Equation
   ROWSUMEQ(i)     'rowsum with error'
   ROWSUM(i)       'row sums'
   COLSUM(j)       'column sums'
   SAMCOEF(i,j)    'define SAM coefficients'
   TSAMEQ(i,j)     'SAM elements in values'
   ASAMEQ(i,j)     'SAM coefficients'
   GDPFCDEF        'define GDP at factor cost'
   GDPDEF          'define GDP at market prices'
   MACROEQ(macro)  'macro aggregates with error'
   ERROR1EQ(i)     'definition of error term 1'
   ERROR2EQ(macro) 'definition of error term 2'
   ERROR3EQ(i,j)   'definition of error term 3'
   SUMW1(i)        'sum of weights 1'
   SUMW2(macro)    'sum of weights 2'
   SUMW3(i,j)      'sum of weights 3'
   ENTROPY         'entropy difference definition';

* Row and column sums estimation and balance
ROWSUMEQ(ii).. Y(ii) =e= ColSum0(ii) + ERR1(ii);

ROWSUM(ii)..   sum(jj, TSAM(ii,jj)) =e= Y(ii);

COLSUM(jj)..   sum(ii, TSAM(ii,jj)) =e= Y(jj);

* Estimating SAM elements from prior values or coefficients
SAMCOEF(ii,jj)$NONZERO(ii,jj).. TSAM(ii,jj) =e= A(ii,jj)*Y(jj);

TSAMEQ(ii,jj)$IVAL(ii,jj)..     TSAM(ii,jj) =e= sam0(ii,jj) + ERR3(ii,jj);

ASAMEQ(ii,jj)$ICOEFF(ii,jj)..   A(ii,jj)    =e= abar0(ii,jj)*EXP(ERR3(ii,jj));

* Macro aggregates measured with error
GDPFCDEF.. MACROV("gdpfc2") =e= TSAM("fac","act");

GDPDEF..   MACROV("gdp2")   =e= TSAM("fac","act") + TSAM("gov","act")
                             -  TSAM("act","gov") + TSAM("gov","com");

MACROEQ(macro).. MACROV(macro) =e= macrov0(macro) + ERR2(macro);

* Definition of errors as probability weighted sums of support sets
ERROR1EQ(ii)..    ERR1(ii)    =e= sum(jwt1, W1(ii,jwt1)*vbar1(ii,jwt1));

ERROR2EQ(macro).. ERR2(macro) =e= sum(jwt2, W2(macro,jwt2)*vbar2(macro,jwt2));

ERROR3EQ(ii,jj)$NONZERO(ii,jj).. ERR3(ii,jj) =e= sum(jwt3, W3(ii,jj,jwt3)*vbar3(ii,jj,jwt3));

* Probabilities must sum to one
SUMW1(ii)..                   sum(jwt1, W1(ii,jwt1))    =e= 1;

SUMW2(macro)..                sum(jwt2, W2(macro,jwt2)) =e= 1;

SUMW3(ii,jj)$NONZERO(ii,jj).. sum(jwt3, W3(ii,jj,jwt3)) =e= 1;

$ifThen set NOCENTROPY
* Cross-entropy objective function, explicit version
ENTROPY.. DENTROPY =e= sum((ii,jj,jwt3)$nonzero(ii,jj),
                       W3(ii,jj,jwt3)*(log(W3(ii,jj,jwt3) + delta) -
                       log(wbar3(ii,jj,jwt3) + delta)))
                    +  sum((ii,jwt1), W1(ii,jwt1)
                    * (log(W1(ii,jwt1)    + delta) -
                       log(wbar1(ii,jwt1) + delta)))
                    +  sum((macro,jwt2), W2(macro,jwt2)
                    * (log(W2(macro,jwt2)    + delta) -
                       log(wbar2(macro,jwt2) + delta)));
$else

* Objective function using GAMS cross-entropy intrinsic function, CENTROPY
ENTROPY.. DENTROPY =e= sum[(ii,jj,jwt3)$nonzero(ii,jj), CENTROPY(W3(ii,jj,jwt3),wbar3(ii,jj,jwt3))]
                    +  sum[(ii,jwt1),    CENTROPY(W1(ii,jwt1),wbar1(ii,jwt1))]
                    +  sum[(macro,jwt2), CENTROPY(W2(macro,jwt2),wbar2(macro,jwt2))];
$endIf

* Define bounds for cell values and fix variables not
* included in the estimation
A.fx(ii,jj)$   (not nonzero(ii,jj)) = 0;
TSAM.fx(ii,jj)$(not nonzero(ii,jj)) = 0;

* Upper and lower bounds on the error weights
W1.lo(ii,jwt1)                         = 0;
W1.up(ii,jwt1)                         = 1;
W2.lo(macro,jwt2)                      = 0;
W2.up(macro,jwt2)                      = 1;
W3.lo(ii,jj,jwt3)$NONZERO(ii,jj)       = 0;
W3.up(ii,jj,jwt3)$NONZERO(ii,jj)       = 1;
W3.fx(ii,jj,jwt3)$(not nonzero(ii,jj)) = 0;

Model SAMENTROP / all /;

option limRow = 100, limCol = 0, solPrint = on, domLim = 100;

SAMENTROP.holdFixed = 1;

solve SAMENTROP using NLP minimizing dentropy;

* Parameters for reporting results
Parameter
   Macsam1(i,j)  'assigned new balanced SAM flows from CE'
   Macsam2(i,j)  'balanced SAM flows in original units'
   percent1(i,j) 'percent change of new SAM from original SAM'
   Diffrnce(i,j) 'differnce btw original SAM and final SAM in values';

macsam1(ii,jj)            = TSAM.L(ii,jj);
macsam1("total",jj)       = sum(ii, macsam1(ii,jj));
macsam1(ii,"total")       = sum(jj, macsam1(ii,jj));
macsam2(i,j)              = macsam1(i,j)*scalesam;
percent1(i,j)$(sam0(i,j)) = 100*(macsam1(i,j) - sam0(i,j))/sam0(i,j);
Diffrnce(i,j)             = macsam1(i,j) - sam0(i,j);
SAMBALCHK(jj)             = TSAM.L('TOTAL',jj) - TSAM.L(jj,'TOTAL');

display sam0, macsam1, SAMBALCHK, Diffrnce, percent1, macsam2, dentropy.l;

gdp00   = macsam1("fac","act") + macsam1("gov","act") - macsam1("act","gov") + macsam1("gov","com");
gdpfc00 = macsam1("fac","act");

display gdp0, gdp00, gdpfc0, gdpfc00, macrov0, macrov.l;

Parameter ANEW(i,j);
* print some stuff
ANEW(ii,jj)       = A.l(ii,jj);
ANEW("total",jj)  = sum(ii, A.L(ii,jj));
ANEW(ii,"total")  = sum(jj, A.L(ii,jj));
ABAR0("total",jj) = sum(ii, ABAR0(ii,jj));
ABAR0(ii,"total") = sum(jj, ABAR0(ii,jj));

display ANEW, ABAR0;

Scalar meanerr1, meanerr2;
meanerr1 = sum(ii, abs(err1.l(ii)))/card(ii);
meanerr2 = sum(macro, abs(err2.l(macro)))/card(macro);
display meanerr1, meanerr2;
