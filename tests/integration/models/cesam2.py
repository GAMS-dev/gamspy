"""
Cross Entropy SAM Estimation (CESAM2)

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

Keywords: nonlinear programming, micro economics, cross entropy, social
accounting matrix
"""

from gamspy import (
    Set,
    Alias,
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Number,
    Card,
    Domain,
)
from gamspy.math import abs, exp, log, centropy
import numpy as np
from gamspy import Problem, Sense


def main(is_centropy=False):
    m = Container()

    SAM_recs = np.array(
        [
            [0, 14827.424, 0, 0, 2101.049, -0.327, 0, 0, 1488.157, 18416.303],
            [
                7917.504,
                0,
                0,
                0,
                6953.332,
                1564.500,
                2518.500,
                2597.798,
                0,
                20751.634,
            ],
            [9805.414, 0, 0, 0, 0, 0, 0, 0, 0, 9805.414],
            [0, 0, 3699.706, 0, 0, 33.000, 0, 0, 0, 3732.706],
            [0, 0, 6000.000, 3300.000, 0, 29.600, 0, 0, 200.000, 9687.915],
            [733.600, 357.400, 74.400, 165.200, 139.500, 0, 0, 0, 0, 1470.100],
            [0, 0, 0, 0, 0, 0, 0, 0, 1712.300, 1712.300],
            [
                0,
                0,
                0,
                150.000,
                649.156,
                -356.673,
                -406.200,
                0,
                2163.857,
                2200.140,
            ],
            [0, 5573.815, 0, 0, 0, 0, 0, 0, 0, 5573.815],
            [
                18456.518,
                20758.639,
                9805.414,
                3732.706,
                9643.037,
                1470.100,
                1712.300,
                2197.798,
                5573.815,
                0,
            ],
        ]
    )

    m.addOptions({"limCol": 0, "limRow": 0, "solPrint": "off"})

    # Set
    i = Set(
        m,
        name="i",
        records=[
            "ACT",
            "COM",
            "FAC",
            "ENT",
            "HOU",
            "GOV",
            "GIN",
            "CAP",
            "ROW",
            "TOTAL",
        ],
        description="i",
    )
    icoeff = Set(
        m,
        name="icoeff",
        domain=[i, i],
        description="SAM elements whose prior is specified as coefficients",
    )
    ival = Set(
        m,
        name="ival",
        domain=[i, i],
        description="SAM elements whose prior is specified as values",
    )
    NONZERO = Set(
        m,
        name="NONZERO",
        domain=[i, i],
        description="SAM elements that can be nonzero and hence estimated",
    )
    ii = Set(
        m, name="ii", domain=[i], description="all accounts in i except total"
    )
    macro = Set(
        m,
        name="macro",
        records=["gdpfc2", "gdp2"],
        description="macro controls",
    )

    # The set jwt defines the dimension of the support set for the error
    # distribution and the number of weights that must be estimated for each
    # error. In this case, we specify an uninformative prior for jwt1,
    # a normal prior for jwt2, and a general two-parameter distribution for
    # jwt3.
    jwt = Set(
        m,
        name="jwt",
        records=[str(ss) for ss in range(1, 8)],
        description="master set of possible weights",
    )
    jwt1 = Set(
        m,
        name="jwt1",
        domain=[jwt],
        records=[str(ss) for ss in range(1, 8)],
        description="set of weights for errors in column sums",
    )
    jwt2 = Set(
        m,
        name="jwt2",
        domain=[jwt],
        records=[str(ss) for ss in range(1, 6)],
        description="set of weights for errors in macro totals",
    )
    jwt3 = Set(
        m,
        name="jwt3",
        domain=[jwt],
        records=[str(ss) for ss in range(1, 4)],
        description="set of weights for errors in cell elements",
    )

    j = Alias(m, name="j", alias_with=i)
    jj = Alias(m, name="jj", alias_with=ii)
    ii[i] = Number(1)
    ii["Total"] = Number(0)

    # Parameters
    stderr1 = Parameter(
        m,
        name="stderr1",
        records=0.05,
        description="standard error of measurement for column sums",
    )
    stderr2 = Parameter(
        m,
        name="stderr2",
        records=0.05,
        description="standard error of measurement for macro totals",
    )
    stderr3 = Parameter(
        m,
        name="stderr3",
        records=0.25,
        description="standard error of measurement for cell elements",
    )
    scalesam = Parameter(
        m,
        name="scalesam",
        records=1e3,
        description="scale factor for scaling initial SAM",
    )
    delta = Parameter(
        m,
        name="delta",
        records=1e-8,
        description="small number for CE objective function",
    )

    # Prior unbalanced proto-SAM
    # ########################    SAM DATABASE       ########################
    # The SAM is unbalanced by adding new rows with bad data
    SAM = Parameter(
        m,
        name="SAM",
        domain=[i, j],
        records=SAM_recs,
        description="prior unbalanced social accounting matrix",
    )

    # Parameters
    SAM0 = Parameter(
        m,
        name="SAM0",
        domain=[i, j],
        description="unbalance prior or proto-SAM transactions matrix",
    )
    SAMBALCHK = Parameter(
        m,
        name="SAMBALCHK",
        domain=[i],
        description="column sums minus row sums in the SAM",
    )
    Abar0 = Parameter(
        m,
        name="Abar0",
        domain=[i, j],
        description="prior SAM coefficient matrix",
    )
    ColSum0 = Parameter(
        m,
        name="ColSum0",
        domain=[i],
        description="targets for macro SAM column totals",
    )
    macrov0 = Parameter(
        m,
        name="macrov0",
        domain=[macro],
        description="target values for macro aggregates",
    )
    vbar1 = Parameter(
        m,
        name="vbar1",
        domain=[i, jwt],
        description="error support set 1 for column sums",
    )
    vbar2 = Parameter(
        m,
        name="vbar2",
        domain=[macro, jwt],
        description="error support set 2 for macro aggregates",
    )
    vbar3 = Parameter(
        m,
        name="vbar3",
        domain=[i, j, jwt],
        description="error support set 3 for SAM elements",
    )
    wbar1 = Parameter(
        m,
        name="wbar1",
        domain=[i, jwt],
        description="weights on error support set 1 for column totals",
    )
    wbar2 = Parameter(
        m,
        name="wbar2",
        domain=[macro, jwt],
        description="weights on error support set 2 for macro aggregates",
    )
    wbar3 = Parameter(
        m,
        name="wbar3",
        domain=[i, j, jwt],
        description="weights on error support set 3 for SAM elements",
    )
    sigmay1 = Parameter(
        m,
        name="sigmay1",
        domain=[i],
        description="prior standard error of column sums",
    )
    sigmay2 = Parameter(
        m,
        name="sigmay2",
        domain=[macro],
        description="prior standard error of macro aggregates",
    )
    sigmay3 = Parameter(
        m,
        name="sigmay3",
        domain=[i, j],
        description="prior standard error of SAM elements",
    )

    # macro control totals
    gdp0 = Parameter(m, name="gdp0", description="base GDP")
    gdpfc0 = Parameter(m, name="gdpfc0", description="base GDP at factor cost")
    gdp00 = Parameter(m, name="gdp00", description="GDP from final SAM")
    gdpfc00 = Parameter(
        m, name="gdpfc00", description="GDP at factor cost from final SAM"
    )

    # ################# Initializing Parameters #################
    SAM["TOTAL", jj] = Sum(ii, SAM[ii, jj])
    SAM[ii, "TOTAL"] = Sum(jj, SAM[ii, jj])

    # Divide SAM entries by scalesam for better scaling.
    # The SAM is scaled to enhance solver efficiency. Nonlinear solvers are
    # more efficient if variables are scaled to be around 1.
    SAM[i, j] = SAM[i, j] / scalesam
    Abar0[ii, jj].where[SAM["TOTAL", jj]] = SAM[ii, jj] / SAM["TOTAL", jj]
    SAM0[ii, jj] = SAM[ii, jj]
    SAM0["TOTAL", jj] = Sum(ii, SAM[ii, jj])
    SAM0[ii, "TOTAL"] = Sum(jj, SAM[ii, jj])
    SAMBALCHK[jj] = SAM0["TOTAL", jj] - SAM0[jj, "TOTAL"]

    # display Abar0, SAM0, SAMBALCHK

    # ########################  CROSS ENTROPY  ##############################
    # Parameters
    NegSam = Parameter(
        m, name="NegSam", domain=[i, j], description="negative SAM values"
    )
    chkset = Parameter(
        m,
        name="chkset",
        domain=[i, j],
        description="check coefficient and value sets",
    )

    # identify negative SAM entries for information
    NegSam[i, j].where[SAM0[i, j] < 0] = SAM[i, j]

    # Define set of elements of SAM that can be nonzero. In this case, only
    # elements which are nonzero in initial SAM.
    NONZERO[ii, jj].where[Abar0[ii, jj]] = Number(1)

    # SAM cells with priors on coefficients. We will also assume they have
    # multiplicative errors.
    acoeff = Set(
        m,
        name="acoeff",
        domain=[i],
        records=["act", "fac", "ent", "hou"],
        description="accounts with prior on column coefficients",
    )

    icoeff[ii, acoeff].where[NONZERO[ii, acoeff]] = Number(1)
    ival[ii, jj].where[(SAM0[ii, jj]) & (~icoeff[ii, jj])] = Number(1)
    chkset[ii, jj] = (
        Number(1).where[ival[ii, jj]]
        + Number(1).where[icoeff[ii, jj]]
        - Number(1).where[NONZERO[ii, jj]]
    )

    # display icoeff, ival, chkset

    # Note that target column sums are being set to average of initial
    # row and column sums. Initial column sums or other values
    # could have been used instead, depending on knowledge of data quality
    # and any other prior information.
    ColSum0[ii] = (SAM[ii, "total"] + SAM["total", ii]) / 2
    gdpfc0.assign = SAM["fac", "act"]
    gdp0.assign = (
        SAM["fac", "act"]
        + SAM["gov", "act"]
        - SAM["act", "gov"]
        + SAM["gov", "com"]
    )

    macrov0["gdp2"] = gdp0
    macrov0["gdpfc2"] = gdpfc0

    # Set standard deviation for errors on column/row totals
    sigmay1[ii] = stderr1 * ColSum0[ii]

    # Set constants for 7-weight error distribution
    # (uninformative uniform prior)
    vbar1[ii, "1"] = -3 * sigmay1[ii]
    vbar1[ii, "2"] = -2 * sigmay1[ii]
    vbar1[ii, "3"] = -1 * sigmay1[ii]
    vbar1[ii, "4"] = 0
    vbar1[ii, "5"] = 1 * sigmay1[ii]
    vbar1[ii, "6"] = 2 * sigmay1[ii]
    vbar1[ii, "7"] = 3 * sigmay1[ii]
    wbar1[ii, jwt1] = 1 / 7

    # Set standard deviation for errors on macro aggregates
    sigmay2[macro] = stderr2 * macrov0[macro]

    # Set constants for 5-weight error distribution (normal prior)
    vbar2[macro, "1"] = -3 * sigmay2[macro]
    vbar2[macro, "2"] = -1.5 * sigmay2[macro]
    vbar2[macro, "3"] = 0
    vbar2[macro, "4"] = 1.5 * sigmay2[macro]
    vbar2[macro, "5"] = 3 * sigmay2[macro]
    wbar2[macro, "1"] = 1 / 162
    wbar2[macro, "2"] = 16 / 81
    wbar2[macro, "3"] = 48 / 81
    wbar2[macro, "4"] = 16 / 81
    wbar2[macro, "5"] = 1 / 162

    for record in SAM.records.itertuples(index=False):
        if record.i != "TOTAL" and record.j != "TOTAL":
            #  Set standard deviation for errors on cell values or coefficients
            #  Additive errors
            sigmay3[record.i, record.j].where[ival[record.i, record.j]] = (
                stderr3 * abs(SAM0[record.i, record.j])
            )
            #  Multiplicative errors
            sigmay3[record.i, record.j].where[
                icoeff[record.i, record.j]
            ] = stderr3
            vbar3[record.i, record.j, "1"] = -3 * sigmay3[record.i, record.j]
            vbar3[record.i, record.j, "2"] = 0
            vbar3[record.i, record.j, "3"] = 3 * sigmay3[record.i, record.j]
            wbar3[record.i, record.j, "1"] = 1 / 18
            wbar3[record.i, record.j, "2"] = 16 / 18
            wbar3[record.i, record.j, "3"] = 1 / 18

    # Variables
    A = Variable(
        m,
        name="A",
        domain=[i, j],
        description="posterior SAM coefficient matrix",
    )
    TSAM = Variable(
        m,
        name="TSAM",
        domain=[i, j],
        description="posterior matrix of SAM transactions",
    )
    MACROV = Variable(
        m, name="MACROV", domain=[macro], description="macro aggregates"
    )
    Y = Variable(m, name="Y", domain=[i], description="row Sum of SAM")
    ERR1 = Variable(
        m, name="ERR1", domain=[i], description="error value on column sums"
    )
    ERR2 = Variable(
        m,
        name="ERR2",
        domain=[macro],
        description="error value for macro aggregates",
    )
    ERR3 = Variable(
        m,
        name="ERR3",
        domain=[i, j],
        description="error value for SAM elements",
    )
    W1 = Variable(
        m,
        name="W1",
        domain=[i, jwt],
        description="error weights for column sums",
    )
    W2 = Variable(
        m,
        name="W2",
        domain=[macro, jwt],
        description="error weights for macro aggregates",
    )
    W3 = Variable(
        m,
        name="W3",
        domain=[i, j, jwt],
        description="error weights for cell elements",
    )
    DENTROPY = Variable(
        m, name="DENTROPY", description="entropy difference (objective)"
    )

    # ########################## INITIALIZE VARIABLES ##################
    A.l[ii, jj] = Abar0[ii, jj]
    TSAM.l[ii, jj] = SAM0[ii, jj]
    Y.l[ii] = ColSum0[ii]
    MACROV.l[macro] = macrov0[macro]
    ERR1.l[ii] = 0.0
    ERR2.l[macro] = 0.0
    ERR3.l[ii, jj].where[NONZERO[ii, jj]] = 0.0
    W1.l[ii, jwt] = wbar1[ii, jwt]
    W2.l[macro, jwt] = wbar2[macro, jwt]
    W3.l[ii, jj, jwt].where[NONZERO[ii, jj]] = wbar3[ii, jj, jwt]
    DENTROPY.L = 0.0

    # ############ CORE EQUATIONS ############
    # Equations
    ROWSUMEQ = Equation(
        m,
        name="ROWSUMEQ",
        domain=[i],
        description="rowsum with error",
    )
    ROWSUM = Equation(m, name="ROWSUM", domain=[i], description="row sums")
    COLSUM = Equation(m, name="COLSUM", domain=[j], description="column sums")
    SAMCOEF = Equation(
        m,
        name="SAMCOEF",
        domain=[i, j],
        description="define SAM coefficients",
    )
    TSAMEQ = Equation(
        m,
        name="TSAMEQ",
        domain=[i, j],
        description="SAM elements in values",
    )
    ASAMEQ = Equation(
        m,
        name="ASAMEQ",
        domain=[i, j],
        description="SAM coefficients",
    )
    GDPFCDEF = Equation(
        m,
        name="GDPFCDEF",
        description="define GDP at factor cost",
    )
    GDPDEF = Equation(
        m,
        name="GDPDEF",
        description="define GDP at market prices",
    )
    MACROEQ = Equation(
        m,
        name="MACROEQ",
        domain=[macro],
        description="macro aggregates with error",
    )
    ERROR1EQ = Equation(
        m,
        name="ERROR1EQ",
        domain=[i],
        description="definition of error term 1",
    )
    ERROR2EQ = Equation(
        m,
        name="ERROR2EQ",
        domain=[macro],
        description="definition of error term 2",
    )
    ERROR3EQ = Equation(
        m,
        name="ERROR3EQ",
        domain=[i, j],
        description="definition of error term 3",
    )
    SUMW1 = Equation(
        m,
        name="SUMW1",
        domain=[i],
        description="Sum of weights 1",
    )
    SUMW2 = Equation(
        m,
        name="SUMW2",
        domain=[macro],
        description="Sum of weights 2",
    )
    SUMW3 = Equation(
        m,
        name="SUMW3",
        domain=[i, j],
        description="Sum of weights 3",
    )
    ENTROPY = Equation(
        m,
        name="ENTROPY",
        description="entropy difference definition",
    )

    # Row and column sums estimation and balance
    ROWSUMEQ[ii] = Y[ii] == ColSum0[ii] + ERR1[ii]

    ROWSUM[ii] = Sum(jj, TSAM[ii, jj]) == Y[ii]

    COLSUM[jj] = Sum(ii, TSAM[ii, jj]) == Y[jj]

    # Estimating SAM elements from prior values or coefficients
    SAMCOEF[ii, jj].where[NONZERO[ii, jj]] = TSAM[ii, jj] == A[ii, jj] * Y[jj]

    TSAMEQ[ii, jj].where[ival[ii, jj]] = (
        TSAM[ii, jj] == SAM0[ii, jj] + ERR3[ii, jj]
    )

    ASAMEQ[ii, jj].where[icoeff[ii, jj]] = A[ii, jj] == Abar0[ii, jj] * exp(
        ERR3[ii, jj]
    )

    # Macro aggregates measured with error
    GDPFCDEF.expr = MACROV["gdpfc2"] == TSAM["fac", "act"]

    GDPDEF.expr = (
        MACROV["gdp2"]
        == TSAM["fac", "act"]
        + TSAM["gov", "act"]
        - TSAM["act", "gov"]
        + TSAM["gov", "com"]
    )

    MACROEQ[macro] = MACROV[macro] == macrov0[macro] + ERR2[macro]

    # Definition of errors as probability weighted sums of support sets
    ERROR1EQ[ii] = ERR1[ii] == Sum(jwt1, W1[ii, jwt1] * vbar1[ii, jwt1])

    ERROR2EQ[macro] = ERR2[macro] == Sum(
        jwt2, W2[macro, jwt2] * vbar2[macro, jwt2]
    )

    ERROR3EQ[ii, jj].where[NONZERO[ii, jj]] = ERR3[ii, jj] == Sum(
        jwt3, W3[ii, jj, jwt3] * vbar3[ii, jj, jwt3]
    )

    # Probabilities must Sum to one
    SUMW1[ii] = Sum(jwt1, W1[ii, jwt1]) == 1

    SUMW2[macro] = Sum(jwt2, W2[macro, jwt2]) == 1

    SUMW3[ii, jj].where[NONZERO[ii, jj]] = Sum(jwt3, W3[ii, jj, jwt3]) == 1

    if not is_centropy:
        # Cross-entropy objective function, explicit version
        ENTROPY.expr = DENTROPY == (
            Sum(
                Domain(ii, jj, jwt3).where[NONZERO[ii, jj]],
                W3[ii, jj, jwt3]
                * (
                    log(W3[ii, jj, jwt3] + delta)
                    - log(wbar3[ii, jj, jwt3] + delta)
                ),
            )
            + Sum(
                [ii, jwt1],
                W1[ii, jwt1]
                * (log(W1[ii, jwt1] + delta) - log(wbar1[ii, jwt1] + delta)),
            )
            + Sum(
                [macro, jwt2],
                W2[macro, jwt2]
                * (
                    log(W2[macro, jwt2] + delta)
                    - log(wbar2[macro, jwt2] + delta)
                ),
            )
        )
    else:
        ENTROPY.expr = DENTROPY == (
            Sum(
                Domain(ii, jj, jwt3).where[NONZERO[ii, jj]],
                centropy(W3[ii, jj, jwt3], wbar3[ii, jj, jwt3]),
            )
            + Sum([ii, jwt1], centropy(W1[ii, jwt1], wbar1[ii, jwt1]))
            + Sum([macro, jwt2], centropy(W2[macro, jwt2], wbar2[macro, jwt2]))
        )

    # Define bounds for cell values and fix variables not
    # included in the estimation
    A.fx[ii, jj].where[~NONZERO[ii, jj]] = 0
    TSAM.fx[ii, jj].where[~NONZERO[ii, jj]] = 0

    # Upper and lower bounds on the error weights
    W1.lo[ii, jwt1] = 0
    W1.up[ii, jwt1] = 1
    W2.lo[macro, jwt2] = 0
    W2.up[macro, jwt2] = 1
    W3.lo[ii, jj, jwt3].where[NONZERO[ii, jj]] = 0
    W3.up[ii, jj, jwt3].where[NONZERO[ii, jj]] = 1
    W3.fx[ii, jj, jwt3].where[~NONZERO[ii, jj]] = 0

    SAMENTROP = Model(
        m,
        name="SAMENTROP",
        equations=m.getEquations(),
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=DENTROPY,
    )

    m.addOptions(
        {
            "limRow": 100,
            "limCol": 0,
            "solPrint": "on",
            "domLim": 100,
            "holdfixedasync": 1,
        }
    )

    SAMENTROP.solve()

    # Parameters for reporting results
    Macsam1 = Parameter(
        m,
        name="Macsam1",
        domain=[i, j],
        description="assigned new balanced SAM flows from CE",
    )
    Macsam2 = Parameter(
        m,
        name="Macsam2",
        domain=[i, j],
        description="balanced SAM flows in original units",
    )
    percent1 = Parameter(
        m,
        name="percent1",
        domain=[i, j],
        description="percent change of new SAM from original SAM",
    )
    Diffrnce = Parameter(
        m,
        name="Diffrnce",
        domain=[i, j],
        description="differnce btw original SAM and final SAM in values",
    )
    ANEW = Parameter(m, name="ANEW", domain=[i, j])

    Macsam1[ii, jj] = TSAM.l[ii, jj]
    Macsam1["total", jj] = Sum(ii, Macsam1[ii, jj])
    Macsam1[ii, "total"] = Sum(jj, Macsam1[ii, jj])
    Macsam2[i, j] = Macsam1[i, j] * scalesam
    percent1[i, j].where[SAM0[i, j]] = (
        100 * (Macsam1[i, j] - SAM0[i, j]) / SAM0[i, j]
    )
    Diffrnce[i, j] = Macsam1[i, j] - SAM0[i, j]
    SAMBALCHK[jj] = TSAM.l["TOTAL", jj] - TSAM.l[jj, "TOTAL"]

    gdp00.assign = (
        Macsam1["fac", "act"]
        + Macsam1["gov", "act"]
        - Macsam1["act", "gov"]
        + Macsam1["gov", "com"]
    )
    gdpfc00.assign = Macsam1["fac", "act"]

    # print some stuff
    ANEW[ii, jj] = A.l[ii, jj]
    ANEW["total", jj] = Sum(ii, A.l[ii, jj])
    ANEW[ii, "total"] = Sum(jj, A.l[ii, jj])
    Abar0["total", jj] = Sum(ii, Abar0[ii, jj])
    Abar0[ii, "total"] = Sum(jj, Abar0[ii, jj])

    meanerr1 = Parameter(m, name="meanerr1")
    meanerr2 = Parameter(m, name="meanerr2")
    meanerr1.assign = Sum(ii, abs(ERR1.l[ii])) / Card(ii)
    meanerr2.assign = Sum(macro, abs(ERR2.l[macro])) / Card(macro)

    print("Objective Function Value: ", round(DENTROPY.records.level[0], 3))
    print("meanerr1: ", round(meanerr1.records.value[0], 3))
    print("meanerr2: ", round(meanerr2.records.value[0], 3))


if __name__ == "__main__":
    main()
