"""
Heat Integrated Distillation Sequences (MINLPHI)

This problem describes a formulation and algorithmic procedure
for obtaining heat-integrated distillation sequences for the separation
of a given multi component feed stream into its pure components products.


Morari, M, and Grossmann, I E, Eds, Chemical Engineering Optimization
Models with GAMS. Computer Aids for Chemical Engineering Corporation,
1991.

Floudas, C A, and Paules IV, G E, A Mixed-Integer Nonlinear Programming
Formulation for the Synthesis of Heat Integrated Distillation Sequence.
Computers and Chemical Engineering 12, 6 (1988), 531-546.

======================================================================

   A MATHEMATICAL PROGRAMMING FORMULATION FOR PROCESS SYNTHESIS

===================================================================

   copyright    G.E. PAULES IV & C.A. FLOUDAS

            *** Dept. of Chemical Engineering ***
                 *** Princeton University ***
                         May 23, 1987


   Algorithm:  The Outer Approximation with Equality Relaxation
          Full Solution with Starting Point from FIXDT

======================================================================

        This formulation provides the Optimal Heat Integrated
    Distillation Sequence with Pressure as a continuous variable
                for a three component separation.
   The Outer Approximation with Equality Relaxation algorithm is
        used in the automatic solution procedure using GAMS

             Components:     a == Hexane
                             b == Benzene
                             c == Heptane

Total feed to superstructure == 396 kgmol/hr

Multicomponent feed composition:
                        a = 0.80
                        b = 0.10
                        c = 0.10

======================================================================


A Superstructure of the form ...



                       _______               _______
                      _|_    |              _|_    |
                     /   \  ( )            /   \  ( )
                     |   |___|__ A         |   |___|___ B
                     |   |                 |   |
           |---------| 1 |                 | 3 |
           |         |   |       ----------|   |
           |         |   |       |         |   |
           |         |   |_______|         |   |
           |         \___/  |  BC          \___/_______ C
    F      |           |   ( )               |     |
  -------->|           |____|                |----( )
  (ABC)    |
           |           _______               _______
           |          _|_    |              _|_    |
           |         /   \  ( )            /   \  ( )
           |         |   |___| AB          |   |___|___ A
           |         |   |   |_____________|   |
           |---------| 2 |                 | 4 |
                     |   |                 |   |
                     |   |                 |   |
                     |   |______ C         |   |_______ B
                     \___/  |              \___/   |
                       |   ( )               |    ( )
                       |____|                |_____|



 is used with binary variables representing:
    a_  the existence of columns in the sequence.
    b_  the selection of heat exchangers for heat integration.
    c_  the selection of hot and cold utilities.


Associated Reference:
_______________________
 "A Mixed-Integer Nonlinear Programming formulation for the
  synthesis of Heat-Integrated Distillation Sequences"

  C.A. Floudas and G.E. Paules IV,  1988.
  Computers and Chemical Engineering vol. 12 no. 6 pp. 531-546

Keywords: mixed integer linear programming, nonlinear programming, chemical engineering,
          distillation sequences, heat integrated distillation
"""
from sys import float_info

import numpy as np

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Container
from gamspy import Domain
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def sqr(x):
    return gams_math.power(x, 2)


def main():
    cont = Container()

    # Set some options
    cont.addOptions(
        {"limCol": 0, "limRow": 0, "bRatio": 1, "domLim": 1000, "optCr": 0}
    )

    # SETS #
    # the set of all columns and their condensers in the superstructure
    i = Set(cont, name="i", records=[f"c-{i}" for i in range(1, 5)])
    # the set of all reboilers in the superstructure
    j = Set(cont, name="j", records=[f"r-{i}" for i in range(1, 5)])
    # the set of all hot utilities available
    hu = Set(cont, name="hu", records=["lp", "ex"])
    # the set of all cold utilities available
    cu = Set(cont, name="cu", records=["cw"])
    # an index for linear fit coefficients
    n = Set(cont, name="n", records=["a", "b"])
    # the set of all intermediate products in superstructure
    m = Set(cont, name="m", records=["ab", "bc"])
    # this set maps columns to produced intermediate products
    pm = Set(
        cont, name="pm", domain=[i, m], records=[("c-1", "bc"), ("c-2", "ab")]
    )
    # this set maps columns to intermediate product feeds
    fm = Set(
        cont, name="fm", domain=[i, m], records=[("c-3", "bc"), ("c-4", "ab")]
    )
    # these sets are for dynamic control of solution algorithm
    km = Set(cont, name="km", records=[f"k-{i}" for i in range(1, 101)])
    k = Set(cont, name="k", domain=[km])
    kiter = Set(cont, name="kiter", domain=[km])
    kdynmax = Set(cont, name="kdynmax", domain=[km])

    # alias sets for condensers and reboilers
    ip = Alias(cont, name="ip", alias_with=i)
    jp = Alias(cont, name="jp", alias_with=j)

    # =====================================================================
    # Definition of "z" parameters for conditional control of model
    # used to map permissible matches between condensers and reboilers
    # and the position of columns in the superstructure
    # =====================================================================

    # PARAMETERS #
    # defines the set of leading columns in the superstructure
    zlead = Parameter(
        cont, name="zlead", domain=[i], records=[["c-1", 1], ["c-2", 1]]
    )

    # defines allowable matches of heat integration for superstructure
    # only permits heat integration between columns in the same sequence
    zcrhx = Parameter(
        cont,
        name="zcrhx",
        domain=[i, j],
        records=[
            ["c-1", "r-3", 1],
            ["c-2", "r-4", 1],
            ["c-3", "r-1", 1],
            ["c-4", "r-2", 1],
        ],
    )

    # Parameter used in pure integer constraint to permit only one
    # direction of heat integration between two columns
    # this would yield an infeasible solution but the constraint
    # is included explicitly to reduce milp solution time
    zlim = Parameter(cont, name="zlim", domain=[i, j])
    zlim[i, j] = Number(1).where[(zcrhx[i, j]) & (Ord(i) < Ord(j))]

    # relates appropriate reboiler to the condenser of same column
    # (preferably should use an alias rather than a different set)
    zcr = Parameter(cont, name="zcr", domain=[i, j])
    zcr[i, j] = Number(1).where[Ord(i) == Ord(j)]

    # =====================================================================
    # Binary variables are divided into 4 classes and variable/parameter
    # names starting with "y"
    #     ycol - column selection
    #     yhx  - heat integration exchanger matches
    #     yhu  - hot utility matches
    #     ycup - cold utiltiy matches

    # These parameters store first guess combination of binary variables
    # used to initialize minlp algorithm and parameterize the minlp
    # primal problem throughout the rest of the iterations
    # =====================================================================
    yhxp = Parameter(
        cont, name="yhxp", domain=[i, j], records=[["c-1", "r-3", 1]]
    )
    yhup = Parameter(
        cont, name="yhup", domain=[hu, j], records=[["lp", "r-1", 1]]
    )
    ycup = Parameter(
        cont,
        name="ycup",
        domain=[i, cu],
        records=[["c-1", "cw", 1], ["c-3", "cw", 1]],
    )
    ycolp = Parameter(
        cont, name="ycolp", domain=[i], records=[["c-1", 1], ["c-3", 1]]
    )

    # =====================================================================
    # These parameters store the values of the binary proposals
    # for all the iterations performed for use in integer cuts
    # and recovering optimal solution
    # =====================================================================
    yhxk = Parameter(cont, name="yhxk", domain=[i, j, km])
    yhuk = Parameter(cont, name="yhuk", domain=[hu, j, km])
    ycuk = Parameter(cont, name="ycuk", domain=[i, cu, km])
    ycolk = Parameter(cont, name="ycolk", domain=[i, km])

    # =====================================================================
    # Declaration of parameters for rest of model
    # =====================================================================
    # mass balances for each sharp separator
    spltfrc = Parameter(
        cont,
        name="spltfrc",
        domain=[i, m],
        records=[["c-1", "bc", 0.20], ["c-2", "ab", 0.90]],
    )
    # minimum condenser temperatures obtained from simulation data
    tcmin = Parameter(
        cont,
        name="tcmin",
        domain=[i],
        records=np.array([341.92, 343.01, 353.54, 341.92]),
    )
    # either hottest hot utility-dtmin or for individual separations
    # 2*dtmin below critical temperature of bottoms product
    trmax = Parameter(cont, name="trmax", domain=[j])

    trmax[j] = 1000

    # ====================================================================
    # scaled cost coefficients for distillation column fits
    # nonlinear fixed-charge cost model
    # cost = fc*y + vc*flow*temp
    # scaling factor = 1000
    # ====================================================================
    fc = Parameter(
        cont,
        name="fc",
        domain=[i],
        records=np.array([151.125, 180.003, 4.2286, 213.42]),
    )
    vc = Parameter(
        cont,
        name="vc",
        domain=[i],
        records=np.array([0.003375, 0.000893, 0.004458, 0.003176]),
    )
    thu = Parameter(
        cont, name="thu", domain=[hu], records=np.array([421.0, 373.0])
    )

    # hot utility cost coeff - gives cost in thousands of dollars per year
    # ucost = q(10e+6 kj/hr)*costhu[hu]
    costhu = Parameter(
        cont, name="costhu", domain=[hu], records=np.array([24.908, 9.139])
    )
    kf = Parameter(
        cont,
        name="kf",
        domain=[i, n],
        records=np.array(
            [[32.4, 0.0225], [25.0, 0.0130], [3.76, 0.0043], [35.1, 0.0156]]
        ),
    )
    af = Parameter(
        cont,
        name="af",
        domain=[i, n],
        records=np.array(
            [[9.541, 1.028], [12.24, 1.050], [8.756, 1.029], [9.181, 1.005]]
        ),
    )

    # =====================================================================
    # define scalar quantities for rest of model
    # =====================================================================
    totflow = Parameter(cont, name="totflow", records=396)
    u = Parameter(cont, name="u", records=1500)
    uint = Parameter(cont, name="uint", records=20)
    fchx = Parameter(cont, name="fchx", records=3.392)
    vchx = Parameter(cont, name="vchx", records=0.0893)
    htc = Parameter(cont, name="htc", records=0.0028)
    dtmin = Parameter(cont, name="dtmin", records=10.0)
    tcin = Parameter(cont, name="tcin", records=305.0)
    tcout = Parameter(cont, name="tcout", records=325.0)
    costcw = Parameter(cont, name="costcw", records=4.65)
    beta = Parameter(cont, name="beta", records=0.52)
    alpha = Parameter(cont, name="alpha", records=0.40)

    # =====================================================================
    # The parameters declared here are assigned throughout the
    # algorithmic procedures.
    # They perform the following tasks in the algorithm
    #     1) transfer of solution data between master and subproblem
    #     2) storage of solution data
    #     3) control of upper and lower bounds in milp master
    #     4) storage of optimal solution
    # =====================================================================

    # Storage of variable levels for each iteration
    # Identifier derived from name of variable with letter "k" appended
    fk = Parameter(cont, name="fk", domain=[i, km])
    qrk = Parameter(cont, name="qrk", domain=[j, km])
    qck = Parameter(cont, name="qck", domain=[i, km])
    qcrk = Parameter(cont, name="qcrk", domain=[i, j, km])
    qhuk = Parameter(cont, name="qhuk", domain=[hu, j, km])
    qcuk = Parameter(cont, name="qcuk", domain=[i, cu, km])
    tck = Parameter(cont, name="tck", domain=[i, km])
    trk = Parameter(cont, name="trk", domain=[j, km])
    lmtdk = Parameter(cont, name="lmtdk", domain=[i, km])

    zoaup = Parameter(cont, name="zoaup", records=np.inf)

    # storage of optimal binary variable combination
    # continuous variable levels are not stored separately as they
    # can be obtained from the xxxk storage parameters above
    yhxopt = Parameter(cont, name="yhxopt", domain=[i, j])
    yhuopt = Parameter(cont, name="yhuopt", domain=[hu, j])
    ycuopt = Parameter(cont, name="ycuopt", domain=[i, cu])
    ycolopt = Parameter(cont, name="ycolopt", domain=[i])

    kopt = Parameter(cont, name="kopt")

    # storage of sign() of Lagrange multiplier from nonlinear equalities
    lmtdmar = Parameter(cont, name="lmtdmar", domain=[i, km])

    # VARIABLES #

    # Free Variables
    zoau = Variable(cont, name="zoau")
    zoal = Variable(cont, name="zoal")
    vqcr = Variable(cont, name="vqcr", domain=[km])
    vqhu = Variable(cont, name="vqhu", domain=[km])
    vqcu = Variable(cont, name="vqcu", domain=[km])

    # Positive Variables
    f = Variable(cont, name="f", type="positive", domain=[i])
    qr = Variable(cont, name="qr", type="positive", domain=[j])
    qc = Variable(cont, name="qc", type="positive", domain=[i])
    qcr = Variable(cont, name="qcr", type="positive", domain=[i, j])
    qhu = Variable(cont, name="qhu", type="positive", domain=[hu, j])
    qcu = Variable(cont, name="qcu", type="positive", domain=[i, cu])
    tc = Variable(cont, name="tc", type="positive", domain=[i])
    tr = Variable(cont, name="tr", type="positive", domain=[j])
    lmtd = Variable(cont, name="lmtd", type="positive", domain=[i])
    sl1 = Variable(cont, name="sl1", type="positive", domain=[i])
    sl2 = Variable(cont, name="sl2", type="positive", domain=[i])
    s1 = Variable(cont, name="s1", type="positive", domain=[i])
    s2 = Variable(cont, name="s2", type="positive", domain=[i])
    s3 = Variable(cont, name="s3", type="positive", domain=[i])
    s4 = Variable(cont, name="s4", type="positive", domain=[i])

    # Binary Variables
    yhx = Variable(cont, name="yhx", type="binary", domain=[i, j])
    yhu = Variable(cont, name="yhu", type="binary", domain=[hu, j])
    ycu = Variable(cont, name="ycu", type="binary", domain=[i, cu])
    ycol = Variable(cont, name="ycol", type="binary", domain=[i])

    # =====================================================================
    # declaration of equations
    # for solution of the nlp subproblems:
    # early versions of GAMS did not permit binary variables to appear
    # in the constraints of a nonlinear programming problem even if
    # they appeared in linear constraints and were fixed at a bound
    # therefore -
    # constraints that contain the binary variables are duplicated:
    # one form contains the declared binary variable and the  other
    # substitutes a parameter that is assigned the current level of
    # the binary variable.  constraints that are duplicated and are to
    # appear in the nlp subproblem model have the letter "n" prepended
    # to the equation name.
    # =====================================================================
    # EQUATIONS #
    nlpobj = Equation(cont, name="nlpobj", type="regular")
    milpcon = Equation(cont, name="milpcon", type="regular", domain=[km])
    evqcr = Equation(cont, name="evqcr", type="regular", domain=[km])
    evqhu = Equation(cont, name="evqhu", type="regular", domain=[km])
    evqcu = Equation(cont, name="evqcu", type="regular", domain=[km])
    lmtdsn = Equation(cont, name="lmtdsn", type="regular", domain=[i])
    lmtdsm = Equation(cont, name="lmtdsm", type="regular", domain=[i, km])
    ntempset = Equation(cont, name="ntempset", type="regular", domain=[i])
    tempset = Equation(cont, name="tempset", type="regular", domain=[i])
    nartrex1 = Equation(cont, name="nartrex1", type="regular", domain=[i])
    artrex1 = Equation(cont, name="artrex1", type="regular", domain=[i])
    nartrex2 = Equation(cont, name="nartrex2", type="regular", domain=[i])
    artrex2 = Equation(cont, name="artrex2", type="regular", domain=[i])
    material = Equation(cont, name="material", type="regular", domain=[m])
    feed = Equation(cont, name="feed", type="regular")
    nmatlog = Equation(cont, name="nmatlog", type="regular", domain=[i])
    matlog = Equation(cont, name="matlog", type="regular", domain=[i])
    duty = Equation(cont, name="duty", type="regular", domain=[i])
    rebcon = Equation(cont, name="rebcon", type="regular", domain=[i, j])
    conheat = Equation(cont, name="conheat", type="regular", domain=[i])
    rebheat = Equation(cont, name="rebheat", type="regular", domain=[j])
    dtminlp = Equation(cont, name="dtminlp", type="regular", domain=[j])
    ndtminc = Equation(cont, name="ndtminc", type="regular", domain=[i])
    dtminc = Equation(cont, name="dtminc", type="regular", domain=[i])
    trtcdef = Equation(cont, name="trtcdef", type="regular", domain=[i, j])
    ndtmincr = Equation(cont, name="ndtmincr", type="regular", domain=[i, j])
    ndtminex = Equation(cont, name="ndtminex", type="regular", domain=[j])
    nhxclog = Equation(cont, name="nhxclog", type="regular", domain=[i, j])
    nhxhulog = Equation(cont, name="nhxhulog", type="regular", domain=[hu, j])
    nhxculog = Equation(cont, name="nhxculog", type="regular", domain=[i, cu])
    nqcqrlog = Equation(cont, name="nqcqrlog", type="regular", domain=[i])
    dtmincr = Equation(cont, name="dtmincr", type="regular", domain=[i, j])
    dtminex = Equation(cont, name="dtminex", type="regular", domain=[j])
    hxclog = Equation(cont, name="hxclog", type="regular", domain=[i, j])
    hxhulog = Equation(cont, name="hxhulog", type="regular", domain=[hu, j])
    hxculog = Equation(cont, name="hxculog", type="regular", domain=[i, cu])
    qcqrlog = Equation(cont, name="qcqrlog", type="regular", domain=[i])

    # these are the pure binary constraints of the minlp
    sequen = Equation(cont, name="sequen", type="regular", domain=[m])
    lead = Equation(cont, name="lead", type="regular")
    limutil = Equation(cont, name="limutil", type="regular", domain=[j])
    hidirect = Equation(cont, name="hidirect", type="regular", domain=[i, j])
    heat = Equation(cont, name="heat", type="regular", domain=[i])
    cuts = Equation(cont, name="cuts", type="regular", domain=[km])

    # =====================================================================
    # equations for nlp subproblems
    # note that some equations are duplicated in structure but
    # given different names in the nlp and milp. these equations
    # involve both continuous and binary variables. In older
    # versions of GAMS, it was not permissible to pose nonlinear
    # models with discrete variables present, even when their values
    # were held fixed (rmidnlp). This required two forms of the equation
    # two be declared: one with the discrete variables present (milp)
    # and one with binary variables replaced by parameters that have
    # been assigned the current levels of their associated binary
    # variables (nlp). These equations start with the letter "n"
    # in the nlp subproblems.
    # =====================================================================
    #                         capital costs

    nlpobj.expr = (
        zoau
        == alpha
        * (
            Sum(i, fc[i] * ycolp[i] + vc[i] * (tc[i] - tcmin[i]) * f[i])
            + Sum(
                Domain(i, j).where[zcrhx[i, j]],
                fchx * yhxp[i, j]
                + (vchx / htc) * (qcr[i, j] / (tc[i] - tr[j] + 1 - ycolp[i])),
            )
            + Sum(
                [i, cu],
                fchx * ycup[i, cu]
                + (vchx / htc) * (qcu[i, cu] / (lmtd[i] + 1 - ycolp[i])),
            )
            + Sum(
                [hu, j],
                fchx * yhup[hu, j]
                + (vchx / htc) * (qhu[hu, j] / (thu[hu] - tr[j])),
            )
        )
        # operating costs
        + beta
        * (
            (costcw * Sum([i, cu], qcu[i, cu]))
            + Sum([hu, j], costhu[hu] * qhu[hu, j])
        )
    )

    lmtdsn[i] = (
        lmtd[i]
        - (2 / 3) * gams_math.sqrt((tc[i] - tcin) * (tc[i] - tcout))
        - (1 / 6) * ((tc[i] - tcin) + (tc[i] - tcout))
        - (sl1[i] - sl2[i])
        == 0
    )

    nartrex1[i] = s1[i] + s2[i] + sl1[i] - u * (1 - ycolp[i]) <= 0

    nartrex2[i] = s3[i] + s4[i] + sl2[i] - u * (1 - ycolp[i]) <= 0

    ntempset[i] = (
        tc[i] + lmtd[i] + Sum(j.where[zcr[i, j]], tr[j]) - u * ycolp[i] <= 0
    )

    material[m] = (
        Sum(i.where[pm[i, m]], spltfrc[i, m] * f[i])
        - Sum(i.where[fm[i, m]], f[i])
        == 0
    )

    feed.expr = Sum(i.where[zlead[i]], f[i]) == totflow

    duty[i] = (
        qc[i]
        - (kf[i, "a"] + kf[i, "b"] * (tc[i] - tcmin[i]))
        - (s3[i] - s4[i])
        == 0
    )

    rebcon[i, j].where[zcr[i, j]] = qr[j] - qc[i] == 0

    conheat[i] = qc[i] == Sum(j.where[zcrhx[i, j]], qcr[i, j]) + Sum(
        cu, qcu[i, cu]
    )

    rebheat[j] = qr[j] == Sum(i.where[zcrhx[i, j]], qcr[i, j]) + Sum(
        hu, qhu[hu, j]
    )

    trtcdef[i, j].where[zcr[i, j]] = (
        tr[j]
        - (af[i, "a"] + af[i, "b"] * (tc[i] - tcmin[i]))
        - (s1[i] - s2[i])
        == 0
    )

    nmatlog[i] = f[i] - u * ycolp[i] <= 0

    ndtminc[i] = (tcmin[i] - tc[i] - u * (1 - ycolp[i])) <= 0

    dtminlp[j] = dtmin - (thu["lp"] - tr[j]) <= 0

    ndtmincr[i, j].where[zcrhx[i, j]] = (
        tr[j] - tc[i] - u * (1 - yhxp[i, j]) + dtmin <= 0
    )

    ndtminex[j] = dtmin - (thu["ex"] - tr[j]) - u * (1 - yhup["ex", j]) <= 0

    nhxclog[i, j].where[zcrhx[i, j]] = qcr[i, j] <= u * yhxp[i, j]

    nhxhulog[hu, j] = qhu[hu, j] <= u * yhup[hu, j]

    nhxculog[i, cu] = qcu[i, cu] <= u * ycup[i, cu]

    nqcqrlog[i] = qc[i] + Sum(j.where[zcr[i, j]], qr[j]) - u * ycolp[i] <= 0

    nlpsub = Model(
        cont,
        name="nlpsub",
        equations=[
            nlpobj,
            lmtdsn,
            nartrex1,
            nartrex2,
            ntempset,
            material,
            feed,
            nmatlog,
            duty,
            rebcon,
            conheat,
            rebheat,
            ndtminc,
            dtminlp,
            trtcdef,
            ndtmincr,
            ndtminex,
            nhxclog,
            nhxhulog,
            nhxculog,
            nqcqrlog,
        ],
        problem="nlp",
        sense="min",
        objective=zoau,
    )

    # ======================================================================
    # Define equations for milp master problems
    # Note: the nonlinear parts of the objective function related
    #       to heat exchanger area have been broken out into separate
    #       constraints to perform their linearizations, only a
    #       contribution term appears in the linearized objective
    #       function milpcon.
    # ======================================================================
    milpcon[k] = zoal >= alpha * (
        Sum(i, fc[i] * ycol[i])
        + fchx
        * (
            Sum(Domain(i, j).where[zcrhx[i, j]], yhx[i, j])
            + Sum([hu, j], yhu[hu, j])
            + Sum([i, cu], ycu[i, cu])
        )
        + Sum(
            i,
            (
                vc[i]
                * (
                    (tck[i, k] - tcmin[i]) * (f[i] - fk[i, k])
                    + fk[i, k] * (tc[i] - tcmin[i])
                )
            ),
        )
        + (vchx / htc) * (vqcr[k] + vqhu[k] + vqcu[k])
    ) + beta * (
        (costcw * Sum([i, cu], qcu[i, cu]))
        + Sum([hu, j], costhu[hu] * qhu[hu, j])
    )

    # ==========================================================================
    # these are the linearized contributions to the objective related
    # to heat exchange.  the appearance of the binary variable storage
    # parameters in the denominator of some of the expressions is done
    # to prevent division by zero during model generation for linearizations
    # done at points where the temperatures were set to zero for unused
    # columns.  the numerator is zero then also and no error is introduced.
    # ==========================================================================
    evqcr[k] = vqcr[k] == Sum(
        Domain(i, j).where[zcrhx[i, j]],
        (
            (qcrk[i, j, k] / (tck[i, k] - trk[j, k] + 1 - ycolk[i, k]))
            + (
                (1 / (tck[i, k] - trk[j, k] + 1 - ycolk[i, k]))
                * (qcr[i, j] - qcrk[i, j, k])
            )
            * ycolk[i, k]
            + (
                (
                    qcrk[i, j, k]
                    / (sqr(tck[i, k] - trk[j, k]) + 1 - ycolk[i, k])
                )
                * ((tr[j] - trk[j, k]) - (tc[i] - tck[i, k]))
            )
        ),
    )

    evqhu[k] = vqhu[k] == Sum(
        [hu, j],
        (
            (qhuk[hu, j, k] / (thu[hu] - trk[j, k]))
            + ((1 / (thu[hu] - trk[j, k])) * (qhu[hu, j] - qhuk[hu, j, k]))
            * Sum(i.where[zcr[i, j]], ycolk[i, k])
            + (
                (qhuk[hu, j, k] / sqr(thu[hu] - trk[j, k]))
                * (tr[j] - trk[j, k])
            )
        ),
    )

    evqcu[k] = vqcu[k] == Sum(
        [i, cu],
        (
            (qcuk[i, cu, k] / (lmtdk[i, k] + 1 - ycolk[i, k]))
            + (
                (1 / (lmtdk[i, k] + 1 - ycolk[i, k]))
                * (qcu[i, cu] - qcuk[i, cu, k])
            )
            * ycolk[i, k]
            - (
                (qcuk[i, cu, k] / (sqr(lmtdk[i, k]) + 1 - ycolk[i, k]))
                * (lmtd[i] - lmtdk[i, k])
            )
        ),
    )

    lmtdsm[i, k] = (
        lmtdmar[i, k]
        * (
            lmtd[i]
            - (2 / 3)
            * gams_math.sqrt((tck[i, k] - tcin) * (tck[i, k] - tcout))
            - (1 / 6) * ((tck[i, k] - tcin) + (tck[i, k] - tcout))
            - (
                (1 / 3)
                * (
                    (
                        (2 * tck[i, k] - (tcin + tcout))
                        / gams_math.sqrt(
                            sqr(tck[i, k])
                            - (tcin + tcout) * tck[i, k]
                            + (tcin * tcout)
                        )
                    )
                    + 1
                )
            )
            * (tc[i] - tck[i, k])
            - (sl1[i] - sl2[i])
        )
        <= 0
    )

    artrex1[i] = s1[i] + s2[i] + sl1[i] - u * (1 - ycol[i]) <= 0

    artrex2[i] = s3[i] + s4[i] + sl2[i] - u * (1 - ycol[i]) <= 0

    tempset[i] = (
        tc[i] + lmtd[i] + Sum(j.where[zcr[i, j]], tr[j]) - u * ycol[i] <= 0
    )

    matlog[i] = f[i] - u * ycol[i] <= 0

    dtminc[i] = (tcmin[i] - tc[i] - u * (1 - ycol[i])) <= 0

    dtmincr[i, j].where[zcrhx[i, j]] = (
        tr[j] - tc[i] - u * (1 - yhx[i, j]) + dtmin <= 0
    )

    dtminex[j] = dtmin - (thu["ex"] - tr[j]) - u * (1 - yhu["ex", j]) <= 0

    hxclog[i, j].where[zcrhx[i, j]] = qcr[i, j] <= u * yhx[i, j]

    hxhulog[hu, j] = qhu[hu, j] <= u * yhu[hu, j]

    hxculog[i, cu] = qcu[i, cu] <= u * ycu[i, cu]

    qcqrlog[i] = qc[i] + Sum(j.where[zcr[i, j]], qr[j]) - u * ycol[i] <= 0

    # pure binary constraints
    # material balances determine sequence
    sequen[m] = (
        Sum(i.where[pm[i, m]], ycol[i]) - Sum(i.where[fm[i, m]], ycol[i]) == 0
    )

    # select 1 sequence
    lead.expr = Sum(i.where[zlead[i]], ycol[i]) == 1

    # limit choice of hot utility to 1
    limutil[j] = Sum(hu, yhu[hu, j]) <= 1

    # only one of the mutual heat integration binaries can be 1
    hidirect[i, j].where[zlim[i, j]] = (
        yhx[i, j]
        + Sum(
            Domain(ip, jp).where[(Ord(ip) == Ord(j)) & (Ord(jp) == Ord(i))],
            yhx[ip, jp],
        )
        <= 1
    )

    # if a column doesn't exist then all binary variables associated
    # with it must also be set to zero
    heat[i] = (
        Sum(
            j.where[zcrhx[i, j]],
            yhx[i, j]
            + Sum(
                Domain(ip, jp).where[
                    (Ord(ip) == Ord(j)) & (Ord(jp) == Ord(i))
                ],
                yhx[ip, jp],
            ),
        )
        + Sum([hu, j], yhu[hu, j].where[zcr[i, j]])
        + Sum(cu, ycu[i, cu])
        - uint * ycol[i]
        <= 0
    )

    # integer cuts
    cuts[k] = (
        Sum(i, gams_math.sign(ycolk[i, k] - 0.5) * ycol[i])
        + Sum(
            Domain(i, j).where[zcrhx[i, j]],
            gams_math.sign(yhxk[i, j, k] - 0.5) * yhx[i, j],
        )
        + Sum([hu, j], gams_math.sign(yhuk[hu, j, k] - 0.5) * yhu[hu, j])
        + Sum([i, cu], gams_math.sign(ycuk[i, cu, k] - 0.5) * ycu[i, cu])
        <= Sum(i, ycolk[i, k])
        + Sum(Domain(i, j).where[zcrhx[i, j]], yhxk[i, j, k])
        + Sum([hu, j], yhuk[hu, j, k])
        + Sum([i, cu], ycuk[i, cu, k])
        - 1
    )

    # ======================================================================
    # declare the milp master problem
    # ======================================================================
    master = Model(
        cont,
        name="master",
        equations=[
            milpcon,
            evqcr,
            evqhu,
            evqcu,
            lmtdsm,
            artrex1,
            artrex2,
            tempset,
            material,
            feed,
            matlog,
            duty,
            rebcon,
            conheat,
            rebheat,
            dtminc,
            dtminlp,
            trtcdef,
            dtmincr,
            dtminex,
            hxclog,
            hxhulog,
            hxculog,
            qcqrlog,
            sequen,
            lead,
            limutil,
            hidirect,
            heat,
            cuts,
        ],
        problem="mip",
        sense="min",
        objective=zoal,
    )

    # =====================================================================
    # all declarations made, start algorithmic procedures

    # initialize the optimal storage parameters to 1st guess
    # =====================================================================
    yhxopt[i, j] = yhxp[i, j]
    yhuopt[hu, j] = yhup[hu, j]
    ycuopt[i, cu] = ycup[i, cu]
    ycolopt[i] = ycolp[i]
    kopt.assign = 1

    # ======================================================================
    # assign the initial configuration to the binary proposal parameter
    # ======================================================================
    kiter["k-1"] = Number(1)

    yhxk[i, j, kiter] = yhxp[i, j]
    yhuk[hu, j, kiter] = yhup[hu, j]
    ycuk[i, cu, kiter] = ycup[i, cu]
    ycolk[i, kiter] = ycolp[i]
    yhx.l[i, j] = yhxp[i, j]
    yhu.l[hu, j] = yhup[hu, j]
    ycu.l[i, cu] = ycup[i, cu]
    ycol.l[i] = ycolp[i]

    # set an arbitrary initial lower bound
    zoal.l.assign = -10e6

    # ======================================================================
    # give the continuous variables a starting point for 1st nlp
    # ======================================================================
    tr.l["r-1"] = 410
    tc.l["c-1"] = 390
    tc.l["c-3"] = 360
    tr.l["r-3"] = 380
    tc.l["c-2"] = 0
    tr.l["r-2"] = 0
    tc.l["c-4"] = 0
    tr.l["r-4"] = 0
    f.l["c-1"] = totflow
    lmtd.l["c-1"] = 75
    lmtd.l["c-3"] = 25
    lmtd.l["c-2"] = 0
    lmtd.l["c-4"] = 0
    qr.l["r-2"] = 0
    qc.l["c-2"] = 0
    qr.l["r-4"] = 0
    qc.l["c-4"] = 0

    # ======================================================================
    # add bounds on tc. A sqrt in equation lmtdsn is defined for tc > tcout
    # and for tc < tcin. The relevant interval is determined for each
    # element of tc based on the initial values given above.
    # ======================================================================
    tc.lo["c-1"] = tcout + 1
    tc.up["c-2"] = tcin - 1
    tc.lo["c-3"] = tcout + 1
    tc.up["c-4"] = tcin - 1

    # ======================================================================
    # bound the reboiler temperatures by their maximum allowable
    # ======================================================================
    tr.up[j] = trmax[j]

    # ======================================================================
    # initialize the dynamic sets for algorithm control
    # ======================================================================
    k[km] = Number(0)
    kiter[km] = Number(0)
    kdynmax[km] = Number(1)

    # ======================================================================
    # major driving loop of algorithm
    # ======================================================================

    for idx, _ in enumerate(kdynmax.toList()):
        #  update the dynamic iteration sets
        # -set kiter to contain only the current iteration element
        # -add to k the current iteration element
        kiter[km] = Number(1).where[Ord(km) == idx + 1]
        k[kiter] = Number(1)

        #  store the current binary combination
        yhxk[i, j, kiter] = yhx.l[i, j]
        yhuk[hu, j, kiter] = yhu.l[hu, j]
        ycuk[i, cu, kiter] = ycu.l[i, cu]
        ycolk[i, kiter] = ycol.l[i]

        #  set the current combination parameters that appear in the nlp constraints
        yhxp[i, j] = yhx.l[i, j]
        yhup[hu, j] = yhu.l[hu, j]
        ycup[i, cu] = ycu.l[i, cu]
        ycolp[i] = ycol.l[i]
        zoal.lo.assign = zoal.l

        # ======================================================================
        # the current levels of the lmtds are moved away from zero
        # to prevent evaluation errors in the next nlp subproblem
        # ======================================================================
        lmtd.l[i] = lmtd.l[i] + 1

        cont.addOptions({"resLim": 15})
        #  solve the nlp subproblem
        nlpsub.solve()

        #  resolve with Conopt to get marginals for lmtdsn, if not provided by used NLP solver *****
        if nlpsub.marginals == 0:
            cont.addOptions({"nlp": "conopt"})
            nlpsub.solve()
            cont.addOptions({"nlp": "%system.nlp%"})

        # ======================================================================
        # update the optimal solution storage parameters if new nlp
        # objective function value is less than the incumbent
        # ======================================================================
        if zoau.l < zoaup:
            yhxopt[i, j] = yhx.l[i, j]
            yhuopt[hu, j] = yhu.l[hu, j]
            ycuopt[i, cu] = ycu.l[i, cu]
            ycolopt[i] = ycol.l[i]
            kopt.assign = idx + 1

        # ======================================================================
        # assign the solution levels of the variables that appear in the
        # nonlinear equations to their corresponding storage parameters
        # ======================================================================
        fk[i, kiter] = f.l[i]
        qrk[j, kiter] = qr.l[j]
        qck[i, kiter] = qc.l[i]
        qcrk[i, j, kiter] = qcr.l[i, j]
        qhuk[hu, j, kiter] = qhu.l[hu, j]
        qcuk[i, cu, kiter] = qcu.l[i, cu]
        tck[i, kiter] = tc.l[i]
        trk[j, kiter] = tr.l[j]
        lmtdk[i, kiter] = lmtd.l[i]

        # ======================================================================
        # assign the sign of marginal values of the nonlinear equalities
        # to the storage parameter lmtdmar
        # ======================================================================
        lmtdmar[i, kiter] = (
            Number(-1)
            * gams_math.sign(lmtdsn.m[i]).where[
                lmtdsn.m[i] != float_info.epsilon
            ]
        )

        # ======================================================================
        # store the smallest nlp objective value for upper bound on master
        # ======================================================================
        zoaup.assign = gams_math.min(zoaup, zoau.l)
        zoal.up.assign = zoaup
        #  protect against numerical errors introduced by the solver
        zoal.lo.assign = gams_math.min(zoal.lo, zoal.up)

        #  now solve the milp master problem
        master.solve()

        print(
            "new binary combination: \n\n",
            f"ycol: {ycol.toDict()}\n\n",
            f"yhx: {yhx.toDict()}\n\n",
            f"yhu: {yhu.toDict()}\n\n",
            f"ycu: {ycu.toDict()}\n\n",
        )

        # ======================================================================
        # check stopping criterion:
        # master problem integer infeasible
        # ======================================================================
        if master.status in [4.0, 10.0, 19.0]:
            kdynmax[km] = Number(0)
            print(
                "stopping criterion met: \n\n",
                f"zoaup: {round(zoaup.toValue(),3)}\n\n",
                f"yhxopt: {yhxopt.toDict()}\n\n",
                f"yhuopt: {yhuopt.toDict()}\n\n",
                f"ycuopt: {ycuopt.toDict()}\n\n",
                f"ycolopt: {ycolopt.toDict()}\n\n",
                f"kopt: {kopt.toValue()}\n\n",
            )
            break


if __name__ == "__main__":
    main()
