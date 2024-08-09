"""
## LICENSETYPE: Demo
## MODELTYPE: NLP


Floudas, C.A., Pardalos, P.M., et al. Handbook of test problems in local and global
optimization. Kluwer Academic Publishers, Dordrecht, 1999
Chapter 8. Section: Phase and Chemical Equilibrium Problems-Equations of State
Test Problem 1, pp. 180-181.

Van der Waals equation, Tangent Plane distance minimization
Ternary System
"""

from __future__ import annotations

import gamspy.math as gams_math
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Parameter,
    Set,
    Sum,
    Variable,
)


def main():
    m = Container()

    # SET #
    i = Set(m, name="i", records=["1", "2", "3"], description="components")

    # ALIAS #
    j = Alias(m, name="j", alias_with=i)

    # PARAMETERS #
    feedmf = Parameter(
        m,
        name="feedmf",
        domain=i,
        description="mole fraction of component i in candidate phase",
    )
    feedz = Parameter(
        m, name="feedz", description="compressibility of candidate phase"
    )
    feedfc = Parameter(
        m,
        name="feedfc",
        domain=i,
        description="fugacity coefficient of component i in candidate phase",
    )
    b = Parameter(
        m,
        name="b",
        domain=i,
        description="Van der Waals pure-component parameter",
    )
    a = Parameter(
        m,
        name="a",
        domain=[i, j],
        description="Van der Waals mixture parameter",
    )

    feedmf["1"] = 0.83
    feedmf["2"] = 0.085
    feedmf["3"] = 0.085

    feedz[...] = 0.55716

    feedfc["1"] = -0.244654
    feedfc["2"] = -1.33572
    feedfc["3"] = -0.457869

    b["1"] = 0.14998
    b["2"] = 0.14998
    b["3"] = 0.14998

    a["1", "1"] = 0.37943
    a["1", "2"] = 0.75885
    a["1", "3"] = 0.48991
    a["2", "1"] = 0.75885
    a["2", "2"] = 0.88360
    a["2", "3"] = 0.23612
    a["3", "1"] = 0.48991
    a["3", "2"] = 0.23612
    a["3", "3"] = 0.63263

    # VARIABLES #
    x = Variable(
        m,
        name="x",
        domain=i,
        description="mole fractio of component i in incipient phase",
    )
    z = Variable(m, name="z", description="compressibility of incipient phase")
    amix = Variable(
        m,
        name="amix",
        description="mixture A parameter (function of composition)",
    )
    bmix = Variable(
        m,
        name="bmix",
        description="mixture B parameter (function of composition)",
    )

    # EQUATIONS #
    eos = Equation(
        m,
        name="eos",
        type="regular",
        description="equation of state constraint",
    )
    defa = Equation(
        m, name="defa", type="regular", description="definition of Amix"
    )
    defb = Equation(
        m, name="defb", type="regular", description="definition of Bmix"
    )
    molesum = Equation(
        m,
        name="molesum",
        type="regular",
        description="mole fractions sum to 1",
    )

    # Objective function to be minimized: tangent plane distance
    dist = (
        Sum(i, x[i] * gams_math.log(x[i]))
        + bmix / (z - bmix)
        - gams_math.log(z - bmix)
        - 2.0 * amix / z
        - Sum(i, x[i] * (gams_math.log(feedmf[i]) + feedfc[i]))
    )

    # Constraints:
    eos[...] = (
        gams_math.power(z, 3)
        - (bmix + 1) * gams_math.power(z, 2)
        + amix * z
        - amix * bmix
        == 0
    )

    defa[...] = amix - Sum(i, Sum(j, a[i, j] * x[i] * x[j])) == 0

    defb[...] = bmix - Sum(i, b[i] * x[i]) == 0

    molesum[...] = Sum(i, x[i]) == 1.0

    # Simple Bounds of variables
    z.lo = 0.001
    x.lo["1"] = 0.001
    x.lo["2"] = 0.001
    x.lo["3"] = 0.001

    phase = Model(
        m,
        name="phase",
        equations=m.getEquations(),
        problem="nlp",
        sense="min",
        objective=dist,
    )
    phase.solve()

    print("x:  \n", x.toDict(), "\n")
    print("z:  \n", round(z.toValue(), 4), "\n")

    # End phase


if __name__ == "__main__":
    main()
