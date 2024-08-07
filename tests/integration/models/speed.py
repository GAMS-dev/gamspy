"""
## GAMSSOURCE: https://www.gams.com/latest/noalib_ml/libhtml/noalib_speed.html
## LICENSETYPE: Demo
## MODELTYPE: NLP


Minimization of the weight of a speed reducer

The weight of the speed reducer is to be minimized subject to constraints
on bending stress of the gear teeth, surface stress, transverse deflections
of the shafts and stresses in the shaft.

Datseris, P., Weight minimization of a speed reducer by heuristic
and decomposition technique. Mechanism and Machine Theory, vol.17, 1982,
pp. 255-262.

Aguirre, A.H., Munoz Zavala, A.E., Villa Diharce, E., Botello Rionada, S.,
COPSO: Constrained optimization via PSO algorithm. Comunicacion Tecnica
No I-07-04/22-02-2007. Center for Research in Mathematics (CIMAT), Mexico.
"""

from __future__ import annotations

import gamspy.math as gams_math
from gamspy import Container, Equation, Model, Variable


def main():
    m = Container()

    # VARIABLES #
    x1 = Variable(m, name="x1")
    x2 = Variable(m, name="x2")
    x3 = Variable(m, name="x3")
    x4 = Variable(m, name="x4")
    x5 = Variable(m, name="x5")
    x6 = Variable(m, name="x6")
    x7 = Variable(m, name="x7")

    # EQUATION #
    g1 = Equation(m, name="g1", type="regular")
    g2 = Equation(m, name="g2", type="regular")
    g3 = Equation(m, name="g3", type="regular")
    g4 = Equation(m, name="g4", type="regular")
    g5 = Equation(m, name="g5", type="regular")
    g6 = Equation(m, name="g6", type="regular")
    g7 = Equation(m, name="g7", type="regular")
    g8 = Equation(m, name="g8", type="regular")
    g9 = Equation(m, name="g9", type="regular")
    g10 = Equation(m, name="g10", type="regular")
    g11 = Equation(m, name="g11", type="regular")

    # Objective function to be minimized:
    g = (
        0.7854
        * x1
        * gams_math.power(x2, 2)
        * (3.3333 * gams_math.power(x3, 2) + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (gams_math.power(x6, 2) + gams_math.power(x7, 2))
        + 7.4777 * (gams_math.power(x6, 3) + gams_math.power(x7, 3))
        + 0.7854 * (x4 * gams_math.power(x6, 2) + x5 * gams_math.power(x7, 2))
    )

    # Constraints:
    g1[...] = 27 / (x1 * gams_math.power(x2, 2) * x3) - 1 <= 0

    g2[...] = (
        397.5 / (x1 * gams_math.power(x2, 2) * gams_math.power(x3, 2)) - 1 <= 0
    )

    g3[...] = (1.93 * gams_math.power(x4, 3)) / (
        x2 * x3 * gams_math.power(x6, 4)
    ) - 1 <= 0

    g4[...] = (1.93 * gams_math.power(x5, 3)) / (
        x2 * x3 * gams_math.power(x7, 4)
    ) - 1 <= 0

    g5[...] = (
        gams_math.sqrt(gams_math.power((745 * x4) / (x2 * x3), 2) + 16900000)
    ) / (110 * gams_math.power(x6, 3)) - 1 <= 0

    g6[...] = (
        gams_math.sqrt(gams_math.power((745 * x5) / (x2 * x3), 2) + 15750000)
    ) / (85 * gams_math.power(x7, 3)) - 1 <= 0

    g7[...] = (x2 * x3) / 40 - 1 <= 0

    g8[...] = (5 * x2) / x1 - 1 <= 0

    g9[...] = x1 / (12 * x2) - 1 <= 0

    g10[...] = (1.5 * x6 + 1.9) / x4 - 1 <= 0

    g11[...] = (1.1 * x7 + 1.9) / x5 - 1 <= 0

    # Bounds on variables
    x1.lo = 2.6
    x1.up = 3.6
    x2.lo = 0.7
    x2.up = 0.8
    x3.lo = 17
    x3.up = 28
    x4.lo = 7.3
    x4.up = 8.3
    x5.lo = 7.8
    x5.up = 8.3
    x6.lo = 2.9
    x6.up = 3.9
    x7.lo = 5.0
    x7.up = 5.5

    speed = Model(
        m,
        name="speed",
        equations=m.getEquations(),
        problem="nlp",
        sense="MIN",
        objective=g,
    )
    speed.solve()

    print("Objective Function Value:  ", round(speed.objective_value, 4))

    # End speed


if __name__ == "__main__":
    main()
