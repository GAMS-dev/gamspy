"""
## GAMSSOURCE: https://www.gams.com/latest/noalib_ml/libhtml/noalib_refrigeration.html
## LICENSETYPE: Demo
## MODELTYPE: NLP


Optimal design of industrial refrigeration system

Paul H and Tay, Optimal design of an industrial refrigeration system.
Proc. of Int. Conf. on Optimization Techniques and Applications, 1987,
pp.427-435.

Pant, M., Thangaraj, R., Singh, V.P., (2009) Optimization of mechanical
design problems using improved differential evolution algorithm.
International Journal of Recent Trends in Engineering, vol.1, No.5,
May 2009, pp.21-25.
"""

from __future__ import annotations

import os

from gamspy import Container, Equation, Model, Parameter, Sense, Variable


def main():
    m = Container(
        system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
    )

    # VARIABLES #
    x = [Variable(m, name=f"x{i+1}") for i in range(14)]

    # EQUATIONS #
    e = [Equation(m, name=f"e{i+1}") for i in range(15)]

    # Objective function to be minimized:
    eobj = (
        63098.88 * x[1] * x[3] * x[11]
        + 5441.5 * x[11] * x[1] ** 2
        + 115055.5 * x[5] * (x[1] ** 1.664)
        + 6172.27 * x[5] * x[1] ** 2
        + 63098.88 * x[0] * x[2] * x[10]
        + 5441.5 * x[10] * x[0] ** 2
        + 115055.5 * x[4] * (x[0] ** 1.664)
        + 6172.27 * x[4] * x[0] ** 2
        + 140.53 * x[0] * x[10]
        + 281.29 * x[2] * x[10]
        + 70.26 * x[0] ** 2
        + 281.29 * x[0] * x[2]
        + 281.29 * x[2] ** 2
        + 14437
        * (x[7] ** 1.8812)
        * (x[11] ** 0.3424)
        * x[6]
        * x[9]
        * (x[0] ** 2)
        / (x[8] * x[13])
        + 20470.2 * (x[6] ** 2.893) * (x[10] * 0.316) * (x[0] * 82)
    )

    # Constaints:
    e[0][...] = 1.524 / x[6] <= 1
    e[1][...] = 1.524 / x[7] <= 1
    e[2][...] = 0.07789 * x[0] - 2 * x[8] / x[6] <= 1
    e[3][...] = (
        7.05305 * (x[0] ** 2) * x[9] / (x[1] * x[7] * x[8] * x[13]) <= 1
    )
    e[4][...] = 0.0833 * x[13] / x[12] <= 1
    e[5][...] = (
        47.136 * x[11] * (x[1] ** 0.333) / x[9]
        - 1.333 * x[7] * (x[12] ** 2.1195)
        + 62.08 * (x[12] ** 2.1195) * (x[7] ** 0.2) / (x[9] * x[11])
        <= 1
    )
    e[6][...] = 0.04771 * x[9] * (x[7] ** 1.8812) * (x[11] ** 0.3424) <= 1
    e[7][...] = 0.0488 * x[8] * (x[6] ** 1.893) * (x[10] ** 0.316) <= 1
    e[8][...] = 0.0099 * x[0] / x[2] <= 1
    e[9][...] = 0.0193 * x[1] / x[3] <= 1
    e[10][...] = 0.0298 * x[0] / x[4] <= 1
    e[11][...] = 0.056 * x[1] / x[5] <= 1
    e[12][...] = 2 / x[8] <= 1
    e[13][...] = 2 / x[9] <= 1
    e[14][...] = x[11] / x[10] <= 1

    # Bounds on variables:
    for v in m.getVariables():
        v.lo = 0.001
        v.up = 5

    refrigeration = Model(
        m,
        name="refrigeration",
        equations=m.getEquations(),
        problem="nlp",
        sense=Sense.MIN,
        objective=eobj,
    )
    refrigeration.solve()

    # REPORTING PARAMETER
    rep = Parameter(m, name="rep", domain=["*", "*"])
    for i in range(14):
        rep[f"x{i+1}", "value"] = x[i].l

    print(
        "Objective Function Value: ",
        round(refrigeration.objective_value, 4),
        "\n",
    )
    print("Solution Summary:\n", rep.pivot().round(3))

    # End refrigeration


if __name__ == "__main__":
    main()
