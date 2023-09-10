"""
Stability of a wire guided Daimler-Benz 0305 bus.

References:
Neculai Andrei, "Models, Test Problems and Applications for
Mathematical Programming". Technical Press, Bucharest, 2003.

Application A38, page 402.
Ackermann, J., et al. " Robust gamma-stability analysis in a plant
parameter space. Automatica, vol. 27, 1991, pp.75.

Floudas, C.A., Pardalos, P.M., et al. "Handbook of Test Problems in
Local and Global Optimization". Kluwer Academic Publishers, Dordrecht,
1999.
Section 7.3.5. Test problem 15, page 102.
"""

from gamspy import Variable, Equation, Model, Container
import gamspy.math as gams_math
from gamspy import Problem, Sense


def main():
    m = Container()

    # Variable
    q1 = Variable(m, name="q1")
    q2 = Variable(m, name="q2")
    w = Variable(m, name="w")
    k = Variable(m, name="k")

    # Equation
    g1 = Equation(m, name="g1")
    g2 = Equation(m, name="g2")
    b1l = Equation(m, name="b1l")
    b1u = Equation(m, name="b1u")
    b2l = Equation(m, name="b2l")
    b2u = Equation(m, name="b2u")

    g1.expr = (gams_math.power(q1, 2)) * (gams_math.power(q2, 2)) * (
        gams_math.power(w, 8)
    ) - (
        1.25 * 1000 * (gams_math.power(q1, 2)) * (gams_math.power(q2, 2))
        + 16.8 * (gams_math.power(q1, 2)) * q2
        + 53.9 * 1000 * q1 * q2
        + 270 * 1000
    ) * (
        gams_math.power(w, 6)
    ) + (
        1.45 * (10**6) * (gams_math.power(q1, 2)) * q2
        + 16.8 * (10**6) * q1 * q2
        + (10**6) * 338
    ) * (
        gams_math.power(w, 4)
    ) - (
        5.72 * (10**6) * (gams_math.power(q1, 2)) * q2
        + 113 * (10**6) * (gams_math.power(q1, 2))
        + 4250 * (10**6) * q1
    ) * (
        gams_math.power(w, 2)
    ) + (
        453 * (10**6) * (gams_math.power(q1, 2))
    ) == 0
    g2.expr = (
        50 * (gams_math.power(q1, 2)) * (gams_math.power(q2, 2))
        + 1080 * q1 * q2
    ) * (gams_math.power(w, 6)) - (
        15.6 * 1000 * (gams_math.power(q1, 2)) * (gams_math.power(q2, 2))
        + 840 * (gams_math.power(q1, 2)) * q2
        + 1.35 * (10**6) * q1 * q2
        + (10**6) * 13.5
    ) * (
        gams_math.power(w, 4)
    ) + (
        6.93 * (10**6) * (gams_math.power(q1, 2)) * q2
        + 911 * (10**6) * q1
        + (10**6) * 4220
    ) * (
        gams_math.power(w, 2)
    ) - (
        528 * (10**6) * (gams_math.power(q1, 2)) + 3640 * (10**6) * q1
    ) == 0

    b1l.expr = 17.5 - 14.5 * k <= q1
    b1u.expr = q1 <= 17.5 + 14.5 * k
    b2l.expr = 20.0 - 15.0 * k <= q2
    b2u.expr = q2 <= 20.0 + 15.0 * k

    q1.lo.assign = 0
    q1.up.assign = 2
    q2.lo.assign = 0
    q2.up.assign = 2
    w.lo.assign = 0
    w.up.assign = 2
    k.lo.assign = 0
    k.up.assign = 2
    q1.l.assign = 0.1
    q2.l.assign = 0.1
    w.l.assign = 0.1
    k.l.assign = 0.1

    benz = Model(
        m,
        name="benz",
        equations=[g1, g2, b1l, b1u, b2l, b2u],
        problem=Problem.NLP,
        sense=Sense.MIN,
        objective=k,
    )
    benz.solve()
    print(benz.objective_value)


if __name__ == "__main__":
    main()
