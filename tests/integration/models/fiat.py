"""
   Analysis of the stability margin of the spark ignition engine
   Fiat Dedra.

   References:
   B.R. Barmish, New tools for robustness of linear systems.
   McMillan Publishing Company, New York, 1994.

   M. Abate, B. Barmish, C. Murillo-Sanchez, R. Tempo, Application of
   some new tools to robust stability analysis of spark ignition engines:
   A case study. IEEE Trans. Contr. syst. tech., vol.2, 1994, pp. 22.

   Neculai Andrei, "Models, Test Problems and Applications for
   Mathematical Programming". Technical Press, Bucharest, 2003.
   Application A41, page 407.

   Floudas, C.A., Pardalos, P.M., et al. "Handbook of Test Problems in
   Local and Global Optimization". Kluwer Academic Publishers, Dordrecht,
   1999.
   Problem 7.3.6. Test problem 16, page 103.
"""
import gamspy.math as gams_math
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Variable


def main():
    m = Container()

    # VARIABLES #
    q1 = Variable(m, name="q1")
    q2 = Variable(m, name="q2")
    q3 = Variable(m, name="q3")
    q4 = Variable(m, name="q4")
    q5 = Variable(m, name="q5")
    q6 = Variable(m, name="q6")
    q7 = Variable(m, name="q7")
    a0 = Variable(m, name="a0")
    a1 = Variable(m, name="a1")
    a2 = Variable(m, name="a2")
    a3 = Variable(m, name="a3")
    a4 = Variable(m, name="a4")
    a5 = Variable(m, name="a5")
    a6 = Variable(m, name="a6")
    a7 = Variable(m, name="a7")
    objval = Variable(
        m, name="objval", description="objective function variable"
    )
    w = Variable(m, name="w", description="frequency")
    k = Variable(m, name="k", description="stability margin")

    # EQUATIONS #
    f = Equation(m, name="f", type="regular", description="Objective function")
    g1 = Equation(m, name="g1", type="regular")
    g2 = Equation(m, name="g2", type="regular")
    b1l = Equation(m, name="b1l", type="regular")
    b1u = Equation(m, name="b1u", type="regular")
    b2l = Equation(m, name="b2l", type="regular")
    b2u = Equation(m, name="b2u", type="regular")
    b3l = Equation(m, name="b3l", type="regular")
    b3u = Equation(m, name="b3u", type="regular")
    b4l = Equation(m, name="b4l", type="regular")
    b4u = Equation(m, name="b4u", type="regular")
    b5l = Equation(m, name="b5l", type="regular")
    b5u = Equation(m, name="b5u", type="regular")
    b6l = Equation(m, name="b6l", type="regular")
    b6u = Equation(m, name="b6u", type="regular")
    b7l = Equation(m, name="b7l", type="regular")
    b7u = Equation(m, name="b7u", type="regular")
    ga0 = Equation(m, name="ga0", type="regular")
    ga1 = Equation(m, name="ga1", type="regular")
    ga2 = Equation(m, name="ga2", type="regular")
    ga3 = Equation(m, name="ga3", type="regular")
    ga4 = Equation(m, name="ga4", type="regular")
    ga5 = Equation(m, name="ga5", type="regular")
    ga6 = Equation(m, name="ga6", type="regular")
    ga7 = Equation(m, name="ga7", type="regular")

    f.expr = objval == k

    g1.expr = (
        -a6 * gams_math.power(w, 6)
        + a4 * gams_math.power(w, 4)
        - a2 * gams_math.power(w, 2)
        + a0
        == 0
    )
    g2.expr = (
        a7 * gams_math.power(w, 6)
        - a5 * gams_math.power(w, 4)
        + a3 * gams_math.power(w, 2)
        - a1
        == 0
    )

    b1l.expr = 3.4329 - 1.02721 * k <= q1
    b1u.expr = q1 <= 3.4320 + 1.02721 * k
    b2l.expr = 0.1627 - 0.06 * k <= q2
    b2u.expr = q2 <= 0.1627 + 0.06 * k
    b3l.expr = 0.1139 - 0.0782 * k <= q3
    b3u.expr = q3 <= 0.1139 + 0.0782 * k
    b4l.expr = 1.2539 - 0.3068 * k <= q4
    b4u.expr = q4 <= 1.2539 + 0.3068 * k
    b5l.expr = 0.0208 - 0.0108 * k <= q5
    b5u.expr = q5 <= 0.0208 + 0.08 * k
    b6l.expr = 5.0247 - 2.4715 * k <= q6
    b6u.expr = q6 <= 5.0247 + 2.4715 * k
    b7l.expr = 1.0 - 2 * k <= q7
    b7u.expr = q7 <= 1.0 + 2 * k

    ga0.expr = (
        a0
        == 6.82079e-05 * q1 * q3 * gams_math.power(q4, 2)
        + 6.82079e-05 * q1 * q2 * q4 * q5
    )

    ga1.expr = a1 == (
        0.00076176 * gams_math.power(q2, 2) * gams_math.power(q5, 2)
        + 0.00076176 * gams_math.power(q3, 2) * gams_math.power(q4, 2)
        + 0.000402141 * q1 * q2 * gams_math.power(q5, 2)
        + 0.00337606 * q1 * q3 * gams_math.power(q4, 2)
        + 6.82079e-05 * q1 * q4 * q5
        + 0.00051612 * gams_math.power(q2, 2) * q5 * q6
        + 0.00337606 * q1 * q2 * q4 * q5
        + 6.82079e-05 * q1 * q2 * q4 * q7
        + 6.28987e-05 * q1 * q2 * q5 * q6
        + 0.000402141 * q1 * q3 * q4 * q5
        + 6.28987e-05 * q1 * q3 * q4 * q6
        + 0.00152352 * q2 * q3 * q4 * q5
        + 0.00051612 * q2 * q3 * q4 * q6
    )

    ga2.expr = a2 == (
        0.000402141 * q1 * gams_math.power(q5, 2)
        + 0.00152352 * q2 * gams_math.power(q5, 2)
        + 0.0552 * gams_math.power(q2, 2) * gams_math.power(q5, 2)
        + 0.0552 * gams_math.power(q3, 2) * gams_math.power(q4, 2)
        + 0.0189477 * q1 * q2 * gams_math.power(q5, 2)
        + 0.034862 * q1 * q3 * gams_math.power(q4, 2)
        + 0.00336706 * q1 * q4 * q5
        + 6.82079e-05 * q1 * q4 * q7
        + 6.28987e-05 * q1 * q5 * q6
        + 0.00152352 * q3 * q4 * q5
        + 0.00051612 * q3 * q4 * q6
        - 0.00234048 * gams_math.power(q3, 2) * q4 * q6
        + 0.034862 * q1 * q2 * q4 * q5
        + 0.0237398 * gams_math.power(q2, 2) * q5 * q6
        + 0.00152352 * gams_math.power(q2, 2) * q5 * q7
        + 0.00051612 * gams_math.power(q2, 2) * q6 * q7
        + 0.00336706 * q1 * q2 * q4 * q7
        + 0.00287416 * q1 * q2 * q5 * q6
        + 0.000804282 * q1 * q2 * q5 * q7
        + 6.28987e-05 * q1 * q2 * q6 * q7
        + 0.0189477 * q1 * q3 * q4 * q5
        + 0.00287416 * q1 * q3 * q4 * q6
        + 0.000402141 * q1 * q3 * q4 * q7
        + 0.1104 * q2 * q3 * q4 * q5
        + 0.0237398 * q2 * q3 * q4 * q6
        + 0.00152352 * q2 * q3 * q4 * q7
        - 0.00234048 * q2 * q3 * q5 * q6
        + 0.00103224 * q2 * q5 * q6
    )

    ga3.expr = a3 == (
        0.189477 * q1 * gams_math.power(q5, 2)
        + 0.1104 * q2 * gams_math.power(q5, 2)
        + 0.00051612 * q5 * q6
        + gams_math.power(q2, 2) * gams_math.power(q5, 2)
        + 0.00076176 * gams_math.power(q2, 2) * gams_math.power(q7, 2)
        + gams_math.power(q3, 2) * gams_math.power(q4, 2)
        + 0.1586 * q1 * q2 * gams_math.power(q5, 2)
        + 0.000402141 * q1 * q2 * gams_math.power(q7, 2)
        + 0.0872 * q1 * q3 * gams_math.power(q4, 2)
        + 0.034862 * q1 * q4 * q5
        + 0.00336706 * q1 * q4 * q7
        + 0.00287416 * q1 * q5 * q6
        + 6.28987e-05 * q1 * q6 * q7
        + 0.00103224 * q2 * q6 * q7
        + 0.1104 * q3 * q4 * q5
        + 0.0237398 * q3 * q4 * q6
        + 0.00152352 * q3 * q4 * q7
        - 0.00234048 * q3 * q5 * q6
        + 0.1826 * gams_math.power(q2, 2) * q5 * q6
        + 0.1104 * gams_math.power(q2, 2) * q5 * q7
        + 0.0237398 * gams_math.power(q2, 2) * q6 * q7
        - 0.0848 * gams_math.power(q3, 2) * q4 * q6
        + 0.0872 * q1 * q2 * q4 * q5
        + 0.034862 * q1 * q2 * q4 * q7
        + 0.0215658 * q1 * q2 * q5 * q6
        + 0.0378954 * q1 * q2 * q5 * q7
        + 0.00287416 * q1 * q2 * q6 * q7
        + 0.1586 * q1 * q3 * q4 * q5
        + 0.0215658 * q1 * q3 * q4 * q6
        + 0.0189477 * q1 * q3 * q4 * q7
        + 2 * q2 * q3 * q4 * q5
        + 0.1826 * q2 * q3 * q4 * q6
        + 0.1104 * q2 * q3 * q4 * q7
        - 0.0848 * q2 * q3 * q5 * q6
        - 0.00234048 * q2 * q3 * q6 * q7
        + 0.00076176 * gams_math.power(q5, 2)
        + 0.0474795 * q2 * q5 * q6
        + 0.000804282 * q1 * q5 * q7
        + 0.00304704 * q2 * q5 * q7
    )

    ga4.expr = a4 == (
        0.1586 * q1 * gams_math.power(q5, 2)
        + 0.000402141 * q1 * gams_math.power(q7, 2)
        + 2 * q2 * gams_math.power(q5, 2)
        + 0.00152352 * q2 * gams_math.power(q7, 2)
        + 0.0237398 * q5 * q6
        + 0.00152352 * q5 * q7
        + 0.00051612 * q6 * q7
        + 0.0552 * gams_math.power(q2, 2) * gams_math.power(q7, 2)
        + 0.0189477 * q1 * q2 * gams_math.power(q7, 2)
        + 0.0872 * q1 * q4 * q5
        + 0.034862 * q1 * q4 * q7
        + 0.0215658 * q1 * q5 * q6
        + 0.00287416 * q1 * q6 * q7
        + 0.0474795 * q2 * q6 * q7
        + 2 * q3 * q4 * q5
        + 0.1826 * q3 * q4 * q6
        + 0.1104 * q3 * q4 * q7
        - 0.0848 * q3 * q5 * q6
        - 0.00234048 * q3 * q6 * q7
        + 2 * gams_math.power(q2, 2) * q5 * q7
        + 0.1826 * gams_math.power(q2, 2) * q6 * q7
        + 0.0872 * q1 * q2 * q4 * q7
        + 0.3172 * q1 * q2 * q5 * q7
        + 0.0215658 * q1 * q2 * q6 * q7
        + 0.1586 * q1 * q3 * q4 * q7
        + 2 * q2 * q3 * q4 * q7
        - 0.0848 * q2 * q3 * q6 * q7
        + 0.0552 * gams_math.power(q5, 2)
        + 0.3652 * q2 * q5 * q6
        + 0.0378954 * q1 * q5 * q7
        + 0.2208 * q2 * q5 * q7
    )

    ga5.expr = (
        a5
        == 0.0189477 * q1 * gams_math.power(q7, 2)
        + 0.1104 * q2 * gams_math.power(q7, 2)
        + 0.1826 * q5 * q6
        + 0.1104 * q5 * q7
        + 0.0237398 * q6 * q7
        + gams_math.power(q2, 2) * gams_math.power(q7, 2)
        + 0.1586 * q1 * q2 * gams_math.power(q7, 2)
        + 0.0872 * q1 * q4 * q7
        + 0.0215658 * q1 * q6 * q7
        + 0.3652 * q2 * q6 * q7
        + 2 * q3 * q4 * q7
        - 0.0848 * q3 * q6 * q7
        + gams_math.power(q5, 2)
        + 0.00076176 * gams_math.power(q7, 2)
        + 0.3172 * q1 * q5 * q7
        + 4 * q2 * q5 * q7
    )

    ga6.expr = a6 == (
        0.1586 * q1 * gams_math.power(q7, 2)
        + 2 * q2 * gams_math.power(q7, 2)
        + 2 * q5 * q7
        + 0.1826 * q6 * q7
        + 0.0552 * gams_math.power(q7, 2)
    )

    ga7.expr = a7 == gams_math.power(q7, 2)

    # Bounds
    # q1.up.assign = 3.4329
    # q2.up.assign = 0.1627
    # q3.up.assign = 0.1139
    # q4.lo.assign = 0.2539
    # q5.up.assign = 0.0208
    # q6.lo.assign = 2.0247
    # q7.lo.assign = 1
    w.lo.assign = 0
    w.up.assign = 10
    k.lo.assign = 0
    k.up.assign = 10

    # Initial point
    q1.l.assign = 0.2
    q2.l.assign = 0.02
    q3.l.assign = 0.1
    q4.l.assign = 0.3
    q5.l.assign = 0
    q6.l.assign = 2
    q7.l.assign = 4.5
    w.l.assign = 0
    k.l.assign = 2

    fiat = Model(
        m,
        name="fiat",
        equations=m.getEquations(),
        problem="NLP",
        sense="MIN",
        objective=objval,
    )

    fiat.solve()
    print("Objective Function Value:  ", round(objval.toValue(), 4))
    # End Fiat


if __name__ == "__main__":
    main()
