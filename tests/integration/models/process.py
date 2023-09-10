"""
Alkylation Process Optimization (PROCESS)

Optimization of a alkylation process.


Bracken, J, and McCormick, G P, Chapter 4. In Selected Applications
of Nonlinear Programming. John Wiley and Sons, New York, 1968.

Keywords: nonlinear programming, alkylation process, chemical engineering
"""

from gamspy import Variable, Equation, Container, Model
from gamspy.math import sqr
from gamspy import Problem, Sense


def main():
    m = Container()

    # Variables
    olefin = Variable(m, name="olefin", type="positive")
    isor = Variable(m, name="isor", type="positive")
    acid = Variable(m, name="acid", type="positive")
    alkylate = Variable(m, name="alkylate", type="positive")
    isom = Variable(m, name="isom", type="positive")
    strength = Variable(m, name="strength", type="positive")
    octane = Variable(m, name="octane", type="positive")
    ratio = Variable(m, name="ratio", type="positive")
    dilute = Variable(m, name="dilute", type="positive")
    f4 = Variable(m, name="f4", type="positive")

    profit = Variable(m, name="profit")
    rangey = Variable(m, name="rangey")
    rangem = Variable(m, name="rangem")
    ranged = Variable(m, name="ranged")
    rangef = Variable(m, name="rangef")

    # Equations
    yield1 = Equation(m, name="yield1")
    rngyield = Equation(m, name="rngyield")
    makeup = Equation(m, name="makeup")
    sdef = Equation(m, name="sdef")
    motor = Equation(m, name="motor")
    rngmotor = Equation(m, name="rngmotor")
    drat = Equation(m, name="drat")
    ddil = Equation(m, name="ddil")
    rngddil = Equation(m, name="rngddil")
    df4 = Equation(m, name="df4")
    rngdf4 = Equation(m, name="rngdf4")
    dprofit = Equation(m, name="dprofit")

    yield1.expr = alkylate == olefin * (
        1.12 + 0.13167 * ratio - 0.00667 * sqr(ratio)
    )
    makeup.expr = alkylate == olefin + isom - 0.22 * alkylate
    sdef.expr = acid == alkylate * dilute * strength / (98 - strength) / 1000
    motor.expr = octane == 86.35 + 1.098 * ratio - 0.038 * sqr(
        ratio
    ) - 0.325 * (89 - strength)
    drat.expr = ratio == (isor + isom) / olefin
    ddil.expr = dilute == 35.82 - 0.222 * f4
    df4.expr = f4 == -133 + 3 * octane
    dprofit.expr = (
        profit
        == 0.063 * alkylate * octane
        - 5.04 * olefin
        - 0.035 * isor
        - 10 * acid
        - 3.36 * isom
    )
    rngyield.expr = rangey * alkylate == olefin * (
        1.12 + 0.13167 * ratio - 0.00667 * sqr(ratio)
    )
    rngmotor.expr = rangem * octane == 86.35 + 1.098 * ratio - 0.038 * sqr(
        ratio
    ) - 0.325 * (89 - strength)
    rngddil.expr = ranged * dilute == 35.82 - 0.222 * f4
    rngdf4.expr = rangef * f4 == -133 + 3 * octane

    # Define Models
    process = Model(
        m,
        name="process",
        equations=[yield1, makeup, sdef, motor, drat, ddil, df4, dprofit],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=profit,
    )
    rproc = Model(
        m,
        name="rproc",
        equations=[
            rngyield,
            makeup,
            sdef,
            rngmotor,
            drat,
            rngddil,
            rngdf4,
            dprofit,
        ],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=profit,
    )

    rangey.lo.assign = 0.9
    rangey.up.assign = 1.1
    rangey.l.assign = 1
    rangem.lo.assign = 0.9
    rangem.up.assign = 1.1
    rangem.l.assign = 1
    ranged.lo.assign = 0.9
    ranged.up.assign = 1.1
    ranged.l.assign = 1
    rangef.lo.assign = 0.9
    rangef.up.assign = 1.1
    rangef.l.assign = 1

    strength.lo.assign = 85
    strength.up.assign = 93
    octane.lo.assign = 90
    octane.up.assign = 95
    ratio.lo.assign = 3
    ratio.up.assign = 12
    dilute.lo.assign = 1.2
    dilute.up.assign = 4
    f4.lo.assign = 145
    f4.up.assign = 162
    olefin.lo.assign = 10
    olefin.up.assign = 2000
    isor.up.assign = 16000
    acid.up.assign = 120
    alkylate.up.assign = 5000
    isom.up.assign = 2000

    olefin.l.assign = 1745
    isor.l.assign = 12000
    acid.l.assign = 110
    alkylate.l.assign = 3048
    isom.l.assign = 1974
    strength.l.assign = 89.2
    octane.l.assign = 92.8
    ratio.l.assign = 8
    dilute.l.assign = 3.6
    f4.l.assign = 145
    profit.l.assign = 872

    process.solve()
    print("Profit in model 'process': {:.2f}".format(profit.records.level[0]))
    rproc.solve()
    print("Profit in model 'rproc': {:.2f}".format(profit.records.level[0]))


if __name__ == "__main__":
    main()
