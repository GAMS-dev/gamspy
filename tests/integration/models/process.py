"""
Alkylation Process Optimization (PROCESS)

Optimization of a alkylation process.


Bracken, J, and McCormick, G P, Chapter 4. In Selected Applications
of Nonlinear Programming. John Wiley and Sons, New York, 1968.

Keywords: nonlinear programming, alkylation process, chemical engineering
"""
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Problem
from gamspy import Sense
from gamspy import Variable
from gamspy.math import sqr


def main():
    m = Container(delayed_execution=True)

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

    yield1.definition = alkylate == olefin * (
        1.12 + 0.13167 * ratio - 0.00667 * sqr(ratio)
    )
    makeup.definition = alkylate == olefin + isom - 0.22 * alkylate
    sdef.definition = (
        acid == alkylate * dilute * strength / (98 - strength) / 1000
    )
    motor.definition = octane == 86.35 + 1.098 * ratio - 0.038 * sqr(
        ratio
    ) - 0.325 * (89 - strength)
    drat.definition = ratio == (isor + isom) / olefin
    ddil.definition = dilute == 35.82 - 0.222 * f4
    df4.definition = f4 == -133 + 3 * octane
    dprofit.definition = (
        profit
        == 0.063 * alkylate * octane
        - 5.04 * olefin
        - 0.035 * isor
        - 10 * acid
        - 3.36 * isom
    )
    rngyield.definition = rangey * alkylate == olefin * (
        1.12 + 0.13167 * ratio - 0.00667 * sqr(ratio)
    )
    rngmotor.definition = (
        rangem * octane
        == 86.35 + 1.098 * ratio - 0.038 * sqr(ratio) - 0.325 * (89 - strength)
    )
    rngddil.definition = ranged * dilute == 35.82 - 0.222 * f4
    rngdf4.definition = rangef * f4 == -133 + 3 * octane

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

    rangey.lo.assignment = 0.9
    rangey.up.assignment = 1.1
    rangey.l.assignment = 1
    rangem.lo.assignment = 0.9
    rangem.up.assignment = 1.1
    rangem.l.assignment = 1
    ranged.lo.assignment = 0.9
    ranged.up.assignment = 1.1
    ranged.l.assignment = 1
    rangef.lo.assignment = 0.9
    rangef.up.assignment = 1.1
    rangef.l.assignment = 1

    strength.lo.assignment = 85
    strength.up.assignment = 93
    octane.lo.assignment = 90
    octane.up.assignment = 95
    ratio.lo.assignment = 3
    ratio.up.assignment = 12
    dilute.lo.assignment = 1.2
    dilute.up.assignment = 4
    f4.lo.assignment = 145
    f4.up.assignment = 162
    olefin.lo.assignment = 10
    olefin.up.assignment = 2000
    isor.up.assignment = 16000
    acid.up.assignment = 120
    alkylate.up.assignment = 5000
    isom.up.assignment = 2000

    olefin.l.assignment = 1745
    isor.l.assignment = 12000
    acid.l.assignment = 110
    alkylate.l.assignment = 3048
    isom.l.assignment = 1974
    strength.l.assignment = 89.2
    octane.l.assignment = 92.8
    ratio.l.assignment = 8
    dilute.l.assignment = 3.6
    f4.l.assignment = 145
    profit.l.assignment = 872

    process.solve()
    print("Profit in model 'process': {:.2f}".format(profit.records.level[0]))
    rproc.solve()
    print("Profit in model 'rproc': {:.2f}".format(profit.records.level[0]))


if __name__ == "__main__":
    main()
