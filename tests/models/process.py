"""
Alkylation Process Optimization (PROCESS)

Optimization of a alkylation process.


Bracken, J, and McCormick, G P, Chapter 4. In Selected Applications
of Nonlinear Programming. John Wiley and Sons, New York, 1968.

Keywords: nonlinear programming, alkylation process, chemical engineering
"""

from gamspy import Variable, Equation, Container, Model
from gamspy.math import power


def main():
    m = Container()

    # gams.transfer.algebra.math seems to miss sqr()
    def sqr(x):
        return power(x, 2)

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
    yield1 = Equation(m, type="eq", name="yield1")
    rngyield = Equation(m, type="eq", name="rngyield")
    makeup = Equation(m, type="eq", name="makeup")
    sdef = Equation(m, type="eq", name="sdef")
    motor = Equation(m, type="eq", name="motor")
    rngmotor = Equation(m, type="eq", name="rngmotor")
    drat = Equation(m, type="eq", name="drat")
    ddil = Equation(m, type="eq", name="ddil")
    rngddil = Equation(m, type="eq", name="rngddil")
    df4 = Equation(m, type="eq", name="df4")
    rngdf4 = Equation(m, type="eq", name="rngdf4")
    dprofit = Equation(m, type="eq", name="dprofit")

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

    m.solve(process, problem="NLP", sense="max", objective_variable=profit)
    print("Profit in model 'process': {:.2f}".format(profit.records.level[0]))
    m.solve(rproc, problem="NLP", sense="max", objective_variable=profit)
    print("Profit in model 'rproc': {:.2f}".format(profit.records.level[0]))


if __name__ == "__main__":
    main()
