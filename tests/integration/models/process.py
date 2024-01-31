"""
## GAMSSOURCE: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_process.html
## LICENSETYPE: Demo
## MODELTYPE: NLP
## KEYWORDS: nonlinear programming, alkylation process, chemical engineering


Alkylation Process Optimization (PROCESS)

Optimization of a alkylation process.


Bracken, J, and McCormick, G P, Chapter 4. In Selected Applications
of Nonlinear Programming. John Wiley and Sons, New York, 1968.
"""

from __future__ import annotations

import os

from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Problem
from gamspy import Sense
from gamspy import Variable
from gamspy.math import sqr


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
    )

    # Variables
    olefin = Variable(
        m, name="olefin", type="positive", description="olefin feed (bpd)"
    )
    isor = Variable(
        m, name="isor", type="positive", description="isobutane recycle (bpd)"
    )
    acid = Variable(
        m,
        name="acid",
        type="positive",
        description="acid addition rate (1000lb per day)",
    )
    alkylate = Variable(
        m, name="alkylate", type="positive", description="alkylate yield (bpd)"
    )
    isom = Variable(
        m, name="isom", type="positive", description="isobutane makeup (bpd)"
    )
    strength = Variable(
        m,
        name="strength",
        type="positive",
        description="acid strength (weight pct)",
    )
    octane = Variable(
        m, name="octane", type="positive", description="motor octane number"
    )
    ratio = Variable(
        m,
        name="ratio",
        type="positive",
        description="isobutane makeup to olefin ratio",
    )
    dilute = Variable(
        m, name="dilute", type="positive", description="acid dilution factor"
    )
    f4 = Variable(
        m, name="f4", type="positive", description="f-4 performance number"
    )

    profit = Variable(m, name="profit")
    rangey = Variable(m, name="rangey")
    rangem = Variable(m, name="rangem")
    ranged = Variable(m, name="ranged")
    rangef = Variable(m, name="rangef")

    # Equations
    yield1 = Equation(
        m, name="yield1", description="alkylate yield definition"
    )
    rngyield = Equation(
        m, name="rngyield", description="ranged alkylate yield definition"
    )
    makeup = Equation(
        m, name="makeup", description="isobutane makeup definition"
    )
    sdef = Equation(m, name="sdef", description="acid strength definition")
    motor = Equation(m, name="motor", description="motor octane number")
    rngmotor = Equation(
        m, name="rngmotor", description="ranged motor octane number"
    )
    drat = Equation(m, name="drat", description="isobutane to olefin ratio")
    ddil = Equation(m, name="ddil", description="dilution definition")
    rngddil = Equation(
        m, name="rngddil", description="ranged dilution definition"
    )
    df4 = Equation(m, name="df4", description="f-4 definition")
    rngdf4 = Equation(m, name="rngdf4", description="ranged f-4 definition")
    dprofit = Equation(m, name="dprofit", description="profit definition")

    yield1[...] = alkylate == olefin * (
        1.12 + 0.13167 * ratio - 0.00667 * sqr(ratio)
    )
    makeup[...] = alkylate == olefin + isom - 0.22 * alkylate
    sdef[...] = acid == alkylate * dilute * strength / (98 - strength) / 1000
    motor[...] = octane == 86.35 + 1.098 * ratio - 0.038 * sqr(
        ratio
    ) - 0.325 * (89 - strength)
    drat[...] = ratio == (isor + isom) / olefin
    ddil[...] = dilute == 35.82 - 0.222 * f4
    df4[...] = f4 == -133 + 3 * octane
    dprofit[...] = (
        profit
        == 0.063 * alkylate * octane
        - 5.04 * olefin
        - 0.035 * isor
        - 10 * acid
        - 3.36 * isom
    )
    rngyield[...] = rangey * alkylate == olefin * (
        1.12 + 0.13167 * ratio - 0.00667 * sqr(ratio)
    )
    rngmotor[...] = rangem * octane == 86.35 + 1.098 * ratio - 0.038 * sqr(
        ratio
    ) - 0.325 * (89 - strength)
    rngddil[...] = ranged * dilute == 35.82 - 0.222 * f4
    rngdf4[...] = rangef * f4 == -133 + 3 * octane

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

    rangey.lo[...] = 0.9
    rangey.up[...] = 1.1
    rangey.l[...] = 1
    rangem.lo[...] = 0.9
    rangem.up[...] = 1.1
    rangem.l[...] = 1
    ranged.lo[...] = 0.9
    ranged.up[...] = 1.1
    ranged.l[...] = 1
    rangef.lo[...] = 0.9
    rangef.up[...] = 1.1
    rangef.l[...] = 1

    strength.lo[...] = 85
    strength.up[...] = 93
    octane.lo[...] = 90
    octane.up[...] = 95
    ratio.lo[...] = 3
    ratio.up[...] = 12
    dilute.lo[...] = 1.2
    dilute.up[...] = 4
    f4.lo[...] = 145
    f4.up[...] = 162
    olefin.lo[...] = 10
    olefin.up[...] = 2000
    isor.up[...] = 16000
    acid.up[...] = 120
    alkylate.up[...] = 5000
    isom.up[...] = 2000

    olefin.l[...] = 1745
    isor.l[...] = 12000
    acid.l[...] = 110
    alkylate.l[...] = 3048
    isom.l[...] = 1974
    strength.l[...] = 89.2
    octane.l[...] = 92.8
    ratio.l[...] = 8
    dilute.l[...] = 3.6
    f4.l[...] = 145
    profit.l[...] = 872

    process.solve()
    print("Profit in model 'process': {:.2f}".format(profit.records.level[0]))
    rproc.solve()
    print("Profit in model 'rproc': {:.2f}".format(profit.records.level[0]))


if __name__ == "__main__":
    main()
