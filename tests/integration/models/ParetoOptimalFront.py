"""
*** Pareto optimal front determination

For more details please refer to Chapter 2 (Gcode2.16), of the following book:
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
--------------------------------------------------------------------------------
Model type: NLP
--------------------------------------------------------------------------------
Contributed by
Dr. Alireza Soroudi
IEEE Senior Member
email: alireza.soroudi@gmail.com
We do request that publications derived from the use of the developed GAMS code
explicitly acknowledge that fact by citing
Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
DOI: doi.org/10.1007/978-3-319-62350-4
"""
import gamspy.math as gams_math
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Number
from gamspy import Parameter
from gamspy import Set
from gamspy import Variable


def sqr(x):
    return gams_math.power(x, 2)


def main():
    m = Container()

    # VARIABLES #
    of1 = Variable(m, name="of1", type="free")
    of2 = Variable(m, name="of2", type="free")
    x1 = Variable(m, name="x1", type="free")
    x2 = Variable(m, name="x2", type="free")

    # EQUATIONS #
    eq1 = Equation(m, name="eq1", type="regular")
    eq2 = Equation(m, name="eq2", type="regular")
    eq3 = Equation(m, name="eq3", type="regular")
    eq4 = Equation(m, name="eq4", type="regular")

    eq1.expr = 4 * x1 - 0.5 * sqr(x2) == of1
    eq2.expr = Number(-1) * sqr(x1) + 5 * x2 == of2
    eq3.expr = 2 * x1 + 3 * x2 <= 10
    eq4.expr = 2 * x1 - x2 >= 0

    x1.lo.assign = 1
    x1.up.assign = 2
    x2.lo.assign = 1
    x2.up.assign = 3

    pareto1 = Model(
        m,
        name="pareto1",
        equations=m.getEquations(),
        problem="nlp",
        sense="max",
        objective=of1,
    )
    pareto2 = Model(
        m,
        name="pareto2",
        equations=m.getEquations(),
        problem="nlp",
        sense="max",
        objective=of2,
    )

    # COUNTER SET #
    counter = Set(m, name="counter", records=[f"c{c}" for c in range(1, 22)])

    # REPORTING PARAMETERS #
    E = Parameter(m, name="E")
    report = Parameter(m, name="report", domain=[counter, "*"])
    ranges = Parameter(m, name="ranges", domain=["*"])

    pareto1.solve()
    ranges["OF1max"] = of1.l
    ranges["OF2min"] = of2.l

    pareto2.solve()
    ranges["OF2max"] = of2.l
    ranges["OF1min"] = of1.l

    for idx, c in enumerate(counter.toList()):
        E.assign = (ranges["OF2max"] - ranges["OF2min"]) * (idx) / (
            Card(counter) - 1
        ) + ranges["OF2min"]
        of2.lo.assign = E
        pareto1.solve()
        report[c, "OF1"] = of1.l
        report[c, "OF2"] = of2.l
        report[c, "E"] = E

    print("Report:  \n", report.pivot().round(4), "\n")
    print("Ranges:  \n", ranges.toDict(), "\n")


if __name__ == "__main__":
    main()
