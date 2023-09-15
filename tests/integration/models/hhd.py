"""
Human Heart Dipole
The human heart dipole problem consists of the experimental electrolytic
determination of the resultant dipole moment in the human heart.

The problem was formulated by:
Nelson, C.V. and Hodgkin, B.C., Determination of magnitudes, directions and
locations of two independent dipoles in a circular conducting region from
boundary potential measurements. IEEE Trans. Biomed. Eng., 28, 1981,
pp.817-823.

Please see:
Neculai Andrei, "Models, Test Problems and Applications for
Mathematical Programming". Technical Press, Bucharest, 2003.
Application U84, page 65. Application A13, page 360.
"""
import gamspy.math as gams_math
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Variable


def main():
    m = Container()

    # Variant 1
    # # SCALARS #
    # summx = Parameter(m, name="summx", records= 0.485)
    # summy = Parameter(m, name="summy", records=-0.0019)
    # suma = Parameter(m, name="suma", records=-0.0581)
    # sumb = Parameter(m, name="sumb", records= 0.015)
    # sumc = Parameter(m, name="sumc", records= 0.105)
    # sumd = Parameter(m, name="sumd", records= 0.0406)
    # sume = Parameter(m, name="sume", records= 0.167)
    # sumf = Parameter(m, name="sumf", records=-0.399)

    # Variant 2
    # # SCALARS #
    # summx = Parameter(m, name="summx", records=-0.69)
    # summy = Parameter(m, name="summy", records=-0.044)
    # suma = Parameter(m, name="suma", records=-1.57)
    # sumb = Parameter(m, name="sumb", records=-1.31)
    # sumc = Parameter(m, name="sumc", records=-2.65)
    # sumd = Parameter(m, name="sumd", records= 2)
    # sume = Parameter(m, name="sume", records=-12.6)
    # sumf = Parameter(m, name="sumf", records= 9.48)

    # Variant 3
    # SCALARS #
    summx = Parameter(m, name="summx", records=-0.816)
    summy = Parameter(m, name="summy", records=-0.017)
    suma = Parameter(m, name="suma", records=-1.826)
    sumb = Parameter(m, name="sumb", records=-0.754)
    sumc = Parameter(m, name="sumc", records=-4.839)
    sumd = Parameter(m, name="sumd", records=-3.259)
    sume = Parameter(m, name="sume", records=-14.023)
    sumf = Parameter(m, name="sumf", records=15.467)

    # VARIABLES #
    x1 = Variable(m, name="x1")
    x2 = Variable(m, name="x2")
    x3 = Variable(m, name="x3")
    x4 = Variable(m, name="x4")
    x5 = Variable(m, name="x5")
    x6 = Variable(m, name="x6")
    x7 = Variable(m, name="x7")
    x8 = Variable(m, name="x8")
    obj = Variable(m, name="obj")

    # EQUATIONS #
    e1 = Equation(m, name="e1", type="regular")
    e2 = Equation(m, name="e2", type="regular")
    e3 = Equation(m, name="e3", type="regular")
    e4 = Equation(m, name="e4", type="regular")
    e5 = Equation(m, name="e5", type="regular")
    e6 = Equation(m, name="e6", type="regular")
    e7 = Equation(m, name="e7", type="regular")
    e8 = Equation(m, name="e8", type="regular")
    e = Equation(m, name="e", type="regular")

    e1.expr = x1 + x2 - summx == 0
    e2.expr = x3 + x4 - summy == 0
    e3.expr = x5 * x1 + x6 * x2 - x7 * x3 - x8 * x4 - suma == 0
    e4.expr = x7 * x1 + x8 * x2 + x5 * x3 + x6 * x4 - sumb == 0
    e5.expr = (
        x1 * (gams_math.power(x5, 2) - gams_math.power(x7, 2))
        - 2 * x3 * x5 * x7
        + x2 * (gams_math.power(x6, 2) - gams_math.power(x8, 2))
        - 2 * x4 * x6 * x8
        - sumc
        == 0
    )
    e6.expr = (
        x3 * (gams_math.power(x5, 2) - gams_math.power(x7, 2))
        - 2 * x1 * x5 * x7
        + x4 * (gams_math.power(x6, 2) - gams_math.power(x8, 2))
        - 2 * x2 * x6 * x8
        - sumd
        == 0
    )
    e7.expr = (
        x1 * x5 * (gams_math.power(x5, 2) - 3 * gams_math.power(x7, 2))
        + x3 * x7 * (gams_math.power(x7, 2) - 3 * gams_math.power(x5, 2))
        + x2 * x6 * (gams_math.power(x6, 2) - 3 * gams_math.power(x8, 2))
        + x4 * x8 * (gams_math.power(x8, 2) - 3 * gams_math.power(x6, 2))
        - sume
        == 0
    )
    e8.expr = (
        x3 * x5 * (gams_math.power(x5, 2) - 3 * gams_math.power(x7, 2))
        + x1 * x7 * (gams_math.power(x7, 2) - 3 * gams_math.power(x5, 2))
        + x4 * x6 * (gams_math.power(x6, 2) - 3 * gams_math.power(x8, 2))
        + x2 * x8 * (gams_math.power(x8, 2) - 3 * gams_math.power(x6, 2))
        - sumf
        == 0
    )

    e.expr = obj == 1

    # Initial point (Variant 1)
    # x1.l.assign = 0.299
    # x2.l.assign = 0.186
    # x3.l.assign = -0.0273
    # x4.l.assign = 0.0254
    # x5.l.assign = -0.474
    # x6.l.assign = 0.474
    # x7.l.assign = -0.0892
    # x8.l.assign = 0.0892

    # Initial point (Variant 2)
    # x1.l.assign = -0.3
    # x2.l.assign = -0.39
    # x3.l.assign =  0.3
    # x4.l.assign = -0.344
    # x5.l.assign = -1.2
    # x6.l.assign =  2.69
    # x7.l.assign =  1.59
    # x8.l.assign = -1.5

    # Initial point (Variant 3)
    x1.l.assign = -0.041
    x2.l.assign = -0.775
    x3.l.assign = 0.03
    x4.l.assign = -0.047
    x5.l.assign = -2.565
    x6.l.assign = 2.565
    x7.l.assign = -0.754
    x8.l.assign = 0.754

    hhd = Model(
        m,
        name="hhd",
        equations=m.getEquations(),
        problem="nlp",
        sense="MIN",
        objective=obj,
    )
    hhd.solve()

    print("Objective Function Value:  ", round(obj.toValue(), 4))

    # End hhd


if __name__ == "__main__":
    main()
