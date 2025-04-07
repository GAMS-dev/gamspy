import sys

from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
    VariableType,
)


def main():
    m = Container()

    f = Set(
        m,
        name="f",
        description="faces on a dice",
        records=[f"face{idx}" for idx in range(1, 7)],
    )
    dice = Set(
        m,
        name="dice",
        description="number of dice",
        records=[f"dice{idx}" for idx in range(1, 7)],
    )

    flo = Parameter(m, name="flo", description="lowest face value", records=1)
    fup = Parameter(
        m, "fup", description="highest face value", records=len(dice) * len(f)
    )

    fp = Alias(m, name="fp", alias_with=f)

    wnx = Variable(m, name="wnx", description="number of wins")
    fval = Variable(
        m,
        name="fval",
        domain=[dice, f],
        description="face value on dice - may be fractional",
    )
    comp = Variable(
        m,
        name="comp",
        domain=[dice, f, fp],
        description="one implies f beats fp",
        type=VariableType.BINARY,
    )

    fval.lo[dice, f] = flo
    fval.up[dice, f] = fup
    fval.fx["dice1", "face1"] = flo

    eq1 = Equation(m, "eq1", domain=dice, description="count the wins")
    eq3 = Equation(
        m,
        "eq3",
        domain=[dice, f, fp],
        description="definition of non-transitive relation",
    )
    eq4 = Equation(
        m,
        "eq4",
        domain=[dice, f],
        description="different face values for a single dice",
    )

    eq1[dice] = Sum((f, fp), comp[dice, f, fp]) == wnx
    eq3[dice, f, fp] = (
        fval[dice, f] + (fup - flo + 1) * (1 - comp[dice, f, fp])
        >= fval[dice.lead(1, type="circular"), fp] + 1
    )
    eq4[dice, f - 1] = fval[dice, f - 1] + 1 <= fval[dice, f]

    xdice = Model(
        m,
        "xdice",
        equations=m.getEquations(),
        problem=Problem.MIP,
        sense=Sense.MAX,
        objective=wnx,
    )
    xdice.freeze(modifiables=[flo])
    xdice.solve(output=sys.stdout)


if __name__ == "__main__":
    main()
