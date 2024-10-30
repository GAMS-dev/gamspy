"""
## GAMSSOURCE: https://gams.com/latest/gamslib_ml/libhtml/gamslib_dice.html
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, dice designment, mathematics, nontransitive dice

Probabilistic dice - an example of a non-transitive relation.
We want to design a set of dice with an integer number on each face
such that on average dice1 beats dice2, and dice2 on average beats
dice3 etc, but diceN has to beat dice1.

MIP codes behave very erratically on such a problem and slight
reformulations can result in dramatic changes in performance.
For example, making the variable fval integer can change performance
so we leave fval free and take the floor to get integer values.

Gardner, M, Scientific American.

Robert A Bosch, Mindsharpener, Optima, MP Society Newsletter, Vol 70,
June 2003, page 8-9

Robert A Bosch, Monochromatic Squares, Optima, MP Society Newsletter,
Vol 71, March 2004, page 6-7
"""

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
        records=[f"dice{idx}" for idx in range(1, 4)],
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
    eq4[dice, f.lag(1)] = fval[dice, f.lag(1)] + 1 <= fval[dice, f]

    xdice = Model(
        m,
        "xdice",
        equations=m.getEquations(),
        problem=Problem.MIP,
        sense=Sense.MAX,
        objective=wnx,
    )
    xdice.solve()
    assert xdice.objective_value == 21


if __name__ == "__main__":
    main()
