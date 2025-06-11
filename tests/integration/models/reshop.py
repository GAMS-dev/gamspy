"""
## LICENSETYPE: Demo
## MODELTYPE: EMP
## KEYWORDS: Extended Mathematical Programming, EMP, RESHOP

"""

import gamspy.math as gpm
from gamspy import Container, Equation, Model, Set, Variable


def ReSHOPAnnotation(m, s):
    return m.addGamsCode("EmbeddedCode ReSHOP:\n" + s + "\nendEmbeddedCode")


def main():
    m = Container(debugging_level="keep")
    t = Set(m, name="m", records=[0, 1])
    a = Set(m, name="a", records=["a0", "a1"])
    beta = 7
    alpha = 6

    x = Variable(m, name="x", domain=[a, t])
    obj = Variable(m, name="obj", domain=[a])

    oterms = a.toList()
    cterms = a.toList()

    # Agent 0
    oterms[0] = (beta / 2 * gpm.sqr(x["a0", "0"]) - alpha * x["a0", "0"]) + (
        1 / 2 * gpm.sqr(x["a0", "1"])
        + 3 * x["a0", "1"] * x["a1", "1"]
        - 4 * x["a0", "1"]
    )
    cterms[0] = x["a0", "1"] - x["a0", "0"]

    # Agent 1
    oterms[1] = x["a1", "0"] + (
        1 / 2 * gpm.sqr(x["a1", "1"])
        + x["a0", "1"] * x["a1", "1"]
        - 3 * x["a1", "1"]
    )
    cterms[1] = x["a1", "1"]

    defobj = Equation(m, name="defobj", domain=a)
    defobj[a] = obj[a] == sum(
        o.where[a.sameAs(f"a{i}")] for i, o in enumerate(oterms)
    )

    cons = Equation(m, name="cons", domain=a)
    cons[a] = (
        sum(c.where[a.sameAs(f"a{i}")] for i, c in enumerate(cterms)) >= 0
    )

    x.lo["a0", "0"] = 0
    x.fx["a1", "0"] = 0

    nash = Model(m, name="nash", equations=[defobj, cons], problem="emp")

    ReSHOPAnnotation(
        m,
        """
    n(a): min obj(a) x(a,'*') defobj(a) cons(a)
    root: Nash(n(a))
    """,
    )

    nash.solve(solver="reshop")
    assert x.toList() == [
        ("a0", "0", 0.8571428571428571),
        ("a0", "1", 2.5),
        ("a1", "0", 0.0),
        ("a1", "1", 0.5),
    ]


if __name__ == "__main__":
    main()
