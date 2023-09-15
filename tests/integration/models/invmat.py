"""
Inverse of a given matrix A of order (n,n).

Two methods are considered.
The first one computes the columns of the inverse x(i) by solving n algebraic
linear systems Ax(i)=e(i), where e(i) are the i-th column of the unity matrix,
i=1,...,n.
The second one determine the inverse by solving the linear system AX=I, where
X is the inverse of A.
"""
import numpy as np

from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def data_records():
    # a records
    a_recs = np.array(
        [
            [1, 2, 3, 4, 4],
            [1, 3, 4, 3, 1],
            [1, 4, 1, 2, 6],
            [2, 4, 1, 1, 1],
            [3, 1, 5, 2, 7],
        ]
    )

    return a_recs


def main():
    m = Container()

    # SET #
    i = Set(m, name="i", records=[f"i{i}" for i in range(1, 6)])

    # ALIASES #
    j = Alias(m, name="j", alias_with=i)
    k = Alias(m, name="k", alias_with=i)

    # PARAMETERS #
    a = Parameter(
        m,
        name="a",
        domain=[i, j],
        records=data_records(),
        description="matrix to be inverted",
    )
    ident = Parameter(
        m, name="ident", domain=[i, j], description="the identity matrix"
    )

    ident[i, i] = 1

    # Method 1. Solving the systems Ax(i)=e(i)
    # -----------------------------------------

    # VARIABLES #
    obj = Variable(
        m,
        name="obj",
        description="variabile associated to a virtual objective",
    )
    col = Variable(
        m,
        name="col",
        domain=[j],
        description="the columns of the inverse matrix",
    )

    # EQUATIONS #
    eobj = Equation(
        m,
        name="eobj",
        type="regular",
        description="name of the virtual objective",
    )
    lin = Equation(
        m,
        name="lin",
        type="regular",
        domain=[i],
        description="name of the equations of the systems to be solved",
    )

    # PARAMETERS #
    b = Parameter(m, name="b", domain=[i], description="Righ-hand Side term")
    ainv = Parameter(
        m, name="ainv", domain=[i, j], description="inverse matrix of A"
    )

    eobj.expr = obj == 0
    lin[i] = Sum(k, a[i, k] * col[k]) == b[i]

    invmat1 = Model(
        m,
        name="invmat1",
        equations=m.getEquations(),
        problem="lp",
        sense="min",
        objective=obj,
    )

    for jj in j.toList():
        b[i] = ident[i, jj]
        invmat1.solve()
        ainv[i, jj] = col.l[i]

    print("a:  \n", a.pivot().round(3))
    print("ainv:  \n", ainv.pivot().round(3))

    # Checking the inverse
    aainv = Parameter(
        m,
        name="aainv",
        domain=[i, k],
        description="matrix A multiplied by ainv",
    )
    aainv[i, k] = Sum(j, a[i, j] * ainv[j, k])
    print("aainv:  \n", aainv.pivot().round(3))

    # Method 2. Solving the system AX=I
    # ----------------------------------

    # VARIABLE #
    ainverse = Variable(
        m, name="ainverse", domain=[i, j], description="the inverse of A"
    )

    # EQUATION #
    lineq = Equation(m, name="lineq", type="regular", domain=[i, j])

    lineq[i, j] = Sum(k, a[i, k] * ainverse[k, j]) == ident[i, j]

    invmat2 = Model(
        m,
        name="invmat2",
        equations=m.getEquations(),
        problem="lp",
        sense="min",
        objective=obj,
    )
    invmat2.solve()

    print("ainverse:  \n", ainverse.pivot().round(3))

    # Checking the inverse
    aainv[i, k] = Sum(j, a[i, j] * ainv[j, k])
    print("aainv:  \n", aainv.pivot().round(3))

    # End of invmat


if __name__ == "__main__":
    main()
