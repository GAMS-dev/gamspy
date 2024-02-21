"""
## GAMSSOURCE: https://www.gams.com/latest/noalib_ml/libhtml/noalib_batchreactor.html
## LICENSETYPE: Demo
## MODELTYPE: NLP


Optimal control of a batch reactor.
Find the optimal temperature profile which gives maximum intermediate product
concentration in a batch reactor with two consecutive reactions. The first
reaction is of second order and the second one is of first order with known
rate constants.

Renfro J.G., Morshedi, A.M., Osbjornsen, O.A., Simultaneous optimization and
solution of systems described by differential/algebraic equations. Computer
and Chemical Engineering, vol.11, 1987, pp.503-517.
"""

from __future__ import annotations

import os

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Problem
from gamspy import Sense
from gamspy import Set
from gamspy import Variable


def main():
    m = Container(
        system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
    )

    # Set
    nh = Set(
        m,
        name="nh",
        records=[str(idx) for idx in range(0, 101)],
        description="Number of subintervals",
    )
    k = Alias(m, name="k", alias_with=nh)

    # Data
    ca_0 = Parameter(
        m, name="ca_0", records=1.0, description="initial value for ca"
    )
    cb_0 = Parameter(
        m, name="cb_0", records=0.0, description="initial value for cb"
    )
    h = Parameter(m, name="h", records=1)

    # Variable
    ca = Variable(
        m, name="ca", domain=nh, description="concentration of component A"
    )
    cb = Variable(
        m, name="cb", domain=nh, description="concentration of component B"
    )
    t = Variable(
        m,
        name="t",
        domain=nh,
        description="temperature of reactor (control variable)",
    )
    k1 = Variable(
        m,
        name="k1",
        domain=nh,
        description="rate constant for the first reaction",
    )
    k2 = Variable(
        m,
        name="k2",
        domain=nh,
        description="rate constant for the second reaction",
    )
    obj = Variable(m, name="obj", description="criterion")

    # Equation
    eobj = Equation(m, name="eobj", description="criterion definition")
    state1 = Equation(
        m, domain=nh, name="state1", description="state equation 1"
    )
    state2 = Equation(
        m, domain=nh, name="state2", description="state equation 2"
    )
    ek1 = Equation(m, domain=nh, name="ek1")
    ek2 = Equation(m, domain=nh, name="ek2")

    eobj[...] = obj == cb["100"]

    ek1[nh[k]] = k1[k] == 4000 * gams_math.exp(-2500 / t[k])
    ek2[nh[k]] = k2[k] == 620000 * gams_math.exp(-5000 / t[k])

    state1[nh[k.lead(1, "linear")]] = ca[k.lead(1, "linear")] == ca[k] + (
        h / 2
    ) * (
        -k1[k] * ca[k] * ca[k]
        - k1[k.lead(1, "linear")]
        * ca[k.lead(1, "linear")]
        * ca[k.lead(1, "linear")]
    )

    state2[nh[k.lead(1, "linear")]] = cb[k.lead(1, "linear")] == cb[k] + (
        h / 2
    ) * (
        k1[k] * ca[k] * ca[k]
        - k2[k] * cb[k]
        + k1[k.lead(1, "linear")]
        * ca[k.lead(1, "linear")]
        * ca[k.lead(1, "linear")]
        - k2[k.lead(1, "linear")] * cb[k.lead(1, "linear")]
    )

    ca.l[nh] = 1.0
    cb.l[nh] = 0.0

    ca.fx["0"] = ca_0
    cb.fx["0"] = cb_0

    t.lo[nh] = 110
    t.up[nh] = 280

    batchReactor = Model(
        m,
        name="batchReactor",
        equations=[eobj, state1, state2, ek1, ek2],
        problem=Problem.NLP,
        sense=Sense.MAX,
        objective=obj,
    )
    batchReactor.solve()

    import math

    assert math.isclose(batchReactor.objective_value, 0.8826, rel_tol=0.001)


if __name__ == "__main__":
    main()
