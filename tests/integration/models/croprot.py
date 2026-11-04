"""
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, logical xor, crop rotation, agriculture, formulations


Corn and Soybean Rotation Planning (CROPROT)

A farm plans which of its fields to plant with corn and which with
soybeans over the next seasons. Corn earns a higher margin, but planting
the same crop on a field in consecutive years lowers yields and increases
fertilizer and pest control costs. A field that switches crops earns a
rotation benefit per acre. A field is rotated in a year if exactly one of
this year's and last year's crops is corn:

  rotated(f,y) = corn(f,y) xor corn(f,y-1)

The farm has contracts with an ethanol plant to deliver the corn of a
minimum number of acres every year, and its grain dryer limits the corn
acres it can harvest.

gamspy.formulations.logical_xor makes rotated(f,y) equal to the exclusive
disjunction, so the farm only earns the rotation benefit on fields that
switch crops.
"""

from __future__ import annotations

import numpy as np

import gamspy.formulations as formulations
from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)


def main():
    m = Container()

    # Sets
    f = Set(
        m,
        name="f",
        records=["north", "creek", "hill", "barn", "south", "east"],
        description="fields",
    )
    y = Set(
        m,
        name="y",
        records=["2026", "2027", "2028", "2029", "2030"],
        description="years",
    )
    plan = Set(
        m,
        name="plan",
        domain=y,
        records=["2027", "2028", "2029", "2030"],
        description="planning years",
    )

    # Parameters
    acres = Parameter(
        m,
        name="acres",
        domain=f,
        records=np.array([120, 80, 150, 60, 100, 90]),
        description="size of field f in acres",
    )
    last_corn = Parameter(
        m,
        name="last_corn",
        domain=f,
        records=np.array([1, 1, 0, 0, 1, 0]),
        description="field f was planted with corn in 2026",
    )
    margin_corn = Parameter(
        m, name="margin_corn", records=290, description="corn margin in $ per acre"
    )
    margin_soy = Parameter(
        m, name="margin_soy", records=230, description="soybean margin in $ per acre"
    )
    benefit = Parameter(
        m,
        name="benefit",
        records=70,
        description="rotation benefit in $ per acre",
    )
    contract = Parameter(
        m,
        name="contract",
        domain=plan,
        records=np.array([300, 380, 350, 280]),
        description="minimum corn acres in year plan",
    )
    dryer = Parameter(
        m, name="dryer", records=420, description="maximum corn acres per year"
    )

    # Variables
    corn = Variable(
        m,
        name="corn",
        type="binary",
        domain=[f, y],
        description="field f is planted with corn in year y",
    )
    corn.fx[f, "2026"] = last_corn[f]

    # Equations
    minimum_corn = Equation(
        m, name="minimum_corn", domain=plan, description="observe the contract"
    )
    minimum_corn[plan] = Sum(f, acres[f] * corn[f, plan]) >= contract[plan]
    maximum_corn = Equation(
        m, name="maximum_corn", domain=plan, description="observe the dryer capacity"
    )
    maximum_corn[plan] = Sum(f, acres[f] * corn[f, plan]) <= dryer

    # rotated[f, y] == corn[f, y] xor corn[f, y - 1]
    rotated, eqs = formulations.logical_xor(corn[f, y], corn[f, y - 1])

    rotation = Model(
        m,
        name="rotation",
        equations=[minimum_corn, maximum_corn, *eqs],
        problem="mip",
        sense=Sense.MAX,
        objective=Sum(
            [f, plan],
            acres[f]
            * (
                margin_corn * corn[f, plan]
                + margin_soy * (1 - corn[f, plan])
                + benefit * rotated[f, plan]
            ),
        ),
    )
    rotation.solve()

    print("Objective Function Value: ", round(rotation.objective_value, 3))
    print(corn.pivot())

    # rotated equals the XOR of this year's and last year's crop
    planted = corn.toDense().round()
    changed = np.abs(planted[:, 1:] - planted[:, :-1])
    if np.abs(rotated.toDense()[:, 1:] - changed).max() > 1e-6:
        raise Exception("rotated does not equal the logical XOR")


if __name__ == "__main__":
    main()
