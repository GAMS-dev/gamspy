"""
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, maximum, fleet planning, formulations


Purchasing an Express Delivery Fleet (FLEETSPEED)

A logistics firm chooses how many vehicles of each type to purchase for an
express delivery fleet under a budget. The fleet must deliver the expected
number of parcels per day, and it should be fast overall, measured by the
speed of the vehicles weighted with the parcels they carry:

  overall = sum(i, capacity(i) * buy(i) * speed(i)) / parcels

Marketing also wants to advertise the top speed of the fleet, which is
determined by the fastest vehicle type in the fleet:

  topspeed = max(speed(i) * in_fleet(i) for all i)

The firm maximizes overall + weight * topspeed. The fastest vehicle carries
few parcels, so a large weight trades the overall speed for a fast flagship
vehicle.

gamspy.formulations.maximum makes the top speed equal to the maximum by
selecting one operand with a binary variable, so the firm only advertises
the vehicle types it purchases.
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
    i = Set(
        m,
        name="i",
        records=[
            "cargo-bike",
            "scooter",
            "compact-van",
            "electric-van",
            "box-truck",
            "sprinter-van",
            "performance-van",
        ],
        description="vehicles",
    )

    # Parameters
    price = Parameter(
        m,
        name="price",
        domain=i,
        records=np.array([8, 6, 35, 45, 60, 50, 80]),
        description="price of vehicle i in thousands of dollars",
    )
    capacity = Parameter(
        m,
        name="capacity",
        domain=i,
        records=np.array([60, 30, 150, 180, 300, 200, 120]),
        description="parcels per day delivered by vehicle i",
    )
    speed = Parameter(
        m,
        name="speed",
        domain=i,
        records=np.array([25, 45, 150, 130, 110, 160, 210]),
        description="top speed of vehicle i in km/h",
    )
    budget = Parameter(
        m, name="budget", records=140, description="budget in thousands of dollars"
    )
    parcels = Parameter(
        m, name="parcels", records=450, description="parcels to deliver per day"
    )
    weight = Parameter(
        m, name="weight", records=1, description="importance of the top speed"
    )

    # Variables
    buy = Variable(
        m,
        name="buy",
        type="integer",
        domain=i,
        description="number of vehicles of type i purchased",
    )
    in_fleet = Variable(
        m,
        name="in_fleet",
        type="binary",
        domain=i,
        description="vehicle type i is in the fleet",
    )

    # Equations
    spend = Equation(m, name="spend", description="observe the budget")
    spend[...] = Sum(i, price[i] * buy[i]) <= budget
    deliver = Equation(m, name="deliver", description="deliver all parcels")
    deliver[...] = Sum(i, capacity[i] * buy[i]) >= parcels
    is_in_fleet = Equation(
        m,
        name="is_in_fleet",
        domain=i,
        description="only purchased vehicle types are in the fleet",
    )
    is_in_fleet[i] = in_fleet[i] <= buy[i]

    # topspeed == max(speed[i] * in_fleet[i] for all i)
    topspeed, eqs = formulations.maximum(speed[i] * in_fleet[i], over=i)

    fleet = Model(
        m,
        name="fleet",
        equations=[spend, deliver, is_in_fleet, *eqs],
        problem="mip",
        sense=Sense.MAX,
        objective=Sum(i, capacity[i] * buy[i] * speed[i]) / parcels + weight * topspeed,
    )
    fleet.solve()

    print("Objective Function Value: ", round(fleet.objective_value, 3))
    print(buy.toDense())

    # The top speed equals the speed of the fastest vehicle type in the fleet
    fastest = (speed.toDense() * in_fleet.toDense()).max()
    if abs(topspeed.toValue() - fastest) > 1e-6:
        raise Exception("top speed does not equal the maximum")


if __name__ == "__main__":
    main()
