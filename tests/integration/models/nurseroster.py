"""
## LICENSETYPE: Demo
## MODELTYPE: MIP
## KEYWORDS: mixed integer linear programming, logical constraints, nurse rostering, scheduling, formulations


Nurse Rostering with Weekend and Supervision Rules (NURSEROSTER)

A hospital ward assigns its nurses to early, late and night shifts for the
next two weeks. Every shift must be staffed, a nurse works at most one
shift per day, and the number of working days of every nurse is limited.
Each nurse works at most one weekend. The roster should also follow rules
from the First International Nurse Rostering Competition, which are
violated at a cost:

  - complete weekends: a nurse works either both days of a weekend or
    neither, i.e. works(n,w,sat) eqv works(n,w,sun),
  - no night shift before a free weekend, i.e. a violation is
    night(n,w,fri) and not weekend(n,w), where
    weekend(n,w) = works(n,w,sat) or works(n,w,sun),
  - supervision: a junior nurse should only work a shift with a senior
    nurse on duty, i.e. work(n,w,d,s) imp senior_on(w,d,s), where
    senior_on(w,d,s) is the OR over the senior nurses.

gamspy.formulations.logical_or, logical_eqv, logical_not, logical_and and
logical_imp make the results equal to the logical expressions. Some of
them are pushed in both directions, e.g. the limit on working weekends
favors weekend(n,w) = 0, while the night shift rule favors
weekend(n,w) = 1.

Haspeslagh, S, De Causmaecker, P, Schaerf, A, and Stolevik, M, The First
International Nurse Rostering Competition 2010. Annals of Operations
Research 218 (2014), 221-236.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    n = Set(
        m,
        name="n",
        records=["ana", "ben", "cleo", "dan", "eva", "finn", "gia"],
        description="nurses",
    )
    senior = Set(
        m, name="senior", domain=n, records=["ana", "ben"], description="seniors"
    )
    junior = Set(
        m,
        name="junior",
        domain=n,
        records=["cleo", "dan", "eva", "finn", "gia"],
        description="juniors",
    )
    w = Set(m, name="w", records=["week1", "week2"], description="weeks")
    d = Set(
        m,
        name="d",
        records=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        description="days of the week",
    )
    s = Set(m, name="s", records=["early", "late", "night"], description="shifts")

    # Parameters
    demand = Parameter(
        m,
        name="demand",
        domain=[d, s],
        records=np.array([[2, 2, 1]] * 5 + [[1, 1, 1]] * 2),
        description="nurses required on day d in shift s",
    )
    min_days = Parameter(
        m, name="min_days", records=7, description="minimum working days of a nurse"
    )
    max_days = Parameter(
        m, name="max_days", records=10, description="maximum working days of a nurse"
    )
    max_weekends = Parameter(
        m,
        name="max_weekends",
        records=1,
        description="maximum working weekends of a nurse",
    )
    cost_incomplete = Parameter(
        m,
        name="cost_incomplete",
        records=30,
        description="cost of an incomplete weekend",
    )
    cost_night = Parameter(
        m,
        name="cost_night",
        records=20,
        description="cost of a night shift before a free weekend",
    )
    cost_unsupervised = Parameter(
        m,
        name="cost_unsupervised",
        records=10,
        description="cost of a junior shift without a senior on duty",
    )

    # Variables
    work = Variable(
        m,
        name="work",
        type="binary",
        domain=[n, w, d, s],
        description="nurse n works shift s on day d of week w",
    )
    works = Variable(
        m,
        name="works",
        type="binary",
        domain=[n, w, d],
        description="nurse n works on day d of week w",
    )

    # Equations
    cover = Equation(
        m,
        name="cover",
        domain=[w, d, s],
        description="staff every shift",
    )
    cover[w, d, s] = Sum(n, work[n, w, d, s]) >= demand[d, s]
    one_shift = Equation(
        m,
        name="one_shift",
        domain=[n, w, d],
        description="at most one shift per day",
    )
    one_shift[n, w, d] = works[n, w, d] == Sum(s, work[n, w, d, s])
    fewest_days = Equation(
        m, name="fewest_days", domain=n, description="minimum working days"
    )
    fewest_days[n] = Sum([w, d], works[n, w, d]) >= min_days
    most_days = Equation(
        m, name="most_days", domain=n, description="maximum working days"
    )
    most_days[n] = Sum([w, d], works[n, w, d]) <= max_days

    # weekend[n, w] == works[n, w, "sat"] or works[n, w, "sun"]
    weekend, weekend_eqs = formulations.logical_or(
        works[n, w, "sat"], works[n, w, "sun"]
    )
    most_weekends = Equation(
        m, name="most_weekends", domain=n, description="maximum working weekends"
    )
    most_weekends[n] = Sum(w, weekend[n, w]) <= max_weekends

    # complete[n, w] == works[n, w, "sat"] eqv works[n, w, "sun"]
    complete, complete_eqs = formulations.logical_eqv(
        works[n, w, "sat"], works[n, w, "sun"]
    )

    # night_before_free[n, w] == work[n, w, "fri", "night"] and not weekend[n, w]
    free, free_eqs = formulations.logical_not(weekend)
    night_before_free, night_eqs = formulations.logical_and(
        work[n, w, "fri", "night"], free
    )

    # senior_on[w, d, s] == or(work[senior, w, d, s] for all senior)
    senior_on, senior_eqs = formulations.logical_or(work[senior, w, d, s], over=senior)

    # supervised[junior, w, d, s] == work[junior, w, d, s] imp senior_on[w, d, s]
    supervised, supervised_eqs = formulations.logical_imp(
        work[junior, w, d, s], senior_on
    )

    roster = Model(
        m,
        name="roster",
        equations=[
            cover,
            one_shift,
            fewest_days,
            most_days,
            most_weekends,
            *weekend_eqs,
            *complete_eqs,
            *free_eqs,
            *night_eqs,
            *senior_eqs,
            *supervised_eqs,
        ],
        problem="mip",
        sense=Sense.MIN,
        objective=cost_incomplete * Sum([n, w], 1 - complete[n, w])
        + cost_night * Sum([n, w], night_before_free[n, w])
        + cost_unsupervised * Sum([junior, w, d, s], 1 - supervised[junior, w, d, s]),
    )
    roster.solve()

    print("Objective Function Value: ", round(roster.objective_value, 3))
    shifts = work.toDense().round()
    names = np.array(["-", *s.toList()])
    assigned = (shifts * np.arange(1, 4)).sum(axis=3).astype(int)
    for k, week in enumerate(w.toList()):
        print(week, pd.DataFrame(names[assigned[:, k]], n.toList(), d.toList()))

    # The results equal the logical expressions
    sat, sun = shifts[:, :, 5].sum(axis=2), shifts[:, :, 6].sum(axis=2)
    worked = np.maximum(sat, sun)
    on_duty = shifts[:2].max(axis=0)
    expected = {
        "weekend": (weekend, worked),
        "complete": (complete, sat == sun),
        "free": (free, 1 - worked),
        "night_before_free": (night_before_free, shifts[:, :, 4, 2] * (1 - worked)),
        "senior_on": (senior_on, on_duty),
        "supervised": (supervised, np.maximum(1 - shifts[2:], on_duty)),
    }
    for name, (variable, values) in expected.items():
        if np.abs(variable.toDense() - values).max() > 1e-6:
            raise Exception(f"{name} does not equal its logical expression")


if __name__ == "__main__":
    main()
