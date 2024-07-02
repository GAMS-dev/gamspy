"""
## LICENSETYPE: Demo
## MODELTYPE: MIP
## DATAFILES: nurses_data.xlsx


The Nurse Assignment Problem (NURSES)

Nurses must be assigned to hospital shifts in accordance with various staffing constraints.
The goal of the model is to find an efficient balance between the different objectives:
* minimize the overall cost of the plan and
* assign shifts as fairly as possible.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Parameter,
    Set,
    Sum,
    Variable,
)


def read_data():
    global df_shifts
    global df_nurses
    global df_vacations
    global df_associations
    global df_incompatibilities

    nurse_xls_file = pd.ExcelFile(
        str(Path(__file__).parent.absolute()) + "/nurses_data.xlsx"
    )

    df_shifts = nurse_xls_file.parse("Shifts")
    df_shifts.index.name = "shiftId"

    df_nurses = nurse_xls_file.parse("Nurses", header=0, index_col=0)
    df_vacations = nurse_xls_file.parse("NurseVacations")
    df_associations = nurse_xls_file.parse("NurseAssociations")
    df_incompatibilities = nurse_xls_file.parse("NurseIncompatibilities")


def preprocess_data():
    # utility to convert a day string e.g. "Monday" to an integer in 0
    def day_to_day_of_week(day):
        return day_of_weeks[day.strip().lower()]

    # utility to calculate the absolute end time of a shift
    def calculate_absolute_endtime(start, end, dow):
        return 24 * dow + end + (24 if start >= end else 0)

    global shifts_id
    global nurses_names
    global nurse_vacations
    global conflicting_shifts

    shifts_id = df_shifts.index
    nurses_names = df_nurses.index

    # Add an extra column `dow` (day of week) which converts the string "day" into an integer in 0..6 (Monday is 0, Sunday is 6).
    days = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    day_of_weeks = dict(zip(days, range(7)))
    df_shifts["dow"] = df_shifts.day.apply(day_to_day_of_week)

    # Compute the absolute start and end times of each shift.
    df_shifts["wstart"] = df_shifts.start_time + 24 * df_shifts.dow
    df_shifts["wend"] = df_shifts.apply(
        lambda row: calculate_absolute_endtime(
            row.start_time, row.end_time, row.dow
        ),
        axis=1,
    )

    # Compute the duration of each shift.
    df_shifts["duration"] = df_shifts.wend - df_shifts.wstart

    # Merge the vacations dataframe with the shifts dataframe to get the vacations of each nurse.
    df_vacations["dow"] = df_vacations.day.apply(day_to_day_of_week)
    nurse_vacations = df_vacations.merge(
        df_shifts.reset_index()[["dow", "shiftId"]]
    )[["nurse", "shiftId"]]

    # Obtain the conflicting shifts
    proc_shifts = (
        df_shifts[["wstart", "wend"]]
        .sort_values(["wstart"])
        .reset_index()[["shiftId", "wstart", "wend"]]
    )
    conflicting_shifts = []
    for row in proc_shifts.itertuples():
        for row2 in proc_shifts.iloc[row[0] + 1 :].itertuples():
            if row.wend > row2.wstart:
                conflicting_shifts.append((row.shiftId, row2.shiftId))


def main():
    read_data()
    preprocess_data()

    # Define container
    m = Container(
        system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
    )

    # Sets
    n = Set(m, name="n", records=nurses_names, description="Nurses")
    s = Set(m, name="s", records=shifts_id, description="Shifts")

    nn = Alias(m, "nn", n)
    ss = Alias(m, "ss", s)

    p = Set(
        m,
        name="p",
        domain=[n, n],
        records=df_associations,
        description="Pair of nurses that prefer to work together",
    )
    i = Set(
        m,
        name="i",
        domain=[n, n],
        records=df_incompatibilities,
        description="Pair of nurses that do not get along together",
    )
    v = Set(
        m,
        name="v",
        domain=[n, s],
        records=nurse_vacations,
        description="Vacations of nurses",
    )
    c = Set(
        m,
        name="c",
        domain=[s, s],
        records=conflicting_shifts,
        description="Conflicting shifts",
    )

    # Parameters
    d = Parameter(
        m,
        name="d",
        domain=s,
        records=df_shifts["duration"],
        description="Duration of shifts    (hours)",
    )
    req = Parameter(
        m,
        name="req",
        domain=s,
        records=df_shifts["min_req"],
        description="Minimum required number of nurses in each shift",
    )
    r = Parameter(
        m,
        name="r",
        domain=n,
        records=df_nurses["pay_rate"],
        description="Pay rate for each nurse",
    )
    mt = Parameter(
        m,
        name="mt",
        records=40,
        description="Maximum weekly hours a nurse can work",
    )

    # Variables
    x = Variable(
        m,
        name="x",
        domain=[n, s],
        type="Binary",
        description="Assignment of nurses to shifts",
    )
    y = Variable(
        m,
        name="y",
        domain=n,
        type="Positive",
        description="Total working time of each nurse",
    )
    salary = Variable(
        m, name="salary", type="free", description="Total salaries spent"
    )

    # Equations
    conflicts = Equation(
        m,
        name="conflicts",
        domain=[n, s, s],
        description="A nurse cannot work in a conflicting shifts",
    )
    vacations = Equation(
        m,
        name="vacations",
        domain=[n, s],
        description="Nurses cannot work in their vacations",
    )
    preferences = Equation(
        m,
        name="preferences",
        domain=[n, n, s],
        description="Nurses that prefer to work together should work together",
    )
    incompatibilities = Equation(
        m,
        name="incompatibilities",
        domain=[n, n, s],
        description=(
            "Nurses that dont get along together should not work together"
        ),
    )
    working_time = Equation(
        m,
        name="working_time",
        domain=n,
        description="Total working time of each nurse",
    )
    max_working_time = Equation(
        m,
        name="max_working_time",
        domain=n,
        description="Maximum working time of each nurse",
    )
    min_required_nurses = Equation(
        m,
        name="min_required_nurses",
        domain=s,
        description="Minimum required number of nurses in each shift",
    )
    total_salary = Equation(
        m, name="total_salary", description="Minimize total salaries spent"
    )

    conflicts[n, s, ss].where[c[s, ss]] = x[n, s] + x[n, ss] <= 1

    vacations[n, s].where[v[n, s]] = x[n, s] == 0

    preferences[n, nn, s].where[p[n, nn]] = x[n, s] == x[nn, s]

    incompatibilities[n, nn, s].where[i[n, nn]] = x[n, s] + x[nn, s] <= 1

    working_time[n] = y[n] == Sum(s, x[n, s] * d[s])

    max_working_time[n] = y[n] <= mt

    min_required_nurses[s] = Sum(n, x[n, s]) >= req[s]

    total_salary[...] = salary == Sum(n, r[n] * y[n])

    nurses = Model(
        m,
        name="nurses",
        problem="MIP",
        equations=m.getEquations(),
        sense="Min",
        objective=salary,
    )

    nurses.solve()

    ########################################################################
    # The model solved above only considers the total salary spent.
    # However, the fairness of the assignment should also be
    # considered; nurses should work similar number of shifts and hours.
    # For that reason, The following model includes more constraints
    # to the model to make it more realistic by mainly minimizing the
    # deviation of the number of shifts and hours worked by each nurse.
    ########################################################################

    sd_shifts1 = x.pivot().sum(axis=1).std()
    sd_hours1 = (
        y.records[["n", "level"]].set_index(y.records["n"])["level"].std()
    )

    avg_shifts = Parameter(
        m,
        name="avg_shifts",
        description="Theoretical Average number of shifts per nurse",
    )
    avg_shifts[...] = Sum(s, req[s]) / len(nurses_names)

    worked = Variable(
        m,
        name="worked",
        domain=n,
        type="positive",
        description="Amount of shifts worked by each nurse",
    )
    overworked = Variable(
        m,
        name="overworked",
        domain=n,
        type="positive",
        description="Overworked shifts by each nurse",
    )
    underworked = Variable(
        m,
        name="underworked",
        domain=n,
        type="positive",
        description="Underworked shifts by each nurse",
    )

    worked_eq = Equation(
        m,
        name="worked_eq",
        domain=n,
        description="Define the amount of shifts worked by each nurse",
    )
    deviation = Equation(
        m,
        name="deviation",
        domain=n,
        description="Define the deviation of each nurse from the average",
    )

    worked_eq[n] = worked[n] == Sum(s, x[n, s])
    deviation[n] = worked[n] == avg_shifts + overworked[n] - underworked[n]

    new_obj = salary + Sum(n, overworked[n]) + Sum(n, underworked[n])

    nurses2 = Model(
        m,
        name="nurses2",
        problem="MIP",
        equations=m.getEquations(),
        sense="Min",
        objective=new_obj,
    )
    nurses2.solve()

    sd_shifts2 = x.pivot().sum(axis=1).std()
    sd_hours2 = (
        y.records[["n", "level"]].set_index(y.records["n"])["level"].std()
    )

    # Display the results
    report = Parameter(
        m, name="report", domain=["*", "*"], description="Result summary"
    )
    report["Model1", "Total Salaries"] = nurses.objective_value
    report["Model1", "sd of # shifts"] = sd_shifts1
    report["Model1", "sd of # hours"] = sd_hours1
    report["Model2", "Total Salaries"] = salary.toValue()
    report["Model2", "sd of # shifts"] = sd_shifts2
    report["Model2", "sd of # hours"] = sd_hours2

    print("\nResults summary:")
    print(report.pivot())


if __name__ == "__main__":
    main()
