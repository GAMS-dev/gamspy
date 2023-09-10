"""
Flow Shop Scheduling - (FLOWSHOP)

A workshop that produces metal pipes on demand for automotive industry
has three machines for bending the pipes, soldering the fastenings,
and assembling the links. The workshop has to produce six items, for
which the durations of the processing steps are given below. Once
started, jobs must be carried out to completion, but the
workpieces(items) may wait between the machines.

Every machine only processes one item at a time. A workpiece(item) may
not overtake any other.

What is the sequence that minimizes the total time for completing all
items (makespan)?


Gueret, C, Prins, C, and Sevaux, M, Applications of Optimization with
Xpress-MP, Translated and revised by Susanne Heipcke. Dash
Optimization, 2002.

Keywords: mixed integer linear programming, relaxed mixed integer programming,
          scenario analysis, GUSS, flow shop scheduling, production planning
"""

import gamspy as gap
import pandas as pd


def flow_shop(process_time_df, last_machine, last_item):
    c = gap.Container()

    # Sets
    i = c.addSet(
        name="i", description="item", records=process_time_df["item"].unique()
    )
    m = c.addSet(
        name="m",
        description="machine",
        records=process_time_df["machine"].unique(),
    )
    k = c.addAlias("k", i)

    # Parameters
    proctime = c.addParameter(
        name="proctime",
        domain=[m, i],
        description="process time of item i on machine m",
        records=process_time_df,
    )

    # Variables
    rank = c.addVariable(
        name="rank",
        type="binary",
        domain=[i, k],
        description="item i has position k",
    )
    start = c.addVariable(
        name="start",
        type="positive",
        domain=[m, k],
        description="start time for job in position k on m",
    )
    comp = c.addVariable(
        name="comp",
        type="positive",
        domain=[m, k],
        description="completion time for job in position k on m",
    )
    totwait = c.addVariable(
        name="totwait",
        type="free",
        description="before first job + times between jobs on last machine",
    )

    # Equations
    oneInPosition = c.addEquation(
        name="oneInPosition",
        domain=[k],
        description="every position gets a jobs",
    )
    oneRankPer = c.addEquation(
        name="oneRankPer",
        domain=[i],
        description="every job is assigned a rank",
    )
    onMachRel = c.addEquation(
        name="onMachRel",
        domain=[m, k],
    )
    perMachRel = c.addEquation(
        name="perMachRel",
        domain=[m, k],
    )
    defComp = c.addEquation(
        name="defComp",
        domain=[m, k],
        description="completion time based on start time and proctime",
    )
    defObj = c.addEquation(
        name="defObj",
        description="completion time of job rank last",
    )

    oneInPosition[k] = gap.Sum(i, rank[i, k]) == 1
    oneRankPer[i] = gap.Sum(k, rank[i, k]) == 1
    onMachRel[m, k.lead(1)] = start[m, k.lead(1)] >= comp[m, k]
    perMachRel[m.lead(1), k] = start[m.lead(1), k] >= comp[m, k]
    defComp[m, k] = comp[m, k] == start[m, k] + gap.Sum(
        i, proctime[m, i] * rank[i, k]
    )

    defObj.expr = totwait >= comp[last_machine, last_item]

    flowshop = gap.Model(
        container=c,
        name="flowshop",
        equations=c.getEquations(),
        problem="MIP",
        sense=gap.Sense.MIN,
        objective=totwait,
    )

    # set optCr to 0
    c.addOptions({"optcr": 0})

    flowshop.solve()


def prepare_data():
    # Prepare data
    process_time = pd.DataFrame(
        [
            ["blending", "i1", 3],
            ["blending", "i2", 6],
            ["blending", "i3", 3],
            ["blending", "i4", 5],
            ["blending", "i5", 5],
            ["blending", "i6", 7],
            ["soldering", "i1", 5],
            ["soldering", "i2", 4],
            ["soldering", "i3", 2],
            ["soldering", "i4", 4],
            ["soldering", "i5", 4],
            ["soldering", "i6", 5],
            ["assembly", "i1", 5],
            ["assembly", "i2", 2],
            ["assembly", "i3", 4],
            ["assembly", "i4", 6],
            ["assembly", "i5", 3],
            ["assembly", "i6", 6],
        ],
        columns=["machine", "item", "value"],
    )

    last_item = process_time["item"].unique()[-1]
    last_machine = process_time["machine"].unique()[-1]

    return process_time, last_machine, last_item


if __name__ == "__main__":
    process_time, last_machine, last_item = prepare_data()
    flow_shop(
        process_time_df=process_time,
        last_machine=last_machine,
        last_item=last_item,
    )
