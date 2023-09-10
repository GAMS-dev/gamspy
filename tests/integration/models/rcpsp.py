"""
Resource-Constrained Project Scheduling Problem (RCPSP)

Resource-constrained project scheduling problem (RCPSP) in a formulation
which encodes the schedule using binary finishing time indicator variables
as first proposed by Pritsker.

Problem and model formulation based on:
Pritsker, A. Alan B., Lawrence J. Waiters, and Philip M. Wolfe. "Multiproject
scheduling with limited resources: A zero-one programming approach."
Management science 16.1 (1969): 93-108.

Contains embedded Python code for parsing instance data from the classic
problem library PSPLIB from Kolisch and Sprecher.

Instance library, generator and file format:
Kolisch, Rainer, and Arno Sprecher. "PSPLIB-a project scheduling problem
library: OR software-ORSEP operations research software exchange program."
European journal of operational research 96.1 (1997): 205-216.
http://www.om-db.wi.tum.de/psplib/main.html

As default the first instance from PSPLIBs subset with 30 projects is solved
to optimality (makespan=43).
"""

from gamspy import Alias, Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Ord, Domain, Sense


# Utility functions
def ints(strs):
    return [int(s) for s in strs]


def my_set(prefix, cardinality):
    return [f"{prefix}{i + 1}" for i in range(cardinality)]


def index_of_line(lines, substr):
    return next(i for i, line in enumerate(lines) if substr in line)


def rhs_part(lines, prefix):
    return lines[index_of_line(lines, prefix)].split(":")[1]


def succs_from_line(line):
    return [f"j{j}" for j in line.split()[3:]]


def column(lines, col, row_start, row_count):
    return [
        int(lines[rowIx].split()[col])
        for rowIx in range(row_start, row_start + row_count)
    ]


def extract(mapping, fields_str):
    return (mapping[field_name] for field_name in fields_str.split(","))


def parse_psplib(filename):
    # Parse data from text file
    with open(filename) as fp:
        lines = fp.readlines()
    njobs = int(rhs_part(lines, "jobs (incl. supersource"))
    nres = int(rhs_part(lines, "- renewable").split()[0])
    nperiods = int(rhs_part(lines, "horizon"))
    prec_offset = index_of_line(lines, "PRECEDENCE RELATIONS:") + 2
    attrs_offset = index_of_line(lines, "REQUESTS/DURATIONS") + 3
    caps_offset = index_of_line(lines, "RESOURCEAVAILABILITIES") + 2

    jobs, res, periods = (
        my_set("j", njobs),
        my_set("r", nres),
        my_set("t", nperiods),
    )
    succs = {
        j: succs_from_line(lines[prec_offset + ix])
        for ix, j in enumerate(jobs)
    }
    job_durations = column(lines, 2, attrs_offset, njobs)
    resource_demands = [
        ints(lines[ix].split()[3:])
        for ix in range(attrs_offset, attrs_offset + njobs)
    ]
    resource_capacities = ints(lines[caps_offset].split())
    return dict(
        jobs=jobs,
        periods=periods,
        res=res,
        succs=succs,
        job_durations=job_durations,
        resource_capacities=resource_capacities,
        resource_demands=resource_demands,
    )


def decorate_with_time_windows(instance):
    jobs, periods, succs, job_durations = extract(
        instance, "jobs,periods,succs,job_durations"
    )

    def compute_earliest_finishing_times():
        eft = {j: 0 for j in jobs}
        for ix, i in enumerate(jobs):
            for j in jobs:
                if j in succs[i]:
                    eft[j] = max(eft[j], eft[i] + job_durations[ix])
        return eft

    def compute_latest_finishing_times():
        lft = {j: len(periods) for j in jobs}
        for i in reversed(jobs):
            for jix, j in reversed(list(enumerate(jobs))):
                if j in succs[i]:
                    lft[i] = min(lft[i], lft[j] - job_durations[jix])
        return lft

    instance["eft"] = compute_earliest_finishing_times()
    instance["lft"] = compute_latest_finishing_times()


def mini_project():
    return dict(
        jobs=["i1", "i2", "i3"],
        periods=["t1", "t2", "t3", "t4"],
        res=["r1"],
        succs=dict(i1=["i2"], i2=["i3"], i3=[]),
        job_durations=[0, 1, 0],
        resource_capacities=[1],
        resource_demands=[[0], [1], [0]],
    )


def fill_records(dataset, symbols):
    sym_fields = "j,t,r,lastJob,actual,pred,tw,fw,capacities,durations,demands"
    (
        j,
        t,
        r,
        lastJob,
        actual,
        pred,
        tw,
        fw,
        capacities,
        durations,
        demands,
    ) = extract(symbols, sym_fields)
    data_fields = (
        "jobs,periods,res,succs,eft,lft,job_durations,"
        "resource_capacities,resource_demands"
    )
    (
        jobs,
        periods,
        res,
        succs,
        eft,
        lft,
        job_durations,
        resource_capacities,
        resource_demands,
    ) = extract(dataset, data_fields)
    j.setRecords(jobs)
    t.setRecords(periods)
    r.setRecords(res)
    lastJob.setRecords(jobs[-1:])
    actual.setRecords(jobs[1:-1])
    pred.setRecords([(i, j) for i in jobs for j in jobs if j in succs[i]])
    tw.setRecords(
        [
            (j, t)
            for j in jobs
            for tix, t in enumerate(periods)
            if eft[j] <= tix + 1 <= lft[j]
        ]
    )
    fw.setRecords(
        [
            (j, t, tau)
            for jix, j in enumerate(jobs)
            for tix, t in enumerate(periods)
            for tauix, tau in enumerate(periods)
            if tix <= tauix <= tix + job_durations[jix] - 1
            and eft[j] <= tix + 1 <= lft[j]
        ]
    )
    capacities.setRecords(
        [(r, resource_capacities[rix]) for rix, r in enumerate(res)]
    )
    durations.setRecords([(j, job_durations[ix]) for ix, j in enumerate(jobs)])
    demands.setRecords(
        [
            (j, r, resource_demands[jix][rix])
            for jix, j in enumerate(jobs)
            for rix, r in enumerate(res)
        ]
    )


# Create model via GTP with algebra
def build_abstract_model():
    m = Container()

    j = Set(m, name="j")
    t = Set(m, name="t")
    r = Set(m, name="r")

    i = Alias(m, name="i", alias_with=j)
    tau = Alias(m, name="tau", alias_with=t)

    lastJob = Set(m, name="lastJob")
    actual = Set(m, name="actual")

    pred = Set(m, name="pred", domain=[i, j])

    tw = Set(m, name="tw", domain=[j, t])
    fw = Set(m, name="fw", domain=[j, t, tau])

    capacities = Parameter(m, name="capacities", domain=[r])
    durations = Parameter(m, name="durations", domain=[j])
    demands = Parameter(m, name="demands", domain=[j, r])

    makespan = Variable(m, name="makespan")
    x = Variable(m, name="x", domain=[j, t], type="Binary")

    objective = Equation(m, name="objective")
    objective.expr = makespan == Sum(
        Domain(j, t).where[tw[j, t] & lastJob[j]], x[j, t] * (Ord(t) - 1)
    )

    once = Equation(m, name="once", domain=[j])
    once[j] = Sum(t.where[tw[j, t]], x[j, t]) == 1

    precedence = Equation(m, name="precedence", domain=[i, j])
    precedence[i, j].where[pred[i, j]] = (
        Sum(t.where[tw[i, t]], Ord(t) * x[i, t])
        <= Sum(t.where[tw[j, t]], Ord(t) * x[j, t]) - durations[j]
    )

    resusage = Equation(m, name="resusage", domain=[r, t])
    resusage[r, t] = (
        Sum(
            j.where[actual[j]],
            demands[j, r] * Sum(tau.where[fw[j, t, tau]], x[j, tau]),
        )
        <= capacities[r]
    )

    rcpsp = Model(
        m,
        name="rcpsp",
        equations=m.getEquations(),
        problem="MIP",
        sense=Sense.MIN,
        objective=makespan,
    )
    makespan.lo.assign = 0

    return dict(
        m=m,
        rcpsp=rcpsp,
        j=j,
        t=t,
        r=r,
        lastJob=lastJob,
        actual=actual,
        pred=pred,
        capacities=capacities,
        durations=durations,
        demands=demands,
        tw=tw,
        fw=fw,
        x=x,
        makespan=makespan,
    )


def display_results(model, dataset):
    m, x, j, t = extract(model, "m,x,j,t")
    res = Parameter(m, name="res", domain=[j, t])
    res[j, t] = x.l[j, t]
    sts = {dataset["jobs"][jix]: t for jix, t in enumerate(res.records["t"])}
    for j, t in sts.items():
        print(f"job {j} starts at period {t}")


def main():
    load_from_file = False
    dataset = parse_psplib("j301_1.sm") if load_from_file else mini_project()
    decorate_with_time_windows(dataset)
    model = build_abstract_model()
    fill_records(dataset, model)
    model["rcpsp"].solve()
    display_results(model, dataset)


if __name__ == "__main__":
    main()
