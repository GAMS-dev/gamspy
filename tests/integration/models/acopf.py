"""
## LICENSETYPE: Demo
## MODELTYPE: NLP
## DATAFILES: acopf.gdx
## KEYWORDS: non linear programming, power flow optimization, rectangular power-voltage formulation


AC optimal power flow model (ACOPF)
-----------------------------------

Description: AC optimal power flow model, rectangular power-voltage formulation

Usage: python acopf.py [options]

Options:
--obj: Objective function, piecewise linear or quadratic. Default="quad"
--timeperiod: Select time period to solve. Default=1
--linelimits: Type of line limit data to use. Default="given"
--genPmin: Data for Generator lower limit. Default="given"
--allon: Option to turn on all gens or lines during solve. Default=none
--slim: Option to use apparent power limits on line. Default=0 (not used)
--qlim: Option to use D-curve constraints. Default=0 (not used)
--wind: Whether to turn off wind turbines. Can only be used with
        PrimeMover,pm_WT
--savesol: Turn on save solution option(1). Default=0
"""

import argparse
from pathlib import Path

from numpy import pi

from gamspy import (
    Container,
    Domain,
    Equation,
    Model,
    Number,
    Ord,
    Parameter,
    Set,
    Smax,
    SpecialValues,
    Sum,
    Variable,
)
from gamspy.math import Max, atan, cos, sin, sqr, sqrt


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timeperiod",
        type=int,
        default=1,
        help="Select time period to solve.",
    )
    parser.add_argument(
        "--obj",
        type=str,
        default="quad",
        help="Objective function; piecewise, linear or quadratic.",
    )
    parser.add_argument(
        "--linelimits",
        type=str,
        default="given",
        help="Type of line limit data to use.",
    )
    parser.add_argument(
        "--genPmin",
        type=str,
        default="given",
        help="Data for Generator lower limit.",
    )
    parser.add_argument(
        "--allon",
        type=str,
        default="0",
        help="Option to turn on all gens or lines during solve.",
    )
    parser.add_argument(
        "--slim",
        type=int,
        default=0,
        help="Option to use apparent power limits on line.",
    )
    parser.add_argument(
        "--qlim",
        type=int,
        default=0,
        help="Option to use D-curve constraints.",
    )
    parser.add_argument(
        "--savesol", type=int, default=0, help="Turn on save solution option."
    )
    parser.add_argument(
        "--wind",
        type=str,
        help=(
            "Whether to turn off wind turbines. Can only be used with"
            " PrimeMover,pm_WT."
        ),
    )

    args = parser.parse_args()

    # Define the container
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/acopf.gdx",
    )

    # ==== SECTION: Data Read-in from input file
    # Sets
    (
        conj,
        costcoefset,
        costptset,
        t,
        bus,
        gen,
        circuit,
        interface,
        interfacemap,
        demandbid,
        demandbidmap,
        fuel_t,
        fuel_s,
        prime_mover,
        bus_t,
        bus_s,
        gen_t,
        gen_s,
        branch_t,
        branch_s,
        line,
        transformer,
        monitored_lines,
        demandbid_t,
        demandbid_s,
    ) = m.getSymbols(
        [
            "conj",
            "costcoefset",
            "costptset",
            "t",
            "bus",
            "gen",
            "circuit",
            "interface",
            "interfacemap",
            "demandbid",
            "demandbidmap",
            "fuel_t",
            "fuel_s",
            "prime_mover",
            "bus_t",
            "bus_s",
            "gen_t",
            "gen_s",
            "branch_t",
            "branch_s",
            "line",
            "transformer",
            "monitored_lines",
            "demandbid_t",
            "demandbid_s",
        ]
    )

    # Aliases
    i, j, c, gen1 = m.getSymbols(["i", "j", "c", "gen1"])

    # Parameters
    (
        baseMVA,
        total_cost,
        businfo,
        geninfo,
        fuelinfo,
        branchinfo,
        interfaceinfo,
        demandbidinfo,
    ) = m.getSymbols(
        [
            "baseMVA",
            "total_cost",
            "businfo",
            "geninfo",
            "fuelinfo",
            "branchinfo",
            "interfaceinfo",
            "demandbidinfo",
        ]
    )

    t.setRecords([args.timeperiod])

    # ==== SECTION: Validity of options
    # linelimits, case insensitive
    if args.linelimits not in ["inf", "uwcalc", "given"]:
        raise ValueError(
            f"Fix invalid option: --linelimit={args.linelimits}. Should be one"
            " of ['inf', 'uwcalc', 'given']"
        )

    # genPmin, case insensitive
    if args.genPmin not in ["0", "uwcalc", "given"]:
        raise ValueError(
            f"Fix invalid option: --genPmin={args.genPmin}. Should be one of"
            " ['0', 'uwcalc', 'given']"
        )

    # allon, case insensitive
    if args.allon not in ["gens", "lines", "both", "0"]:
        raise ValueError(
            f"Fix invalid option: --allon={args.allon}. Should be one of"
            " ['gens', 'lines', 'both', '0']"
        )

    # ==== SECTION: Data Declaration (extracted/manipulated from datafile)
    # -- All OPF models
    # Parameters
    type = Parameter(
        m,
        name="type",
        domain=bus,
        description=(
            "bus type (probably irrelevant, but gives reference bus[es])"
        ),
    )
    pf = Parameter(
        m, name="pf", domain=bus, description="bus demand power factor"
    )
    Pd = Parameter(
        m, name="Pd", domain=bus, description="bus real power demand"
    )

    Pg = Parameter(
        m, name="Pg", domain=gen, description="gen real power output"
    )
    Pmax = Parameter(
        m, name="Pmax", domain=gen, description="gen maximum real power output"
    )
    Pmin = Parameter(
        m, name="Pmin", domain=gen, description="gen minimum real power output"
    )
    Va = Parameter(m, name="Va", domain=bus, description="bus voltage angle")

    Vm = Parameter(
        m, name="Vm", domain=bus, description="bus voltage magnitude"
    )
    MaxVm = Parameter(
        m,
        name="MaxVm",
        domain=bus,
        description="maximum bus voltage magnitude",
    )
    MinVm = Parameter(
        m,
        name="MinVm",
        domain=bus,
        description="minimum bus voltage magnitude",
    )
    Gs = Parameter(
        m, name="Gs", domain=bus, description="bus shunt conductance"
    )

    atBus = Parameter(
        m, name="atBus", domain=[gen, bus], description="Location of generator"
    )
    status = Parameter(
        m, name="status", domain=gen, description="generator status"
    )

    costcoef = Parameter(
        m,
        name="costcoef",
        domain=[gen, costcoefset],
        description="gen cost coefficients",
    )
    costpts_x = Parameter(
        m,
        name="costpts_x",
        domain=[gen, costptset],
        description="gen cost breakpoints (piecewise linear)",
    )
    costpts_y = Parameter(
        m,
        name="costpts_y",
        domain=[gen, costptset],
        description="gen cost breakpoints (piecewise linear)",
    )

    costmodel = Parameter(
        m, name="costmodel", domain=gen, description="gen cost model type"
    )
    numcostpts = Parameter(
        m,
        name="numcostpts",
        domain=gen,
        description="gen cost number of piecewise points",
    )
    numcostcoef = Parameter(
        m,
        name="numcostcoef",
        domain=gen,
        description="gen cost number of coefficients",
    )
    noloadcost = Parameter(
        m,
        name="noloadcost",
        domain=gen,
        description=(
            "generator no load operating cost for piecewise cost functions"
        ),
    )

    r = Parameter(m, name="r", domain=[i, j, c], description="line resistance")
    x = Parameter(m, name="x", domain=[i, j, c], description="line reactance")
    B = Parameter(
        m, name="B", domain=[i, j, c], description="line susceptance"
    )
    ratio = Parameter(
        m, name="ratio", domain=[i, j, c], description="transformer tap ratio"
    )
    angle = Parameter(
        m, name="angle", domain=[i, j, c], description="transformer tap angle"
    )
    rateA = Parameter(
        m, name="rateA", domain=[i, j, c], description="line power limits (MW)"
    )
    currentrate = Parameter(
        m,
        name="currentrate",
        domain=[i, j, c],
        description="line current limits",
    )
    branchstatus = Parameter(
        m, name="branchstatus", domain=[i, j, c], description="line status"
    )
    interfaceLimit = Parameter(
        m,
        name="interfaceLimit",
        domain=[interface],
        description="Limit on power across each interface",
    )

    # Bus type
    type[bus] = businfo[bus, "type", "given"]
    # Power factor
    pf[bus] = businfo[bus, "pf", "given"]
    # Bus demand(real power)
    Pd[bus] = businfo[bus, "Pd", f"{args.timeperiod}"] / baseMVA

    # Bus shunt conductance
    Gs[bus] = businfo[bus, "Gs", "given"] / baseMVA

    atBus[gen, bus].where[geninfo[gen, "atBus", bus]] = 1
    Pg[gen] = geninfo[gen, "Pg", f"{args.timeperiod}"] / baseMVA

    # Maximum power generation
    Pmax[gen] = geninfo[gen, "Pmax", "given"] / baseMVA

    # Minimum power generation options
    if args.genPmin == 0:
        Pmin[gen] = 0
    else:
        Pmin[gen] = geninfo[gen, "Pmin", f"{args.genPmin}"] / baseMVA

    # Voltage angle
    Va[bus] = businfo[bus, "Va", f"{args.timeperiod}"] * pi / 180
    # Voltage magnitude information
    Vm[bus] = businfo[bus, "Vm", f"{args.timeperiod}"]
    MaxVm[bus] = businfo[bus, "maxVm", "given"]
    MinVm[bus] = businfo[bus, "minVm", "given"]

    # Initial generator commitment
    status[gen] = geninfo[gen, "status", f"{args.timeperiod}"]

    # Initial branch status (active/not connected)
    branchstatus[i, j, c].where[line[i, j, c]] = branchinfo[
        i, j, c, "branchstatus", f"{args.timeperiod}"
    ]

    # Define original cost model in dataset
    costmodel[gen] = geninfo[gen, "costmodel", "given"]

    # No load cost
    noloadcost[gen] = geninfo[gen, "noloadcost", "given"]

    # Quadratic objective function
    numcostpts[gen] = geninfo[gen, "numcostpts", "given"]
    costcoef[gen, costcoefset].where[geninfo[gen, "costcoef", costcoefset]] = (
        geninfo[gen, "costcoef", costcoefset]
    )

    # Piecewise linear information
    numcostcoef[gen] = geninfo[gen, "numcostcoef", "given"]
    costpts_x[gen, costptset].where[geninfo[gen, "costpts_x", costptset]] = (
        geninfo[gen, "costpts_x", costptset]
    )
    costpts_y[gen, costptset].where[geninfo[gen, "costpts_y", costptset]] = (
        geninfo[gen, "costpts_y", costptset]
    )

    # Line resistance (r) and reactance (x)
    r[i, j, c].where[line[i, j, c]] = branchinfo[i, j, c, "r", "given"]
    x[i, j, c].where[line[i, j, c]] = branchinfo[i, j, c, "x", "given"]

    # Line limit (active power)
    rateA[i, j, c].where[line[i, j, c]] = (
        branchinfo[i, j, c, "rateA", f"{args.linelimits}"] / baseMVA
    )
    rateA[j, i, c].where[line[i, j, c]] = (
        branchinfo[i, j, c, "rateA", f"{args.linelimits}"] / baseMVA
    )

    # If linelimits=inf, no monitored lines
    if args.linelimits == "inf":
        monitored_lines[i, j, c] = 0

    # Limit on power across each interface
    interfaceLimit[interface] = (
        interfaceinfo[interface, f"{args.timeperiod}", "rateA"] / baseMVA
    )

    # Line current
    currentrate[i, j, c].where[line[i, j, c]] = branchinfo[
        i, j, c, "currentrateA", f"{args.linelimits}"
    ]
    currentrate[j, i, c].where[line[i, j, c]] = branchinfo[
        i, j, c, "currentrateA", f"{args.linelimits}"
    ]

    # Take down all lines to buses marked as "isolated"
    branchstatus[i, j, c].where[(type[i] == 4) | (type[j] == 4)] = 0

    # Line susceptance
    B[i, j, c].where[line[i, j, c]] = -x[i, j, c] / (
        sqr(r[i, j, c]) + sqr(x[i, j, c])
    )
    B[j, i, c].where[B[i, j, c]] = B[i, j, c]

    # transformer tap ratios and angles
    ratio[i, j, c].where[line[i, j, c]] = branchinfo[i, j, c, "ratio", "given"]
    ratio[j, i, c].where[ratio[i, j, c]] = ratio[i, j, c]
    angle[i, j, c].where[line[i, j, c]] = (
        branchinfo[i, j, c, "angle", "given"] * pi / 180
    )
    angle[j, i, c].where[angle[i, j, c]] = -angle[i, j, c]

    # ---- AC model data types
    # Parameters
    Qd = Parameter(
        m, name="Qd", domain=bus, description="bus reactive power demand"
    )

    Qg = Parameter(
        m, name="Qg", domain=gen, description="gen reactive power output"
    )
    Qmax = Parameter(
        m,
        name="Qmax",
        domain=gen,
        description="gen maximum reactive power output",
    )
    Qmin = Parameter(
        m,
        name="Qmin",
        domain=gen,
        description="gen minimum reactive power output",
    )

    Bs = Parameter(
        m, name="Bs", domain=bus, description="bus shunt susceptance"
    )
    _ = Parameter(
        m,
        name="yb",
        domain=[i, j, conj],
        description="Bus admittance matrix, Ybus",
    )

    g = Parameter(
        m, name="g", domain=[i, j, c], description="line conductance"
    )
    bc = Parameter(
        m, name="bc", domain=[i, j, c], description="line charging susceptance"
    )
    Bswitched = Parameter(
        m,
        name="Bswitched",
        domain=[bus, bus_s],
        description="susceptance of switched shunts",
    )
    numBswitched = Parameter(
        m,
        name="numBswitched",
        domain=[bus, bus_s],
        description=(
            "number of each type of switched shunt elements at each bus"
        ),
    )

    # Reactive power information
    Qd[bus] = businfo[bus, "Qd", f"{args.timeperiod}"] / baseMVA
    Qmax[gen] = geninfo[gen, "Qmax", "given"] / baseMVA
    Qmin[gen] = geninfo[gen, "Qmin", "given"] / baseMVA
    Qg[gen] = geninfo[gen, "Qg", f"{args.timeperiod}"] / baseMVA

    # Bus shunt conductance and susceptance
    Bs[bus] = businfo[bus, "Bs", "given"] / baseMVA

    # line conductance
    g[i, j, c].where[line[i, j, c]] = r[i, j, c] / (
        sqr(r[i, j, c]) + sqr(x[i, j, c])
    )
    g[j, i, c].where[g[i, j, c]] = g[i, j, c]

    # line charging conductance
    bc[i, j, c].where[line[i, j, c]] = branchinfo[i, j, c, "bc", "given"]
    bc[j, i, c].where[bc[i, j, c]] = bc[i, j, c]

    # number and susceptance of switched shunt element data
    numBswitched[bus, bus_s] = businfo[bus, "switchedelements", bus_s]
    Bswitched[bus, bus_s] = businfo[bus, "switchedBs", bus_s] / baseMVA

    # ==== SECTION: Additional Model Options
    # -- %allon% options
    if args.allon == "gens":
        status[gen] = 1
    elif args.allon == "lines":
        branchstatus[i, j, c].where[line[i, j, c]] = 1
    elif args.allon == "both":
        status[gen] = 1
        branchstatus[i, j, c].where[line[i, j, c]] = 1

    # ===== SECTION: DATA MANIPULATION
    # --- Define load, gen buses and active lines
    # Sets
    load = Set(m, name="load", domain=bus, description="Load buses")
    isGen = Set(m, name="isGen", domain=bus, description="Generator buses")
    activeGen = Set(
        m, name="activeGen", domain=bus, description="Active generator buses"
    )
    _ = Set(m, name="isLine", domain=[i, j], description="Active (i,j) line")

    load[bus].where[Sum(gen, atBus[gen, bus]) == 0] = 1
    isGen[bus].where[~(load[bus])] = 1
    activeGen[bus].where[
        isGen[bus] & (Sum(gen.where[atBus[gen, bus]], status[gen]) > 0)
    ] = 1

    # ===== SECTION: VARIABLE DEFINITION
    # Free variables
    V_P = Variable(
        m,
        name="V_P",
        domain=gen,
        description="Real power generation of generator",
    )
    V_Q = Variable(
        m,
        name="V_Q",
        domain=gen,
        description="Reactive power generation of generator",
    )

    V_real = Variable(
        m, name="V_real", domain=i, description="Real part of bus voltage"
    )
    V_imag = Variable(
        m, name="V_imag", domain=i, description="Imaginary part of bus voltage"
    )

    V_LineP = Variable(
        m,
        name="V_LineP",
        domain=[i, j, c],
        description="Real power flowing from bus i towards bus j on line c",
    )
    V_LineQ = Variable(
        m,
        name="V_LineQ",
        domain=[i, j, c],
        description=(
            "Reactive power flowing from bus i towards bus j on line c"
        ),
    )
    V_interfaceP = Variable(
        m,
        name="V_interfaceP",
        domain=[i, j, c],
        description="Real power flowing on interface (i,j,c)",
    )

    V_objcost = Variable(
        m, name="V_objcost", description="Total cost of objective function"
    )

    # Positive variables
    V_shunt = Variable(
        m,
        name="V_shunt",
        domain=[bus, bus_s],
        description="Bus shunt susceptance",
    )
    V_pw_cost = Variable(
        m, name="V_pw_cost", domain=gen, description="Generator piecewise cost"
    )
    V_Pd_elastic = Variable(
        m,
        name="V_Pd_elastic",
        domain=demandbid,
        description="Elastic incremental demand",
    )
    V_demandbid_rev = Variable(
        m,
        name="V_demandbid_rev",
        domain=demandbid,
        description="Revenue from elastic incremental demand",
    )

    # ===== SECTION: EQUATION DEFINITION
    # Equations
    c_SLimit = Equation(
        m,
        name="c_SLimit",
        domain=[i, j, c],
        description="Apparent power limit on line ijc",
    )
    c_V_limit_lo = Equation(
        m,
        name="c_V_limit_lo",
        domain=i,
        description="Limit voltage magnitude on a line",
    )
    c_V_limit_up = Equation(
        m,
        name="c_V_limit_up",
        domain=i,
        description="Limit voltage magnitude on a line",
    )

    c_LinePij = Equation(
        m,
        name="c_LinePij",
        domain=[i, j, c],
        description="Real power flowing from bus i into bus j along line c",
    )
    c_LinePji = Equation(
        m,
        name="c_LinePji",
        domain=[i, j, c],
        description="Real power flowing from bus j into bus i along line c",
    )
    c_LineQij = Equation(
        m,
        name="c_LineQij",
        domain=[i, j, c],
        description=(
            "Reactive power flowing from bus i into bus j along line c"
        ),
    )
    c_LineQji = Equation(
        m,
        name="c_LineQji",
        domain=[i, j, c],
        description=(
            "Reactive power flowing from bus j into bus i along line c"
        ),
    )

    c_BalanceP = Equation(
        m,
        name="c_BalanceP",
        domain=bus,
        description="Balance of real power for bus",
    )
    c_BalanceQ = Equation(
        m,
        name="c_BalanceQ",
        domain=bus,
        description="Balance of reactive power for bus",
    )

    c_InterfaceP = Equation(
        m,
        name="c_InterfaceP",
        domain=[i, j, c],
        description=(
            "Definition of real power on interfaces involving (i,j,c) at time"
        ),
    )
    c_InterfaceLimit = Equation(
        m,
        name="c_InterfaceLimit",
        domain=interface,
        description="Limit of real power on interface at time t",
    )

    c_pw_cost = Equation(
        m,
        name="c_pw_cost",
        domain=[gen, costptset],
        description="Generator piecewise cost functions",
    )
    c_obj = Equation(m, name="c_obj", description="Objective function")

    # ===== SECTION: EQUATIONS PART 1
    # Apparent power limit on line ijc
    c_SLimit[i, j, c].where[branchstatus[i, j, c] | branchstatus[j, i, c]] = (
        sqr(V_LineP[i, j, c]) + sqr(V_LineQ[i, j, c]) <= sqr(rateA[i, j, c])
    )

    # Limit voltage magnitude on a line
    c_V_limit_lo[i] = sqr(V_real[i]) + sqr(V_imag[i]) >= sqr(MinVm[i])

    # Limit voltage magnitude on a line
    c_V_limit_up[i] = sqr(V_real[i]) + sqr(V_imag[i]) <= sqr(MaxVm[i])

    # Real power flowing from bus i into bus j along line c
    c_LinePij[i, j, c].where[branchstatus[i, j, c]] = V_LineP[i, j, c] == (
        g[i, j, c] / sqr(ratio[i, j, c])
    ) * (sqr(V_real[i]) + sqr(V_imag[i])) - (1 / ratio[i, j, c]) * (
        (g[i, j, c] * cos(angle[i, j, c]) - B[i, j, c] * sin(angle[i, j, c]))
        * (V_real[i] * V_real[j] + V_imag[i] * V_imag[j])
        + (B[i, j, c] * cos(angle[i, j, c]) + g[i, j, c] * sin(angle[i, j, c]))
        * (V_real[j] * V_imag[i] - V_real[i] * V_imag[j])
    )

    # Real power flowing from bus j into bus i along line c
    c_LinePji[i, j, c].where[branchstatus[i, j, c]] = V_LineP[j, i, c] == g[
        i, j, c
    ] * (sqr(V_real[j]) + sqr(V_imag[j])) - (1 / ratio[i, j, c]) * (
        (g[i, j, c] * cos(angle[i, j, c]) + B[i, j, c] * sin(angle[i, j, c]))
        * (V_real[j] * V_real[i] + V_imag[j] * V_imag[i])
        + (B[i, j, c] * cos(angle[i, j, c]) - g[i, j, c] * sin(angle[i, j, c]))
        * (V_real[i] * V_imag[j] - V_real[j] * V_imag[i])
    )

    # Reactive power flowing from bus i into bus j along line c
    c_LineQij[i, j, c].where[branchstatus[i, j, c]] = V_LineQ[i, j, c] == -(
        (B[i, j, c] + bc[i, j, c] / 2) / sqr(ratio[i, j, c])
    ) * (sqr(V_real[i]) + sqr(V_imag[i])) - (1 / ratio[i, j, c]) * (
        (g[i, j, c] * cos(angle[i, j, c]) - B[i, j, c] * sin(angle[i, j, c]))
        * (V_real[j] * V_imag[i] - V_real[i] * V_imag[j])
        - (B[i, j, c] * cos(angle[i, j, c]) + g[i, j, c] * sin(angle[i, j, c]))
        * (V_real[i] * V_real[j] + V_imag[i] * V_imag[j])
    )

    # Reactive power flowing from bus j into bus i along line c
    c_LineQji[i, j, c].where[branchstatus[i, j, c]] = V_LineQ[j, i, c] == -(
        B[i, j, c] + bc[i, j, c] / 2
    ) * (sqr(V_real[j]) + sqr(V_imag[j])) - (1 / ratio[i, j, c]) * (
        (g[i, j, c] * cos(angle[i, j, c]) + B[i, j, c] * sin(angle[i, j, c]))
        * (V_real[i] * V_imag[j] - V_real[j] * V_imag[i])
        - (B[i, j, c] * cos(angle[i, j, c]) - g[i, j, c] * sin(angle[i, j, c]))
        * (V_real[j] * V_real[i] + V_imag[j] * V_imag[i])
    )

    # Balance of real power for bus
    c_BalanceP[i].where[type[i] != 4] = Sum(
        gen.where[atBus[gen, i] & status[gen]], V_P[gen]
    ) - Pd[i] == Sum(
        Domain(j, c).where[branchstatus[i, j, c]], V_LineP[i, j, c]
    ) + Sum(Domain(j, c).where[branchstatus[j, i, c]], V_LineP[i, j, c]) + Gs[
        i
    ] * (sqr(V_real[i]) + sqr(V_imag[i]))

    # Balance of reactive power for bus
    c_BalanceQ[i].where[type[i] != 4] = Sum(
        gen.where[atBus[gen, i] & status[gen]], V_Q[gen]
    ) - Qd[i] == Sum(
        Domain(j, c).where[branchstatus[i, j, c]], V_LineQ[i, j, c]
    ) + Sum(Domain(j, c).where[branchstatus[j, i, c]], V_LineQ[i, j, c]) - Bs[
        i
    ] * (sqr(V_real[i]) + sqr(V_imag[i])) - (
        sqr(V_real[i]) + sqr(V_imag[i])
    ) * Sum(
        bus_s.where[~bus_s.sameAs("given")],
        Bswitched[i, bus_s] * V_shunt[i, bus_s],
    )

    # Definition of real power on interfaces involving [i,j,c] at time t
    # Since we only care about interfaces in the specified direction, we don't need abs(LinePower)
    c_InterfaceP[i, j, c].where[
        (branchstatus[i, j, c] | branchstatus[j, i, c])
        & (Sum(interface.where[interfacemap[interface, i, j]], 1) >= 1)
    ] = V_interfaceP[i, j, c] == V_LineP[i, j, c]

    # Limit of real power on interface at time t
    c_InterfaceLimit[interface] = (
        Sum(
            Domain(i, j, c).where[
                interfacemap[interface, i, j]
                & (branchstatus[i, j, c] | branchstatus[j, i, c])
            ],
            V_interfaceP[i, j, c],
        )
        <= interfaceLimit[interface]
    )

    # Set costmodel variable
    if args.obj == "pwl":
        costmodel[gen] = 1
    elif args.obj in ["quad", "linear"]:
        costmodel[gen] = 2
    elif args.obj == "0":
        costmodel[gen] = 0
    else:
        raise ValueError(
            f"Fix invalid option: --obj={args.obj}. Should be one of ['pwl',"
            " 'quad', 'linear', '0']"
        )

    # -- Convexity Check
    # Not part of system of equations
    # LP/QCP/NLP can't handle nonconvex piecewise linear cost functions
    thisgen = Set(m, name="thisgen", domain=[gen])

    cur_slope = Parameter(m, name="cur_slope")
    next_slope = Parameter(m, name="next_slope")

    for idx, gen_ in enumerate(gen.toList()):
        if not (
            (status.records.at[idx, "value"])
            and (costmodel.records.at[idx, "value"] == 1)
            and (numcostpts.records.at[idx, "value"] > 2)
        ):
            continue

        next_slope[...] = (costpts_y[gen_, "2"] - costpts_y[gen_, "1"]) / (
            costpts_x[gen_, "2"] - costpts_x[gen_, "1"]
        )

        for idx2, cps_ in enumerate(costptset.toList()):
            if not (idx2 + 1) < (numcostpts.records.at[idx, "value"] - 1):
                continue

            cur_slope[...] = next_slope

            # Define the queries
            query1 = costpts_x.records.query(
                f'gen == "{idx + 1}" & costptset == "{idx2 + 2}"'
            ).value.values
            query1 = query1[0] if len(query1) > 0 else None

            query2 = costpts_x.records.query(
                f'gen == "{idx + 1}" & costptset == "{idx2 + 1}"'
            ).value.values
            query2 = query2[0] if len(query2) > 0 else None

            if (idx2 + 1 < numcostpts.records.at[idx, "value"] - 2) and (
                query1 == query2
            ):
                raise ValueError("Zero-length piecewise segment detected")

            next_slope[...] = (
                costpts_y[gen_, f"{int(cps_) + 2}"]
                - costpts_y[gen_, f"{int(cps_) + 1}"]
            ) / (
                costpts_x[gen_, f"{int(cps_) + 2}"]
                - costpts_x[gen_, f"{int(cps_) + 1}"]
            )

            if cur_slope.toValue() - next_slope.toValue() > 1e-8:
                thisgen[gen1] = False
                thisgen[gen_] = True
                print("thisgen: ", thisgen.toList())
                raise Exception(
                    "Nonconvex piecewise linear costs not supported"
                )

    # ===== SECTION: EQUATIONS PART 2
    # Defining piecewise linear generator cost curves
    # P is in per-unit, costpts_x is in MW, and costpts_y is in $/hr
    c_pw_cost[gen, costptset].where[
        status[gen]
        & (Ord(costptset) < numcostpts[gen])
        & (costmodel[gen] == 1)
    ] = (
        V_pw_cost[gen]
        >= (
            (costpts_y[gen, costptset + 1] - costpts_y[gen, costptset])
            / (costpts_x[gen, costptset + 1] - costpts_x[gen, costptset])
        )
        * (V_P[gen] * baseMVA - costpts_x[gen, costptset])
        + costpts_y[gen, costptset]
    )

    # Piecewise linear objective function
    if args.obj == "pwl":
        c_obj[...] = V_objcost == Sum(
            gen.where[costmodel[gen] == 1], V_pw_cost[gen]
        )

    # Quadratic objective function
    elif args.obj == "quad":
        c_obj[...] = V_objcost == Sum(
            gen.where[status[gen] & (costmodel[gen] == 2)],
            costcoef[gen, "0"]
            + costcoef[gen, "1"] * V_P[gen] * baseMVA
            + costcoef[gen, "2"] * sqr(V_P[gen] * baseMVA),
        )

    # Linear objective function
    elif args.obj == "linear":
        c_obj[...] = V_objcost == Sum(
            gen.where[(status[gen]) & (costmodel[gen] == 2)],
            costcoef[gen, "0"] + costcoef[gen, "1"] * V_P[gen] * baseMVA,
        )

    # D-curve limits
    # Reactive power circle constraints (see Dan's pdf for derivation).
    # Add to a model when qlim=1
    if args.qlim == 1:
        R_max = Parameter(m, name="R_max", domain=gen)
        nameplate_pf = Parameter(m, name="nameplate_pf", domain=gen)
        Qfield = Parameter(m, name="Qfield", domain=gen)
        Rfield = Parameter(m, name="Rfield", domain=gen)
        Qend = Parameter(m, name="Qend", domain=gen)
        Rend = Parameter(m, name="Rend", domain=gen)

        R_max[gen] = geninfo[gen, "R_max", "given"] / baseMVA
        nameplate_pf[gen] = geninfo[gen, "nameplate_pf", "given"] / baseMVA
        Qfield[gen] = geninfo[gen, "Qfield", "given"] / baseMVA
        Rfield[gen] = geninfo[gen, "Rfield", "given"] / baseMVA
        Qend[gen] = geninfo[gen, "Qend", "given"] / baseMVA
        Rend[gen] = geninfo[gen, "Rend", "given"] / baseMVA

        c_Armature = Equation(
            m,
            name="c_Armature",
            domain=gen,
            description="Armature current limit for reactive power",
        )
        c_Field = Equation(
            m,
            name="c_Field",
            domain=gen,
            description="Field current limit for reactive power",
        )
        c_Heating = Equation(
            m,
            name="c_Heating",
            domain=gen,
            description="End region heating limit for reactive power",
        )

        c_Armature[gen].where[
            status[gen] & (Qfield[gen] != SpecialValues.EPS)
        ] = sqr(V_P[gen]) + sqr(V_Q[gen]) <= sqr(R_max[gen])

        c_Field[gen].where[
            status[gen]
            & (Qfield[gen] != SpecialValues.NA)
            & (Qfield[gen] != SpecialValues.EPS)
        ] = sqr(V_P[gen]) + sqr(V_Q[gen] - Qfield[gen]) <= sqr(Rfield[gen])

        c_Heating[gen].where[
            status[gen]
            & (Qend[gen] != SpecialValues.NA)
            & (Qend[gen] != SpecialValues.EPS)
        ] = sqr(V_P[gen]) + sqr(V_Q[gen] - Qend[gen]) <= sqr(Rend[gen])

        # To represent the lower portion of the capability curve as a horizonrtal line when Qend is not physical
        # To use rectangular constraints when size of box is not big enough to be inside of the D-curve
        V_Q.lo[gen].where[
            (status[gen]) & (Qend[gen] == SpecialValues.NA)
            | (Qfield[gen] == SpecialValues.EPS)
        ] = Qmin[gen]
        # To represent the upper portion of the capability curve as a horizonrtal line when Qfield is not physical
        # To use rectangular constraints when size of box is not big enough to be inside of the D-curve
        V_Q.up[gen].where[
            (status[gen]) & (Qfield[gen] == SpecialValues.NA)
            | (Qfield[gen] == SpecialValues.EPS)
        ] = Qmax[gen]

    # ===== SECTION: MODEL DEFINITION
    equation_list = [
        c_V_limit_lo,
        c_V_limit_up,
        c_LinePij,
        c_LinePji,
        c_LineQij,
        c_LineQji,
        c_BalanceP,
        c_BalanceQ,
        c_InterfaceP,
        c_InterfaceLimit,
        c_pw_cost,
        c_obj,
    ]
    if args.qlim == 1:
        equation_list.extend([c_Armature, c_Field, c_Heating])
    if args.slim == 1:
        equation_list.append(c_SLimit)
    acopf = Model(
        m,
        name="acopf",
        problem="nlp",
        equations=equation_list,
        sense="min",
        objective=V_objcost,
    )

    # ===== SECTION: VARIABLE BOUNDS
    # Generator active power generation limits
    V_P.lo[gen].where[status[gen]] = Pmin[gen]
    V_P.up[gen].where[status[gen]] = Pmax[gen]
    V_P.fx[gen].where[~status[gen]] = 0

    # Generator reactive power generation limits
    # Does not impose Qmax, Qmin limits when the D-curve contraint is applied
    if args.qlim == 0:
        V_Q.lo[gen].where[status[gen]] = Qmin[gen]
        V_Q.up[gen].where[status[gen]] = Qmax[gen]
    V_Q.fx[gen].where[~status[gen]] = 0

    # Bus voltage magnitude limits
    V_real.lo[bus] = -MaxVm[bus]
    V_real.up[bus] = MaxVm[bus]
    V_imag.lo[bus] = -MaxVm[bus]
    V_imag.up[bus] = MaxVm[bus]
    V_imag.fx[bus].where[type[bus] == 3] = 0

    if args.slim != 1:
        # Line real power flow limits
        V_LineP.lo[i, j, c].where[branchstatus[i, j, c]] = -rateA[i, j, c]
        V_LineP.up[i, j, c].where[branchstatus[i, j, c]] = rateA[i, j, c]
        V_LineP.lo[j, i, c].where[branchstatus[i, j, c]] = -rateA[i, j, c]
        V_LineP.up[j, i, c].where[branchstatus[i, j, c]] = rateA[i, j, c]

    if args.wind == 1:
        # Needed to avoid compilation error. Puts strings into UEL
        _ = Set(m, name="winddesc", records=["PrimeMover", "pm_WT"])
        # Wind turbines are not reliable sources of power, treated differently
        windTurbine = Parameter(m, name="windTurbine", domain=[gen])
        windTurbine[gen].where[geninfo[gen, "PrimeMover", "pm_WT"] == 1] = 1
        V_P.fx[gen].where[windTurbine[gen]] = 0

    # Bus shunt susceptance
    V_shunt.up[bus, bus_s] = numBswitched[bus, bus_s]

    # Elastic demand not considered
    V_Pd_elastic.fx[demandbid] = 0
    V_demandbid_rev.fx[demandbid] = 0

    # ===== SECTION: VARIABLE INITIAL STARTING POINTS
    V_shunt.l[bus, bus_s] = 1

    # Set initial conditions
    V_P.l[gen].where[status[gen]] = (Pmin[gen] + Pmax[gen]) / 2
    V_Q.l[gen].where[status[gen]] = (Qmin[gen] + Qmax[gen]) / 2
    V_real.l[bus] = (MinVm[bus] + MaxVm[bus]) / 2
    # V_imag can stay 0, since angles are allowed to range in (-pi, pi)
    V_imag.l[bus] = 0

    # Calculate line power and objective value from P, Q, V_real, V_imag
    V_LineP.l[i, j, c].where[branchstatus[i, j, c]] = (
        g[i, j, c] / sqr(ratio[i, j, c])
    ) * (sqr(V_real.l[i]) + sqr(V_imag.l[i])) - (1 / ratio[i, j, c]) * (
        (g[i, j, c] * cos(angle[i, j, c]) - B[i, j, c] * sin(angle[i, j, c]))
        * (V_real.l[i] * V_real.l[j] + V_imag.l[i] * V_imag.l[j])
        + (B[i, j, c] * cos(angle[i, j, c]) + g[i, j, c] * sin(angle[i, j, c]))
        * (V_real.l[j] * V_imag.l[i] - V_real.l[i] * V_imag.l[j])
    )

    V_LineP.l[j, i, c].where[branchstatus[i, j, c]] = g[i, j, c] * (
        sqr(V_real.l[j]) + sqr(V_imag.l[j])
    ) - (1 / ratio[i, j, c]) * (
        (g[i, j, c] * cos(angle[i, j, c]) + B[i, j, c] * sin(angle[i, j, c]))
        * (V_real.l[j] * V_real.l[i] + V_imag.l[j] * V_imag.l[i])
        + (B[i, j, c] * cos(angle[i, j, c]) - g[i, j, c] * sin(angle[i, j, c]))
        * (V_real.l[i] * V_imag.l[j] - V_real.l[j] * V_imag.l[i])
    )

    V_LineQ.l[i, j, c].where[branchstatus[i, j, c]] = -(
        (B[i, j, c] + bc[i, j, c] / 2) / sqr(ratio[i, j, c])
    ) * (sqr(V_real.l[i]) + sqr(V_imag.l[i])) - (1 / ratio[i, j, c]) * (
        (g[i, j, c] * cos(angle[i, j, c]) - B[i, j, c] * sin(angle[i, j, c]))
        * (V_real.l[j] * V_imag.l[i] - V_real.l[i] * V_imag.l[j])
        - (B[i, j, c] * cos(angle[i, j, c]) + g[i, j, c] * sin(angle[i, j, c]))
        * (V_real.l[i] * V_real.l[j] + V_imag.l[i] * V_imag.l[j])
    )

    V_LineQ.l[j, i, c].where[branchstatus[i, j, c]] = -(
        B[i, j, c] + bc[i, j, c] / 2
    ) * (sqr(V_real.l[j]) + sqr(V_imag.l[j])) - (1 / ratio[i, j, c]) * (
        (g[i, j, c] * cos(angle[i, j, c]) + B[i, j, c] * sin(angle[i, j, c]))
        * (V_real.l[i] * V_imag.l[j] - V_real.l[j] * V_imag.l[i])
        - (B[i, j, c] * cos(angle[i, j, c]) - g[i, j, c] * sin(angle[i, j, c]))
        * (V_real.l[j] * V_real.l[i] + V_imag.l[j] * V_imag.l[i])
    )

    V_objcost.l = Sum(
        gen,
        costcoef[gen, "2"] * V_P.l[gen] * baseMVA
        + costcoef[gen, "1"] * sqr(V_P.l[gen] * baseMVA),
    )

    V_pw_cost.l[gen].where[status[gen] & (costmodel[gen] == 1)] = Max(
        0,
        Smax(
            costptset.where[Ord(costptset) < numcostpts[gen]],
            (
                (costpts_y[gen, costptset + 1] - costpts_y[gen, costptset])
                / (costpts_x[gen, costptset + 1] - costpts_x[gen, costptset])
            )
            * (V_P.l[gen] * baseMVA - costpts_x[gen, costptset])
            + costpts_y[gen, costptset]
            - noloadcost[gen],
        ),
    )

    if args.obj.casefold() == "linear":
        V_objcost.l = Sum(
            gen.where[(status[gen]) & (costmodel[gen] == 2)],
            costcoef[gen, "0"] + costcoef[gen, "1"] * V_P.l[gen] * baseMVA,
        ) + Sum(
            gen.where[status[gen] & (costmodel[gen] == 1)],
            V_pw_cost.l[gen] + noloadcost[gen],
        )

    else:
        V_objcost.l = Sum(
            gen.where[status[gen] & (costmodel[gen] == 2)],
            costcoef[gen, "0"]
            + costcoef[gen, "1"] * V_P.l[gen] * baseMVA
            + costcoef[gen, "2"] * sqr(V_P.l[gen] * baseMVA),
        ) + Sum(
            gen.where[status[gen] & (costmodel[gen] == 1)],
            V_pw_cost.l[gen] + noloadcost[gen],
        )

    print(f"costcoef: \n{costcoef.pivot()}\n\n")

    # ===== SECTION: MODEL OPTIONS AND SOLVE
    # ---- Basic options
    acopf.solve()

    # ==== SECTION: Solution Analysis
    # See if model is solved
    infeas = Parameter(
        m,
        name="infeas",
        description="Number of infeasibilities from model solve",
    )

    infeas.setRecords(acopf.num_infeasibilities)
    print("Number of infeasibilities from model solve: ", infeas.toValue())

    # Declaration needs to be made outside loop
    lines_at_limit = Set(
        m,
        name="lines_at_limit",
        domain=[i, j, c],
        description="lines at their bound",
    )

    total_cost = Parameter(
        m, name="total_cost", description="Cost of objective function"
    )
    LMP = Parameter(
        m, name="LMP", domain=bus, description="Locational marginal price"
    )
    LineSP = Parameter(
        m,
        name="LineSP",
        domain=[i, j, c],
        description="Marginal price of active power on line (i,j,c)",
    )
    shuntB = Parameter(
        m,
        name="shuntB",
        domain=i,
    )

    if infeas.toValue() == 0:
        # Final Objective function value
        total_cost[...] = V_objcost.l
        # Generator real power solution
        Pg[gen] = V_P.l[gen]
        # Generator reactive power solution
        Qg[gen] = V_Q.l[gen]
        # Voltage magnitude solution
        Vm[bus] = sqrt(sqr(V_real.l[bus]) + sqr(V_imag.l[bus]))
        # Voltage angle solution
        Va[bus].where[V_real.l[bus] > 0] = (
            atan(V_imag.l[bus] / V_real.l[bus]) * 180 / pi
        )
        Va[bus].where[V_real.l[bus] == 0] = (
            atan(V_imag.l[bus] / V_real.l[bus]) * 180 / pi + 180
        )
        # Bus shunt solution
        shuntB[i] = Sum(bus_s, V_shunt.l[i, bus_s] * Bswitched[i, bus_s])
        # Locational marginal price of bus at time t
        LMP[bus] = c_BalanceP.m[bus]
        # Marginal for active power on a line
        LineSP[i, j, c].where[branchstatus[i, j, c]] = V_LineP.m[i, j, c]
        LineSP[j, i, c].where[branchstatus[i, j, c]] = V_LineP.m[j, i, c]

    # Find which lines are at their limits
    lines_at_limit[i, j, c].where[
        branchstatus[i, j, c] | branchstatus[j, i, c]
    ] = Number(1).where[
        sqr(rateA[i, j, c]) - sqr(V_LineP.l[i, j, c]) - sqr(V_LineQ.l[i, j, c])
        <= 1e-4
    ]
    print("lines at their bound: ", lines_at_limit.records)
    print("Model status: ", acopf.status.name)
    print("Objective Function Value: ", round(acopf.objective_value, 3))

    # Write to GDX
    if args.savesol == 1:
        m._gdx_write(
            "acopf_solution.gdx",
            ["Pg", "Qg", "Vm", "Va", "shuntB", "total_cost", "LMP", "LineSP"],
            [],
            False,
            "string",
            False,
        )


if __name__ == "__main__":
    main()
