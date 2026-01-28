"""
## GAMSSOURCE: https://gams.com/latest/gamslib_ml/libhtml/gamslib_indus89.html
## LICENSETYPE: Requires license
## MODELTYPE: NLP
## DATAFILES: indus89.gdx

This file contains the basic data and definition of the surface water
system. Data is complete for year 1988. Some parameters could be
computed for future years using growth rates provided in this file,
others had to be estimated and entered. Enter the year for which the
setup is desired in Set isr (Set isr should have only one entry).

This is a semiautomatic translation of the original GAMS model.

Ahmad, M, and Kutcher, G P, Irrigation Planning with Environmental
Considerations - A Case Study of Pakistans's Indus Basin. Tech. rep.,
The World Bank, 1992.

Changes for year 2000 runs:
   Growth of crop yields set to a maximum of 3%
   insert this line after growthcy parameter:
   growthcy[c,z].where[growthcy(c,z) > 3] = 3.0

Keywords: linear programming, irrigation engineering, agricultural economics,
          resource allocation, water management, surface water system, water
          distribution, agricultural production, irrigation planning
"""

import math
from pathlib import Path

import gamspy.math as gp_math
from gamspy import (
    Alias,
    Card,
    Container,
    Domain,
    Equation,
    Model,
    Ord,
    Parameter,
    Set,
    SpecialValues,
    Sum,
    Variable,
)


def main():
    container = Container()

    z = Set(container, "z", domain=["*"], description="agroclimatic zones")
    pv = Set(container, "pv", domain=["*"], description="provinces and country")
    p1 = Set(container, "pv1", domain=pv, description="provinces")
    pv2 = Set(container, "pv2", domain=[pv], description="punjab and sind")
    pvz = Set(container, "pvz", domain=[pv, z], description="province to zone map")
    cq = Set(container, "cq", domain=["*"], description="crop and livestock products")
    cc = Set(container, "cc", domain=[cq], description="consumable commodities")
    c = Set(container, "c", domain=[cq], description="crops")
    cf = Set(container, "cf", domain=[c], description="fodder crops")
    cnf = Set(container, "cnf", domain=[c], description="non-fodder crops")
    t = Set(container, "t", domain=["*"], description="technology")
    s = Set(container, "s", domain=["*"], description="sequence")
    w = Set(container, "w", domain=["*"], description="water stress level")
    g = Set(container, "g", domain=["*"], description="ground water quality types")
    gf = Set(container, "gf", domain=[g], description="fresh ground water sub-zone")
    gs = Set(container, "gs", domain=[g], description="saline ground water sub-zone")
    _ = Set(container, "t1", description="sub zones by gw quality")
    r1 = Set(container, "r1", domain=["*"], description="resources")
    dc = Set(
        container, "dc", domain=[r1], description="characteristics of canal command"
    )
    sa = Set(container, "sa", domain=["*"], description="subareas")
    wce = Set(container, "wce", domain=[dc], description="watercourse efficiencies")
    m1 = Set(container, "m1", domain=["*"], description="months and seasons")
    m = Set(container, "m", domain=[m1], description="months")
    wcem = Set(
        container,
        "wcem",
        domain=[wce, m],
        description="mapping from season to months for watercourse efficiencies",
    )
    sea = Set(container, "sea", domain=[m1], description="seasons")
    seam = Set(
        container,
        "seam",
        domain=[sea, m],
        description="mapping from seasons to months",
    )
    sea1 = Set(container, "sea1", domain=["*"], description="")
    sea1m = Set(container, "sea1m", domain=[sea1, m], description="")
    ci = Set(container, "ci", domain=["*"], description="crop input outputs")
    p2 = Set(container, "p2", domain=[ci], description="")
    a = Set(container, "a", domain=["*"], description="animal types")
    _ = Set(container, "ai", domain=["*"], description="animals input output")
    q = Set(container, "q", domain=[cq], description="livestock commodities")
    nt = Set(container, "nt", domain=["*"], description="nutrients for animals")
    is_renamed = Set(
        container, "is", domain=["*"], description="irrigation system scenarios"
    )
    ps = Set(container, "ps", domain=["*"], description="price scenarios")
    isr = Set(
        container,
        "isr",
        domain=[is_renamed],
        description="irrigation system scenario for this run",
        is_singleton=True,
    )
    baseyear = Parameter(
        container, "baseyear", domain=[], description="base year for crop yields"
    )
    land = Parameter(
        container,
        "land",
        domain=[c, z, t, s, w, m],
        description="land occupation by month",
    )
    tech = Set(
        container,
        "tech",
        domain=[z, c, t, s, w],
        description="technology availability indicator",
    )
    bullock = Parameter(
        container,
        "bullock",
        domain=[c, z, t, s, w, m],
        description="bullock power requirements (bullock pair hours per month)",
    )
    labor = Parameter(
        container,
        "labor",
        domain=[c, z, t, s, w, m],
        description="labor requirements for crops (man hours)",
    )
    water = Parameter(
        container,
        "water",
        domain=[c, z, t, s, w, m],
        description="water requirements (acre feet per acre)",
    )
    tractor = Parameter(
        container,
        "tractor",
        domain=[c, z, t, s, w, m],
        description="tractor requirements (tractor hours per acre)",
    )
    sylds = Parameter(
        container,
        "sylds",
        domain=[c, z, t, s, w, ci],
        description="straw yield and seed data",
    )
    fert = Parameter(
        container,
        "fert",
        domain=[p2, c, z],
        description="fertilizer applications (kg per acre)",
    )
    fertgr = Parameter(
        container,
        "fertgr",
        domain=[c],
        description="fertilizer application growth rate  percent",
    )
    natyield = Parameter(
        container,
        "natyield",
        domain=[c],
        description="national crop yields 1988 for standard technologies (kgs)",
    )
    yldprpv = Parameter(
        container,
        "yldprpv",
        domain=[c, pv],
        description="province yields proportion of national 1987-88",
    )
    yldprzs = Parameter(
        container,
        "yldprzs",
        domain=[c, z],
        description="zones yields as proportion of province-standard technologies",
    )
    yldprzo = Parameter(
        container,
        "yldprzo",
        domain=[c, s, w],
        description="yields as proportion of standard technologies",
    )
    growthcy = Parameter(
        container,
        "growthcy",
        domain=[c, z],
        description="growth rate of crop yields from 1988 base (percent)",
    )
    weedy = Parameter(
        container,
        "weedy",
        domain=[z, sea, c],
        description="weed yields by crop (tonns per acer)",
    )
    graz = Parameter(
        container,
        "graz",
        domain=[z, sea],
        description="grazing from slack land (tonns per acre)",
    )
    yield_renamed = Parameter(
        container,
        "yield",
        domain=[c, t, s, w, z],
        description="yield by zone crop technology in metric tonns",
    )
    growthcyf = Parameter(
        container,
        "growthcyf",
        domain=[c, z],
        description="growth factor for crop yields using growthcy",
    )
    iolive = Parameter(
        container,
        "iolive",
        domain=[a, z, "*"],
        description="livestock input output coefficients by zones",
    )
    sconv = Parameter(
        container,
        "sconv",
        domain=[nt, sea, c],
        description="tdn and dp conversion factor from crop straw",
    )
    repco = Parameter(
        container, "repco", domain=[], description="reproductive coefficient"
    )
    gr = Parameter(
        container,
        "gr",
        domain=[],
        description="required proportion of green fodder in total fodder",
    )
    growthq = Parameter(
        container,
        "growthq",
        domain=[],
        description="growth rate of milk and meat yields (percent)",
    )
    bp = Parameter(
        container,
        "bp",
        domain=[m],
        description="draft power available per bullock(hours per month)",
    )
    cnl = Set(
        container,
        "cnl",
        domain=["*"],
        description="irrigation canals in the indus river irrigation system",
    )
    pvcnl = Set(
        container, "pvcnl", domain=[pv, cnl], description="province to canals map"
    )
    gwfg = Set(
        container,
        "gwfg",
        domain=[cnl, sa, g],
        description="subarea identification by the groundwater quality",
    )
    comdef = Parameter(
        container,
        "comdef",
        domain=[is_renamed, dc, cnl],
        description="canal command characteristics",
    )
    subdef = Parameter(
        container,
        "subdef",
        domain=[sa, cnl],
        description="sub-area definition (proportion of cca) by canals",
    )
    zsa = Set(
        container,
        "zsa",
        domain=[z, cnl, sa],
        description="canal-subarea to agroclimatic zone mapping",
    )
    gwf = Set(
        container,
        "gwf",
        domain=[cnl, sa],
        description="subareas with fresh ground water",
    )
    carea = Parameter(
        container,
        "carea",
        domain=[cnl, "*"],
        description="cca classified by groundwater quality for each canal",
    )
    evap = Parameter(
        container, "evap", domain=[cnl, m], description="pan evaporation (feet)"
    )
    rain = Parameter(container, "rain", domain=[cnl, m], description="rain (inches)")
    divpost = Parameter(
        container,
        "divpost",
        domain=[cnl, m1],
        description="average (1976-77 to 1987-88) canal diversions (maf)",
    )
    gwt = Parameter(
        container, "gwt", domain=[cnl, m], description="public tunewell pumpage (kaf)"
    )
    dep1 = Parameter(
        container,
        "dep1",
        domain=[cnl, is_renamed, "*"],
        description="depth to water table (feet)",
    )
    dep2 = Parameter(
        container,
        "dep2",
        domain=[is_renamed, cnl, m],
        description="depth to water table (feet)",
    )
    depth = Parameter(
        container, "depth", domain=[cnl, m], description="depth to groundwater (feet)"
    )
    efr = Parameter(
        container, "efr", domain=[cnl, m], description="effective rainfall in feet"
    )
    eqevap = Parameter(
        container,
        "eqevap",
        domain=[cnl, m],
        description="evaporation from the equaifer (feet)",
    )
    subirr = Parameter(
        container,
        "subirr",
        domain=[cnl, m],
        description="water supplied by capillary action from the aquifer",
    )
    subirrfac = Parameter(
        container,
        "subirrfac",
        domain=[z],
        description="maximum sub-irrigation in saline areas as proportion of crop req. (net of rain)",
    )
    drc = Parameter(
        container, "drc", domain=[], description="run-off portion of rainfall"
    )
    the1 = Parameter(
        container,
        "the1",
        domain=[],
        description="portion of equaifer evaporation used by crops",
    )
    n = Set(container, "n", domain=["*"], description="nodes of the indus river system")
    i = Set(container, "i", domain=["*"], description="system inflows")
    nc = Set(container, "nc", domain=[n, cnl], description="node to canal map")
    n1 = Alias(container, "n1", alias_with=n)
    nn = Set(
        container, "nn", domain=[n, n1], description="water flow system node to node"
    )
    ni = Set(
        container, "ni", domain=[n, i], description="node to rim station inflow map"
    )
    nb = Set(container, "nb", domain=[n], description="")
    ncap = Parameter(
        container,
        "ncap",
        domain=[n, n1],
        description="node to node transfer capacity (maf)",
    )
    lloss = Parameter(
        container, "lloss", domain=[n, n1], description="link canal loss factors"
    )
    lceff = Parameter(
        container,
        "lceff",
        domain=[n, n1],
        description="link canal efficiency from head to tail",
    )
    cd = Set(container, "cd", domain=["*"], description="")
    rivercd = Parameter(
        container,
        "rivercd",
        domain=[n, cd],
        description="coefficients for river routing",
    )
    riverb = Parameter(
        container,
        "riverb",
        domain=[n, n1],
        description="coefficients for river routing",
    )
    s58 = Set(container, "s58", domain=["*"], description="")
    infl5080 = Parameter(
        container,
        "infl5080",
        domain=[s58, i, m1],
        description="system inflows measured atthe rim stations (maf)",
    )
    tri = Parameter(
        container,
        "tri",
        domain=[s58, n1, n, m1],
        description="tributary inflows (maf)",
    )
    inflow = Parameter(
        container,
        "inflow",
        domain=[i, m],
        description="inflows for this run           (maf)",
    )
    trib = Parameter(
        container,
        "trib",
        domain=[n1, n, m],
        description="tributary inflows for this run (maf)",
    )
    rrcap = Parameter(
        container,
        "rrcap",
        domain=[n, is_renamed],
        description="live storage capacity of reservoirs (maf)",
    )
    rulelo = Parameter(
        container, "rulelo", domain=[n, m], description="lower rule curve"
    )
    ruleup = Parameter(
        container, "ruleup", domain=[n, m], description="upper rule curve"
    )
    revapl = Parameter(
        container,
        "revapl",
        domain=[n, m],
        description="evaporation losses from reservoirs (kaf)",
    )
    pow = Set(container, "pow", domain=["*"], description="")
    pn = Set(container, "pn", domain=[n], description="nodes with power house")
    v = Set(container, "v", domain=["*"], description="")
    powerchar = Parameter(
        container,
        "powerchar",
        domain=[n, "*", v],
        description="power generation chractersitics of hrdro stations",
    )
    rcap = Parameter(
        container, "rcap", domain=[n], description="live capacity of resrvoirs (maf)"
    )
    rep7 = Parameter(container, "rep7", domain=["*", "*"], description="")
    rep8 = Parameter(container, "rep8", domain=["*", "*", "*"], description="")
    p3 = Set(container, "p3", domain=["*"], description="")
    prices = Parameter(
        container, "prices", domain=[ps, cq, p3], description="1988  prices"
    )
    finsdwtpr = Parameter(
        container,
        "finsdwtpr",
        domain=[c, ps, "*"],
        description="prices of seed (rs per kg) and water (rs per acre)",
    )
    ecnsdwtpr = Parameter(
        container,
        "ecnsdwtpr",
        domain=[c, ps, "*"],
        description="prices of seed(rs per kg) and water (rs per acre)",
    )
    p1 = Set(container, "p1", domain=["*"], description="")
    p11 = Set(container, "p11", domain=["*"], description="")
    pri1 = Parameter(
        container,
        "pri1",
        domain=[ps, p11, p1],
        description="fertilizer tubewell tractor and protein prices",
    )
    wageps = Parameter(
        container,
        "wageps",
        domain=[ps, p11, m],
        description="wage rates rs per man hour",
    )
    lstd = Parameter(
        container,
        "lstd",
        domain=[],
        description="standard labor limit (hours per month)",
    )
    trcap = Parameter(
        container,
        "trcap",
        domain=[],
        description="tractor capacity in tractor hours per month",
    )
    twcap = Parameter(
        container,
        "twcap",
        domain=[],
        description="nameplate capacity of the private tubewell (af per month)",
    )
    ntwucap = Parameter(
        container,
        "ntwucap",
        domain=[],
        description="effective capacity of new tubewells (af per month)",
    )
    twefac = Parameter(
        container,
        "twefac",
        domain=[],
        description="factor to convert wc losses to from private tubewell losses",
    )
    labfac = Parameter(
        container,
        "labfac",
        domain=[],
        description="factor to convert wage to the reservation wage",
    )
    twutil = Parameter(
        container,
        "twutil",
        domain=["*"],
        description="effective capacity of tubewells (proportion of name plate capacity)",
    )
    totprod = Parameter(
        container,
        "totprod",
        domain=[z, cq],
        description="total production 1988 (000's tons)",
    )
    farmcons = Parameter(
        container,
        "farmcons",
        domain=[z, cq],
        description="on-farm consumption 1988 (000's tons)",
    )
    demand = Parameter(
        container,
        "demand",
        domain=["*", cq],
        description="market demand by zone (000 tons or million liters)",
    )
    cowf = Parameter(
        container,
        "cowf",
        domain=[],
        description="adjustment factor for cows population in the irrigated areas",
    )
    buff = Parameter(
        container,
        "buff",
        domain=[],
        description="adjustment factor for buffloes pop.   in the irrigated areas",
    )
    elast = Parameter(
        container,
        "elast",
        domain=[cq],
        description="elasticity of demand for crop and livestock comodities",
    )
    growthrd = Parameter(
        container,
        "growthrd",
        domain=[cq],
        description="growth rate of reference demand (percent)",
    )
    consratio = Parameter(
        container,
        "consratio",
        domain=[z, g],
        description="proportion of consumption by growundwater type",
    )
    natexp = Parameter(
        container, "natexp", domain=[cq], description="national exports (000 tons)"
    )
    explimit = Parameter(
        container, "explimit", domain=[z, cq], description="export limits by zone"
    )
    explimitgr = Parameter(
        container,
        "explimitgr",
        domain=[],
        description="growth rate of export limits (percent)",
    )
    exppv = Parameter(
        container,
        "exppv",
        domain=[pv, cq],
        description="provincial exports as proportion of national",
    )
    expzo = Parameter(
        container,
        "expzo",
        domain=[z, cq],
        description="zonal exports as proportion of provincial",
    )
    sr1 = Set(container, "sr1", domain=[dc], description="")
    g1 = Alias(container, "g1", alias_with=g)
    zwt = Parameter(
        container,
        "zwt",
        domain=[z, cnl, sa],
        description="weighting factor to map rain evap and efficiencies to zones",
    )
    eqevapz = Parameter(
        container,
        "eqevapz",
        domain=[z, m],
        description="evaporation from the equaifer by acz (feet)",
    )
    subirrz = Parameter(
        container,
        "subirrz",
        domain=[z, m],
        description="subirrigation by acz                 (feet)",
    )
    efrz = Parameter(
        container,
        "efrz",
        domain=[z, m],
        description="effective rain  by acz               (feet)",
    )
    resource = Parameter(
        container,
        "resource",
        domain=[z, g, r1],
        description="endowments by acz and groundwater quality",
    )
    cneff = Parameter(
        container,
        "cneff",
        domain=[cnl],
        description="canal efficiency from canal head to the watercourse head",
    )
    wceff = Parameter(
        container,
        "wceff",
        domain=[cnl, m],
        description="watercourse command delivery efficiency",
    )
    tweff = Parameter(
        container,
        "tweff",
        domain=[cnl, m],
        description="delivery efficiency from private tubewell to the root zone",
    )
    cneffz = Parameter(
        container,
        "cneffz",
        domain=[z],
        description="weighted canal delivery efficiency from canal head to watercourse head",
    )
    tweffz = Parameter(
        container,
        "tweffz",
        domain=[z, m],
        description="weighted private tubewell delivery efficiency by zone",
    )
    wceffz = Parameter(
        container,
        "wceffz",
        domain=[z, m],
        description="weighted water course command delivery efficiency by zone",
    )
    fleffz = Parameter(
        container,
        "fleffz",
        domain=[z],
        description="weighted field efficiency by zone",
    )
    canalwz = Parameter(
        container,
        "canalwz",
        domain=[z, g, m],
        description="canal water availablility at the canal head (maf)",
    )
    canalwrtz = Parameter(
        container,
        "canalwrtz",
        domain=[z, g, m],
        description="canal water availablility at the root zone (maf)",
    )
    gwtsa = Parameter(
        container,
        "gwtsa",
        domain=[cnl, sa, m],
        description="government tubewell pumpage by canal and subarea (kaf)",
    )
    gwt1 = Parameter(
        container,
        "gwt1",
        domain=[z, g, m],
        description="public tubewell pumpage at the root zone (kaf)",
    )
    ratiofs = Parameter(
        container,
        "ratiofs",
        domain=[z, g],
        description="fresh and saline cca as a proportion off total",
    )
    ftt = Set(container, "ftt", domain=[r1], description="")
    res88 = Parameter(
        container, "res88", domain=[r1, z], description="available resources 1988"
    )
    croparea = Parameter(
        container,
        "croparea",
        domain=[z, c],
        description="cropped area 1988 (000's acres)",
    )
    growthres = Parameter(
        container,
        "growthres",
        domain=[r1, z],
        description="growth rate of farm population tractors and tubwells (percent)",
    )
    orcharea = Parameter(
        container,
        "orcharea",
        domain=[z],
        description="area under orchards by zone (thousand acres)",
    )
    orchgrowth = Parameter(
        container, "orchgrowth", domain=[z], description="growth rate of orchard area"
    )
    scmillcap = Parameter(
        container,
        "scmillcap",
        domain=[z],
        description="sugarcane mill capacity (thousand tonns per year)",
    )
    cnl1 = Set(
        container, "cnl1", domain=[cnl], description="canals excluding nwfp canals"
    )
    postt = Parameter(
        container,
        "postt",
        domain=["*", "*"],
        description="average canal diversions by season",
    )
    protarb = Parameter(
        container,
        "protarb",
        domain=["*", "*"],
        description="diversions as proportion of total (punjab and sind) ost tarbela",
    )
    psr = Set(
        container,
        "psr",
        domain=[ps],
        description="price scenario for the model (financial prices)",
        is_singleton=True,
    )
    psr1 = Set(
        container,
        "psr1",
        domain=[ps],
        description="price scenario for report (economic prices)",
        is_singleton=True,
    )
    z1 = Set(container, "z1", domain=[z], description="zone selection for this run")
    cn = Set(container, "cn", domain=[cq], description="comodities  endogenous prices")
    ccn = Set(
        container,
        "ccn",
        domain=[cq],
        description="crop comodities with endogenous prices",
    )
    qn = Set(
        container,
        "qn",
        domain=[cq],
        description="livestock comodities endogenous prices",
    )
    ncn = Set(
        container,
        "ncn",
        domain=[cq],
        description="crops with fixed prices excluding fodder",
    )
    ce = Set(container, "ce", domain=[cq], description="exportable comodities")
    cm = Set(
        container, "cm", domain=[cq], description="comodities which could be imported"
    )
    ex = Set(
        container,
        "ex",
        domain=[z, g],
        description="to check fresh or saline area within a zone",
    )
    techc = Set(container, "techc", domain=[z, cq], description="comodities by zones")
    tec = Parameter(
        container,
        "tec",
        domain=[c, t, s, w, z],
        description="crop technology disabled for 1988 run",
    )
    big = Parameter(
        container,
        "big",
        domain=[],
        description="big number used for artifical production",
    )
    pawat = Parameter(
        container, "pawat", domain=[], description="big number for artificial water"
    )
    pafod = Parameter(
        container, "pafod", domain=[], description="big number for artifical fodder"
    )
    divnwfp = Parameter(
        container,
        "divnwfp",
        domain=[m],
        description="monthy diversion to the nwfp zone (maf)",
    )
    rval = Parameter(
        container,
        "rval",
        domain=[n],
        description="value of water stored in the reservoirs",
    )
    fsalep = Parameter(
        container,
        "fsalep",
        domain=[cq],
        description="financial sale price for crop and livestock comodities (rs per kg or per liter)",
    )
    pp = Parameter(
        container,
        "pp",
        domain=[],
        description="financial purchase price of protein                                (rs per kgs)",
    )
    misc = Parameter(
        container, "misc", domain=["*"], description="financial miscellenious prices"
    )
    seedp = Parameter(
        container,
        "seedp",
        domain=[c],
        description="financial seed price                                               (rs per kgs)",
    )
    wage = Parameter(
        container,
        "wage",
        domain=[m],
        description="financial wage rates                                          (rs per man-hour)",
    )
    miscct = Parameter(
        container,
        "miscct",
        domain=[c],
        description="financial water charges and miscillenious costs                   (rs per acre)",
    )
    esalep = Parameter(
        container,
        "esalep",
        domain=[cq],
        description="economic sale price for crop and livestock comodities  (rs per kg or per liter)",
    )
    epp = Parameter(
        container,
        "epp",
        domain=[],
        description="economic price of protein concentrate                              (rs per kgs)",
    )
    emisc = Parameter(
        container, "emisc", domain=["*"], description="economic miscellenious prices"
    )
    eseedp = Parameter(
        container, "eseedp", domain=[c], description="economic seed price"
    )
    ewage = Parameter(
        container,
        "ewage",
        domain=[m],
        description="economic wage rate                                            (rs per man-hour)",
    )
    emiscct = Parameter(
        container,
        "emiscct",
        domain=[c],
        description="economic water charges and miscillenious costs                    (rs per acre)",
    )
    importp = Parameter(
        container, "importp", domain=[cq], description="import prices for the scenario"
    )
    exportp = Parameter(
        container, "exportp", domain=[cq], description="export prices for the scenario"
    )
    wnr = Parameter(
        container,
        "wnr",
        domain=[c, z, t, s, w, m],
        description="water requirements net of rain",
    )
    tolcnl = Parameter(
        container,
        "tolcnl",
        domain=[],
        description="allowed deviation from proportional allocation by canal",
    )
    tolpr = Parameter(
        container,
        "tolpr",
        domain=[],
        description="allowed deviation from proportional allocation by province",
    )
    tolnwfp = Parameter(
        container, "tolnwfp", domain=[], description="nwfp diversion tolerance"
    )
    beta = Parameter(
        container,
        "beta",
        domain=[cq, z1],
        description="gradient comodities demand curve",
    )
    alpha = Parameter(
        container, "alpha", domain=[cq, z1], description="demand curve intecept"
    )
    betaf = Parameter(container, "betaf", domain=[], description="beta factor")
    p = Set(container, "p", domain=["*"], description="grid points for linearization")
    pmax = Parameter(
        container, "pmax", domain=[cq, z1], description="maximum price for segments"
    )
    pmin = Parameter(
        container, "pmin", domain=[cq, z1], description="minimum price for segments"
    )
    qmax = Parameter(
        container, "qmax", domain=[cq, z1], description="max national consumption"
    )
    qmin = Parameter(
        container, "qmin", domain=[cq, z1], description="min national consumption"
    )
    incr = Parameter(container, "incr", domain=[cq, z1], description="increment")
    ws = Parameter(
        container,
        "ws",
        domain=[cq, z1, p],
        description="welfare segments                     (million rupees)",
    )
    rs = Parameter(
        container,
        "rs",
        domain=[cq, z1, p],
        description="revenue definition                   (million rupees)",
    )
    qs = Parameter(
        container,
        "qs",
        domain=[cq, z1, p],
        description="quantity definition (thousand tons or million liters)",
    )
    endpr = Parameter(
        container,
        "endpr",
        domain=[cq, z1, p],
        description="price                       (rupees per kgs or liter)",
    )
    cps = Variable(
        container,
        "cps",
        domain=[],
        description="consumer plus producers surplus (million rupees)",
        type="free",
    )
    acost = Variable(
        container,
        "acost",
        domain=[z, g],
        description="farm cost in                                                    (million rupees)",
        type="positive",
    )
    ppc = Variable(
        container,
        "ppc",
        domain=[z, g, sea],
        description="purchases of protein concentrates                         (thousand metric tons)",
        type="positive",
    )
    x = Variable(
        container,
        "x",
        domain=[z, g, c, t, s, w],
        description="cropped area by technology                                      (thousand acres)",
        type="positive",
    )
    animal = Variable(
        container,
        "animal",
        domain=[z, g, a],
        description="production of livestock type a                                       (thousands)",
        type="positive",
    )
    prodt = Variable(
        container,
        "prodt",
        domain=[z, g, cq],
        description="production (crop commodities 000 metric tons livestock comm mill. kgs or liters)",
        type="positive",
    )
    proda = Variable(
        container,
        "proda",
        domain=[z, g, cq],
        description="artificial supply",
        type="positive",
    )
    import_renamed = Variable(
        container,
        "import",
        domain=[z, cq],
        description="import of comodities      (crop comm. 000 m. tons livestock mill. kgs or liters)",
        type="positive",
    )
    export = Variable(
        container,
        "export",
        domain=[z, cq],
        description="export of comodities                                          (000 metric tonns)",
        type="positive",
    )
    consump = Variable(
        container,
        "consump",
        domain=[z, g, cq],
        description="on farm consumption                                           (000 metric tonns)",
        type="positive",
    )
    familyl = Variable(
        container,
        "familyl",
        domain=[z, g, m],
        description="family labor used                                            (million man hours)",
        type="positive",
    )
    hiredl = Variable(
        container,
        "hiredl",
        domain=[z, g, m],
        description="hired labor used                                             (million man hours)",
        type="positive",
    )
    itw = Variable(
        container,
        "itw",
        domain=[z],
        description="investment in increased private tubewell capacity                (kaf per month)",
        type="positive",
    )
    tw = Variable(
        container,
        "tw",
        domain=[z, m],
        description="private tubewell water used  by month m                                    (kaf)",
        type="positive",
    )
    itr = Variable(
        container,
        "itr",
        domain=[z, g],
        description="investment in increased tractor capacity             (000 tractor-hrs per month)",
        type="positive",
    )
    ts = Variable(
        container,
        "ts",
        domain=[z, g, m],
        description="private tractor services use by month                             (thousand hrs)",
        type="positive",
    )
    f = Variable(
        container,
        "f",
        domain=[n, n1, m],
        description="flow to node n from node n1                                                (maf)",
        type="positive",
    )
    rcont = Variable(
        container,
        "rcont",
        domain=[n, m],
        description="end of the month resrvoir contents                                         (maf)",
        type="positive",
    )
    canaldiv = Variable(
        container,
        "canaldiv",
        domain=[cnl, m],
        description="canal diversion at the canal head                                          (maf)",
        type="positive",
    )
    cnldivsea = Variable(
        container,
        "cnldivsea",
        domain=[cnl, sea],
        description="canal diversion by season                                                  (maf)",
        type="positive",
    )
    prsea = Variable(
        container,
        "prsea",
        domain=[pv, sea],
        description="canal diversion by province (Sind and Punjab)                              (maf)",
        type="positive",
    )
    tcdivsea = Variable(
        container,
        "tcdivsea",
        domain=[sea],
        description="total canal diversion in Sind and Punjab by season                         (maf)",
        type="positive",
    )
    wdivrz = Variable(
        container,
        "wdivrz",
        domain=[z1, g, m],
        description="surface water diversion at the root zone                                   (kaf)",
        type="positive",
    )
    slkland = Variable(
        container,
        "slkland",
        domain=[z, g, m],
        description="slack land                                                      (thousand acres)",
        type="positive",
    )
    slkwater = Variable(
        container,
        "slkwater",
        domain=[z, g, m],
        description="slack water at the root zone                                               (kaf)",
        type="positive",
    )
    artfod = Variable(
        container,
        "artfod",
        domain=[z1, g, sea],
        description="artificial fodder supply  equaivalent of rab-fod                     (000 tonns)",
        type="positive",
    )
    artwater = Variable(
        container,
        "artwater",
        domain=[z, g, m],
        description="water from imaginary source at the root zone                               (kaf)",
        type="positive",
    )
    artwaternd = Variable(
        container,
        "artwaternd",
        domain=[n, m],
        description="water from imaginary source at nodes                                       (maf)",
        type="positive",
    )
    nat = Variable(
        container,
        "nat",
        domain=[cq, z, p],
        description="provincial demand linearized",
        type="positive",
    )
    natn = Variable(
        container,
        "natn",
        domain=[cq, z],
        description="provincial demand non-linear",
        type="positive",
    )
    objz = Equation(
        container,
        "objz",
        domain=[],
        description="objective function for the zone model linear version      (million rupees)",
    )
    objzn = Equation(
        container,
        "objzn",
        domain=[],
        description="objective function for the zone model non-linear version  (million rupees)",
    )
    objn = Equation(
        container,
        "objn",
        domain=[],
        description="objective function for the indus model linear version     (million rupees)",
    )
    objnn = Equation(
        container,
        "objnn",
        domain=[],
        description="objective function for the indus model non-linear version (million rupees)",
    )
    cost = Equation(
        container,
        "cost",
        domain=[z, g],
        description="annual farm cost                                          (million rupees)",
    )
    conv = Equation(
        container,
        "conv",
        domain=[z, cq],
        description="convex combination for aggregate consumption",
    )
    demnat = Equation(
        container,
        "demnat",
        domain=[z, cq],
        description="provincial demand balance linear              (000 tons or million liters)",
    )
    demnatn = Equation(
        container,
        "demnatn",
        domain=[z, cq],
        description="zonal demand balance non-linear               (000 tons or million liters)",
    )
    ccombal = Equation(
        container,
        "ccombal",
        domain=[z, g, c],
        description="commodity balances for crops                                    (000 tons)",
    )
    qcombal = Equation(
        container,
        "qcombal",
        domain=[z, g, cq],
        description="livestock comodity balances                         (000 tons or m liters)",
    )
    consbal = Equation(
        container,
        "consbal",
        domain=[z, g, cq],
        description="consumption balance                                 (000 tons or m liters)",
    )
    laborc = Equation(
        container,
        "laborc",
        domain=[z, g, m],
        description="monthly labor constraint                               (million man hours)",
    )
    fodder = Equation(
        container,
        "fodder",
        domain=[z, g, sea],
        description="seasonal maintenance of fodder supplies                  (000 metric tons)",
    )
    protein = Equation(
        container,
        "protein",
        domain=[z, g, sea],
        description="protein requirements of livestock by season              (000 metric tons)",
    )
    grnfdr = Equation(
        container,
        "grnfdr",
        domain=[z, g, sea],
        description="green fodder requirements                                (000 metric tons)",
    )
    bdraft = Equation(
        container,
        "bdraft",
        domain=[z, g, m],
        description="bullock draft power constraint                     (million bullock hours)",
    )
    brepco = Equation(
        container,
        "brepco",
        domain=[z, g],
        description="bullock reproduction constraint",
    )
    bullockc = Equation(
        container,
        "bullockc",
        domain=[z1],
        description="bullock population constraint                               (000 bullocks)",
    )
    tdraft = Equation(
        container,
        "tdraft",
        domain=[z, g, m],
        description="tractor draft power balance                            (000 tractor hours)",
    )
    trcapc = Equation(
        container,
        "trcapc",
        domain=[z, m],
        description="tractor capacity constraint                            (000 tractor hours)",
    )
    twcapc = Equation(
        container,
        "twcapc",
        domain=[z, m],
        description="tubewell capacity constraint                                         (kaf)",
    )
    landc = Equation(
        container,
        "landc",
        domain=[z, g, m],
        description="land constraint                                                (000 acres)",
    )
    orchareac = Equation(
        container,
        "orchareac",
        domain=[z],
        description="orchard area constraint                                        (000 acres)",
    )
    scmillc = Equation(
        container,
        "scmillc",
        domain=[z],
        description="sugar cane to mill constraint                                  (000 acres)",
    )
    waterbaln = Equation(
        container,
        "waterbaln",
        domain=[z, g, m],
        description="water balance at the root zone                                       (kaf)",
    )
    watalcz = Equation(
        container,
        "watalcz",
        domain=[z, g, m],
        description="surface water by zone                                                (kaf)",
    )
    subirrc = Equation(
        container,
        "subirrc",
        domain=[z, g, m],
        description="subirrigation constraint                                             (kaf)",
    )
    nbal = Equation(
        container,
        "nbal",
        domain=[n, m],
        description="water balance at a node                                              (maf)",
    )
    watalcsea = Equation(
        container,
        "watalcsea",
        domain=[cnl, sea],
        description="water allocations by season                                          (maf)",
    )
    divsea = Equation(
        container,
        "divsea",
        domain=[sea],
        description="total canal diversions in Sind and Punjab                            (maf)",
    )
    divcnlsea = Equation(
        container,
        "divcnlsea",
        domain=[cnl, sea],
        description="canal diversion by season                                            (maf)",
    )
    watalcpro = Equation(
        container,
        "watalcpro",
        domain=[pv, sea],
        description="water allocation by province                                         (maf)",
    )
    prseaw = Equation(
        container,
        "prseaw",
        domain=[pv, sea],
        description="diversions by province and season                                    (maf)",
    )
    nwfpalc = Equation(
        container,
        "nwfpalc",
        domain=[m],
        description="water allocations to the nwfp acz                                    (maf)",
    )
    container.loadRecordsFromGdx(str(Path(__file__).parent.absolute()) + "/indus89.gdx")

    cnf[c] = 1
    cnf[cf] = 0
    pvz["pakistan", z] = 1
    sea1m["annual", m] = 1
    tech[z, c, t, s, w].where[Sum(m, land[c, z, t, s, w, m])] = 1
    bullock[c, z, t, s, w, m] = bullock[c, z, t, s, w, m] * 2
    fert[p2, c, z] = fert[p2, c, z] * (
        (1 + (fertgr[c] / 100)) ** (isr.val - baseyear[...])
    )
    yield_renamed[c, t, "standard", "standard", z] = Sum(
        pv.where[pvz[pv, z]], (((natyield[c] / 1000) * yldprpv[c, pv]) * yldprzs[c, z])
    )
    yield_renamed[c, t, s, w, z].where[yldprzo[c, s, w]] = (
        yield_renamed[c, t, "standard", "standard", z] * yldprzo[c, s, w]
    )
    growthcyf[c, z] = (1 + (growthcy[c, z] / 100)) ** (isr.val - baseyear[...])
    yield_renamed[c, t, s, w, z] = yield_renamed[c, t, s, w, z] * growthcyf[c, z]
    print(baseyear.records)
    print(growthcyf.records)
    print(fert.records)
    bp[m] = 96
    bp["may"] = 77
    bp["jun"] = 77
    iolive[a, z, q] = iolive[a, z, q] * (
        (1 + (growthq[...] / 100)) ** (isr.val - baseyear[...])
    )

    print(iolive.records)
    gwfg[cnl, sa, "saline"].where[subdef[sa, cnl]] = 1
    gwfg[cnl, sa, "saline"].where[gwf[cnl, sa]] = 0
    gwfg[cnl, sa, "fresh"].where[gwf[cnl, sa]] = 1
    carea[cnl, g] = Sum(
        sa.where[gwfg[cnl, sa, g]], (subdef[sa, cnl] * comdef[isr, "cca", cnl])
    )
    carea[cnl, "total"] = Sum(g, carea[cnl, g])
    print(carea.records)
    depth[cnl, m] = dep1[cnl, isr, "depth"]
    depth[cnl, m].where[(dep2[isr, cnl, m] != 0)] = dep2[isr, cnl, m]
    efr[cnl, m] = (
        ((1 - drc[...]) - (1 - comdef[isr, "flde", cnl])) * rain[cnl, m]
    ) / 12
    eqevap[cnl, m] = gp_math.Min(1, (10.637 / (depth[cnl, m] ** 2.558))) * evap[cnl, m]
    subirr[cnl, m] = eqevap[cnl, m] * the1[...]

    print(depth.records)
    nb[n] = 1
    nb["a-sea"] = 0
    lceff[n, n1].where[lloss[n, n1]] = 1 - lloss[n, n1]
    riverb[n, n1].where[(riverb[n, n1] == 0)] = 1
    rivercd[n, "d"].where[(rivercd[n, "d"] == 0)] = 1
    inflow[i, m] = infl5080["50", i, m]
    trib[n1, n, m] = tri["50", n1, n, m]
    rcap[n] = rrcap[n, isr]
    powerchar[pn, "r-cap", v] = gp_math.Max(
        0, (powerchar[pn, "r-cap", v] - (powerchar[pn, "r-cap", "26"] - rcap[pn]))
    )
    powerchar[pn, "r-cap", "1"] = 0
    rep7[i, m] = inflow[i, m]
    rep7[i, sea1] = Sum(m.where[sea1m[sea1, m]], inflow[i, m])
    rep7["total", m1] = Sum(i, rep7[i, m1])
    print("system inflows at rim stations (maf)")
    print(rep7.records)
    rep8[n, n1, m] = trib[n, n1, m]
    rep8[n, n1, sea1] = Sum(m.where[sea1m[sea1, m]], trib[n, n1, m])
    print("tributory inflow in (maf)")
    print(rep8.records)
    rep8[n, n1, m1] = 0
    rep8[pn, v, pow] = powerchar[pn, pow, v]

    print(rcap.records)
    print("r-ele reservoir elevation (feet from spd)")
    print("p-cap installed capacity of the power house at r-ele")
    print("g-cap generation capability (kwh per af) at r-ele")
    print("r-cap live reservoir capacity (maf) at r-ele")
    print(rep8.records)
    demand[z, "cow-milk"] = demand[z, "cow-milk"] * cowf[...]
    demand[z, "buff-milk"] = demand[z, "buff-milk"] * buff[...]
    growthrd[cq].where[(growthrd[cq] == 0)] = 3
    consratio[z, "saline"] = 1 - consratio[z, "fresh"]
    explimit[z, cq] = (natexp[cq] * Sum(pv.where[pvz[pv, z]], exppv[pv, cq])) * expzo[
        z, cq
    ]
    explimit[z, cq] = (
        (1 + (explimitgr[...] / 100)) ** (isr.val - baseyear[...])
    ) * explimit[z, cq]
    resource[z, g, sr1] = Sum(
        Domain(cnl, sa).where[(zsa[z, cnl, sa].where[gwfg[cnl, sa, g]])],
        (comdef[isr, sr1, cnl] * subdef[sa, cnl]),
    )
    zwt[z, cnl, sa].where[zsa[z, cnl, sa]] = (
        comdef[isr, "cca", cnl] * subdef[sa, cnl]
    ) / Sum(g, resource[z, g, "cca"])
    cneff[cnl] = comdef[isr, "ceff", cnl]
    wceff[cnl, m] = Sum(wce.where[wcem[wce, m]], comdef[isr, wce, cnl])
    tweff[cnl, m] = (
        1 - ((1 - (wceff[cnl, m] / comdef[isr, "flde", cnl])) * twefac[...])
    ) * comdef[isr, "flde", cnl]
    cneffz[z] = Sum(Domain(cnl, sa), (comdef[isr, "ceff", cnl] * zwt[z, cnl, sa]))
    fleffz[z] = Sum(Domain(cnl, sa), (comdef[isr, "flde", cnl] * zwt[z, cnl, sa]))
    canalwrtz[z, g, m] = Sum(
        Domain(cnl, sa).where[(zsa[z, cnl, sa].where[gwfg[cnl, sa, g]])],
        (
            ((divpost[cnl, m] * subdef[sa, cnl]) * comdef[isr, "ceff", cnl])
            * wceff[cnl, m]
        ),
    )
    gwtsa[cnl, sa, m].where[carea[cnl, "fresh"]] = (
        ((subdef[sa, cnl].where[gwf[cnl, sa]]) * comdef[isr, "cca", cnl])
        / carea[cnl, "fresh"]
    ) * gwt[cnl, m]
    ratiofs[z, g] = resource[z, g, "cca"] / Sum(g1, resource[z, g1, "cca"])
    canalwz[z, g, m] = Sum(
        Domain(cnl, sa).where[(zsa[z, cnl, sa].where[gwfg[cnl, sa, g]])],
        (divpost[cnl, m] * subdef[sa, cnl]),
    )
    eqevapz[z, m] = Sum(Domain(cnl, sa), (eqevap[cnl, m] * zwt[z, cnl, sa]))
    subirrz[z, m] = Sum(Domain(cnl, sa), (subirr[cnl, m] * zwt[z, cnl, sa]))
    efrz[z, m] = Sum(Domain(cnl, sa), (efr[cnl, m] * zwt[z, cnl, sa]))
    tweffz[z, m] = Sum(Domain(cnl, sa), (tweff[cnl, m] * zwt[z, cnl, sa]))
    wceffz[z, m] = Sum(Domain(cnl, sa), (wceff[cnl, m] * zwt[z, cnl, sa]))
    gwt1[z, "fresh", m] = Sum(
        Domain(cnl, sa).where[zsa[z, cnl, sa]], (gwtsa[cnl, sa, m] * wceff[cnl, m])
    )
    print(wceff.records)
    print(tweff.records)
    print(gwtsa.records)
    print(gwt1.records)
    demand[z, cq] = demand[z, cq] * (
        (1 + (growthrd[cq] / 100)) ** (isr.val - baseyear[...])
    )
    orcharea[z] = croparea[z, "orchard"] * (
        (1 + (orchgrowth[z] / 100)) ** (isr.val - baseyear[...])
    )
    resource[z, g, ftt] = (res88[ftt, z] * ratiofs[z, g]) * (
        (1 + (growthres[ftt, z] / 100)) ** (isr.val - baseyear[...])
    )
    ntwucap[...] = twutil["new"] * twcap[...]
    resource[z, "fresh", "tubewells"].where[resource[z, "fresh", "cca"]] = Sum(
        g, resource[z, g, "tubewells"]
    )
    resource[z, "saline", "tubewells"] = 0
    resource[z, "fresh", "twc"].where[resource[z, "fresh", "cca"]] = (
        (resource[z, "fresh", "tubewells"] * twcap[...]) * twutil["existing"]
    ) / 1000

    print(resource.records)
    print(totprod.records)
    print(farmcons.records)
    print(consratio.records)
    print(demand.records)
    print(explimit.records)
    cnl1[cnl] = 1
    cnl1[cnl].where[pvcnl["nwfp", cnl]] = 0
    postt[cnl, sea] = Sum(m.where[seam[sea, m]], divpost[cnl, m])
    postt[pv2, sea] = Sum(cnl.where[pvcnl[pv2, cnl]], postt[cnl, sea])
    protarb[cnl1, sea] = (0.999 * postt[cnl1, sea]) / (
        postt["punjab", sea] + postt["sind", sea]
    )
    protarb[pv2, sea] = (0.999 * postt[pv2, sea]) / (
        postt["punjab", sea] + postt["sind", sea]
    )

    print(cnl1.records)
    print(postt.records)
    print(protarb.records)
    ex[z1, g].where[resource[z1, g, "cca"]] = 1
    print(cq.records)
    print(cn.records)
    print(ccn.records)
    print(qn.records)
    print(ncn.records)
    print(ce.records)
    print(cm.records)
    print(ex.records)
    print(fert.records)
    fsalep[cq] = prices[psr, cq, "financial"]
    pp[...] = pri1[psr, "financial", "protein"]
    misc[p1] = pri1[psr, "financial", p1]
    seedp[c] = finsdwtpr[c, psr, "seed"]
    wage[m] = wageps[psr, "financial", m]
    miscct[c] = finsdwtpr[c, psr, "water"] + finsdwtpr[c, psr, "miscc"]
    importp[cq] = prices[psr, cq, "import"]
    exportp[cq] = prices[psr, cq, "export"]
    esalep[cq] = prices[psr1, cq, "economic"]
    epp[...] = pri1[psr1, "economic", "protein"]
    emisc[p1] = pri1[psr1, "economic", p1]
    eseedp[c] = ecnsdwtpr[c, psr1, "seed"]
    ewage[m] = wageps[psr1, "economic", m]
    emiscct[c] = ecnsdwtpr[c, psr1, "water"] + ecnsdwtpr[c, psr1, "miscc"]
    wnr[c, z1, t, s, w, m] = gp_math.Max(0, (water[c, z1, t, s, w, m] - efrz[z1, m]))
    tec["cotton", t, s, w, "prw"] = 1
    tec["maize", t, s, w, "prw"] = 1
    tech[z1, c, t, s, w].where[tec[c, t, s, w, z1]] = 0
    techc[z1, c].where[Sum(Domain(t, s, w), tech[z1, c, t, s, w])] = 1
    techc[z1, cf] = 0
    techc[z1, q] = 1
    print(techc.records)
    beta[cn, z1].where[demand[z1, cn]] = (fsalep[cn] / demand[z1, cn]) / elast[cn]
    alpha[cn, z1] = fsalep[cn] - (beta[cn, z1] * demand[z1, cn])
    pmin[cn, z1] = 0.5 * fsalep[cn]
    pmax[cn, z1] = gp_math.Min(alpha[cn, z1], (2 * fsalep[cn]))
    qmin[cn, z1].where[beta[cn, z1]] = (pmax[cn, z1] - alpha[cn, z1]) / beta[cn, z1]
    qmax[cn, z1].where[beta[cn, z1]] = (pmin[cn, z1] - alpha[cn, z1]) / beta[cn, z1]
    incr[cn, z1] = (qmax[cn, z1] - qmin[cn, z1]) / (Card(p) - 1)
    qs[cn, z1, p] = qmin[cn, z1] + (incr[cn, z1] * (Ord(p) - 1))
    ws[cn, z1, p] = (alpha[cn, z1] * qs[cn, z1, p]) + (
        (betaf[...] * beta[cn, z1]) * gp_math.sqr(qs[cn, z1, p])
    )
    rs[cn, z1, p] = (alpha[cn, z1] * qs[cn, z1, p]) + (
        beta[cn, z1] * gp_math.sqr(qs[cn, z1, p])
    )
    endpr[cn, z1, p] = alpha[cn, z1] + (beta[cn, z1] * qs[cn, z1, p])
    print(pmax.records)
    print(pmin.records)
    print(qmax.records)
    print(qmin.records)
    print(incr.records)
    print(qs.records)
    print(ws.records)
    print(rs.records)
    print(endpr.records)
    print(alpha.records)
    print(beta.records)
    objz[...] = cps[...] == (
        (
            (
                Sum(
                    z1,
                    Sum(
                        ex[z1, g],
                        (
                            (
                                (
                                    (
                                        Sum(ncn, (fsalep[ncn] * prodt[z1, g, ncn]))
                                        - acost[z1, g]
                                    )
                                    - (Sum(sea, artfod[z1, g, sea]) * pafod[...])
                                )
                                - (Sum(m, artwater[z1, g, m]) * pawat[...])
                            )
                            - Sum(techc[z1, cq], (proda[z1, g, cq] * big[...]))
                        ),
                    ),
                )
                - Sum(z1, Sum(techc[z1, cm], (import_renamed[z1, cm] * importp[cm])))
            )
            + Sum(z1, Sum(techc[z1, ce], (export[z1, ce] * exportp[ce])))
        )
        + Sum(z1, Sum((techc[z1, cn], p), (nat[cn, z1, p] * ws[cn, z1, p])))
    )
    objzn[...] = cps[...] == (
        (
            (
                Sum(
                    z1,
                    Sum(
                        ex[z1, g],
                        (
                            (
                                (
                                    (
                                        Sum(ncn, (fsalep[ncn] * prodt[z1, g, ncn]))
                                        - acost[z1, g]
                                    )
                                    - (Sum(sea, artfod[z1, g, sea]) * pafod[...])
                                )
                                - (Sum(m, artwater[z1, g, m]) * pawat[...])
                            )
                            - Sum(techc[z1, cq], (proda[z1, g, cq] * big[...]))
                        ),
                    ),
                )
                - Sum(z1, Sum(techc[z1, cm], (import_renamed[z1, cm] * importp[cm])))
            )
            + Sum(z1, Sum(techc[z1, ce], (export[z1, ce] * exportp[ce])))
        )
        + Sum(
            z1,
            Sum(
                techc[z1, cn],
                (
                    (alpha[cn, z1] * natn[cn, z1])
                    + ((betaf[...] * beta[cn, z1]) * gp_math.sqr(natn[cn, z1]))
                ),
            ),
        )
    )
    objn[...] = cps[...] == (
        (
            (
                (
                    Sum(
                        z1,
                        Sum(
                            ex[z1, g],
                            (
                                (
                                    (
                                        (
                                            Sum(ncn, (fsalep[ncn] * prodt[z1, g, ncn]))
                                            - acost[z1, g]
                                        )
                                        - (Sum(sea, artfod[z1, g, sea]) * pafod[...])
                                    )
                                    - (Sum(m, artwater[z1, g, m]) * pawat[...])
                                )
                                - Sum(techc[z1, cq], (proda[z1, g, cq] * big[...]))
                            ),
                        ),
                    )
                    - Sum(
                        z1, Sum(techc[z1, cm], (import_renamed[z1, cm] * importp[cm]))
                    )
                )
                + Sum(z1, Sum(techc[z1, ce], (export[z1, ce] * exportp[ce])))
            )
            + Sum(z1, Sum((techc[z1, cn], p), (nat[cn, z1, p] * ws[cn, z1, p])))
        )
        + Sum(
            Domain(n, m),
            (
                (
                    (-(artwaternd[n, m] * pawat[...]))
                    + (rval[n] * (rcont[n, m].where[rcap[n]]))
                )
                + (rval["a-sea"] * f["a-sea", "kotri-b", m])
            ),
        )
    )
    objnn[...] = cps[...] == (
        (
            (
                (
                    Sum(
                        z1,
                        Sum(
                            ex[z1, g],
                            (
                                (
                                    (
                                        (
                                            Sum(ncn, (fsalep[ncn] * prodt[z1, g, ncn]))
                                            - acost[z1, g]
                                        )
                                        - (Sum(sea, artfod[z1, g, sea]) * pafod[...])
                                    )
                                    - (Sum(m, artwater[z1, g, m]) * pawat[...])
                                )
                                - Sum(techc[z1, cq], (proda[z1, g, cq] * big[...]))
                            ),
                        ),
                    )
                    - Sum(
                        z1, Sum(techc[z1, cm], (import_renamed[z1, cm] * importp[cm]))
                    )
                )
                + Sum(z1, Sum(techc[z1, ce], (export[z1, ce] * exportp[ce])))
            )
            + Sum(
                z1,
                Sum(
                    techc[z1, cn],
                    (
                        (alpha[cn, z1] * natn[cn, z1])
                        + ((betaf[...] * beta[cn, z1]) * gp_math.sqr(natn[cn, z1]))
                    ),
                ),
            )
        )
        + Sum(
            Domain(n, m),
            (
                (
                    (-(artwaternd[n, m] * pawat[...]))
                    + (rval[n] * (rcont[n, m].where[rcap[n]]))
                )
                + (rval["a-sea"] * f["a-sea", "kotri-b", m])
            ),
        )
    )
    cost[ex[z1, g]] = acost[z1, g] == (
        (
            (
                (
                    (
                        (
                            (
                                Sum(
                                    tech[z1, c, t, s, w],
                                    (
                                        (
                                            (
                                                Sum(p2, (fert[p2, c, z1] * misc[p2]))
                                                + miscct[c]
                                            )
                                            + (seedp[c] * sylds[c, z1, t, s, w, "seed"])
                                        )
                                        * x[z1, g, c, t, s, w]
                                    ),
                                )
                                + Sum(
                                    m,
                                    (
                                        (misc["twopc"] * (tw[z1, m].where[gf[g]]))
                                        + (misc["tropc"] * ts[z1, g, m])
                                    ),
                                )
                            )
                            + (misc["twinvt"] * (itw[z1].where[gf[g]]))
                        )
                        + (misc["trinvt"] * itr[z1, g])
                    )
                    + Sum(a, (iolive[a, z1, "fix-cost"] * animal[z1, g, a]))
                )
                / 1000
            )
            + Sum(sea, (pp[...] * ppc[z1, g, sea]))
        )
        + Sum(m, (((familyl[z1, g, m] * labfac[...]) + hiredl[z1, g, m]) * wage[m]))
    )
    conv[techc[z1, cn]] = Sum(p, nat[cn, z1, p]) <= 1
    demnat[techc[z1, cq]] = (
        (
            Sum(
                ex[z1, g],
                (
                    (prodt[z1, g, cq] - (consump[z1, g, cq].where[cc[cq]]))
                    + proda[z1, g, cq]
                ),
            )
            - (export[z1, cq].where[ce[cq]])
        )
        + (import_renamed[z1, cq].where[cm[cq]])
    ) >= (Sum(p, (nat[cq, z1, p] * qs[cq, z1, p])).where[cn[cq]])
    demnatn[techc[z1, cq]] = (
        (
            Sum(
                ex[z1, g],
                (
                    (prodt[z1, g, cq] - (consump[z1, g, cq].where[cc[cq]]))
                    + proda[z1, g, cq]
                ),
            )
            - (export[z1, cq].where[ce[cq]])
        )
        + (import_renamed[z1, cq].where[cm[cq]])
    ) >= (natn[cq, z1].where[cn[cq]])
    ccombal[ex[z1, g], cnf[c]] = (
        Sum(
            tech[z1, c, t, s, w],
            (yield_renamed[c, t, s, w, z1] * x[z1, g, c, t, s, w]),
        )
        == prodt[z1, g, c]
    )
    qcombal[ex[z1, g], q] = (
        Sum(a, (iolive[a, z1, q] * animal[z1, g, a])) / 1000
    ) == prodt[z1, g, q]
    consbal[ex[z1, g], cc].where[techc[z1, cc]] = (
        prodt[z1, g, cc] + proda[z1, g, cc]
    ) >= consump[z1, g, cc]
    laborc[ex[z1, g], m] = (
        (
            Sum(tech[z1, c, t, s, w], (labor[c, z1, t, s, w, m] * x[z1, g, c, t, s, w]))
            + Sum(a, (iolive[a, z1, "labor"] * animal[z1, g, a]))
        )
        / 1000
    ) <= (familyl[z1, g, m] + hiredl[z1, g, m])
    fodder[ex[z1, g], sea] = Sum(a, (iolive[a, z1, "tdn"] * animal[z1, g, a])) <= (
        (
            Sum(
                tech[z1, c, t, s, w],
                (
                    (
                        (
                            (
                                yield_renamed[c, t, s, w, z1]
                                * sylds[c, z1, t, s, w, "straw-yld"]
                            )
                            * sconv["tdn", sea, c]
                        )
                        + (weedy[z1, sea, c] * sconv["tdn", "rabi", "rab-fod"])
                    )
                    * x[z1, g, c, t, s, w]
                ),
            )
            + (
                (Sum(m, slkland[z1, g, m]) * graz[z1, sea])
                * sconv["tdn", "rabi", "rab-fod"]
            )
        )
        + (artfod[z1, g, sea] * sconv["tdn", "rabi", "rab-fod"])
    )
    protein[ex[z1, g], sea] = Sum(a, (iolive[a, z1, "dp"] * animal[z1, g, a])) <= (
        (
            (
                ppc[z1, g, sea]
                + Sum(
                    tech[z1, c, t, s, w],
                    (
                        (
                            (
                                (
                                    yield_renamed[c, t, s, w, z1]
                                    * sylds[c, z1, t, s, w, "straw-yld"]
                                )
                                * sconv["dp", sea, c]
                            )
                            + (weedy[z1, sea, c] * sconv["dp", "rabi", "rab-fod"])
                        )
                        * x[z1, g, c, t, s, w]
                    ),
                )
            )
            + (
                (Sum(m, slkland[z1, g, m]) * graz[z1, sea])
                * sconv["dp", "rabi", "rab-fod"]
            )
        )
        + (artfod[z1, g, sea] * sconv["dp", "rabi", "rab-fod"])
    )
    grnfdr[ex[z1, g], sea] = (
        gr[...] * Sum(a, (iolive[a, z1, "tdn"] * animal[z1, g, a]))
    ) <= (
        (
            Sum(
                tech[z1, cf, t, s, w],
                (
                    (yield_renamed[cf, t, s, w, z1] * sconv["tdn", sea, cf])
                    * x[z1, g, cf, t, s, w]
                ),
            )
            + Sum(
                tech[z1, c, t, s, w],
                (
                    (weedy[z1, sea, c] * sconv["tdn", "rabi", "rab-fod"])
                    * x[z1, g, c, t, s, w]
                ),
            )
        )
        + (artfod[z1, g, sea] * sconv["tdn", "rabi", "rab-fod"])
    )
    bdraft[ex[z1, g], m] = (
        Sum(tech[z1, c, t, s, w], (bullock[c, z1, t, s, w, m] * x[z1, g, c, t, s, w]))
        / 1000
    ) <= ((bp[m] * animal[z1, g, "bullock"]) / 1000)
    brepco[ex[z1, g]] = animal[z1, g, "bullock"] <= (repco[...] * animal[z1, g, "cow"])
    bullockc[z1] = Sum(ex[z1, g], animal[z1, g, "bullock"]) <= res88["bullocks", z1]
    tdraft[z1, g, m].where[ex[z1, g]] = (
        Sum(tech[z1, c, t, s, w], (tractor[c, z1, t, s, w, m] * x[z1, g, c, t, s, w]))
        == ts[z1, g, m]
    )
    trcapc[z1, m] = Sum(ex[z1, g], ts[z1, g, m]) <= Sum(
        g, (((resource[z1, g, "tractors"] / 1000) + itr[z1, g]) * trcap[...])
    )
    twcapc[z1, m].where[ex[z1, "fresh"]] = tw[z1, m] <= (
        resource[z1, "fresh", "twc"] + (ntwucap[...] * itw[z1])
    )
    landc[ex[z1, g], m] = (
        Sum(tech[z1, c, t, s, w], (land[c, z1, t, s, w, m] * x[z1, g, c, t, s, w]))
        + slkland[z1, g, m]
    ) == (resource[z1, g, "cca"] * 1000)
    orchareac[z1] = (
        Sum(
            Domain(g, t, s, w).where[ex[z1, g]],
            (x[z1, g, "orchard", t, s, w].where[tech[z1, "orchard", t, s, w]]),
        )
        <= orcharea[z1]
    )
    scmillc[z1] = Sum(ex[z1, g], prodt[z1, g, "sc-mill"]) <= scmillcap[z1]
    waterbaln[ex[z1, g], m] = (
        Sum(
            tech[z1, c, t, s, w],
            (
                gp_math.Max(
                    (
                        wnr[c, z1, t, s, w, m]
                        - (subirrz[z1, m] * land[c, z1, t, s, w, m])
                    ),
                    0,
                )
                * x[z1, g, c, t, s, w]
            ),
        )
        + slkwater[z1, g, m]
    ) == (
        (
            ((tweffz[z1, m] * (tw[z1, m].where[gf[g]])) + gwt1[z1, g, m])
            + artwater[z1, g, m]
        )
        + wdivrz[z1, g, m]
    )
    watalcz[ex[z1, g], m] = wdivrz[z1, g, m] == Sum(
        zsa[z1, cnl, sa].where[gwfg[cnl, sa, g]],
        ((((cneff[cnl] * wceff[cnl, m]) * canaldiv[cnl, m]) * subdef[sa, cnl]) * 1000),
    )
    divcnlsea[cnl, sea] = cnldivsea[cnl, sea] == Sum(seam[sea, m], canaldiv[cnl, m])
    prseaw[pv2, sea] = prsea[pv2, sea] == Sum(cnl1, cnldivsea[cnl1, sea])
    divsea[sea] = tcdivsea[sea] == Sum(pv2, prsea[pv2, sea])
    watalcsea[cnl1, sea] = (
        (protarb[cnl1, sea] * (1 - tolcnl[...])) * tcdivsea[sea]
    ) <= cnldivsea[cnl1, sea]
    watalcpro[pv2, sea] = (
        (protarb[pv2, sea] * (1 - tolpr[...])) * tcdivsea[sea]
    ) <= prsea[pv2, sea]
    nwfpalc[m] = Sum(cnl, canaldiv[cnl, m]) >= ((1 - tolnwfp[...]) * divnwfp[m])
    subirrc[ex[z1, g], m].where[gs[g]] = wdivrz[z1, g, m] >= (
        (1 - subirrfac[z1])
        * Sum(tech[z1, c, t, s, w], (wnr[c, z1, t, s, w, m] * x[z1, g, c, t, s, w]))
    )
    nbal[nb[n], m] = (
        (
            (
                (
                    (
                        (
                            Sum(ni[n, i], inflow[i, m])
                            + Sum(
                                n1,
                                (
                                    (rivercd[n, "d"] * trib[n1, n, m])
                                    + (rivercd[n, "c"] * trib[n1, n, m.lag(1)])
                                ),
                            )
                        )
                        + Sum(
                            nn[n, n1],
                            (
                                (f[n, n1, m] * (lceff[n1, n].where[lceff[n1, n]]))
                                + (
                                    (
                                        (riverb[n, n1] * f[n, n1, m])
                                        + (rivercd[n, "c"] * f[n, n1, m.lag(1)])
                                    ).where[(lceff[n1, n] == 0)]
                                )
                            ),
                        )
                    )
                    - Sum(nn[n1, n], f[n1, n, m])
                )
                + (
                    ((rcont[n, m.lag(1)] - rcont[n, m]) - (revapl[n, m] / 1000)).where[
                        rcap[n]
                    ]
                )
            )
            - Sum(nc[n, cnl], canaldiv[cnl, m])
        )
        + artwaternd[n, m]
    ) == 0
    artwaternd.lo[n, m] = 0
    f.up[n, n1, m] = SpecialValues.POSINF
    f.up[n, n1, m].where[(ncap[n1, n] != 0)] = ncap[n1, n]
    familyl.up[z1, g, m] = (resource[z1, g, "farmpop"] * lstd[...]) / 1000
    consump.fx[z1, g, cq] = farmcons[z1, cq] * consratio[z1, g]
    export.up[z1, ce] = explimit[z1, ce]
    itr.fx[z1, g] = 0
    itw.fx[z1] = 0
    canaldiv.up[cnl, m] = comdef[isr, "ccap", cnl]
    canaldiv.lo[cnl, m] = divpost[cnl, m]
    rcont.lo[n, m] = (rulelo[n, m] * rcap[n]) / 100
    rcont.up[n, m] = (ruleup[n, m] * rcap[n]) / 100
    tolnwfp[...] = 1
    trib["chasma-r", "taunsa-b", m] = 0
    trib["tarbela-r", "kalabagh-r", m] = 0
    inflow["haro", m] = 0
    inflow["soan", m] = 0
    wsiszn = Model(
        container,
        name="wsiszn",
        equations=[
            objzn,
            demnatn,
            scmillc,
            cost,
            ccombal,
            qcombal,
            consbal,
            laborc,
            fodder,
            protein,
            grnfdr,
            bdraft,
            brepco,
            tdraft,
            trcapc,
            twcapc,
            landc,
            orchareac,
            waterbaln,
            watalcz,
            subirrc,
        ],
        problem="NLP",
        sense="MAX",
        objective=cps,
    )

    wsiszn.solve(solver="conopt")
    assert math.isclose(wsiszn.objective_value, 124599.00034157003, abs_tol=1e-3), (
        wsiszn.objective_value
    )


if __name__ == "__main__":
    main()
