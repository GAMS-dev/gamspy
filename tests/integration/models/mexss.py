"""
Mexico Steel - Small Static (MEXSS)

A simplified representation of the Mexican steel sector is used
to introduce a process industry production and distribution
scheduling problem.


Kendrick, D, Meeraus, A, and Alatorre, J, The Planning of Investment
Programs in the Steel Industry. The Johns Hopkins University Press,
Baltimore and London, 1984.

A scanned version of this out-of-print book is accessible at
http://www.gams.com/docs/pdf/steel_investment.pdf

Keywords: linear programming, production problem, distribution problem,
scheduling,
          micro economics, steel industry
"""

from gamspy import Set, Parameter, Variable, Equation, Model, Container
from gamspy import Sum, Sense
import pandas


def main():
    cont = Container()

    # Prepare data
    steel_plants = ["ahmsa", "fundidora", "sicartsa", "hylsa", "hylsap"]
    markets = ["mexico-df", "monterrey", "guadalaja"]
    commodities = [
        "pellets",
        "coke",
        "nat-gas",
        "electric",
        "scrap",
        "pig-iron",
        "sponge",
        "steel",
    ]
    final_products = ["steel"]
    intermediate_products = ["sponge", "pig-iron"]
    raw_materials = ["pellets", "coke", "nat-gas", "electric", "scrap"]
    processes = ["pig-iron", "sponge", "steel-oh", "steel-el", "steel-bof"]
    productive_units = [
        "blast-furn",
        "openhearth",
        "bof",
        "direct-red",
        "elec-arc",
    ]

    io_coefficients = pandas.DataFrame(
        [
            ["pellets", "pig-iron", -1.58],
            ["pellets", "sponge", -1.38],
            ["coke", "pig-iron", -0.63],
            ["nat-gas", "sponge", -0.57],
            ["electric", "steel-el", -0.58],
            ["scrap", "steel-oh", -0.33],
            ["scrap", "steel-bof", -0.12],
            ["pig-iron", "pig-iron", 1.00],
            ["pig-iron", "steel-oh", -0.77],
            ["pig-iron", "steel-bof", -0.95],
            ["sponge", "sponge", 1.00],
            ["sponge", "steel-el", -1.09],
            ["steel", "steel-oh", 1.00],
            ["steel", "steel-el", 1.00],
            ["steel", "steel-bof", 1.00],
        ]
    )

    capacity_utilization = pandas.DataFrame(
        [
            ["blast-furn", "pig-iron", 1.0],
            ["openhearth", "steel-oh", 1.0],
            ["bof", "steel-bof", 1.0],
            ["direct-red", "sponge", 1.0],
            ["elec-arc", "steel-el", 1.0],
        ]
    )

    capacities_of_units = pandas.DataFrame(
        [
            ["blast-furn", "ahmsa", 3.25],
            ["blast-furn", "fundidora", 1.40],
            ["blast-furn", "sicartsa", 1.10],
            ["openhearth", "ahmsa", 1.50],
            ["openhearth", "fundidora", 0.85],
            ["bof", "ahmsa", 2.07],
            ["bof", "fundidora", 1.50],
            ["bof", "sicartsa", 1.30],
            ["direct-red", "hylsa", 0.98],
            ["direct-red", "hylsap", 1.00],
            ["elec-arc", "hylsa", 1.13],
            ["elec-arc", "hylsap", 0.56],
        ]
    )

    rail_distances = pandas.DataFrame(
        [
            ["ahmsa", "mexico-df", 1204],
            ["ahmsa", "monterrey", 218],
            ["ahmsa", "guadalaja", 1125],
            ["ahmsa", "export", 739],
            ["fundidora", "mexico-df", 1017],
            ["fundidora", "guadalaja", 1030],
            ["fundidora", "export", 521],
            ["sicartsa", "mexico-df", 819],
            ["sicartsa", "monterrey", 1305],
            ["sicartsa", "guadalaja", 704],
            ["hylsa", "mexico-df", 1017],
            ["hylsa", "guadalaja", 1030],
            ["hylsa", "export", 521],
            ["hylsap", "mexico-df", 185],
            ["hylsap", "monterrey", 1085],
            ["hylsap", "guadalaja", 760],
            ["hylsap", "export", 315],
            ["import", "mexico-df", 428],
            ["import", "monterrey", 521],
            ["import", "guadalaja", 300],
        ]
    )

    product_prices = pandas.DataFrame(
        [
            ["pellets", "domestic", 18.7],
            ["coke", "domestic", 52.17],
            ["nat-gas", "domestic", 14.0],
            ["electric", "domestic", 24.0],
            ["scrap", "domestic", 105.0],
            ["steel", "import", 150],
            ["steel", "export", 140],
        ]
    )

    demand_distribution = pandas.DataFrame(
        [["mexico-df", 55], ["monterrey", 30], ["guadalaja", 15]]
    )

    dt = 5.209  # total demand for final goods in 1979
    rse = 40  # raw steel equivalence
    eb = 1.0  # export bound

    # Set
    i = Set(
        cont,
        name="i",
        records=pandas.DataFrame(steel_plants),
        description="steel plants",
    )
    j = Set(
        cont,
        name="j",
        records=pandas.DataFrame(markets),
        description="markets",
    )
    c = Set(
        cont,
        name="c",
        records=pandas.DataFrame(commodities),
        description="commidities",
    )
    cf = Set(
        cont,
        name="cf",
        records=pandas.DataFrame(final_products),
        domain=[c],
        description="final products",
    )
    ci = Set(
        cont,
        name="ci",
        records=pandas.DataFrame(intermediate_products),
        domain=[c],
        description="intermediate products",
    )
    cr = Set(
        cont,
        name="cr",
        records=pandas.DataFrame(raw_materials),
        domain=[c],
        description="raw materials",
    )
    p = Set(
        cont,
        name="p",
        records=pandas.DataFrame(processes),
        description="processes",
    )
    m = Set(
        cont,
        name="m",
        records=pandas.DataFrame(productive_units),
        description="productive units",
    )

    # Data
    a = Parameter(
        cont,
        name="a",
        domain=[c, p],
        records=io_coefficients,
        description="input-output coefficients",
    )
    b = Parameter(
        cont,
        name="b",
        domain=[m, p],
        records=capacity_utilization,
        description="capacity utilization",
    )
    k = Parameter(
        cont,
        name="k",
        domain=[m, i],
        records=capacities_of_units,
        description="capacities of productive units",
    )
    dd = Parameter(
        cont,
        name="dd",
        domain=[j],
        records=demand_distribution,
        description="distribution of demand",
    )
    d = Parameter(
        cont, name="d", domain=[c, j], description="demand for steel in 1979"
    )

    d["steel", j] = dt * (1 + rse / 100) * dd[j] / 100

    rd = Parameter(
        cont,
        name="rd",
        domain=["*", "*"],
        records=rail_distances,
        description="rail distances from plants to markets",
    )

    muf = Parameter(
        cont,
        name="muf",
        domain=[i, j],
        description="transport rate: final products",
    )
    muv = Parameter(
        cont, name="muv", domain=[j], description="transport rate: imports"
    )
    mue = Parameter(
        cont, name="mue", domain=[i], description="transport rate: exports"
    )

    muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]
    muv[j] = (2.48 + 0.0084 * rd["import", j]).where[rd["import", j]]
    mue[i] = (2.48 + 0.0084 * rd[i, "export"]).where[rd[i, "export"]]

    prices = Parameter(
        cont,
        name="prices",
        domain=[c, "*"],
        records=product_prices,
        description="product prices (us$ per unit)",
    )

    pd = Parameter(cont, name="pd", domain=[c], description="domestic prices")
    pv = Parameter(cont, name="pv", domain=[c], description="import prices")
    pe = Parameter(cont, name="pe", domain=[c], description="export prices")

    pd[c] = prices[c, "domestic"]
    pv[c] = prices[c, "import"]
    pe[c] = prices[c, "export"]

    # Variable
    z = Variable(
        cont,
        name="z",
        domain=[p, i],
        type="Positive",
        description="process level",
    )
    x = Variable(
        cont,
        name="x",
        domain=[c, i, j],
        type="Positive",
        description="shipment of final products",
    )
    u = Variable(
        cont,
        name="u",
        domain=[c, i],
        type="Positive",
        description="purchase of domestic materials",
    )
    v = Variable(
        cont, name="v", domain=[c, j], type="Positive", description="imports"
    )
    e = Variable(
        cont, name="e", domain=[c, i], type="Positive", description="exports"
    )
    phi = Variable(cont, name="phi", description="total cost")
    phipsi = Variable(cont, name="phipsi", description="raw material cost")
    philam = Variable(cont, name="philam", description="transport cost")
    phipi = Variable(cont, name="phipi", description="import cost")
    phieps = Variable(cont, name="phieps", description="export revenue")

    # Equation declaration
    mbf = Equation(
        cont,
        name="mbf",
        domain=[c, i],
        description="material balances: final products",
    )
    mbi = Equation(
        cont,
        name="mbi",
        domain=[c, i],
        description="material balances: intermediates",
    )
    mbr = Equation(
        cont,
        name="mbr",
        domain=[c, i],
        description="material balances: raw materials",
    )
    cc = Equation(
        cont,
        name="cc",
        domain=[m, i],
        description="capacity constraint",
    )
    mr = Equation(
        cont,
        name="mr",
        domain=[c, j],
        description="market requirements",
    )
    me = Equation(
        cont,
        name="me",
        domain=[c],
        description="maximum export",
    )
    obj = Equation(cont, name="obj", description="accounting: total cost")
    apsi = Equation(
        cont,
        name="apsi",
        description="accounting: raw material cost",
    )
    alam = Equation(
        cont,
        name="alam",
        description="accounting: transport cost",
    )
    api = Equation(cont, name="api", description="accounting: import cost")
    aeps = Equation(
        cont,
        name="aeps",
        description="accounting: export cost",
    )

    # Equation definition
    mbf[cf, i] = Sum(p, a[cf, p] * z[p, i]) >= Sum(j, x[cf, i, j]) + e[cf, i]
    mbi[ci, i] = Sum(p, a[ci, p] * z[p, i]) >= 0
    mbr[cr, i] = Sum(p, a[cr, p] * z[p, i]) + u[cr, i] >= 0
    cc[m, i] = Sum(p, b[m, p] * z[p, i]) <= k[m, i]
    mr[cf, j] = Sum(i, x[cf, i, j]) + v[cf, j] >= d[cf, j]
    me[cf] = Sum(i, e[cf, i]) <= eb
    obj.expr = phi == phipsi + philam + phipi - phieps
    apsi.expr = phipsi == Sum((cr, i), pd[cr] * u[cr, i])
    alam.expr = philam == Sum((cf, i, j), muf[i, j] * x[cf, i, j]) + Sum(
        (cf, j), muv[j] * v[cf, j]
    ) + Sum((cf, i), mue[i] * e[cf, i])
    api.expr = phipi == Sum((cf, j), pv[cf] * v[cf, j])
    aeps.expr = phieps == Sum((cf, i), pe[cf] * e[cf, i])

    mexss = Model(
        cont,
        name="mexss",
        equations=cont.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=phi,
    )

    mexss.solve()
    print(mexss.objective_value)


if __name__ == "__main__":
    main()
