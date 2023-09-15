"""
Indexation model using the co-movements approach

ComovementsModel.gms: Indexation model using the co-movements approach
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 7.2.2
Last modified: Apr 2008.
"""
from pathlib import Path

from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import ModelStatus
from gamspy import Parameter
from gamspy import Sum
from gamspy import Variable


def main():
    gdx_file = str(Path(__file__).parent.absolute()) + "/InputData.gdx"
    m = Container(load_from=gdx_file)

    # SETS #
    BB, EE, BxE, SS = m.getSymbols(["BB2", "EE", "BxE2", "SS"])

    # PARAMETERS #
    # From GDX file
    (
        BondPrices1,
        BondPrices0,
        ExchangeRates0,
        ExchangeRates1,
        IndexReturns,
    ) = m.getSymbols(
        ["data2", "data3", "ExchangeRates0", "ExchangeRates1", "IndexReturns"]
    )

    BondReturns = Parameter(m, name="BondReturns", domain=[SS, BB])
    ExchangeRatesReturns = Parameter(
        m, name="ExchangeRatesReturns", domain=[SS, EE]
    )
    ExpectedReturns = Parameter(m, name="ExpectedReturns", domain=[BB])

    # SCALAR #
    EpsTolerance = Parameter(m, name="EpsTolerance")

    i = Alias(m, name="i", alias_with=BB)
    l = Alias(m, name="l", alias_with=SS)
    e = Alias(m, name="e", alias_with=EE)

    BondReturns[l, i] = (BondPrices1[l, i] - BondPrices0[i]) / BondPrices0[i]
    ExchangeRatesReturns[l, e] = (
        ExchangeRates1[l, e] - ExchangeRates0[e]
    ) / ExchangeRates0[e]

    # Calculate bond returns in USD currency

    BondReturns[l, i].where[BxE[i, "DEM"]] = (
        BondReturns[l, i].where[BxE[i, "DEM"]] + ExchangeRatesReturns[l, "DEM"]
    )
    BondReturns[l, i].where[BxE[i, "CHF"]] = (
        BondReturns[l, i].where[BxE[i, "CHF"]] + ExchangeRatesReturns[l, "CHF"]
    )

    ExpectedReturns[i] = (1.0 / Card(l)) * Sum(l, BondReturns[l, i])

    # VARIABLES #
    z = Variable(m, name="z", type="free")
    x = Variable(m, name="x", type="positive", domain=[i])

    # EQUATIONS #
    ObjDef = Equation(m, name="ObjDef", type="regular")
    NormalCon = Equation(m, name="NormalCon", type="regular")
    TrackingConL = Equation(m, name="TrackingConL", type="regular", domain=[l])
    TrackingConG = Equation(m, name="TrackingConG", type="regular", domain=[l])

    ObjDef.expr = z == Sum(i, ExpectedReturns[i] * x[i])

    TrackingConG[l] = (
        Sum(i, BondReturns[l, i] * x[i]) - IndexReturns[l] >= -EpsTolerance
    )

    TrackingConL[l] = (
        Sum(i, BondReturns[l, i] * x[i]) - IndexReturns[l] <= EpsTolerance
    )

    NormalCon.expr = Sum(i, x[i]) == 1.0

    IndexFund = Model(
        m,
        name="IndexFund",
        equations=[ObjDef, TrackingConG, TrackingConL, NormalCon],
        problem="LP",
        sense="MAX",
        objective=z,
    )

    EpsTolerance.assign = 0.1

    while IndexFund.status in [
        ModelStatus.OptimalGlobal,
        ModelStatus.OptimalLocal,
    ]:
        IndexFund.solve()
        if IndexFund.status in [
            ModelStatus.OptimalGlobal,
            ModelStatus.OptimalLocal,
        ]:
            print(
                f"EpsTolerance: {round(EpsTolerance.toValue(),3)}\t-->\t z:"
                f" {round(z.toValue(),6)}"
            )
        EpsTolerance.assign = EpsTolerance - 0.01

    EpsTolerance.assign = EpsTolerance + 0.02
    IndexFund.solve()

    x_recs = {k: round(v, 3) for k, v in x.toDict().items() if v != 0}
    print(f"\nz:  {round(z.toValue(),3)}\nx: ", x_recs, "\n")


if __name__ == "__main__":
    main()
