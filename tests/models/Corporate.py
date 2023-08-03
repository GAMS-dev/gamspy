"""
Corporate bond indexation model

Corporate.gms: Corporate bond indexation model
Consiglio, Nielsen and Zenios.
PRACTICAL FINANCIAL OPTIMIZATION: A Library of GAMS Models, Section 8.3
Last modified: May 2008.
"""

from pathlib import Path
from gamspy import (
    Parameter,
    Variable,
    Equation,
    Container,
    Model,
    Sum,
    Card,
    Number,
    Sense,
)
import gamspy.math as gams_math


def main():
    # Define container
    m = Container(
        load_from=str(Path(__file__).parent.absolute()) + "/Corporate.gdx",
    )

    # SETS #
    BroadAssetClassOne, BroadAssetClassTwo, BroadAssetClassThree, ACTIVE = (
        m.getSymbols(
            [
                "BroadAssetClassOne",
                "BroadAssetClassTwo",
                "BroadAssetClassThree",
                "ACTIVE",
            ]
        )
    )

    # ALIASES
    i, l, j, m1, m2, m3, a = m.getSymbols(
        ["i", "l", "j", "m1", "m2", "m3", "a"]
    )

    # PARAMETERS #
    AssetReturns = m.getSymbols(["AssetReturns"])[0]
    BroadWeights = Parameter(
        m,
        name="BroadWeights",
        domain=[j],
        description="Weights of the broad asset classes",
    )
    AssetWeights = Parameter(
        m,
        name="AssetWeights",
        domain=[i],
        description="Weights of each asset in the index",
    )

    # Assign weights randomly

    AssetWeights[i] = gams_math.uniform(1, 10)

    # DISPLAY  AssetWeights

    # Normalize the random weights
    WeightsSum = Parameter(m, name="WeightsSum")

    WeightsSum.assign = Sum(i, AssetWeights[i])

    AssetWeights[i] = AssetWeights[i] / WeightsSum

    BroadWeights["BA_1"] = Sum(m1, AssetWeights[m1])
    BroadWeights["BA_2"] = Sum(m2, AssetWeights[m2])
    BroadWeights["BA_3"] = Sum(m3, AssetWeights[m3])

    IndexReturns = Parameter(
        m,
        name="IndexReturns",
        domain=[l],
        description="Index return scenarios",
    )
    BroadAssetReturns = Parameter(
        m,
        name="BroadAssetReturns",
        domain=[j, l],
        description="Broad asset class return scenarios",
    )
    Benchmark = Parameter(
        m,
        name="Benchmark",
        domain=[l],
        description="Current benchmark scenario returns",
    )

    BroadAssetReturns["BA_1", l] = Sum(
        m1, AssetWeights[m1] * AssetReturns[m1, l]
    )
    BroadAssetReturns["BA_2", l] = Sum(
        m2, AssetWeights[m2] * AssetReturns[m2, l]
    )
    BroadAssetReturns["BA_3", l] = Sum(
        m3, AssetWeights[m3] * AssetReturns[m3, l]
    )

    IndexReturns[l] = Sum(j, BroadWeights[j] * BroadAssetReturns[j, l])

    CurrentWeight = Parameter(
        m,
        name="CurrentWeight",
        description="Current weight for tactcal allocation",
    )
    EpsTolerance = Parameter(m, name="EpsTolerance", description="Tolerance")
    pr = Parameter(
        m, name="pr", domain=[l], description="Scenario probability"
    )

    pr[l] = 1.0 / Card(l)

    # VARIABLES #
    x = Variable(
        m,
        name="x",
        type="positive",
        domain=[i],
        description="Percentage invested in each security",
    )
    z = Variable(
        m,
        name="z",
        type="positive",
        domain=[j],
        description="Percentages invested in each broad asset class",
    )
    PortRet = Variable(
        m, name="PortRet", domain=[l], description="Portfolio returns"
    )
    ObjValue = Variable(
        m, name="ObjValue", description="Objective function value"
    )

    # EQUATIONS #
    ObjDef = Equation(
        m,
        name="ObjDef",
        type="eq",
        description=(
            "Objective function for the strategic model (Expected return)"
        ),
    )
    BroadPortRetDef = Equation(
        m,
        name="BroadPortRetDef",
        type="eq",
        domain=[l],
        description="Portfolio return definition for broad asset classes",
    )
    PortRetDef = Equation(
        m,
        name="PortRetDef",
        type="eq",
        domain=[l],
        description="Portfolio return definition",
    )
    BroadNormalCon = Equation(
        m,
        name="BroadNormalCon",
        type="eq",
        description=(
            "Equation defining the normalization contraint for broad asset"
            " classes"
        ),
    )
    NormalCon = Equation(
        m,
        name="NormalCon",
        type="eq",
        description="Equation defining the normalization contraint",
    )
    MADCon = Equation(
        m, name="MADCon", type="geq", domain=[l], description="MAD constraints"
    )

    ObjDef.definition = ObjValue == Sum(l, pr[l] * PortRet[l])

    BroadPortRetDef[l] = PortRet[l] == Sum(j, z[j] * BroadAssetReturns[j, l])

    PortRetDef[l] = PortRet[l] == Sum(a, x[a] * AssetReturns[a, l])

    MADCon[l] = PortRet[l] >= Benchmark[l] - EpsTolerance

    BroadNormalCon.definition = Sum(j, z[j]) == 1.0

    NormalCon.definition = Sum(a, x[a]) == CurrentWeight

    m.addOptions({"SOLVEOPT": "REPLACE"})

    # MODELS #
    StrategicModel = Model(
        m,
        name="StrategicModel",
        equations=[ObjDef, BroadPortRetDef, MADCon, BroadNormalCon],
        problem="LP",
        sense=Sense.MAX,
        objective_variable=ObjValue,
    )
    TacticalModel = Model(
        m,
        name="TacticalModel",
        equations=[ObjDef, PortRetDef, MADCon, NormalCon],
        problem="LP",
        sense=Sense.MAX,
        objective_variable=ObjValue,
    )

    # Solve strategic model

    Benchmark[l] = IndexReturns[l]

    EpsTolerance.assign = 0.02

    StrategicModel.solve()

    print("## Strategic Asset Allocation ##")
    print("z: \n", z.records.level.to_list())

    # Solve tactical model for Broad Asset 1 (BA_1)

    CurrentWeight.assign = z.l["BA_1"]

    Benchmark[l] = BroadAssetReturns["BA_1", l]

    EpsTolerance.assign = 0.02

    ACTIVE[i] = BroadAssetClassOne[i]

    if CurrentWeight.records.value[0] > 0.05:
        m.solve(
            TacticalModel,
            problem="LP",
            sense=Sense.MAX,
            objective_variable=ObjValue,
        )

        print("\n## Model BA_1 ##")
        print("\nx: \n", x.records.level.tolist())

    # Solve tactical model for Broad Asset 2 (BA_2)

    CurrentWeight.assign = z.l["BA_2"]

    ACTIVE[i] = BroadAssetClassTwo[i]

    Benchmark[l] = BroadAssetReturns["BA_2", l]

    EpsTolerance.assign = 0.03

    if CurrentWeight.records.value[0] > 0.05:
        TacticalModel.solve()

        print("\n## Model BA_2 ##")
        print("\nx: \n", x.records.level.tolist())

    # Solve tactical model for Broad Asset 3 (BA_3)

    CurrentWeight.assign = z.l["BA_3"]

    ACTIVE[i] = BroadAssetClassThree[i]

    Benchmark[l] = BroadAssetReturns["BA_3", l]

    EpsTolerance.assign = 0.02

    if CurrentWeight.records.value[0] > 0.05:
        TacticalModel.solve()

        print("\n## Model BA_3 ##")
        print("\nx: \n", x.records.level.tolist())

    # Solve integrated model

    CurrentWeight.assign = 1.0

    Benchmark[l] = IndexReturns[l]

    EpsTolerance.assign = 0.02

    ACTIVE[i] = Number(1)

    TacticalModel.solve()
    print("\n## Model Integrated ##")
    print("x: \n", x.records.level.tolist())

    print("\nObjective Function Value: ", round(ObjValue.records.level[0], 3))


if __name__ == "__main__":
    main()
