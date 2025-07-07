from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

import gamspy as gp
from gamspy import Container, ModelStatus
from gamspy.exceptions import ValidationError
from gamspy.formulations.ml import DecisionTreeStruct, RegressionTree
from gamspy.math import dim

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    tree_args = {
        "children_left": np.array([1, 2, -1, -1, -1]),
        "children_right": np.array([4, 3, -1, -1, -1]),
        "feature": np.array([0, 1, -2, -2, -2]),
        "threshold": np.array([5.5, 4.5, -2.0, -2.0, -2.0]),
        "value": np.array([[15.6], [11.25], [10.0], [15.0], [33.0]]),
        "capacity": 5,
        "n_features": 2,
    }
    in_data = np.array(
        [
            [2, 3],
            [3, 1],
            [1, 2],
            [5, 6],
            [6, 4],
        ]
    )

    output = np.array([10, 10, 10, 15, 33])
    par_input = gp.Parameter(m, domain=dim(in_data.shape), records=in_data)
    x = gp.Variable(m, "x", domain=dim(in_data.shape), type="positive")
    yield m, tree_args, in_data, output, par_input, x
    m.close()


def test_regression_tree_bad_init(data):
    m, tree_args, *_ = data

    tree = DecisionTreeStruct(**tree_args)

    # No Container
    pytest.raises(TypeError, RegressionTree, tree)

    # No regressor
    pytest.raises(TypeError, RegressionTree, m)

    # wrong container object
    pytest.raises(ValidationError, RegressionTree, "m", tree)

    # wrong regressor type, it must be either sklearn.tree.DecisionTreeRegressor or DecisionTreeStruct object
    pytest.raises(ValidationError, RegressionTree, m, tree_args)

    # initializing the formulation with untrained sklearn.tree
    tree = DecisionTreeRegressor(random_state=42)
    pytest.raises(ValidationError, RegressionTree, m, tree)


def test_regression_tree_incomplete_data(data):
    m, tree_args, *_ = data

    # tree instance with missing attribute, children_left
    rm_tree_args = tree_args.copy()
    rm_tree_args.pop("children_left")
    broken_tree = DecisionTreeStruct(**rm_tree_args)
    pytest.raises(ValidationError, RegressionTree, m, broken_tree)

    # tree instance with missing attribute, children_right
    rm_tree_args = tree_args.copy()
    rm_tree_args.pop("children_right")
    broken_tree = DecisionTreeStruct(**rm_tree_args)
    pytest.raises(ValidationError, RegressionTree, m, broken_tree)

    # tree instance with missing attribute, features
    rm_tree_args = tree_args.copy()
    rm_tree_args.pop("feature")
    broken_tree = DecisionTreeStruct(**rm_tree_args)
    pytest.raises(ValidationError, RegressionTree, m, broken_tree)

    # tree instance with missing attribute, threshold
    rm_tree_args = tree_args.copy()
    rm_tree_args.pop("threshold")
    broken_tree = DecisionTreeStruct(**rm_tree_args)
    pytest.raises(ValidationError, RegressionTree, m, broken_tree)

    # tree instance with missing attribute, value
    rm_tree_args = tree_args.copy()
    rm_tree_args.pop("value")
    broken_tree = DecisionTreeStruct(**rm_tree_args)
    pytest.raises(ValidationError, RegressionTree, m, broken_tree)

    # cannot assign values to attributes to once initialized dataclass
    broken_tree = DecisionTreeStruct(**tree_args)
    with pytest.raises(FrozenInstanceError):
        broken_tree.n_features = -2

    # wrong value for attribute, capacity
    broken_tree = DecisionTreeStruct(**tree_args)
    with pytest.raises(FrozenInstanceError):
        broken_tree.capacity = -2


def test_regression_tree_bad_call(data):
    m, tree_args, _, _, _, x = data

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    # missing required input
    pytest.raises(TypeError, rt)

    # no input dimension
    in_data_1 = gp.Variable(m)
    pytest.raises(ValidationError, rt, in_data_1)

    # wrong input dimension, model has 2 features not 1
    in_data_2 = gp.Variable(m, domain=dim((5, 1)))
    pytest.raises(ValidationError, rt, in_data_2)

    # wrong type of input
    in_data_3 = gp.Set(m, domain=dim((5, 2)))
    pytest.raises(ValidationError, rt, in_data_3)

    # wrong value type for M either float or int
    pytest.raises(ValidationError, rt, x, "M")


def test_regression_tree_with_trained_sklearn_tree(data):
    m, _, in_data, output, par_input, _ = data

    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X=in_data, y=output)
    rt = RegressionTree(m, tree)

    out, eqns = rt(par_input)

    model = gp.Model(
        m,
        "regressionTree",
        equations=eqns,
        problem="MIP",
    )
    model.solve()

    assert np.allclose(out.toDense().flatten(), output)
    assert model.status == ModelStatus(1)


def test_regression_tree_valid_variable(data):
    m, tree_args, _, output, par_input, x = data

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    x.fx[:, 0] = par_input[:, 0]
    x.fx[:, 1] = par_input[:, 1]

    # check the prediction of the decions tree on the trained data
    out, eqns = rt(x)

    model = gp.Model(
        m,
        "regressionTree",
        equations=eqns,
        problem="MIP",
    )
    model.solve()

    assert np.allclose(out.toDense().flatten(), output)
    assert model.status == ModelStatus(1)


def test_regression_tree_valid_parameter(data):
    m, tree_args, _, output, par_input, _ = data

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    # check the prediction of the decions tree on the trained data using parameter
    out, eqns = rt(par_input)

    model = gp.Model(
        m,
        "regressionTree",
        equations=eqns,
        problem="MIP",
    )
    model.solve()

    assert np.allclose(out.toDense().flatten(), output)
    assert model.status == ModelStatus(1)


def test_regression_tree_var_up(data):
    m, tree_args, in_data, _, par_input, x = data

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    x.fx[:, 0] = par_input[:, 0]
    x.up[:, 1] = int(max(in_data[:, 1]))

    out, eqns = rt(x)
    s = out.domain

    model_max = gp.Model(
        m,
        "regTree_max",
        equations=eqns,
        sense="max",
        problem="mip",
        objective=gp.Sum(s, out[s]),
    )

    model_min = gp.Model(
        m,
        "regTree_min",
        equations=eqns,
        sense="min",
        problem="mip",
        objective=gp.Sum(s, out[s]),
    )

    model_max.solve()
    max_out = np.array([15, 15, 15, 15, 33])
    assert np.allclose(out.toDense().flatten(), max_out)
    assert model_max.objective_value == 93
    assert model_max.status == ModelStatus(1)

    model_min.solve()
    min_out = np.array([10, 10, 10, 10, 33])
    assert np.allclose(out.toDense().flatten(), min_out)
    assert model_min.objective_value == 73
    assert model_min.status == ModelStatus(1)


def test_regression_tree_put_M(data):
    m, tree_args, _, _, _, x = data

    tree = DecisionTreeStruct(**tree_args)
    rt1 = RegressionTree(m, tree, name_prefix="test_big_m")
    rt2 = RegressionTree(m, tree, name_prefix="test_bound_big_m")

    x.up[:, 0] = 7
    x.up[:, 1] = 7

    _, eqns_m = rt1(x, M=1e3)
    _, eqns = rt2(x)

    m1 = gp.Model(
        m,
        "regTree_bigm",
        equations=eqns_m,
        problem="mip",
    )

    m1.solve()

    m2 = gp.Model(
        m,
        "regTree_bound_bigM",
        equations=eqns,
        problem="mip",
    )
    m2.solve()

    le_cons = [
        ele
        for ele in eqns_m
        if ele.name.startswith("e_test_big_m_link_indctr_feature_le_")
    ][0]
    df = le_cons.records
    res_bigm = df[df["level"] == 1e3][
        [dom.name for dom in le_cons.domain]
    ].to_numpy(dtype=int)

    le_cons = [
        ele
        for ele in eqns
        if ele.name.startswith("e_test_bound_big_m_link_indctr_feature_le_")
    ][0]
    df = le_cons.records
    res_m = df[df["level"] > 0][[dom.name for dom in le_cons.domain]].to_numpy(
        dtype=int
    )
    active = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [2, 0, 0],
            [2, 1, 0],
            [3, 0, 0],
            [3, 1, 0],
            [4, 0, 0],
            [4, 1, 0],
        ]
    )

    assert np.allclose(res_bigm, active)
    assert m1.status == ModelStatus(1)

    assert np.allclose(res_m, active)
    assert m2.status == ModelStatus(1)


def test_regression_tree_add_equation(data):
    m, tree_args, _, _, par_input, x = data

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    x.fx[:, 0] = par_input[:, 0]
    x.up[:, 1] = 7

    out, eqns = rt(x)

    s = out.domain

    # force second feature to be more than 26
    e1 = gp.Equation(m)
    e1[...] = gp.Sum(s, x[..., 1]) >= 26

    m1 = gp.Model(
        m,
        "regTree_add_equation",
        equations=eqns,
        sense="min",
        problem="mip",
        objective=gp.Sum(s, out[s]),
    )

    m2 = gp.Model(
        m,
        "regTree_add_equation_cons",
        equations=eqns + [e1],
        sense="min",
        problem="mip",
        objective=gp.Sum(s, out[s]),
    )

    m1.solve()
    o1 = out.toDense().flatten()

    m2.solve()
    o2 = out.toDense().flatten()

    output = np.array([10, 10, 10, 10, 33])

    assert m1.objective_value == 73
    assert np.allclose(o1, output)
    assert m1.status == ModelStatus(1)
    output[0] = 15
    assert m2.objective_value == 78
    assert np.allclose(o2, output)
    assert m2.status == ModelStatus(1)


def test_regression_tree_multi_output(data):
    m, tree_args, _, _, par_input, _ = data

    tree_args["value"] = np.array(
        [[15.6, 14.6], [11.25, 15.5], [10.0, 14.0], [15.0, 20.0], [33.0, 11.0]]
    )
    output = np.array([[10, 14], [10, 14], [10, 14], [15, 20], [33, 11]])

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    out, eqns = rt(par_input)
    s = out.domain

    m1 = gp.Model(
        m,
        "regTree_add_equation",
        equations=eqns,
        sense="min",
        problem="mip",
        objective=gp.Sum(s, out[s]),
    )

    m1.solve()
    o1 = out.toDense()

    assert len(s[1]) == output.shape[-1]
    assert m1.objective_value == 151
    assert np.allclose(o1, output)
    assert m1.status == ModelStatus(1)


def test_regression_tree_multi_output_equation(data):
    m, tree_args, _, _, par_input, x = data

    tree_args["value"] = np.array(
        [[15.6, 14.6], [11.25, 15.5], [10.0, 14.0], [15.0, 20.0], [33.0, 11.0]]
    )
    output1 = np.array(
        [
            [15.0, 20.0],
            [15.0, 20.0],
            [15.0, 20.0],
            [15.0, 20.0],
            [33.0, 11.0],
        ]
    )
    output2 = np.array(
        [
            [10.0, 14.0],
            [10.0, 14.0],
            [15.0, 20.0],
            [15.0, 20.0],
            [33.0, 11.0],
        ]
    )

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    x.fx[:, 0] = par_input[:, 0]
    x.up[:, 1] = 7

    out, eqns = rt(x)
    s = out.domain

    # force second feature to be less than 26
    e1 = gp.Equation(m)
    e1[...] = gp.Sum(s, x[..., 1]) <= 26

    m1 = gp.Model(
        m,
        "regTree_add_equation",
        equations=eqns,
        sense="max",
        problem="mip",
        objective=gp.Sum(s, out[s]),
    )

    m2 = gp.Model(
        m,
        "regTree_add_equation_cons",
        equations=eqns + [e1],
        sense="max",
        problem="mip",
        objective=gp.Sum(s, out[s]),
    )

    m1.solve()
    o1 = out.toDense()

    m2.solve()
    o2 = out.toDense()

    assert m1.objective_value == 184
    assert np.allclose(o1, output1)
    assert m1.status == ModelStatus(1)
    assert m2.objective_value == 162
    assert np.allclose(o2, output2)
    assert m2.status == ModelStatus(1)


def test_regression_tree_string_features(data):
    m, tree_args, in_data, output, _, _ = data

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    samples = gp.Set(
        m, "set_of_samples", domain=["*"], records=[f"s{i}" for i in range(5)]
    )
    features = gp.Set(
        m, "set_of_features", domain=["*"], records=[f"f{i}" for i in range(2)]
    )

    rt_input = gp.Parameter(
        m,
        "new_input",
        domain=[samples, features],
        records=[(f"s{i}", "f0", ele) for i, ele in enumerate(in_data[:, 0])]
        + [(f"s{i}", "f1", ele) for i, ele in enumerate(in_data[:, 1])],
    )

    out, eqns = rt(rt_input)

    model = gp.Model(
        m,
        "regressionTree_string_feat",
        equations=eqns,
        problem="MIP",
    )
    model.solve()

    assert np.allclose(out.toDense().flatten(), output)
    assert model.status == ModelStatus(1)


def test_regression_tree_no_upper_bound(data):
    m, tree_args, in_data, _, _, _ = data

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    x = gp.Variable(m, name="x_free", domain=dim(in_data.shape))
    x.lo[...] = 5
    # x.up is inf

    # check the prediction of the decions tree on the trained data
    out, eqns = rt(x)

    model = gp.Model(
        m,
        "regressionTree_up_bnd",
        equations=eqns,
        problem="MIP",
    )
    model.solve()

    expected_out = [15, 15, 15, 15, 15]

    assert np.allclose(out.toDense().flatten(), expected_out)
    assert model.status == ModelStatus(1)


def test_regression_tree_no_lower_bound(data):
    m, tree_args, in_data, _, _, _ = data

    tree = DecisionTreeStruct(**tree_args)
    rt = RegressionTree(m, tree)

    x = gp.Variable(m, name="x_free", domain=dim(in_data.shape))
    # x.lo is -inf
    x.up[...] = 7

    # check the prediction of the decions tree on the trained data
    out, eqns = rt(x)

    model = gp.Model(
        m,
        "regressionTree_lo_bnd",
        equations=eqns,
        problem="MIP",
    )
    model.solve()

    expected_out = [10, 10, 10, 10, 10]

    assert np.allclose(out.toDense().flatten(), expected_out)
    assert model.status == ModelStatus(1)
