from __future__ import annotations

import numpy as np
import pytest

import gamspy as gp
from gamspy import Container, ModelStatus
from gamspy.exceptions import ValidationError
from gamspy.formulations.ml import RegressionTree
from gamspy.math import dim

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    tree_dict = {
        "capacity": 5,
        "children_left": np.array([1, 2, -1, -1, -1]),
        "children_right": np.array([4, 3, -1, -1, -1]),
        "feature": np.array([0, 1, -2, -2, -2]),
        "n_features": 2,
        "threshold": np.array([5.5, 4.5, -2.0, -2.0, -2.0]),
        "value": np.array([[15.6], [11.25], [10.0], [15.0], [33.0]]),
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
    yield m, tree_dict, in_data, output, par_input, x
    m.close()


def test_RegressionTree_bad_init(data):
    m, *_ = data

    # wrong container object
    pytest.raises(ValidationError, RegressionTree, "m")

    # wrong regressor type, it must be either None or a sklearn.tree.DecisionTreeRegressor object
    pytest.raises(ValidationError, RegressionTree, m, [])


def test_RegressionTree_incomplete_data(data):
    m, tree_dict, _, _, par_input, _ = data
    rt = RegressionTree(m)

    rt.children_left = tree_dict["children_left"]
    rt.children_right = tree_dict["children_right"]
    rt.capacity = tree_dict["capacity"]
    rt.feature = tree_dict["feature"]
    rt.value = tree_dict["value"]
    rt.threshold = tree_dict["threshold"]

    # n_features is missing: rt.n_features = tree_dict["n_features"]
    pytest.raises(ValidationError, rt, par_input)

    rt.n_features = -2
    # only positive integer allowed
    pytest.raises(ValidationError, rt, par_input)


def test_RegressionTree_bad_call(data):
    m, tree_dict, _, _, _, x = data
    rt = RegressionTree(m)

    for key, value in tree_dict.items():
        setattr(rt, key, value)

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


def test_RegressionTree_valid_variable(data):
    m, tree_dict, _, output, par_input, x = data
    rt = RegressionTree(m)

    for key, value in tree_dict.items():
        setattr(rt, key, value)

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


def test_RegressionTree_valid_parameter(data):
    m, tree_dict, _, output, par_input, _ = data
    rt = RegressionTree(m)

    for key, value in tree_dict.items():
        setattr(rt, key, value)

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


def test_RegressionTree_var_up(data):
    m, tree_dict, in_data, _, par_input, x = data
    rt = RegressionTree(m)

    for key, value in tree_dict.items():
        setattr(rt, key, value)

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


def test_RegressionTree_put_M(data):
    m, tree_dict, _, _, _, x = data
    rt1 = RegressionTree(m, name_prefix="test_big_m")
    rt2 = RegressionTree(m, name_prefix="test_bound_big_m")

    for key, value in tree_dict.items():
        setattr(rt1, key, value)
        setattr(rt2, key, value)

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


def test_RegressionTree_add_equation(data):
    m, tree_dict, _, _, par_input, x = data
    rt = RegressionTree(m)

    for key, value in tree_dict.items():
        setattr(rt, key, value)

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


def test_RegressionTree_multi_output(data):
    m, tree_dict, _, _, par_input, _ = data
    tree_dict["value"] = np.array(
        [[15.6, 14.6], [11.25, 15.5], [10.0, 14.0], [15.0, 20.0], [33.0, 11.0]]
    )
    output = np.array([[10, 14], [10, 14], [10, 14], [15, 20], [33, 11]])
    rt = RegressionTree(m)

    for key, value in tree_dict.items():
        setattr(rt, key, value)

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


def test_RegressionTree_multi_output_equation(data):
    m, tree_dict, _, _, par_input, x = data

    tree_dict["value"] = np.array(
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
    rt = RegressionTree(m)

    for key, value in tree_dict.items():
        setattr(rt, key, value)

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
