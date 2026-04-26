from __future__ import annotations

import numpy as np
import pytest

import gamspy as gp
from gamspy import ModelStatus
from gamspy.exceptions import ValidationError
from gamspy.formulations.nn import (
    AvgPool2d,
    MaxPool2d,
    MinPool2d,
    _MPool2d,
)
from gamspy.formulations.result import FormulationResult
from gamspy.math import dim


@pytest.mark.unit
def test_max_pooling(data):
    m, _w1, _b1, _inp, par_input, _ii, *_ = data

    mp1 = MaxPool2d(m, 2)
    mp2 = MaxPool2d(m, (2, 1))
    mp3 = MaxPool2d(m, 3, stride=(1, 1))
    mp4 = MaxPool2d(m, 4, stride=(3, 2), padding=2)

    out, eqs = mp1(par_input)
    out2, eqs2 = mp2(par_input)
    out3, eqs3 = mp3(par_input)
    out4, eqs4 = mp4(par_input)
    obj = (
        gp.Sum(out.domain, out)
        + gp.Sum(out2.domain, out2)
        + gp.Sum(out3.domain, out3)
        + gp.Sum(out4.domain, out4)
    )
    model = gp.Model(
        m,
        "maxpool",
        equations=[*eqs, *eqs2, *eqs3, *eqs4],
        objective=obj,
        sense="min",
        problem="MIP",
    )
    model.solve()
    expected_out = np.array(
        [
            [[[0.64615000, 0.90273000], [0.66672000, 0.83690000]]],
            [[[0.94010000, 0.80008000], [0.93950000, 0.97884000]]],
            [[[0.92599000, 0.86881000], [0.90938000, 0.51248000]]],
        ]
    )
    assert np.allclose(out.toDense(), expected_out)

    expected_out_2 = np.array(
        [
            [
                [
                    [
                        0.64615000,
                        0.40183000,
                        0.90273000,
                        0.89937000,
                        0.77734000,
                    ],
                    [
                        0.42091000,
                        0.66672000,
                        0.83690000,
                        0.49197000,
                        0.82491000,
                    ],
                ]
            ],
            [
                [
                    [
                        0.52554000,
                        0.94010000,
                        0.75623000,
                        0.80008000,
                        0.55513000,
                    ],
                    [
                        0.67074000,
                        0.93950000,
                        0.95293000,
                        0.97884000,
                        0.60580000,
                    ],
                ]
            ],
            [
                [
                    [
                        0.92599000,
                        0.69301000,
                        0.69718000,
                        0.86881000,
                        0.87645000,
                    ],
                    [
                        0.49372000,
                        0.90938000,
                        0.49188000,
                        0.51248000,
                        0.64488000,
                    ],
                ]
            ],
        ]
    )
    assert np.allclose(out2.toDense(), expected_out_2)

    expected_out_3 = np.array(
        [
            [
                [
                    [0.90273000, 0.90273000, 0.90273000],
                    [0.90273000, 0.90273000, 0.90273000],
                    [0.83690000, 0.83690000, 0.83690000],
                ]
            ],
            [
                [
                    [0.95293000, 0.97884000, 0.97884000],
                    [0.95293000, 0.97884000, 0.97884000],
                    [0.95293000, 0.97884000, 0.97884000],
                ]
            ],
            [
                [
                    [0.92599000, 0.90938000, 0.87645000],
                    [0.92599000, 0.90938000, 0.87645000],
                    [0.90938000, 0.90938000, 0.64488000],
                ]
            ],
        ]
    )
    assert np.allclose(out3.toDense(), expected_out_3)

    expected_out_4 = np.array(
        [
            [
                [
                    [0.64615000, 0.90273000, 0.90273000],
                    [0.66672000, 0.90273000, 0.90273000],
                ]
            ],
            [
                [
                    [0.94010000, 0.94010000, 0.80008000],
                    [0.94010000, 0.97884000, 0.97884000],
                ]
            ],
            [
                [
                    [0.92599000, 0.92599000, 0.87645000],
                    [0.92599000, 0.92599000, 0.87645000],
                ]
            ],
        ]
    )
    assert np.allclose(out4.toDense(), expected_out_4)


@pytest.mark.unit
def test_pooling_with_bounds(data):
    m, _w1, _b1, inp, par_input, _ii, *_ = data
    mp1 = MinPool2d(m, 2)
    mp2 = MaxPool2d(m, 2)
    ap1 = AvgPool2d(m, 2)
    var_input = gp.Variable(m, domain=dim(inp.shape))

    # input bounds should be passed to the output as well
    var_input.lo["0", "0", var_input.domain[2], var_input.domain[3]] = 10
    var_input.up["0", "0", var_input.domain[2], var_input.domain[3]] = 20

    var_input.lo["1", "0", var_input.domain[2], var_input.domain[3]] = 1
    var_input.up["1", "0", var_input.domain[2], var_input.domain[3]] = 100

    var_input.lo["2", "0", var_input.domain[2], var_input.domain[3]] = -50
    var_input.up["2", "0", var_input.domain[2], var_input.domain[3]] = 50

    out, _ = mp1(var_input)
    out2, _ = mp2(var_input)
    out3, _ = ap1(var_input)

    out4, _ = mp1(par_input)
    out5, _ = mp2(par_input)
    out6, _ = ap1(par_input)

    for recs in [out.records, out2.records, out3.records]:
        assert (recs[recs["DenseDim3_1"] == "0"]["lower"] == 10).all()
        assert (recs[recs["DenseDim3_1"] == "0"]["upper"] == 20).all()

        assert (recs[recs["DenseDim3_1"] == "1"]["lower"] == 1).all()
        assert (recs[recs["DenseDim3_1"] == "1"]["upper"] == 100).all()

        assert (recs[recs["DenseDim3_1"] == "2"]["lower"] == -50).all()
        assert (recs[recs["DenseDim3_1"] == "2"]["upper"] == 50).all()

    exp_lb = np.array(
        [
            [[[0.27341, 0.29883], [0.40205, 0.23754]]],
            [[[0.48203, 0.364], [0.37691, 0.19674]]],
            [[[0.54127, 0.03598], [0.19366, 0.043283]]],
        ]
    )

    exp_ub = np.array(
        [
            [[[0.64615, 0.90273], [0.66672, 0.8369]]],
            [[[0.9401, 0.80008], [0.9395, 0.97884]]],
            [[[0.92599, 0.86881], [0.90938, 0.51248]]],
        ]
    )

    for recs in [out4.records, out5.records, out6.records]:
        assert np.allclose(np.array(recs.upper).reshape(out4.shape), exp_ub)
        assert np.allclose(np.array(recs.lower).reshape(out4.shape), exp_lb)


@pytest.mark.unit
def test_pooling_return_formulation_result(data):
    m, _, _, inp, _, _, *_ = data
    mp1 = MinPool2d(m, 2)
    mp2 = MaxPool2d(m, 2)
    ap1 = AvgPool2d(m, 2)
    var_input = gp.Variable(m, domain=dim(inp.shape))

    # input bounds should be passed to the output as well
    var_input.lo["0", "0", var_input.domain[2], var_input.domain[3]] = 10
    var_input.up["0", "0", var_input.domain[2], var_input.domain[3]] = 20

    var_input.lo["1", "0", var_input.domain[2], var_input.domain[3]] = 1
    var_input.up["1", "0", var_input.domain[2], var_input.domain[3]] = 100

    var_input.lo["2", "0", var_input.domain[2], var_input.domain[3]] = -50
    var_input.up["2", "0", var_input.domain[2], var_input.domain[3]] = 50

    result1 = mp1(var_input)
    result2 = mp2(var_input)
    result3 = ap1(var_input)

    for i, res in enumerate([result1, result2, result3]):
        assert isinstance(res, FormulationResult), "Expected a FormulationResult object"
        assert isinstance(res.result, gp.Variable), "Expected the output variable"
        assert isinstance(res.variables_created["output"], gp.Variable), (
            "Expected the output variable"
        )
        assert all(
            isinstance(v, gp.Equation) for v in res.equations_created.values()
        ), "Expected a list of Equations"
        assert all(isinstance(v, gp.Set) for v in res.sets_created.values()), (
            "Expected a list of Sets"
        )
        assert "in_out_matching_1" in res.sets_created, "Expected this set"

        if i != 2:  # skip for avgpool
            assert "bigM" in res.parameters_created, "Expected this parameter"
            assert "aux_variable" in res.variables_created, "Expected this variable"

        out = res.result.records

        assert (out[out["DenseDim3_1"] == "0"]["lower"] == 10).all()
        assert (out[out["DenseDim3_1"] == "0"]["upper"] == 20).all()

        assert (out[out["DenseDim3_1"] == "1"]["lower"] == 1).all()
        assert (out[out["DenseDim3_1"] == "1"]["upper"] == 100).all()

        assert (out[out["DenseDim3_1"] == "2"]["lower"] == -50).all()
        assert (out[out["DenseDim3_1"] == "2"]["upper"] == 50).all()


@pytest.mark.unit
def test_mpooling_with_complex_bounds(data):
    m, *_ = data

    max_pool = MaxPool2d(m, (2, 3), name_prefix="maxpool1")
    min_pool = MinPool2d(m, (2, 3), name_prefix="minpool1")

    data = np.array(
        [
            [
                [
                    [2, -5, -40, -3, -np.inf, 3],
                    [100, 0, 2, 2, 10, -5],
                    [np.inf, -5, 0, 0, 0, 0],
                    [-2, 4, -2, 0, 0, 0],
                ],
                [
                    [4, 1, 1, -2, 3, 4],
                    [4, 1, 3, 3, -1, -5],
                    [3, -3, -2, -5, -3, -6],
                    [-2, 4, 4, -5, 0, -5],
                ],
            ],
            [
                [
                    [-4, 1, -1, -3, 3, 0],
                    [-12, -3, -5, -4, -3, 4],
                    [0, -4, 1, 3, -3, 2],
                    [-1, -5, -2, -3, 3, 2],
                ],
                [
                    [-4, -5, 4, 2, -2, 2],
                    [4, -5, -3, -1, 3, 0],
                    [4, 2, -3, -1, -1, 0],
                    [40, 2, 1, 5, 2, 0],
                ],
            ],
        ]
    )

    ub_data = np.where(data < 0, 0, data)
    lb_data = np.where(data > 0, 0, data)

    lb = gp.Parameter(m, domain=dim((2, 2, 4, 6)), records=lb_data)
    ub = gp.Parameter(m, domain=dim((2, 2, 4, 6)), records=ub_data)

    par = gp.Parameter(m, domain=dim((2, 2, 4, 6)), records=data)
    var = gp.Variable(m, domain=dim((2, 2, 4, 6)))

    var.lo[...] = lb[...]
    var.up[...] = ub[...]

    out1, _ = max_pool(par, propagate_bounds=False)
    out2, _ = min_pool(par, propagate_bounds=False)
    out3, _ = max_pool(par)
    out4, _ = min_pool(par)

    for name in ["minpool1", "maxpool1"]:
        output_var_found = False
        matching_set_found = False
        for sym_name in m.data:
            if sym_name.startswith(f"v_{name}_output"):
                output_var_found = True
            elif sym_name.startswith(f"s_{name}_in_out_matching"):
                matching_set_found = True

        assert output_var_found, f"{name} output var not found"
        assert matching_set_found, f"{name} match set not found"

    out5, _ = max_pool(var, propagate_bounds=False)
    out6, _ = min_pool(var, propagate_bounds=False)
    out7, _ = max_pool(var)
    out8, _ = min_pool(var)

    exp_ub_par = np.array(
        [
            [[[100, 10], [np.inf, 0]], [[4.0, 4.0], [4.0, 0.0]]],
            [[[1.0, 4.0], [1.0, 3.0]], [[4.0, 3.0], [40.0, 5.0]]],
        ]
    )

    exp_lb_par = np.array(
        [
            [[[-40.0, -np.inf], [-5.0, 0.0]], [[1.0, -5.0], [-3.0, -6.0]]],
            [[[-12.0, -4.0], [-5.0, -3.0]], [[-5.0, -2.0], [-3.0, -1.0]]],
        ]
    )

    exp_ub_var = np.where(exp_ub_par < 0, 0, exp_ub_par)
    exp_lb_var = np.where(exp_lb_par > 0, 0, exp_lb_par)

    assert out1.records is None
    assert out2.records is None
    assert out5.records is None
    assert out6.records is None

    assert np.allclose(np.array(out3.records.lower).reshape(out3.shape), exp_lb_par)
    assert np.allclose(np.array(out3.records.upper).reshape(out3.shape), exp_ub_par)

    assert np.allclose(np.array(out4.records.lower).reshape(out4.shape), exp_lb_par)
    assert np.allclose(np.array(out4.records.upper).reshape(out4.shape), exp_ub_par)

    assert np.allclose(np.array(out7.records.lower).reshape(out7.shape), exp_lb_var)
    assert np.allclose(np.array(out7.records.upper).reshape(out7.shape), exp_ub_var)

    assert np.allclose(np.array(out8.records.lower).reshape(out8.shape), exp_lb_var)
    assert np.allclose(np.array(out8.records.upper).reshape(out8.shape), exp_ub_var)


@pytest.mark.unit
def test_min_pooling(data):
    m, _w1, _b1, _inp, par_input, _ii, *_ = data
    mp1 = MinPool2d(m, 2)
    mp2 = MinPool2d(m, (2, 1))
    mp3 = MinPool2d(m, 3, stride=(1, 1))
    mp4 = MinPool2d(m, 4, stride=(3, 2), padding=2)
    out, eqs = mp1(par_input)
    out2, eqs2 = mp2(par_input)
    out3, eqs3 = mp3(par_input)
    out4, eqs4 = mp4(par_input)
    obj = (
        gp.Sum(out.domain, out)
        + gp.Sum(out2.domain, out2)
        + gp.Sum(out3.domain, out3)
        + gp.Sum(out4.domain, out4)
    )
    model = gp.Model(
        m,
        "minpool",
        equations=[*eqs, *eqs2, *eqs3, *eqs4],
        objective=obj,
        sense="min",
        problem="MIP",
    )
    model.solve()
    expected_out = np.array(
        [
            [[[0.27341000, 0.29883000], [0.40205000, 0.23754000]]],
            [[[0.48203000, 0.36400000], [0.37691000, 0.19674000]]],
            [[[0.54127000, 0.03598000], [0.19366000, 0.04328300]]],
        ]
    )
    assert np.allclose(out.toDense(), expected_out)

    expected_out_2 = np.array(
        [
            [
                [
                    [
                        0.54191000,
                        0.27341000,
                        0.78980000,
                        0.29883000,
                        0.17423000,
                    ],
                    [
                        0.40205000,
                        0.55509000,
                        0.67382000,
                        0.23754000,
                        0.64736000,
                    ],
                ]
            ],
            [
                [
                    [
                        0.48203000,
                        0.92097000,
                        0.71608000,
                        0.36400000,
                        0.02026600,
                    ],
                    [
                        0.37691000,
                        0.44264000,
                        0.50225000,
                        0.19674000,
                        0.37108000,
                    ],
                ]
            ],
            [
                [
                    [
                        0.56254000,
                        0.54127000,
                        0.61562000,
                        0.03598000,
                        0.48145000,
                    ],
                    [
                        0.38243000,
                        0.19366000,
                        0.18858000,
                        0.04328300,
                        0.26750000,
                    ],
                ]
            ],
        ]
    )
    assert np.allclose(out2.toDense(), expected_out_2)

    expected_out_3 = np.array(
        [
            [
                [
                    [0.27341000, 0.27341000, 0.17423000],
                    [0.27341000, 0.23754000, 0.23754000],
                    [0.33891000, 0.23754000, 0.23754000],
                ]
            ],
            [
                [
                    [0.37691000, 0.36400000, 0.02026600],
                    [0.37691000, 0.19674000, 0.02026600],
                    [0.16820000, 0.06974300, 0.00961020],
                ]
            ],
            [
                [
                    [0.38243000, 0.03598000, 0.03598000],
                    [0.18858000, 0.04328300, 0.04328300],
                    [0.18858000, 0.04328300, 0.04328300],
                ]
            ],
        ]
    )
    assert np.allclose(out3.toDense(), expected_out_3)

    expected_out_4 = np.array(
        [
            [
                [
                    [0.27341000, 0.27341000, 0.17423000],
                    [0.27341000, 0.23754000, 0.23754000],
                ]
            ],
            [
                [
                    [0.48203000, 0.36400000, 0.02026600],
                    [0.16820000, 0.06974300, 0.00961020],
                ]
            ],
            [
                [
                    [0.54127000, 0.03598000, 0.03598000],
                    [0.19366000, 0.04328300, 0.04328300],
                ]
            ],
        ]
    )
    assert np.allclose(out4.toDense(), expected_out_4)


@pytest.mark.unit
def test_avg_pooling(data):
    m, _w1, _b1, _inp, par_input, _ii, *_ = data
    ap1 = AvgPool2d(m, 2, name_prefix="avgpool1")
    ap2 = AvgPool2d(m, (2, 1))
    ap3 = AvgPool2d(m, 3, stride=(1, 1))
    ap4 = AvgPool2d(m, 4, stride=(3, 2), padding=2)
    out, eqs = ap1(par_input)
    out2, eqs2 = ap2(par_input)
    out3, eqs3 = ap3(par_input)
    out4, eqs4 = ap4(par_input)
    out5, _ = ap4(par_input, propagate_bounds=False)

    output_var_found = False
    matching_set_found = False
    for sym_name in m.data:
        if sym_name.startswith("v_avgpool1_output"):
            output_var_found = True
        elif sym_name.startswith("s_avgpool1_in_out_matching_1"):
            matching_set_found = True

    assert output_var_found
    assert matching_set_found

    # test that records are not created when propagate_bounds is False
    assert out5.records is None

    obj = (
        gp.Sum(out.domain, out)
        + gp.Sum(out2.domain, out2)
        + gp.Sum(out3.domain, out3)
        + gp.Sum(out4.domain, out4)
    )
    model = gp.Model(
        m,
        "avgpool",
        equations=[*eqs, *eqs2, *eqs3, *eqs4],
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = np.array(
        [
            [[[0.46582499, 0.72268248], [0.51119250, 0.56005752]]],
            [[[0.71716005, 0.65909749], [0.60744750, 0.65768999]]],
            [[[0.68070245, 0.55439746], [0.49479753, 0.30905575]]],
        ]
    )

    expected_out_2 = np.array(
        [
            [
                [
                    [
                        0.59403002,
                        0.33761999,
                        0.84626496,
                        0.59909999,
                        0.47578499,
                    ],
                    [
                        0.41148001,
                        0.61090499,
                        0.75536001,
                        0.36475500,
                        0.73613501,
                    ],
                ]
            ],
            [
                [
                    [
                        0.50378501,
                        0.93053502,
                        0.73615503,
                        0.58204001,
                        0.28769800,
                    ],
                    [
                        0.52382499,
                        0.69106996,
                        0.72758996,
                        0.58779001,
                        0.48843998,
                    ],
                ]
            ],
            [
                [
                    [
                        0.74426496,
                        0.61714000,
                        0.65639997,
                        0.45239499,
                        0.67895001,
                    ],
                    [
                        0.43807501,
                        0.55151999,
                        0.34022999,
                        0.27788150,
                        0.45618999,
                    ],
                ]
            ],
        ]
    )

    expected_out_3 = np.array(
        [
            [
                [
                    [0.57630998, 0.58742785, 0.62838334],
                    [0.58594882, 0.54855663, 0.63237786],
                    [0.55955440, 0.58058667, 0.65012449],
                ]
            ],
            [
                [
                    [0.73447669, 0.81874776, 0.63881731],
                    [0.66924220, 0.71879554, 0.57156295],
                    [0.62388670, 0.62716144, 0.50318259],
                ]
            ],
            [
                [
                    [0.64658892, 0.59617889, 0.53859448],
                    [0.54380774, 0.50185585, 0.50105363],
                    [0.41795224, 0.40988812, 0.32560670],
                ]
            ],
        ]
    )

    expected_out_4 = np.array(
        [
            [
                [
                    [0.11645625, 0.29712689, 0.24014375],
                    [0.22423249, 0.52836502, 0.48937631],
                ]
            ],
            [
                [
                    [0.17929001, 0.34406438, 0.20073663],
                    [0.29626748, 0.61241204, 0.37906685],
                ]
            ],
            [
                [
                    [0.17017561, 0.30877501, 0.22346812],
                    [0.28089315, 0.49685395, 0.33070874],
                ]
            ],
        ]
    )

    assert np.allclose(out.toDense(), expected_out)
    assert np.allclose(out2.toDense(), expected_out_2)
    assert np.allclose(out3.toDense(), expected_out_3)
    assert np.allclose(out4.toDense(), expected_out_4)


@pytest.mark.unit
def test_avg_pool_bounds_neg(data):
    m, _w1, _b1, _, _par_input, _ii, *_ = data
    inp = np.array(
        [
            [
                [
                    [-0.64615, 0.40183, 0.7898, 0.89937, 0.17423],
                    [0.54191, 0.27341, 0.90273, 0.29883, 0.77734],
                    [0.40205, 0.55509, 0.67382, -0.49197, 0.64736],
                    [0.42091, 0.66672, 0.8369, 0.23754, 0.82491],
                    [0.38872, -0.33891, 0.75287, 0.67146, 0.71429],
                ]
            ],
            [
                [
                    [0.64615, 0.40183, 0.7898, 0.89937, 0.17423],
                    [0.54191, 0.27341, 0.91273, 0.29883, 0.77734],
                    [0.40205, 0.55509, 0.67382, 0.49197, 0.64736],
                    [0.42091, 0.66672, 0.8369, 0.23754, 0.82491],
                    [0.38872, 0.33891, 0.75287, 0.67146, 0.71429],
                ]
            ],
            [
                [
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                ]
            ],
        ]
    )
    new_par = gp.Parameter(m, domain=dim(inp.shape), records=inp)
    ap1 = AvgPool2d(m, 4, padding=2)

    out, eqs = ap1(new_par)

    recs = out.records

    # Maximum (-1) is scaled by scaling factor (1/4)
    exp_ub = np.array(
        [
            [[[0.54191, 0.90273], [0.66672, 0.8369]]],
            [[[0.64615, 0.91273], [0.66672, 0.8369]]],
            [[[-1 / 4, -1 / 4], [-1 / 4, -1 / 4]]],
        ]
    )

    # Positive Minimum values are scaled by scaling factor (1/4)
    exp_lb = np.array(
        [
            [[[-0.64615, 0.17423 / 4], [-0.33891, -0.49197]]],
            [[[0.27341 / 4, 0.17423 / 4], [0.33891 / 4, 0.23754 / 4]]],
            [[[-1.0, -1.0], [-1.0, -1.0]]],
        ]
    )

    assert np.allclose(np.array(recs.upper).reshape(out.shape), exp_ub)
    assert np.allclose(np.array(recs.lower).reshape(out.shape), exp_lb)

    model = gp.Model(
        m,
        "avgpool_edge",
        equations=[*eqs],
        problem="mip",
        objective=out["0", "0", "0", "0"] + 1,
        sense="min",
    )

    model.solve()
    # bounds shouldn't make it infeasible
    assert model.status == ModelStatus.OptimalGlobal


@pytest.mark.unit
def test_pool_call_bad(data):
    m, _w1, _b1, _inp, _par_input, _ii, *_ = data
    avgpool1 = AvgPool2d(m, (2, 2))
    minpool1 = MinPool2d(m, (2, 2))
    maxpool1 = MaxPool2d(m, (2, 2))

    new_par = gp.Parameter(m, "new_par", domain=dim([10]))
    new_var = gp.Variable(m, "new_var", domain=dim([10]))

    par2 = gp.Parameter(m, "par2", domain=dim([2, 2, 4, 10]))
    var2 = gp.Variable(m, "var2", domain=dim([2, 2, 4, 10]))

    for pool in [avgpool1, minpool1, maxpool1]:
        with pytest.raises(ValidationError):
            pool("asd")
        with pytest.raises(ValidationError):
            pool(5)
        with pytest.raises(ValidationError):
            pool(new_par)
        with pytest.raises(ValidationError):
            pool(new_var)
        with pytest.raises(ValidationError):
            pool(par2, propagate_bounds="True")
        with pytest.raises(ValidationError):
            pool(var2, propagate_bounds="True")

    with pytest.raises(ValidationError):
        _MPool2d("sup", m, (2, 2))


@pytest.mark.unit
def test_pool_check_str_method(data):
    m, *_ = data
    avgpool1 = AvgPool2d(m, (2, 2))
    minpool1 = MinPool2d(m, (2, 2))
    maxpool1 = MaxPool2d(m, (2, 2))

    expected = "AvgPool2d(\n  kernel_size=(2, 2)\n  stride=(2, 2)\n  padding=(0, 0)\n)"
    assert str(avgpool1) == expected

    expected = "MinPool2d(\n  kernel_size=(2, 2)\n  stride=None\n  padding=0\n  name_prefix=None\n)"
    assert str(minpool1) == expected

    expected = "MaxPool2d(\n  kernel_size=(2, 2)\n  stride=None\n  padding=0\n  name_prefix=None\n)"
    assert str(maxpool1) == expected
