from __future__ import annotations

import numpy as np
import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations.nn import (
    Conv1d,
    Conv2d,
)
from gamspy.formulations.result import FormulationResult
from gamspy.math import dim


@pytest.mark.unit
def test_conv_bad_init(data):
    m, *_ = data
    for layer in [Conv1d, Conv2d]:
        # in channel must be integer
        with pytest.raises(ValidationError):
            layer(m, 2.5, 4, 3)
        with pytest.raises(ValidationError):
            layer(m, "2", 4, 3)
        # out channel must be integer
        with pytest.raises(ValidationError):
            layer(m, 2, 4.1, 3)
        with pytest.raises(ValidationError):
            layer(m, "2", 4.1, 3)

        # in channel must be positive
        with pytest.raises(ValidationError):
            layer(m, -4, 4, 3)

        # out channel must be positive
        with pytest.raises(ValidationError):
            layer(m, 4, -4, 3)

        # kernel_size must be integer
        with pytest.raises(ValidationError):
            layer(m, 4, 4, 3.5)

        # stride must be integer
        with pytest.raises(ValidationError):
            layer(m, 4, 4, 3, stride=0.4)

        # padding when string must be valid or same
        with pytest.raises(ValidationError):
            layer(m, 1, 2, 3, 1, "asd")

        # same padding requires stride = 1
        with pytest.raises(ValidationError):
            layer(m, 1, 2, 3, 2, "same")

        # bias must be a bool
        with pytest.raises(ValidationError):
            layer(m, 4, 4, 3, bias=10)

    # kernel size must be integer or tuple of integer
    bad_values = [(3, "a"), ("a", 3), 2.4, -1, 0]
    for bad_value in bad_values:
        with pytest.raises(ValidationError):
            Conv2d(m, 4, 4, bad_value)
    # stride size must be integer or tuple of integer
    for bad_value in bad_values:
        with pytest.raises(ValidationError):
            Conv2d(m, 4, 4, 3, bad_value)
    # padding size must be integer or tuple of integer
    for bad_value in bad_values[:-1]:
        with pytest.raises(ValidationError):
            Conv2d(m, 4, 4, 3, 1, bad_value)

    with pytest.raises(ValidationError):
        Conv2d(m, 1, 3, 2, padding=(1, 2, 3))


@pytest.mark.unit
def test_conv_load_weights(data):
    m, *_ = data

    w1 = np.random.rand(2, 1, 3, 3)
    b1 = np.random.rand(2)

    w2 = np.random.rand(2, 1, 3)
    b2 = np.random.rand(2)

    for layer, (w, b) in zip([Conv1d, Conv2d], [(w2, b2), (w1, b1)], strict=False):
        conv1 = layer(m, in_channels=1, out_channels=2, kernel_size=3, bias=True)
        conv2 = layer(m, in_channels=1, out_channels=2, kernel_size=3, bias=False)
        # needs bias as well
        with pytest.raises(ValidationError):
            conv1.load_weights(w)

        # conv2 does not have bias
        with pytest.raises(ValidationError):
            conv2.load_weights(w, b)

        # test bad shape
        bad1 = np.random.rand(1)
        with pytest.raises(ValidationError):
            conv1.load_weights(bad1, b)
        with pytest.raises(ValidationError):
            conv1.load_weights(w, bad1)

        bad2 = np.random.rand(2, 2, 2, 2)
        with pytest.raises(ValidationError):
            conv1.load_weights(bad2, b)

        bad3 = np.random.rand(6)
        with pytest.raises(ValidationError):
            conv1.load_weights(w, bad3)

        bad4 = np.random.rand(6, 2)
        with pytest.raises(ValidationError):
            conv1.load_weights(w, bad4)

        bad5 = np.random.rand(2, 2, 2)
        with pytest.raises(ValidationError):
            conv1.load_weights(bad5, b)


@pytest.mark.unit
def test_conv_same_indices(data):
    m, *_ = data
    for layer, l in zip([Conv1d, Conv2d], [3, 4], strict=False):
        w1 = np.random.rand(*[4] * l)
        b1 = np.random.rand(4)
        conv1 = layer(m, 4, 4, 4, bias=True, name_prefix="conv1")
        conv1.load_weights(w1, b1)

        inp = gp.Variable(m, domain=dim([4] * l))
        _out, _eqs = conv1(inp)

        output_var_found = False
        weight_par_found = False
        bias_par_found = False
        for sym_name in m.data:
            if sym_name.startswith("v_conv1_output"):
                output_var_found = True
            elif sym_name.startswith("p_conv1_weight"):
                weight_par_found = True
            elif sym_name.startswith("p_conv1_bias"):
                bias_par_found = True

        assert output_var_found
        assert weight_par_found
        assert bias_par_found

        # this produces an output that is 4 x 4 too
        conv2 = layer(m, 4, 4, 4, padding=3, stride=2, bias=True)
        conv2.load_weights(w1, b1)
        _out2, _eqs2 = conv2(inp)
        conv2.load_weights(w1, b1)


@pytest.mark.unit
def test_conv_reloading_weights(data):
    m, *_ = data
    for layer, random_shape in zip(
        [Conv1d, Conv2d], [(2, 1, 3), (2, 1, 3, 3)], strict=False
    ):
        conv1 = layer(m, in_channels=1, out_channels=2, kernel_size=3, bias=True)
        w1 = np.random.rand(*random_shape)
        b1 = np.random.rand(2)
        conv1.load_weights(w1, b1)

        w1 = np.ones(random_shape)
        b1 = np.ones(2)
        conv1.load_weights(w1, b1)

        assert np.allclose(w1, conv1.weight.toDense())
        assert np.allclose(b1, conv1.bias.toDense())


@pytest.mark.unit
def test_conv_make_variable(data):
    m, *_ = data
    for layer, shape_len, padding_type in zip(
        [Conv1d, Conv2d], [3, 4], ["valid", "same"], strict=False
    ):
        conv1 = layer(
            m,
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            bias=True,
            padding=padding_type,
        )
        conv1.make_variable()
        assert conv1.weight.records is None
        assert conv1.bias.records is None

        conv2 = layer(
            m,
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            bias=True,
            padding=padding_type,
        )
        # setting init_weights initializes bias and weight
        conv2.make_variable(init_weights=True)
        assert conv2.weight.records is not None
        assert conv2.bias.records is not None

        w1 = np.random.rand(*[2, 1, 3, 3][:shape_len])
        b1 = np.random.rand(2)
        with pytest.raises(ValidationError):
            conv1.load_weights(w1, b1)
        assert isinstance(conv1.weight, gp.Variable)
        assert isinstance(conv1.bias, gp.Variable)
        inp = gp.Variable(m, domain=dim([4, 1, 4, 4][:shape_len]))
        _out, _eqs = conv1(inp)


@pytest.mark.unit
def test_conv_load_weight_make_var(data):
    m, *_ = data
    for layer, shape_len in zip([Conv1d, Conv2d], [3, 4], strict=False):
        conv1 = layer(m, in_channels=1, out_channels=2, kernel_size=3, bias=True)
        w1 = np.random.rand(*[2, 1, 3, 3][:shape_len])
        b1 = np.random.rand(2)
        conv1.load_weights(w1, b1)
        assert isinstance(conv1.weight, gp.Parameter)
        assert isinstance(conv1.bias, gp.Parameter)
        with pytest.raises(ValidationError):
            conv1.make_variable()


@pytest.mark.unit
def test_conv_call_bad(data):
    m, *_ = data
    for layer, shape_len in zip([Conv1d, Conv2d], [3, 4], strict=False):
        conv1 = layer(m, 4, 4, 4, bias=True)
        inp = gp.Variable(m, domain=dim([4, 4, 4, 4][:shape_len]))
        # requires initialization before
        with pytest.raises(ValidationError):
            conv1(inp)

        w1 = np.random.rand(*[4, 4, 4, 4][:shape_len])
        b1 = np.random.rand(4)
        conv1.load_weights(w1, b1)

        # needs 3 or 4 dimension
        bad_inp = gp.Variable(m, domain=dim([4] * (shape_len - 1)))
        with pytest.raises(ValidationError):
            conv1(bad_inp)

        # in channel must match 4
        # batch x in_channel x height x width
        bad_inp_2 = gp.Variable(m, domain=dim([10, 3, 4, 4][:shape_len]))
        with pytest.raises(ValidationError):
            conv1(bad_inp_2)

        # propagate_bounds must be a boolean
        with pytest.raises(ValidationError):
            conv1(inp, propagate_bounds="True")


@pytest.mark.unit
def test_conv2d_simple_correctness(data):
    m, w1, b1, _inp, par_input, *_ = data
    conv1 = Conv2d(m, 1, 2, 3)
    conv1.load_weights(w1, b1)
    out, eqs = conv1(par_input)
    obj = gp.Sum(out.domain, out)
    model = gp.Model(
        m,
        "convolve",
        equations=eqs,
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = np.array(
        [
            [
                [
                    [3.40817582, 3.3172471, 3.45373787],
                    [3.56033042, 2.98572956, 3.78199492],
                    [3.37529918, 3.47334837, 3.70609844],
                ],
                [
                    [3.48893453, 3.78932057, 4.0210465],
                    [3.47019698, 3.4842526, 4.21685095],
                    [3.4510945, 3.5865001, 3.97044084],
                ],
            ],
            [
                [
                    [4.11079177, 4.50948556, 3.71462732],
                    [3.69309437, 4.21991099, 3.47390714],
                    [3.55334825, 3.24612233, 2.60289597],
                ],
                [
                    [4.19617192, 4.89775484, 4.12360071],
                    [4.284388, 4.68931071, 3.95501428],
                    [3.60870751, 4.34399436, 4.02012768],
                ],
            ],
            [
                [
                    [3.86066434, 3.34271449, 3.5020279],
                    [3.19546317, 3.14612869, 2.99550383],
                    [2.30479297, 2.35869599, 1.8917482],
                ],
                [
                    [3.84418946, 3.8675763, 3.50556872],
                    [3.99821923, 3.56950653, 3.48662108],
                    [2.81866521, 3.16280288, 2.24684035],
                ],
            ],
        ]
    )

    assert np.allclose(out.toDense(), expected_out)


@pytest.mark.unit
def test_conv1d_simple_correctness(data):
    m, _, _, _, _, _, par_input_2, *_ = data

    w1 = np.array(
        [
            [[0.98727534, 0.94129724, 0.44578929]],
            [[0.45728722, 0.15647212, 0.56943917]],
        ]
    )
    b1 = np.array([2.2, -0.4])

    conv1 = Conv1d(m, 1, 2, 3)
    conv1.load_weights(w1, b1)
    out, eqs = conv1(par_input_2)
    obj = gp.Sum(out.domain, out)
    model = gp.Model(
        m,
        "convolve",
        equations=eqs,
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = np.array(
        [
            [
                [3.56825381, 3.74108292, 3.90399443],
                [0.40809439, 0.41946991, 0.20110516],
            ],
            [
                [3.92287844, 3.98335548, 3.53671043],
                [0.41505584, 0.34675258, 0.31888293],
            ],
            [
                [3.57567320, 3.40667563, 3.13680175],
                [0.33893762, -0.02290649, 0.19859786],
            ],
        ]
    )

    assert np.allclose(out.toDense(), expected_out)


@pytest.mark.unit
def test_conv_with_same_padding_odd_kernel(data):
    # when kernel size is odd
    m, _w1, _b1, _inp, par_input, _ii, par_input_2 = data

    keep_same_1 = np.array(
        [
            [
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            ]
        ]
    )

    keep_same_2 = np.array([[[0, 1, 0]]])

    for layer, weight, inp_par in zip(
        [Conv1d, Conv2d],
        [keep_same_2, keep_same_1],
        [par_input_2, par_input],
        strict=False,
    ):
        conv1 = layer(m, 1, 1, 3, padding="same", bias=True)
        add_one = np.array([1])
        conv1.load_weights(weight, add_one)

        out, eqs = conv1(inp_par)
        obj = gp.Sum(out.domain, out)
        model = gp.Model(
            m,
            "convolve",
            equations=eqs,
            objective=obj,
            sense="min",
            problem="LP",
        )
        model.solve()
        assert np.allclose(out.toDense(), inp_par.toDense() + 1)


@pytest.mark.unit
def test_conv2d_with_same_padding_even_kernel(data):
    # when kernel size is even
    m, _w1, _b1, inp, par_input, *_ = data

    conv1 = Conv2d(m, 1, 1, 2, padding="same", bias=False)
    keep_same = np.array(
        [
            [
                [
                    [1, 0],
                    [0, 0],
                ]
            ]
        ]
    )
    conv1.load_weights(keep_same)

    out, eqs = conv1(par_input)
    obj = gp.Sum(out.domain, out)
    model = gp.Model(
        m,
        "convolve",
        equations=eqs,
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()
    assert np.allclose(out.toDense(), inp)


@pytest.mark.unit
def test_conv2d_with_same_padding_even_kernel_2(data):
    # when kernel size is odd
    m, w1, b1, _inp, par_input, *_ = data

    conv1 = Conv2d(m, 1, 2, 2, padding="same", bias=True)
    conv1.load_weights(w1[:, :, :2, :2], b1)

    out, eqs = conv1(par_input)
    obj = gp.Sum(out.domain, out)
    model = gp.Model(
        m,
        "convolve",
        equations=eqs,
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = np.array(
        [
            [
                [
                    [
                        1.625492362,
                        1.831020399,
                        2.074811448,
                        1.949193639,
                        1.320707505,
                    ],
                    [
                        1.676575121,
                        1.825233887,
                        2.045062188,
                        1.750769834,
                        1.518886772,
                    ],
                    [
                        1.739361268,
                        2.182634389,
                        1.871283841,
                        1.768675940,
                        1.604790037,
                    ],
                    [
                        1.456248628,
                        1.901950465,
                        2.224367490,
                        1.933974614,
                        1.600900934,
                    ],
                    [
                        0.792228761,
                        0.803098530,
                        1.008683930,
                        0.970616023,
                        0.929699696,
                    ],
                ],
                [
                    [
                        1.989642237,
                        2.331308922,
                        2.732532466,
                        2.291186359,
                        1.165601731,
                    ],
                    [
                        1.911615902,
                        2.289122410,
                        2.402451849,
                        2.164851606,
                        1.686090769,
                    ],
                    [
                        2.073993977,
                        2.581295621,
                        2.232052203,
                        2.237492228,
                        1.662072268,
                    ],
                    [
                        1.920337806,
                        2.559152521,
                        2.467719982,
                        2.293567862,
                        1.772644566,
                    ],
                    [
                        1.171485701,
                        1.429711668,
                        1.779075395,
                        1.730294061,
                        1.242035090,
                    ],
                ],
            ],
            [
                [
                    [
                        2.116932766,
                        2.511731088,
                        2.270286885,
                        1.503704789,
                        0.865328703,
                    ],
                    [
                        2.005340874,
                        2.716170531,
                        2.641717508,
                        2.325581132,
                        1.094038321,
                    ],
                    [
                        1.787231320,
                        1.931375667,
                        1.735580121,
                        1.599498191,
                        1.193099388,
                    ],
                    [
                        1.695021842,
                        2.159671421,
                        1.620438092,
                        0.764438783,
                        0.761570065,
                    ],
                    [
                        0.712479172,
                        1.006866173,
                        1.001301633,
                        0.599276914,
                        0.567524580,
                    ],
                ],
                [
                    [
                        2.701160679,
                        3.078839627,
                        2.552822524,
                        1.793918220,
                        1.096404400,
                    ],
                    [
                        2.609892372,
                        3.238318246,
                        3.105523792,
                        2.355734422,
                        0.912041700,
                    ],
                    [
                        2.320455077,
                        2.795927285,
                        2.643403462,
                        2.337474751,
                        1.353383002,
                    ],
                    [
                        2.146012877,
                        2.377427706,
                        1.724706729,
                        1.053355860,
                        0.907902065,
                    ],
                    [
                        1.236397505,
                        1.873128785,
                        1.419942866,
                        0.611034633,
                        0.544359913,
                    ],
                ],
            ],
            [
                [
                    [
                        2.277089480,
                        2.014599858,
                        2.182283439,
                        2.105873144,
                        1.563848010,
                    ],
                    [
                        2.191232684,
                        2.167709605,
                        1.808435209,
                        1.751474886,
                        1.243114652,
                    ],
                    [
                        1.426328333,
                        1.398039294,
                        1.059038334,
                        1.427672402,
                        1.254717934,
                    ],
                    [
                        1.591004737,
                        1.380324608,
                        1.283963575,
                        1.136078616,
                        0.934567582,
                    ],
                    [
                        0.761236039,
                        0.896806832,
                        0.714964186,
                        0.838569987,
                        0.586811741,
                    ],
                ],
                [
                    [
                        2.539326575,
                        2.440968140,
                        2.240982654,
                        2.072073217,
                        1.528195358,
                    ],
                    [
                        2.847413970,
                        2.568970141,
                        2.449173707,
                        2.541086065,
                        1.560280100,
                    ],
                    [
                        2.019722140,
                        2.050710014,
                        1.544939781,
                        1.731307436,
                        1.179856643,
                    ],
                    [
                        1.776538432,
                        1.377596789,
                        1.256271748,
                        1.402568837,
                        1.201103294,
                    ],
                    [
                        1.270046896,
                        1.296650580,
                        1.131551899,
                        1.093507107,
                        0.581513691,
                    ],
                ],
            ],
        ]
    )

    assert np.allclose(out.toDense(), expected_out)


@pytest.mark.unit
def test_conv2d_with_padding(data):
    m, w1, b1, _inp, par_input, *_ = data

    conv_with_valid_padding = Conv2d(m, 1, 2, 3, padding="valid")
    assert conv_with_valid_padding.padding == (0, 0, 0, 0)

    conv1 = Conv2d(m, 1, 2, 3, padding=(2, 1))
    conv1.load_weights(w1, b1)
    out, eqs = conv1(par_input)
    obj = gp.Sum(out.domain, out)
    model = gp.Model(
        m,
        "convolve",
        equations=eqs,
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = np.array(
        [
            [
                [
                    [
                        1.09482886,
                        1.45187392,
                        1.60871321,
                        1.28820095,
                        0.90917775,
                    ],
                    [
                        1.73266432,
                        2.77629864,
                        2.72445678,
                        3.01090006,
                        1.89135463,
                    ],
                    [
                        2.00526839,
                        3.40817582,
                        3.31724710,
                        3.45373787,
                        2.36433349,
                    ],
                    [
                        2.04022241,
                        3.56033042,
                        2.98572956,
                        3.78199492,
                        2.16053738,
                    ],
                    [
                        2.04495384,
                        3.37529918,
                        3.47334837,
                        3.70609844,
                        2.26612500,
                    ],
                    [
                        1.51940592,
                        2.38417569,
                        2.42072500,
                        3.12349376,
                        1.93397461,
                    ],
                    [
                        0.80250535,
                        1.24912158,
                        1.21058621,
                        1.44216377,
                        0.97061602,
                    ],
                ],
                [
                    [
                        0.71071167,
                        1.40481570,
                        1.24067886,
                        1.45165663,
                        1.42892661,
                    ],
                    [
                        1.28715275,
                        2.29089282,
                        2.13384626,
                        2.81963343,
                        1.57319673,
                    ],
                    [
                        1.96462678,
                        3.48893453,
                        3.78932057,
                        4.02104650,
                        2.84586726,
                    ],
                    [
                        1.83858827,
                        3.47019698,
                        3.48425260,
                        4.21685095,
                        2.49355534,
                    ],
                    [
                        1.89531428,
                        3.45109450,
                        3.58650010,
                        3.97044084,
                        2.97442924,
                    ],
                    [
                        1.70490171,
                        2.77092927,
                        2.97687223,
                        3.29542150,
                        2.29356786,
                    ],
                    [
                        1.05067524,
                        1.67584386,
                        1.87953213,
                        2.25758827,
                        1.73029406,
                    ],
                ],
            ],
            [
                [
                    [
                        1.39211618,
                        1.60463084,
                        1.39657212,
                        1.31346770,
                        0.90085603,
                    ],
                    [
                        2.34173815,
                        3.21871490,
                        3.30662930,
                        2.38772268,
                        1.59456021,
                    ],
                    [
                        2.87777551,
                        4.11079177,
                        4.50948556,
                        3.71462732,
                        2.05095603,
                    ],
                    [
                        2.58540365,
                        3.69309437,
                        4.21991099,
                        3.47390714,
                        2.53743692,
                    ],
                    [
                        2.52497112,
                        3.55334825,
                        3.24612233,
                        2.60289597,
                        1.62478153,
                    ],
                    [
                        1.43317117,
                        2.46939237,
                        2.31797748,
                        1.85099642,
                        0.76443878,
                    ],
                    [
                        1.01444597,
                        1.22324974,
                        1.04919098,
                        1.00713376,
                        0.59927691,
                    ],
                ],
                [
                    [
                        0.82438483,
                        1.34110666,
                        1.60881756,
                        1.44935530,
                        0.95429445,
                    ],
                    [
                        1.55741715,
                        2.55611183,
                        2.95835033,
                        2.25227659,
                        1.92893181,
                    ],
                    [
                        2.52818950,
                        4.19617192,
                        4.89775484,
                        4.12360071,
                        2.81645886,
                    ],
                    [
                        2.34508430,
                        4.28438800,
                        4.68931071,
                        3.95501428,
                        2.59088597,
                    ],
                    [
                        2.29573969,
                        3.60870751,
                        4.34399436,
                        4.02012768,
                        2.40634409,
                    ],
                    [
                        1.72826637,
                        2.80660800,
                        2.53608556,
                        1.97699944,
                        1.05335586,
                    ],
                    [
                        1.14225034,
                        1.80023063,
                        1.91985060,
                        1.42638087,
                        0.61103463,
                    ],
                ],
            ],
            [
                [
                    [
                        1.15375539,
                        1.42140982,
                        1.03719591,
                        1.11218177,
                        0.77030560,
                    ],
                    [
                        2.17736580,
                        2.86705014,
                        2.67669292,
                        2.58957975,
                        1.62148771,
                    ],
                    [
                        2.86727624,
                        3.86066434,
                        3.34271449,
                        3.50202790,
                        2.37209718,
                    ],
                    [
                        2.22385497,
                        3.19546317,
                        3.14612869,
                        2.99550383,
                        2.02820282,
                    ],
                    [
                        2.19571800,
                        2.30479297,
                        2.35869599,
                        1.89174820,
                        1.60889262,
                    ],
                    [
                        1.30250046,
                        1.82027845,
                        1.70167408,
                        1.70161924,
                        1.13607862,
                    ],
                    [
                        0.96057942,
                        0.88614164,
                        1.21777939,
                        0.74357013,
                        0.83856999,
                    ],
                ],
                [
                    [
                        0.73515794,
                        1.31735508,
                        1.15219339,
                        1.33486383,
                        0.62699899,
                    ],
                    [
                        1.42969176,
                        2.65849114,
                        2.33068117,
                        2.07437727,
                        1.85093786,
                    ],
                    [
                        2.51938809,
                        3.84418946,
                        3.86757630,
                        3.50556872,
                        2.60153026,
                    ],
                    [
                        2.41933486,
                        3.99821923,
                        3.56950653,
                        3.48662108,
                        2.65975312,
                    ],
                    [
                        2.04366991,
                        2.81866521,
                        3.16280288,
                        2.24684035,
                        2.25052283,
                    ],
                    [
                        1.47083491,
                        1.98213489,
                        1.61027923,
                        1.70643899,
                        1.40256884,
                    ],
                    [
                        1.15567758,
                        1.40792860,
                        1.65096810,
                        1.16312964,
                        1.09350711,
                    ],
                ],
            ],
        ]
    )

    assert np.allclose(out.toDense(), expected_out)


@pytest.mark.unit
def test_conv2d_with_stride(data):
    m, w1, b1, _inp, par_input, *_ = data
    conv1 = Conv2d(m, 1, 2, 3, stride=(2, 1))
    conv1.load_weights(w1, b1)
    out, eqs = conv1(par_input)
    obj = gp.Sum(out.domain, out)
    model = gp.Model(
        m,
        "convolve",
        equations=eqs,
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = np.array(
        [
            [
                [
                    [3.40817582, 3.31724710, 3.45373787],
                    [3.37529918, 3.47334837, 3.70609844],
                ],
                [
                    [3.48893453, 3.78932057, 4.02104650],
                    [3.45109450, 3.58650010, 3.97044084],
                ],
            ],
            [
                [
                    [4.11079177, 4.50948556, 3.71462732],
                    [3.55334825, 3.24612233, 2.60289597],
                ],
                [
                    [4.19617192, 4.89775484, 4.12360071],
                    [3.60870751, 4.34399436, 4.02012768],
                ],
            ],
            [
                [
                    [3.86066434, 3.34271449, 3.50202790],
                    [2.30479297, 2.35869599, 1.89174820],
                ],
                [
                    [3.84418946, 3.86757630, 3.50556872],
                    [2.81866521, 3.16280288, 2.24684035],
                ],
            ],
        ],
    )

    assert np.allclose(out.toDense(), expected_out)


@pytest.mark.unit
def test_conv2d_with_padding_and_stride(data):
    m, w1, b1, _inp, par_input, *_ = data
    conv1 = Conv2d(m, 1, 2, 3, stride=(2, 1), padding=(1, 2))
    conv1.load_weights(w1, b1)
    out, eqs = conv1(par_input)
    obj = gp.Sum(out.domain, out)
    model = gp.Model(
        m,
        "convolve",
        equations=eqs,
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = np.array(
        [
            [
                [
                    [
                        1.28484820,
                        1.73266432,
                        2.77629864,
                        2.72445678,
                        3.01090006,
                        1.89135463,
                        0.95047736,
                    ],
                    [
                        1.39675216,
                        2.04022241,
                        3.56033042,
                        2.98572956,
                        3.78199492,
                        2.16053738,
                        1.77149393,
                    ],
                    [
                        1.03489579,
                        1.51940592,
                        2.38417569,
                        2.42072500,
                        3.12349376,
                        1.93397461,
                        1.60090093,
                    ],
                ],
                [
                    [
                        0.91730681,
                        1.28715275,
                        2.29089282,
                        2.13384626,
                        2.81963343,
                        1.57319673,
                        1.39243096,
                    ],
                    [
                        1.15649834,
                        1.83858827,
                        3.47019698,
                        3.48425260,
                        4.21685095,
                        2.49355534,
                        2.48715936,
                    ],
                    [
                        0.96652008,
                        1.70490171,
                        2.77092927,
                        2.97687223,
                        3.29542150,
                        2.29356786,
                        1.77264457,
                    ],
                ],
            ],
            [
                [
                    [
                        1.17758352,
                        2.34173815,
                        3.21871490,
                        3.30662930,
                        2.38772268,
                        1.59456021,
                        1.04624809,
                    ],
                    [
                        1.51316670,
                        2.58540365,
                        3.69309437,
                        4.21991099,
                        3.47390714,
                        2.53743692,
                        1.20767189,
                    ],
                    [
                        1.06347776,
                        1.43317117,
                        2.46939237,
                        2.31797748,
                        1.85099642,
                        0.76443878,
                        0.76157006,
                    ],
                ],
                [
                    [
                        0.85609347,
                        1.55741715,
                        2.55611183,
                        2.95835033,
                        2.25227659,
                        1.92893181,
                        0.88178639,
                    ],
                    [
                        1.16830351,
                        2.34508430,
                        4.28438800,
                        4.68931071,
                        3.95501428,
                        2.59088597,
                        1.27239681,
                    ],
                    [
                        1.04895946,
                        1.72826637,
                        2.80660800,
                        2.53608556,
                        1.97699944,
                        1.05335586,
                        0.90790207,
                    ],
                ],
            ],
            [
                [
                    [
                        1.49460230,
                        2.17736580,
                        2.86705014,
                        2.67669292,
                        2.58957975,
                        1.62148771,
                        1.24506132,
                    ],
                    [
                        1.66749775,
                        2.22385497,
                        3.19546317,
                        3.14612869,
                        2.99550383,
                        2.02820282,
                        1.44059232,
                    ],
                    [
                        1.01903323,
                        1.30250046,
                        1.82027845,
                        1.70167408,
                        1.70161924,
                        1.13607862,
                        0.93456758,
                    ],
                ],
                [
                    [
                        0.97981128,
                        1.42969176,
                        2.65849114,
                        2.33068117,
                        2.07437727,
                        1.85093786,
                        1.66978918,
                    ],
                    [
                        1.42419598,
                        2.41933486,
                        3.99821923,
                        3.56950653,
                        3.48662108,
                        2.65975312,
                        2.18652189,
                    ],
                    [
                        0.97384675,
                        1.47083491,
                        1.98213489,
                        1.61027923,
                        1.70643899,
                        1.40256884,
                        1.20110329,
                    ],
                ],
            ],
        ]
    )

    assert np.allclose(out.toDense(), expected_out)


@pytest.mark.unit
def test_conv2d_propagate_bounds_general(data):
    m, *_ = data

    w1 = np.random.rand(3, 1, 2, 2)
    b1 = np.random.rand(3)

    inp_lower = np.random.rand(16, 1, 24, 24)
    inp_upper = np.random.rand(16, 1, 24, 24)

    lo_inp = gp.Parameter(m, domain=dim((16, 1, 24, 24)), records=inp_lower)
    up_inp = gp.Parameter(m, domain=dim((16, 1, 24, 24)), records=inp_upper)

    # in_channels=1, out_channels=3, kernel_size=2x2
    conv1 = Conv2d(m, 1, 3, 2)
    conv1.load_weights(w1, b1)

    # in_channels=1, out_channels=3, kernel_size=2x2, bias=False
    conv2 = Conv2d(m, 1, 3, 2, bias=False)
    conv2.load_weights(w1)

    # 16 images, 1 channels, 24 by 24
    inp = gp.Variable(m, domain=dim((16, 1, 24, 24)))

    # Unbounded input results in unbounded output
    out1, _ = conv1(inp, propagate_bounds=True)
    out2, _ = conv2(inp, propagate_bounds=True)

    assert out1.up.records is None
    assert out1.lo.records is None
    assert out2.up.records is None
    assert out2.lo.records is None

    # Bounded input with "propagate_bounds = False" results in unbounded output
    inp.lo[...] = lo_inp[...]
    inp.up[...] = up_inp[...]

    out3, _ = conv1(inp, propagate_bounds=False)
    out4, _ = conv2(inp, propagate_bounds=False)

    assert out3.up.records is None
    assert out3.lo.records is None
    assert out4.up.records is None
    assert out4.lo.records is None

    # Bounded input with "propagate_bounds = True" results in bounded output
    out5, _ = conv1(inp, propagate_bounds=True)
    out6, _ = conv2(inp, propagate_bounds=True)

    assert out5.up.records is not None
    assert out5.lo.records is not None
    assert out6.up.records is not None
    assert out6.lo.records is not None


@pytest.mark.unit
def test_conv_propagate_bounds_zero_weights_unbounded_input(data):
    m, *_ = data

    for layer, shape_len in zip([Conv1d, Conv2d], [3, 4], strict=False):
        w1 = np.zeros((3, 1, 2, 2)[:shape_len])
        b1 = np.random.rand(3)

        # in_channels=1, out_channels=3, kernel_size=2
        conv1 = layer(m, 1, 3, 2)
        conv1.load_weights(w1, b1)

        # in_channels=1, out_channels=3, kernel_size=2x2, bias=False
        conv2 = layer(m, 1, 3, 2, bias=False)
        conv2.load_weights(w1)

        # 16 images, 1 channels, 24 by 24
        inp = gp.Variable(m, domain=dim((16, 1, 24, 24)[:shape_len]))

        out1, _ = conv1(inp, propagate_bounds=True)
        out2, _ = conv2(inp, propagate_bounds=True)

        # When bias is present, the output bounds should be equal to the bias
        if layer == Conv1d:
            assert np.allclose(
                np.array(out1.records.upper).reshape(out1.shape),
                b1[:, np.newaxis],
            )
            assert np.allclose(
                np.array(out1.records.lower).reshape(out1.shape),
                b1[:, np.newaxis],
            )
        else:
            assert np.allclose(
                np.array(out1.records.upper).reshape(out1.shape),
                b1[:, np.newaxis, np.newaxis],
            )
            assert np.allclose(
                np.array(out1.records.lower).reshape(out1.shape),
                b1[:, np.newaxis, np.newaxis],
            )

        # When bias is not present, the output bounds should be zeros
        assert np.allclose(
            np.array(out2.records.upper).reshape(out2.shape),
            np.zeros(out2.shape),
        )
        assert np.allclose(
            np.array(out2.records.lower).reshape(out2.shape),
            np.zeros(out2.shape),
        )


@pytest.mark.unit
def test_conv_propagate_bounds_input_bounded_by_zero(data):
    m, *_ = data

    for layer, shape_len in zip([Conv1d, Conv2d], [3, 4], strict=False):
        w1 = np.random.rand(*(3, 1, 2, 2)[:shape_len])
        b1 = np.random.rand(3)

        # in_channels=1, out_channels=3, kernel_size=2
        conv1 = layer(m, 1, 3, 2)
        conv1.load_weights(w1, b1)

        # in_channels=1, out_channels=3, kernel_size=2, bias=False
        conv2 = layer(m, 1, 3, 2, bias=False)
        conv2.load_weights(w1)

        # 16 images, 1 channels, 24 by 24
        inp = gp.Variable(m, domain=dim((16, 1, 24, 24)[:shape_len]))

        # Input bounded with zeros results in bounded output with zeros (bias not present) or bias (bias present)
        inp.lo[...] = 0
        inp.up[...] = 0

        out1, _ = conv1(inp, propagate_bounds=True)
        out2, _ = conv2(inp, propagate_bounds=True)

        # When bias is present, the output bounds should be equal to the bias
        if layer == Conv2d:
            assert np.allclose(
                np.array(out1.records.upper).reshape(out1.shape),
                b1[:, np.newaxis, np.newaxis],
            )
            assert np.allclose(
                np.array(out1.records.lower).reshape(out1.shape),
                b1[:, np.newaxis, np.newaxis],
            )
        else:
            assert np.allclose(
                np.array(out1.records.upper).reshape(out1.shape),
                b1[:, np.newaxis],
            )
            assert np.allclose(
                np.array(out1.records.lower).reshape(out1.shape),
                b1[:, np.newaxis],
            )

        # When bias is not present, the output bounds should be zeros
        assert np.allclose(
            np.array(out2.records.upper).reshape(out2.shape),
            np.zeros(out2.shape),
        )
        assert np.allclose(
            np.array(out2.records.lower).reshape(out2.shape),
            np.zeros(out2.shape),
        )


@pytest.mark.unit
def test_conv2d_propagate_bounds_complex_bounds(data):
    m, *_ = data

    w1 = np.array([[[[3, -3], [-2, 0]], [[0, -2], [-1, -3]], [[2, 0], [-4, 1]]]])

    b1 = np.array([3])

    inp_lower = np.array(
        [
            [
                [
                    [-2, -np.inf, -4, -1],
                    [-2, -1, -1, -np.inf],
                    [0, -5, -3, -1],
                    [-3, 0, -1, -np.inf],
                ],
                [
                    [-2, -4, -2, -1],
                    [-3, -1, -5, -3],
                    [-2, -1, -1, -1],
                    [-1, -1, 0, -2],
                ],
                [
                    [-1, -2, -1, -5],
                    [0, -3, -3, -1],
                    [-2, 0, -1, -2],
                    [-5, -3, -5, 0],
                ],
            ],
            [
                [
                    [-2, -3, 0, 0],
                    [-3, -4, -2, -1],
                    [-2, -1, -5, -4],
                    [-5, 0, -4, -5],
                ],
                [
                    [-1, -2, -5, 0],
                    [-3, -1, -3, -2],
                    [-1, -5, -3, -1],
                    [-2, -2, 0, -3],
                ],
                [
                    [-3, -3, -np.inf, -1],
                    [-np.inf, 0, -5, -2],
                    [-4, -np.inf, -3, -3],
                    [-4, -3, -3, -np.inf],
                ],
            ],
        ]
    )

    inp_upper = np.array(
        [
            [
                [
                    [4, 2, 4, 4],
                    [3, 2, np.inf, 4],
                    [2, 1, 2, np.inf],
                    [3, np.inf, 3, 3],
                ],
                [
                    [3, 2, 3, 1],
                    [3, 3, 4, 1],
                    [np.inf, 4, 3, np.inf],
                    [3, 2, 4, 2],
                ],
                [[1, 2, 1, 2], [3, 4, np.inf, 4], [4, 2, 4, 2], [1, 3, 3, 4]],
            ],
            [
                [[np.inf, 4, 3, 4], [2, 4, 3, 3], [4, 2, 2, 4], [2, 2, 1, 3]],
                [
                    [np.inf, 3, 3, 2],
                    [2, 4, np.inf, 4],
                    [4, 2, 3, 4],
                    [4, 2, 2, 2],
                ],
                [
                    [np.inf, 1, 4, 2],
                    [3, 1, 3, 1],
                    [2, 4, 2, 2],
                    [3, 2, np.inf, 2],
                ],
            ],
        ]
    )

    exp_up = np.array([[[[np.inf, 54], [67, 54]]], [[[np.inf, 54], [68, 58]]]])

    exp_lo = np.array(
        [
            [[[-48, -np.inf], [-34, -np.inf]]],
            [[[-57, -np.inf], [-50, -np.inf]]],
        ]
    )

    lo_inp = gp.Parameter(m, domain=dim((2, 3, 4, 4)), records=inp_lower)
    up_inp = gp.Parameter(m, domain=dim((2, 3, 4, 4)), records=inp_upper)

    # in_channels=3, out_channels=1, kernel_size=2x2, stride=2
    conv1 = Conv2d(m, 3, 1, 2, 2)
    conv1.load_weights(w1, b1)

    # 2 images, 3 channels, 4 by 4
    inp = gp.Variable(m, domain=dim((2, 3, 4, 4)))
    inp.lo[...] = lo_inp[...]
    inp.up[...] = up_inp[...]

    out, _ = conv1(inp, propagate_bounds=True)

    assert np.allclose(np.array(out.records.upper).reshape(out.shape), exp_up)
    assert np.allclose(np.array(out.records.lower).reshape(out.shape), exp_lo)


@pytest.mark.unit
def test_conv2d_propagate_bounds_with_same_padding(data):
    m, *_ = data

    w1 = np.array(
        [
            [
                [
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                ]
            ]
        ]
    )

    inp_lower = np.array(
        [
            [
                [
                    [-1, -2, -3, -3, 0],
                    [-1, 0, -1, -3, -4],
                    [-1, -1, -3, -3, 0],
                    [-1, -1, -3, -2, -5],
                    [-2, -4, -5, 0, -4],
                ]
            ],
            [
                [
                    [-1, -1, -2, -2, -5],
                    [-2, -4, 0, -4, -3],
                    [-4, -2, -2, -3, -2],
                    [-4, -4, -1, -2, -2],
                    [-4, -4, -5, 0, -3],
                ]
            ],
        ]
    )

    inp_upper = np.array(
        [
            [
                [
                    [1, 4, 1, 4, 4],
                    [4, 4, 2, 3, 4],
                    [4, 3, 1, 1, 1],
                    [2, 4, 2, 3, 1],
                    [3, 1, 4, 3, 2],
                ]
            ],
            [
                [
                    [3, 2, 4, 3, 3],
                    [4, 1, 2, 3, 3],
                    [1, 2, 1, 3, 3],
                    [3, 1, 2, 3, 3],
                    [4, 2, 1, 3, 1],
                ]
            ],
        ]
    )

    exp_up = np.array(
        [
            [
                [
                    [1.0, 5.0, 4.0, 8.0, 4.0],
                    [5.0, 8.0, 7.0, 6.0, 7.0],
                    [5.0, 7.0, 3.0, 7.0, 4.0],
                    [6.0, 10.0, 3.0, 10.0, 4.0],
                    [3.0, 2.0, 5.0, 6.0, 4.0],
                ]
            ],
            [
                [
                    [7.0, 2.0, 8.0, 6.0, 3.0],
                    [6.0, 4.0, 6.0, 7.0, 5.0],
                    [5.0, 5.0, 7.0, 5.0, 7.0],
                    [7.0, 10.0, 4.0, 8.0, 6.0],
                    [4.0, 6.0, 5.0, 4.0, 3.0],
                ]
            ],
        ]
    )

    exp_lo = np.array(
        [
            [
                [
                    [-5.0, -4.0, -6.0, -7.0, 0.0],
                    [-4.0, -2.0, -6.0, -5.0, -8.0],
                    [-5.0, -7.0, -10.0, -6.0, -3.0],
                    [-2.0, -9.0, -9.0, -5.0, -6.0],
                    [-2.0, -6.0, -9.0, -2.0, -7.0],
                ]
            ],
            [
                [
                    [-2.0, -3.0, -5.0, -5.0, -5.0],
                    [-4.0, -8.0, -5.0, -11.0, -6.0],
                    [-5.0, -8.0, -6.0, -8.0, -5.0],
                    [-6.0, -6.0, -6.0, -4.0, -5.0],
                    [-4.0, -7.0, -6.0, -2.0, -6.0],
                ]
            ],
        ]
    )

    lo_inp = gp.Parameter(m, domain=dim((2, 1, 5, 5)), records=inp_lower)
    up_inp = gp.Parameter(m, domain=dim((2, 1, 5, 5)), records=inp_upper)

    # in_channels=1, out_channels=1, kernel_size=3x3, padding=same
    conv1 = Conv2d(m, 1, 1, 3, padding="same", bias=False)
    conv1.load_weights(w1)

    # 2 images, 1 channel, 5 by 5
    inp = gp.Variable(m, domain=dim((2, 1, 5, 5)))
    inp.lo[...] = lo_inp[...]
    inp.up[...] = up_inp[...]

    out, _ = conv1(inp, propagate_bounds=True)

    assert np.allclose(np.array(out.records.lower).reshape(out.shape), exp_lo)
    assert np.allclose(np.array(out.records.upper).reshape(out.shape), exp_up)


@pytest.mark.unit
def test_conv2d_propagate_bounds_with_same_padding_even_input(data):
    m, *_ = data

    np.random.seed(42)

    w1 = np.random.randint(-5, 5, (2, 2, 2, 2))

    inp_lower = np.random.randint(-5, 0, (2, 2, 2, 2))

    inp_upper = np.random.randint(0, 5, (2, 2, 2, 2))

    lo_inp = gp.Parameter(m, domain=dim((2, 2, 2, 2)), records=inp_lower)
    up_inp = gp.Parameter(m, domain=dim((2, 2, 2, 2)), records=inp_upper)

    # in_channels=2, out_channels=2, kernel_size=2x2, padding=same
    conv = Conv2d(m, 2, 2, 2, padding="same", bias=False)
    conv.load_weights(w1)

    # 2 images, 2 channels, 2 by 2
    inp = gp.Variable(m, domain=dim((2, 2, 2, 2)))
    inp.lo[...] = lo_inp[...]
    inp.up[...] = up_inp[...]

    out, _ = conv(inp, propagate_bounds=True)

    exp_lo = np.array(
        [
            [
                [[-56, -17], [-28, -6]],
                [[-30, -20], [-21, -12]],
            ],
            [
                [[-34, -18], [-23, -7]],
                [[-30, -16], [-24, -14]],
            ],
        ]
    )

    exp_up = np.array(
        [
            [[[31, 22], [19, 5]], [[41, 12], [28, 10]]],
            [[[40, 15], [16, 4]], [[37, 16], [15, 8]]],
        ]
    )

    assert np.allclose(np.array(out.records.lower).reshape(out.shape), exp_lo)
    assert np.allclose(np.array(out.records.upper).reshape(out.shape), exp_up)


@pytest.mark.unit
def test_conv1d_return_formulation_result(data):
    m, _, _, _, _, _, par_input_2, *_ = data

    w1 = np.array(
        [
            [[0.98727534, 0.94129724, 0.44578929]],
            [[0.45728722, 0.15647212, 0.56943917]],
        ]
    )
    b1 = np.array([2.2, -0.4])

    conv1 = Conv1d(m, 1, 2, 3)
    conv1.load_weights(w1, b1)
    result = conv1(par_input_2)

    assert isinstance(result, FormulationResult), "Expected a FormulationResult object"
    assert isinstance(result.result, gp.Variable), "Expected the output variable"
    assert isinstance(result.variables_created["output"], gp.Variable), (
        "Expected the output variable"
    )
    assert all(isinstance(v, gp.Equation) for v in result.equations_created.values()), (
        "Expected a list of Equations"
    )

    out = result.result
    obj = gp.Sum(out.domain, out)
    model = gp.Model(
        m,
        "convolve",
        equations=result.equations_created.values(),
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = np.array(
        [
            [
                [3.56825381, 3.74108292, 3.90399443],
                [0.40809439, 0.41946991, 0.20110516],
            ],
            [
                [3.92287844, 3.98335548, 3.53671043],
                [0.41505584, 0.34675258, 0.31888293],
            ],
            [
                [3.57567320, 3.40667563, 3.13680175],
                [0.33893762, -0.02290649, 0.19859786],
            ],
        ]
    )

    assert np.allclose(out.toDense(), expected_out)


@pytest.mark.unit
def test_conv2d_return_formulation_result(data):
    m, w1, b1, _inp, par_input, *_ = data
    conv1 = Conv2d(m, 1, 2, 3)
    conv1.load_weights(w1, b1)
    result = conv1(par_input)

    assert isinstance(result, FormulationResult), "Expected a FormulationResult object"
    assert isinstance(result.result, gp.Variable), "Expected the output variable"
    assert isinstance(result.variables_created["output"], gp.Variable), (
        "Expected the output variable"
    )
    assert all(isinstance(v, gp.Equation) for v in result.equations_created.values()), (
        "Expected a list of Equations"
    )

    out = result.result
    obj = gp.Sum(out.domain, out)

    model = gp.Model(
        m,
        "convolve",
        equations=result.equations_created.values(),
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = np.array(
        [
            [
                [
                    [3.40817582, 3.3172471, 3.45373787],
                    [3.56033042, 2.98572956, 3.78199492],
                    [3.37529918, 3.47334837, 3.70609844],
                ],
                [
                    [3.48893453, 3.78932057, 4.0210465],
                    [3.47019698, 3.4842526, 4.21685095],
                    [3.4510945, 3.5865001, 3.97044084],
                ],
            ],
            [
                [
                    [4.11079177, 4.50948556, 3.71462732],
                    [3.69309437, 4.21991099, 3.47390714],
                    [3.55334825, 3.24612233, 2.60289597],
                ],
                [
                    [4.19617192, 4.89775484, 4.12360071],
                    [4.284388, 4.68931071, 3.95501428],
                    [3.60870751, 4.34399436, 4.02012768],
                ],
            ],
            [
                [
                    [3.86066434, 3.34271449, 3.5020279],
                    [3.19546317, 3.14612869, 2.99550383],
                    [2.30479297, 2.35869599, 1.8917482],
                ],
                [
                    [3.84418946, 3.8675763, 3.50556872],
                    [3.99821923, 3.56950653, 3.48662108],
                    [2.81866521, 3.16280288, 2.24684035],
                ],
            ],
        ]
    )

    assert np.allclose(out.toDense(), expected_out)
