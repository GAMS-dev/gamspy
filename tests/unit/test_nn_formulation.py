from __future__ import annotations

import numpy as np
import pytest

import gamspy as gp
from gamspy import Container, ModelStatus
from gamspy.exceptions import ValidationError
from gamspy.formulations import flatten_dims
from gamspy.formulations.nn import (
    AvgPool2d,
    Conv2d,
    Linear,
    MaxPool2d,
    MinPool2d,
    _MPool2d,
)
from gamspy.math import dim

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = Container()
    w1 = np.array(
        [
            [
                [
                    [0.513957, 0.08810022, 0.60686814],
                    [0.86008108, 0.83876174, 0.55791719],
                    [0.3062239, 0.40856229, 0.66757332],
                ]
            ],
            [
                [
                    [0.99005985, 0.74292407, 0.66991401],
                    [0.58952114, 0.7207975, 0.38511354],
                    [0.97109817, 0.11883609, 0.24657315],
                ]
            ],
        ]
    )
    b1 = np.array([0.56258535, 0.53484524])
    inp = np.array(
        [
            [
                [
                    [0.64615, 0.40183, 0.7898, 0.89937, 0.17423],
                    [0.54191, 0.27341, 0.90273, 0.29883, 0.77734],
                    [0.40205, 0.55509, 0.67382, 0.49197, 0.64736],
                    [0.42091, 0.66672, 0.8369, 0.23754, 0.82491],
                    [0.38872, 0.33891, 0.75287, 0.67146, 0.71429],
                ]
            ],
            [
                [
                    [0.52554, 0.92097, 0.75623, 0.364, 0.55513],
                    [0.48203, 0.9401, 0.71608, 0.80008, 0.020266],
                    [0.37691, 0.9395, 0.95293, 0.97884, 0.6058],
                    [0.67074, 0.44264, 0.50225, 0.19674, 0.37108],
                    [0.1682, 0.72016, 0.84165, 0.069743, 0.0096102],
                ]
            ],
            [
                [
                    [0.56254, 0.54127, 0.69718, 0.03598, 0.48145],
                    [0.92599, 0.69301, 0.61562, 0.86881, 0.87645],
                    [0.38243, 0.90938, 0.49188, 0.51248, 0.2675],
                    [0.49372, 0.19366, 0.18858, 0.043283, 0.64488],
                    [0.28109, 0.61501, 0.20582, 0.5289, 0.047137],
                ]
            ],
        ]
    )

    par_input = gp.Parameter(m, domain=dim(inp.shape), records=inp)
    ii = gp.Set(m, "ii", records=range(20))
    yield m, w1, b1, inp, par_input, ii
    m.close()


def test_conv2d_bad_init(data):
    m, *_ = data
    # in channel must be integer
    pytest.raises(ValidationError, Conv2d, m, 2.5, 4, 3)
    pytest.raises(ValidationError, Conv2d, m, "2", 4, 3)
    # out channel must be integer
    pytest.raises(ValidationError, Conv2d, m, 2, 4.1, 3)
    pytest.raises(ValidationError, Conv2d, m, "2", 4.1, 3)
    # in channel must be positive
    pytest.raises(ValidationError, Conv2d, m, -4, 4, 3)
    # out channel must be positive
    pytest.raises(ValidationError, Conv2d, m, 4, -4, 3)

    # padding when string must be valid or same
    pytest.raises(ValidationError, Conv2d, m, 1, 2, 3, 1, "asd")

    # same padding requires stride = 1
    pytest.raises(ValidationError, Conv2d, m, 1, 2, 3, 2, "same")

    # kernel size must be integer or tuple of integer
    bad_values = [(3, "a"), ("a", 3), 2.4, -1, 0]
    for bad_value in bad_values:
        pytest.raises(ValidationError, Conv2d, m, 4, 4, bad_value)
    # stride size must be integer or tuple of integer
    for bad_value in bad_values:
        pytest.raises(ValidationError, Conv2d, m, 4, 4, 3, bad_value)
    # stride size must be integer or tuple of integer
    for bad_value in bad_values[:-1]:
        pytest.raises(ValidationError, Conv2d, m, 4, 4, 3, 1, bad_value)

    # bias must be a bool
    pytest.raises(ValidationError, Conv2d, m, 4, 4, 3, bias=10)


def test_conv2d_load_weights(data):
    m, *_ = data
    conv1 = Conv2d(m, in_channels=1, out_channels=2, kernel_size=3, bias=True)
    conv2 = Conv2d(m, in_channels=1, out_channels=2, kernel_size=3, bias=False)

    w1 = np.random.rand(2, 1, 3, 3)
    b1 = np.random.rand(2)

    # needs bias as well
    pytest.raises(ValidationError, conv1.load_weights, w1)

    # conv2 does not have bias
    pytest.raises(ValidationError, conv2.load_weights, w1, b1)

    # test bad shape
    bad1 = np.random.rand(1)
    pytest.raises(ValidationError, conv1.load_weights, bad1, b1)
    pytest.raises(ValidationError, conv1.load_weights, w1, bad1)

    bad2 = np.random.rand(2, 2, 2, 2)
    pytest.raises(ValidationError, conv1.load_weights, bad2, b1)

    bad3 = np.random.rand(6)
    pytest.raises(ValidationError, conv1.load_weights, w1, bad3)

    bad4 = np.random.rand(6, 2)
    pytest.raises(ValidationError, conv1.load_weights, w1, bad4)


def test_conv2d_same_indices(data):
    m, *_ = data
    conv1 = Conv2d(m, 4, 4, 4, bias=True)
    w1 = np.random.rand(4, 4, 4, 4)
    b1 = np.random.rand(4)
    conv1.load_weights(w1, b1)

    inp = gp.Variable(m, domain=dim([4, 4, 4, 4]))
    out, eqs = conv1(inp)

    # this produces an output that is 4 x 4 too
    conv2 = Conv2d(m, 4, 4, 4, padding=3, stride=2, bias=True)
    conv2.load_weights(w1, b1)
    out2, eqs2 = conv2(inp)
    conv2.load_weights(w1, b1)


def test_conv2d_reloading_weights(data):
    m, *_ = data
    conv1 = Conv2d(m, in_channels=1, out_channels=2, kernel_size=3, bias=True)
    w1 = np.random.rand(2, 1, 3, 3)
    b1 = np.random.rand(2)
    conv1.load_weights(w1, b1)

    w1 = np.ones((2, 1, 3, 3))
    b1 = np.ones(2)
    conv1.load_weights(w1, b1)

    assert np.allclose(w1, conv1.weight.toDense())
    assert np.allclose(b1, conv1.bias.toDense())


def test_conv2d_make_variable(data):
    m, *_ = data
    conv1 = Conv2d(m, in_channels=1, out_channels=2, kernel_size=3, bias=True)
    conv1.make_variable()
    w1 = np.random.rand(2, 1, 3, 3)
    b1 = np.random.rand(2)
    pytest.raises(ValidationError, conv1.load_weights, w1, b1)
    assert isinstance(conv1.weight, gp.Variable)
    assert isinstance(conv1.bias, gp.Variable)
    inp = gp.Variable(m, domain=dim([4, 1, 4, 4]))
    out, eqs = conv1(inp)


def test_conv2d_load_weight_make_var(data):
    m, *_ = data
    conv1 = Conv2d(m, in_channels=1, out_channels=2, kernel_size=3, bias=True)
    w1 = np.random.rand(2, 1, 3, 3)
    b1 = np.random.rand(2)
    conv1.load_weights(w1, b1)
    assert isinstance(conv1.weight, gp.Parameter)
    assert isinstance(conv1.bias, gp.Parameter)
    pytest.raises(ValidationError, conv1.make_variable)


def test_conv2d_call_bad(data):
    m, *_ = data
    conv1 = Conv2d(m, 4, 4, 4, bias=True)
    inp = gp.Variable(m, domain=dim([4, 4, 4, 4]))
    # requires initialization before
    pytest.raises(ValidationError, conv1, inp)

    w1 = np.random.rand(4, 4, 4, 4)
    b1 = np.random.rand(4)
    conv1.load_weights(w1, b1)

    # needs 4 dimension
    bad_inp = gp.Variable(m, domain=dim([4, 4, 4]))
    pytest.raises(ValidationError, conv1, bad_inp)

    # in channel must match 4
    # batch x in_channel x height x width
    bad_inp_2 = gp.Variable(m, domain=dim([10, 3, 4, 4]))
    pytest.raises(ValidationError, conv1, bad_inp_2)


def test_conv2d_simple_correctness(data):
    m, w1, b1, inp, par_input, _ = data
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


def test_conv2d_with_same_padding_odd_kernel(data):
    # when kernel size is odd
    m, w1, b1, inp, par_input, _ = data

    conv1 = Conv2d(m, 1, 1, 3, padding="same", bias=True)
    keep_same = np.array(
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
    add_one = np.array([1])
    conv1.load_weights(keep_same, add_one)

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
    assert np.allclose(out.toDense(), inp + 1)


def test_conv2d_with_same_padding_even_kernel(data):
    # when kernel size is even
    m, w1, b1, inp, par_input, _ = data

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


def test_conv2d_with_same_padding_even_kernel_2(data):
    # when kernel size is odd
    m, w1, b1, inp, par_input, _ = data

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


def test_conv2d_with_padding(data):
    m, w1, b1, inp, par_input, _ = data

    conv_with_valid_padding = Conv2d(m, 1, 2, 3, padding="valid")
    assert conv_with_valid_padding.padding == (0, 0)

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


def test_conv2d_with_stride(data):
    m, w1, b1, inp, par_input, _ = data
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


def test_conv2d_with_padding_and_stride(data):
    m, w1, b1, inp, par_input, _ = data
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


def test_max_pooling(data):
    m, w1, b1, inp, par_input, ii = data

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


def test_pooling_with_bounds(data):
    m, w1, b1, inp, par_input, ii = data
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

    assert np.allclose(
        np.array(out4.records.upper).reshape(out4.shape), exp_ub
    )
    assert np.allclose(
        np.array(out4.records.lower).reshape(out4.shape), exp_lb
    )

    assert np.allclose(
        np.array(out5.records.upper).reshape(out5.shape), exp_ub
    )
    assert np.allclose(
        np.array(out5.records.lower).reshape(out5.shape), exp_lb
    )


def test_mpooling_with_complex_bounds(data):
    m, *_ = data

    max_pool = MaxPool2d(m, (2, 3))
    min_pool = MinPool2d(m, (2, 3))

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

    assert np.allclose(
        np.array(out3.records.lower).reshape(out3.shape), exp_lb_par
    )
    assert np.allclose(
        np.array(out3.records.upper).reshape(out3.shape), exp_ub_par
    )

    assert np.allclose(
        np.array(out4.records.lower).reshape(out4.shape), exp_lb_par
    )
    assert np.allclose(
        np.array(out4.records.upper).reshape(out4.shape), exp_ub_par
    )

    assert np.allclose(
        np.array(out7.records.lower).reshape(out7.shape), exp_lb_var
    )
    assert np.allclose(
        np.array(out7.records.upper).reshape(out7.shape), exp_ub_var
    )

    assert np.allclose(
        np.array(out8.records.lower).reshape(out8.shape), exp_lb_var
    )
    assert np.allclose(
        np.array(out8.records.upper).reshape(out8.shape), exp_ub_var
    )


def test_min_pooling(data):
    m, w1, b1, inp, par_input, ii = data
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


def test_avg_pooling(data):
    m, w1, b1, inp, par_input, ii = data
    ap1 = AvgPool2d(m, 2)
    ap2 = AvgPool2d(m, (2, 1))
    ap3 = AvgPool2d(m, 3, stride=(1, 1))
    ap4 = AvgPool2d(m, 4, stride=(3, 2), padding=2)
    out, eqs = ap1(par_input)
    out2, eqs2 = ap2(par_input)
    out3, eqs3 = ap3(par_input)
    out4, eqs4 = ap4(par_input)
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


def test_avg_pool_bounds_neg(data):
    m, w1, b1, _, par_input, ii = data
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

    # nothing gets scaled
    assert (recs[recs["DenseDim3_1"] == "0"]["lower"] == -0.64615).all()
    assert (recs[recs["DenseDim3_1"] == "0"]["upper"] == 0.90273).all()

    # positive lower bounds must be scaled due to padding
    assert (recs[recs["DenseDim3_1"] == "1"]["lower"] == (0.17423 / 4)).all()
    assert (recs[recs["DenseDim3_1"] == "1"]["upper"] == 0.91273).all()

    # negative upper bounds must be scaled due to padding
    assert (recs[recs["DenseDim3_1"] == "2"]["lower"] == -1).all()
    assert (recs[recs["DenseDim3_1"] == "2"]["upper"] == (-1) / 4).all()

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


def test_pool_call_bad(data):
    m, w1, b1, inp, par_input, ii = data
    avgpool1 = AvgPool2d(m, (2, 2))
    minpool1 = MinPool2d(m, (2, 2))
    maxpool1 = MaxPool2d(m, (2, 2))

    new_par = gp.Parameter(m, "new_par", domain=dim([10]))
    new_var = gp.Variable(m, "new_var", domain=dim([10]))

    for pool in [avgpool1, minpool1, maxpool1]:
        pytest.raises(ValidationError, pool, "asd")
        pytest.raises(ValidationError, pool, 5)
        pytest.raises(ValidationError, pool, new_par)
        pytest.raises(ValidationError, pool, new_var)

    pytest.raises(ValidationError, _MPool2d, "sup", m, (2, 2))

    pytest.raises(ValidationError, minpool1, new_var, "true")
    pytest.raises(ValidationError, maxpool1, new_var, "true")


def test_flatten_bad(data):
    m, w1, b1, inp, par_input, ii = data
    # should only work for parameter or variable
    pytest.raises(ValidationError, flatten_dims, w1, [2, 3])
    pytest.raises(ValidationError, flatten_dims, par_input, [0])  # single dim
    pytest.raises(ValidationError, flatten_dims, par_input, [])  # no dim
    pytest.raises(ValidationError, flatten_dims, par_input, ["a", "b"])
    pytest.raises(ValidationError, flatten_dims, par_input, [-1, 0])
    pytest.raises(ValidationError, flatten_dims, par_input, [5, 6])
    pytest.raises(
        ValidationError, flatten_dims, par_input, [1, 3]
    )  # non consecutive
    pytest.raises(
        ValidationError,
        flatten_dims,
        par_input,
        [0, 1],
        propagate_bounds="True",
    )  # propagate_bounds not bool

    i = gp.Set(m, "i")
    j = gp.Set(m, "j")
    k = gp.Set(m, "k")
    var1 = gp.Variable(m, "var1", domain=[i, j, k])  # j, k not populated yet
    pytest.raises(ValidationError, flatten_dims, var1, [1, 2])


def test_flatten_par(data):
    m, w1, b1, inp, par_input, ii = data
    # 3x1x5x5 -> 3x25
    par_flattened, eqs = flatten_dims(par_input, [1, 2, 3])
    out_data = par_flattened.toDense()
    assert np.allclose(out_data, inp.reshape(3, 25))
    assert eqs == []  # for parameters no equation needed

    # 3x1x5x5 -> 75
    par_flattened, eqs = flatten_dims(par_input, [0, 1, 2, 3])
    out_data = par_flattened.toDense()
    assert np.allclose(out_data, inp.reshape(75))
    assert eqs == []  # for parameters no equation needed

    # 3x1x5x5 -> 3x1x25
    par_flattened, eqs = flatten_dims(par_input, [2, 3])
    out_data = par_flattened.toDense()
    assert np.allclose(out_data, inp.reshape(3, 1, 25))
    assert eqs == []  # for parameters no equation needed

    # test flatten par with copied domain as well
    data = np.random.rand(20, 20, 20, 20)
    par = gp.Parameter(m, "par_iii", domain=[ii, ii, ii, ii], records=data)

    par_flattened, eqs = flatten_dims(par, [2, 3])
    out_data = par_flattened.toDense()
    assert np.allclose(out_data, data.reshape(20, 20, 400))
    assert eqs == []  # for parameters no equation needed


def test_flatten_par_with_no_records(data):
    m, *_ = data

    par = gp.Parameter(m, "par", domain=dim([10, 5]))

    # 10x5 -> 50
    par_flattened, eqs = flatten_dims(par, [0, 1])

    assert par_flattened.toDense() is None
    assert par_flattened.shape == (50,)
    assert eqs == []  # for parameters no equation needed


def test_flatten_var_copied_domain(data):
    m, w1, b1, inp, par_input, ii = data

    a1 = gp.Alias(m, "ii2", alias_with=ii)
    a2 = gp.Alias(m, "ii3", alias_with=ii)
    a3 = gp.Alias(m, "ii4", alias_with=ii)

    var = gp.Variable(
        m,
        "var_ii",
        domain=[ii, ii, ii, ii],
    )
    data = np.random.rand(20, 20, 20, 20)

    fix_var = gp.Parameter(m, "var_ii_data", domain=var.domain, records=data)
    var.fx[ii, a1, a2, a3] = fix_var[ii, a1, a2, a3]
    var_2, eqs = flatten_dims(var, [2, 3])
    var_3, eqs_2 = flatten_dims(var_2, [0, 1])
    var_4, eqs_3 = flatten_dims(var_3, [0, 1])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs, *eqs_2, *eqs_3],
        problem="lp",
        objective=var_4["240"] + 1,
        sense="min",
    )

    model.solve()

    out_data = var_3.toDense()
    assert np.allclose(out_data, data.reshape(400, 400))

    out_data_2 = var_4.toDense()
    assert np.allclose(out_data_2, data.reshape(400 * 400))


def test_flatten_2d_propagate_bounds(data):
    m, *_ = data
    i = gp.Set(m, name="i", records=[f"i{i}" for i in range(1, 41)])
    j = gp.Set(m, name="j", records=[f"j{j}" for j in range(1, 51)])
    var = gp.Variable(m, name="var", domain=[i, j])

    # If the variable is unbounded, the bounds are not propagated even if propagate_bounds is True
    var_1, eqs_1 = flatten_dims(var, [0, 1], propagate_bounds=True)
    var_2, eqs_2 = flatten_dims(var, [0, 1], propagate_bounds=False)
    assert var_1.records == var_2.records

    # If the variable is bounded, the bounds are propagated
    bound_up = np.random.rand(40, 50) * 5
    bound_lo = np.random.rand(40, 50) * -5
    upper = gp.Parameter(m, name="upper", domain=[i, j], records=bound_up)
    lower = gp.Parameter(m, name="lower", domain=[i, j], records=bound_lo)
    var.up[...] = upper[...]
    var.lo[...] = lower[...]

    var_3, eqs_3 = flatten_dims(var, [0, 1])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs_1, *eqs_2, *eqs_3],
        problem="lp",
        objective=var_3["240"] + 1,
        sense="min",
    )

    model.solve()

    assert np.allclose(
        np.array(var_3.records.lower.tolist()), bound_lo.reshape(2000)
    )
    assert np.allclose(
        np.array(var_3.records.upper.tolist()), bound_up.reshape(2000)
    )


def test_flatten_3d_propagate_bounds(data):
    m, *_ = data
    i = gp.Set(m, name="i", records=[f"i{i}" for i in range(1, 41)])
    j = gp.Set(m, name="j", records=[f"j{j}" for j in range(1, 51)])
    k = gp.Set(m, name="k", records=[f"k{k}" for k in range(1, 21)])
    var = gp.Variable(m, name="var", domain=[i, j, k])
    bounds_set = gp.Set(m, name="bounds_set", records=["lb", "ub"])

    # If the variable is unbounded, the bounds are not propagated even if propagate_bounds is True
    var_1, eqs_1 = flatten_dims(var, [0, 1], propagate_bounds=True)
    var_2, eqs_2 = flatten_dims(var, [1, 2], propagate_bounds=True)
    assert (var_1.records is None) and (var_2.records is None)
    assert var_1.shape == (2000, 20) and var_2.shape == (40, 1000)

    # If the variable is bounded, the bounds are propagated
    bound_up = np.random.rand(40, 50, 20) * 5
    bound_lo = np.random.rand(40, 50, 20) * -5
    all_bounds = np.stack([bound_lo, bound_up], axis=0)

    bounds = gp.Parameter(
        m, name="bounds", domain=[bounds_set, i, j, k], records=all_bounds
    )

    var.up[...] = bounds[("ub", i, j, k)]
    var.lo[...] = bounds[("lb", i, j, k)]

    var_3, eqs_3 = flatten_dims(var, [0, 1])
    var_4, eqs_4 = flatten_dims(var, [1, 2])
    var_5, eqs_5 = flatten_dims(var, [0, 1, 2])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs_1, *eqs_2, *eqs_3, *eqs_4, *eqs_5],
        problem="lp",
        objective=var_3["140", "k4"] + 1,
        sense="min",
    )

    model.solve()

    var_3_bounds = gp.Parameter(
        m, name="var_3_bounds", domain=[bounds_set, *var_3.domain]
    )
    var_3_bounds[("lb",) + tuple(var_3.domain)] = var_3.lo[...]
    var_3_bounds[("ub",) + tuple(var_3.domain)] = var_3.up[...]

    var_4_bounds = gp.Parameter(
        m, name="var_4_bounds", domain=[bounds_set, *var_4.domain]
    )
    var_4_bounds[("lb",) + tuple(var_4.domain)] = var_4.lo[...]
    var_4_bounds[("ub",) + tuple(var_4.domain)] = var_4.up[...]

    var_5_bounds = gp.Parameter(
        m, name="var_5_bounds", domain=[bounds_set, *var_5.domain]
    )
    var_5_bounds[("lb",) + tuple(var_5.domain)] = var_5.lo[...]
    var_5_bounds[("ub",) + tuple(var_5.domain)] = var_5.up[...]

    assert np.allclose(var_3_bounds.toDense(), all_bounds.reshape(2, 2000, 20))
    assert np.allclose(var_4_bounds.toDense(), all_bounds.reshape(2, 40, 1000))
    assert np.allclose(var_5_bounds.toDense(), all_bounds.reshape(2, 40000))


def test_flatten_propagate_zero_bounds(data):
    m, *_ = data
    var = gp.Variable(m, name="var1", domain=dim([30, 40, 10, 5]))

    var.up[...] = 0
    var.lo[...] = 0

    var_1, eqs_1 = flatten_dims(var, [0, 1])
    var_2, eqs_2 = flatten_dims(var, [0, 1, 2, 3])
    var_3, eqs_3 = flatten_dims(var_1, [1, 2])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs_1, *eqs_2, *eqs_3],
        problem="lp",
        objective=5 * var_1["0", "0", "0"] + 1,
        sense="min",
    )

    model.solve()

    expected_bounds = np.zeros([30, 40, 10, 5])

    # Because the bounds are zero, there are no way, currently, to represent them as an array
    assert np.allclose(
        np.array(var_1.records.upper).reshape(var_1.shape),
        expected_bounds.reshape(var_1.shape),
    )
    assert np.allclose(
        np.array(var_1.records.lower).reshape(var_1.shape),
        expected_bounds.reshape(var_1.shape),
    )

    assert np.allclose(
        np.array(var_2.records.upper).reshape(var_2.shape),
        expected_bounds.reshape(var_2.shape),
    )
    assert np.allclose(
        np.array(var_2.records.lower).reshape(var_2.shape),
        expected_bounds.reshape(var_2.shape),
    )

    assert np.allclose(
        np.array(var_3.records.upper).reshape(var_3.shape),
        expected_bounds.reshape(var_3.shape),
    )
    assert np.allclose(
        np.array(var_3.records.lower).reshape(var_3.shape),
        expected_bounds.reshape(var_3.shape),
    )


def test_flatten_more_complex_propagate_bounds(data):
    m, *_ = data
    var = gp.Variable(m, name="var", domain=dim([2, 4, 5]))
    bounds_set = gp.Set(m, name="bounds_set", records=["lb", "ub"])

    bound_up = np.array(
        [
            [
                [1.6873254, np.inf, 4.64399079, np.inf, 0.85146007],
                [4.31392932, 1.99165668, 4.19013802, 3.77449253, 0],
                [np.inf, 4.13450595, 4.25880061, 1.529363, 2.54171194],
                [1.79348688, 2.04002383, 0.19198094, 4.14445882, 4.72650868],
            ],
            [
                [1.54070398, np.inf, 0, 3.55077501, 2.12700496],
                [0.13939228, 1.10668786, 0.23710837, 3.61857607, 1.64761417],
                [1.80097419, 0.89434166, 1.46039526, 1.31960681, np.inf],
                [2.50636193, 1.3920737, np.inf, 3.35616509, 4.98534911],
            ],
        ]
    )

    bound_lo = -bound_up
    all_bounds = np.stack([bound_lo, bound_up], axis=0)

    bounds = gp.Parameter(
        m, name="bounds", domain=[bounds_set, *var.domain], records=all_bounds
    )

    var.up[...] = bounds[("ub",) + tuple(var.domain)]
    var.lo[...] = bounds[("lb",) + tuple(var.domain)]

    var_1, eqs_1 = flatten_dims(var, [0, 1])

    model = gp.Model(
        m,
        "flatten_everything",
        equations=[*eqs_1],
        problem="lp",
        objective=5 * var_1["5", "3"] + 1,
        sense="min",
    )

    model.solve()

    var_1_bounds = gp.Parameter(
        m, name="var_1_bounds", domain=[bounds_set, *var_1.domain]
    )
    var_1_bounds[("lb",) + tuple(var_1.domain)] = var_1.lo[...]
    var_1_bounds[("ub",) + tuple(var_1.domain)] = var_1.up[...]

    assert np.allclose(var_1_bounds.toDense(), all_bounds.reshape(2, 8, 5))


def test_linear_bad_init(data):
    m, *_ = data
    # in feature must be integer
    pytest.raises(ValidationError, Linear, m, 2.5, 4, False)
    pytest.raises(ValidationError, Linear, m, "2", 4, True)
    # out feature must be integer
    pytest.raises(ValidationError, Linear, m, 2, 4.1, True)
    pytest.raises(ValidationError, Linear, m, 2, "4.1", False)
    # in feature must be positive
    pytest.raises(ValidationError, Linear, m, -4, 4, False)
    # out feature must be positive
    pytest.raises(ValidationError, Linear, m, 4, -4, False)
    # bias must be a bool
    pytest.raises(ValidationError, Linear, m, 4, 4, bias=10)


def test_linear_load_weights(data):
    m, *_ = data
    lin1 = Linear(m, 1, 2, bias=True)
    lin2 = Linear(m, 1, 2, bias=False)

    w1 = np.random.rand(2, 1)
    b1 = np.random.rand(2)

    # needs bias as well
    pytest.raises(ValidationError, lin1.load_weights, w1)

    # conv2 does not have bias
    pytest.raises(ValidationError, lin2.load_weights, w1, b1)

    # test bad shape
    bad1 = np.random.rand(1)
    pytest.raises(ValidationError, lin1.load_weights, bad1, b1)
    pytest.raises(ValidationError, lin1.load_weights, w1, bad1)

    bad2 = np.random.rand(2, 2)
    pytest.raises(ValidationError, lin1.load_weights, bad2, b1)

    bad3 = np.random.rand(6)
    pytest.raises(ValidationError, lin1.load_weights, w1, bad3)

    bad4 = np.random.rand(6, 2)
    pytest.raises(ValidationError, lin1.load_weights, w1, bad4)


def test_linear_same_indices(data):
    m, *_ = data
    lin1 = Linear(m, 4, 4, bias=True)
    w1 = np.random.rand(4, 4)
    b1 = np.random.rand(4)
    lin1.load_weights(w1, b1)
    inp = gp.Variable(m, domain=dim([4, 4, 4, 4]))
    out, eqs = lin1(inp)
    lin2 = Linear(m, 4, 4, bias=True)
    lin2.load_weights(w1, b1)
    out2, eqs2 = lin2(inp)


def test_linear_reloading_weights(data):
    m, *_ = data
    lin1 = Linear(m, 1, 2, bias=True)
    w1 = np.random.rand(2, 1)
    b1 = np.random.rand(2)
    lin1.load_weights(w1, b1)

    w1 = np.ones((2, 1))
    b1 = np.ones(2)
    lin1.load_weights(w1, b1)

    assert np.allclose(w1, lin1.weight.toDense())
    assert np.allclose(b1, lin1.bias.toDense())


def test_linear_make_variable(data):
    m, *_ = data
    lin1 = Linear(m, 4, 2)
    lin1.make_variable()
    w1 = np.random.rand(2, 4)
    b1 = np.random.rand(2)
    pytest.raises(ValidationError, lin1.load_weights, w1, b1)
    assert isinstance(lin1.weight, gp.Variable)
    assert isinstance(lin1.bias, gp.Variable)
    inp = gp.Variable(m, domain=dim([4, 1, 2, 4]))
    out, eqs = lin1(inp)
    assert len(out.domain) == 4
    assert len(set([x.name for x in out.domain])) == 4


def test_linear_load_weight_make_var(data):
    m, *_ = data
    lin1 = Linear(m, 1, 2, bias=True)
    w1 = np.random.rand(2, 1)
    b1 = np.random.rand(2)
    lin1.load_weights(w1, b1)
    assert isinstance(lin1.weight, gp.Parameter)
    assert isinstance(lin1.bias, gp.Parameter)
    pytest.raises(ValidationError, lin1.make_variable)


def test_linear_call_bad(data):
    m, *_ = data
    lin1 = Linear(m, 4, 4, bias=True)
    inp = gp.Variable(m, domain=dim([4, 4, 4, 4]))
    # requires initialization before
    pytest.raises(ValidationError, lin1, inp)

    w1 = np.random.rand(4, 4)
    b1 = np.random.rand(4)
    lin1.load_weights(w1, b1)

    # needs at least 1 dim
    bad_inp = gp.Variable(m, domain=[])
    pytest.raises(ValidationError, lin1, bad_inp)

    # in channel must match 4
    bad_inp_2 = gp.Variable(m, domain=dim([10, 3, 4, 5]))
    pytest.raises(ValidationError, lin1, bad_inp_2)


def test_linear_simple_correctness(data):
    m, _, _, _, par_input, _ = data
    lin1 = Linear(m, 5, 128, bias=True)
    lin2 = Linear(m, 128, 64, bias=False)
    w1 = np.random.rand(128, 5)
    b1 = np.random.rand(128)
    w2 = np.random.rand(64, 128)

    lin1.load_weights(w1, b1)
    lin2.load_weights(w2)

    out1, eqs1 = lin1(par_input)
    out2, eqs2 = lin2(out1)

    obj = gp.Sum(out2.domain, out2)

    model = gp.Model(
        m,
        "affine",
        equations=[*eqs1, *eqs2],
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()

    expected_out = (par_input.toDense() @ w1.T + b1) @ w2.T
    assert np.allclose(out2.toDense(), expected_out)


def test_linear_bias_domain_conflict(data):
    m, *_ = data
    lin1 = Linear(m, 20, 30, bias=True)
    w1 = np.random.rand(30, 20)
    b1 = np.random.rand(30)
    lin1.load_weights(w1, b1)

    input_data = np.random.rand(30, 20, 30, 20)
    par_input = gp.Parameter(
        m, domain=dim([30, 20, 30, 20]), records=input_data
    )
    out1, eqs1 = lin1(par_input)

    last_domain = out1.domain[-1].name
    # get rhs of equality
    definition = eqs1[0].getDefinition().split("=e=")[1]
    # 1 from weight 1 from bias
    assert definition.count(last_domain) == 2

    obj = gp.Sum(out1.domain, out1)

    model = gp.Model(
        m,
        "affine2",
        equations=eqs1,
        objective=obj,
        sense="min",
        problem="LP",
    )
    model.solve()
    expected_out = input_data @ w1.T + b1
    assert np.allclose(out1.toDense(), expected_out)


def test_linear_propagate_bounds_non_boolean(data):
    m, *_ = data
    lin1 = Linear(m, 20, 30, bias=True)
    w1 = np.random.rand(30, 20)
    b1 = np.random.rand(30)
    lin1.load_weights(w1, b1)

    par_input = gp.Parameter(m, domain=dim([30, 20, 30, 20]))
    pytest.raises(ValidationError, lin1, par_input, "True")


def test_linear_propagate_bounded_input(data):
    m, *_ = data
    lin1 = Linear(m, 4, 3)
    w1 = np.random.rand(3, 4)
    b1 = np.random.rand(3)
    lin1.load_weights(w1, b1)

    # bounds for the input
    xlb = np.random.randint(-5, 1, (2, 4))
    xub = np.random.randint(1, 5, (2, 4))

    x_lb = gp.Parameter(m, "x_lb", domain=dim([2, 4]), records=xlb)
    x_ub = gp.Parameter(m, "x_ub", domain=dim([2, 4]), records=xub)

    x = gp.Variable(m, "x", domain=dim([2, 4]))
    x.up[...] = x_ub[...]
    x.lo[...] = x_lb[...]

    out1, _ = lin1(x)

    wpos = np.maximum(w1, 0)
    wneg = np.minimum(w1, 0)

    expected_lb = np.dot(xlb, wpos.T) + np.dot(xub, wneg.T) + b1
    expected_ub = np.dot(xub, wpos.T) + np.dot(xlb, wneg.T) + b1

    # check if the bounds are propagated correctly
    assert np.allclose(
        np.array(out1.lo.records.lower).reshape(2, 3), expected_lb
    )
    assert np.allclose(
        np.array(out1.up.records.upper).reshape(2, 3), expected_ub
    )


def test_linear_propagate_unbounded_input(data):
    m, *_ = data
    lin1 = Linear(m, 20, 30, bias=True)
    w1 = np.random.rand(30, 20)
    b1 = np.random.rand(30)
    lin1.load_weights(w1, b1)

    x = gp.Variable(m, "x", domain=dim([30, 20, 30, 20]))

    out1, _ = lin1(x)

    # Data to bounds is not added; which means its unbounded
    assert out1.lo.records is None
    assert out1.up.records is None


def test_linear_propagate_partially_bounded_input(data):
    m, *_ = data
    lin1 = Linear(m, 4, 3, bias=False)
    w1 = np.array([[-3, 2, 1, 0], [1, -1, 0, 1], [0, 1, -1, 3]])
    lin1.load_weights(w1)

    x = gp.Variable(m, "x", domain=dim([2, 4]))

    xlb = np.array([[-4, -np.inf, -5, 0], [-1, -2, -1, -4]])
    xub = np.array([[4, 5, np.inf, 0], [np.inf, 1, 2, 1]])

    x_lb = gp.Parameter(m, "x_lb", domain=dim([2, 4]), records=xlb)
    x_ub = gp.Parameter(m, "x_ub", domain=dim([2, 4]), records=xub)

    x.up[...] = x_ub[...]
    x.lo[...] = x_lb[...]

    out1, _ = lin1(x)

    out1_lb = gp.Parameter(m, "out1_lb", domain=dim([2, 3]))
    out1_ub = gp.Parameter(m, "out1_ub", domain=dim([2, 3]))

    out1_lb[...] = out1.lo[...]
    out1_ub[...] = out1.up[...]

    expected_lb = np.array([[-np.inf, -9, -np.inf], [-np.inf, -6, -16]])

    expected_ub = np.array([[np.inf, np.inf, 10], [7, np.inf, 5]])

    # check if the bounds are propagated correctly
    assert np.allclose(out1_lb.toDense(), expected_lb)
    assert np.allclose(out1_ub.toDense(), expected_ub)


def test_linear_propagate_unbounded_input_with_zero_weight(data):
    m, *_ = data
    lin1 = Linear(m, 20, 30, bias=False)
    w1 = np.zeros((30, 20))
    lin1.load_weights(w1)

    x = gp.Variable(m, "x", domain=dim([30, 20, 30, 20]))

    out1, _ = lin1(x)

    out1_ub = np.array(out1.up.records.upper).reshape(30, 20, 30, 30)
    out1_lb = np.array(out1.lo.records.lower).reshape(30, 20, 30, 30)

    expected_bounds = np.zeros((30, 20, 30, 30))

    # check if the bounds are zeros, since the weights are all zeros
    assert np.allclose(out1_ub, expected_bounds)
    assert np.allclose(out1_lb, expected_bounds)


def test_linear_propagate_zero_bounds(data):
    m, *_ = data
    lin1 = Linear(m, 4, 3, bias=False)
    w1 = np.random.rand(3, 4)
    lin1.load_weights(w1)

    x = gp.Variable(m, "x", domain=dim([2, 4]))

    x.up[...] = 0
    x.lo[...] = 0

    out1, _ = lin1(x)

    expected_bounds = np.zeros((2, 3))

    assert np.allclose(
        np.array(out1.records.upper).reshape(out1.shape), expected_bounds
    )
    assert np.allclose(
        np.array(out1.records.lower).reshape(out1.shape), expected_bounds
    )

    lin2 = Linear(m, 4, 3, bias=True)
    b1 = np.random.rand(3)
    lin2.load_weights(w1, b1)

    out2, _ = lin2(x)

    expected_bounds = np.stack((b1, b1))

    assert np.allclose(
        np.array(out2.records.upper).reshape(out1.shape), expected_bounds
    )
    assert np.allclose(
        np.array(out2.records.lower).reshape(out1.shape), expected_bounds
    )
