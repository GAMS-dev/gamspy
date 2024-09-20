from __future__ import annotations

import gamspy as gp
import numpy as np
import pytest
from gamspy import Container
from gamspy.exceptions import ValidationError
from gamspy.formulations.nn import Conv2d
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

    yield m, w1, b1, inp
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


def test_same_indices(data):
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


def test_reloading_weights(data):
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


def test_make_variable(data):
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


def test_load_weight_make_var(data):
    m, *_ = data
    conv1 = Conv2d(m, in_channels=1, out_channels=2, kernel_size=3, bias=True)
    w1 = np.random.rand(2, 1, 3, 3)
    b1 = np.random.rand(2)
    conv1.load_weights(w1, b1)
    assert isinstance(conv1.weight, gp.Parameter)
    assert isinstance(conv1.bias, gp.Parameter)
    pytest.raises(ValidationError, conv1.make_variable)


def test_call_bad(data):
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


def test_simple_correctness(data):
    m, w1, b1, inp = data
    conv1 = Conv2d(m, 1, 2, 3)
    conv1.load_weights(w1, b1)
    par_input = gp.Parameter(m, domain=dim(inp.shape), records=inp)
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


def test_with_padding(data):
    m, w1, b1, inp = data
    conv1 = Conv2d(m, 1, 2, 3, padding=(2, 1))
    conv1.load_weights(w1, b1)
    par_input = gp.Parameter(m, domain=dim(inp.shape), records=inp)
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


def test_with_stride(data):
    m, w1, b1, inp = data
    conv1 = Conv2d(m, 1, 2, 3, stride=(2, 1))
    conv1.load_weights(w1, b1)
    par_input = gp.Parameter(m, domain=dim(inp.shape), records=inp)
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


def test_with_padding_and_stride(data):
    m, w1, b1, inp = data
    conv1 = Conv2d(m, 1, 2, 3, stride=(2, 1), padding=(1, 2))
    conv1.load_weights(w1, b1)
    par_input = gp.Parameter(m, domain=dim(inp.shape), records=inp)
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
