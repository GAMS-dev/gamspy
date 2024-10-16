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

    conv1 = Conv2d(m, 1, 2, 3, padding="same", bias=True)
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
                        1.732664325,
                        2.776298639,
                        2.724456784,
                        3.010900060,
                        1.891354627,
                    ],
                    [
                        2.005268388,
                        3.408175819,
                        3.317247105,
                        3.453737875,
                        2.364333495,
                    ],
                    [
                        2.040222406,
                        3.560330422,
                        2.985729559,
                        3.781994916,
                        2.160537378,
                    ],
                    [
                        2.044953839,
                        3.375299180,
                        3.473348372,
                        3.706098438,
                        2.266124998,
                    ],
                    [
                        1.519405918,
                        2.384175689,
                        2.420725000,
                        3.123493757,
                        1.933974614,
                    ],
                ],
                [
                    [
                        1.287152749,
                        2.290892822,
                        2.133846258,
                        2.819633428,
                        1.573196728,
                    ],
                    [
                        1.964626780,
                        3.488934532,
                        3.789320574,
                        4.021046498,
                        2.845867257,
                    ],
                    [
                        1.838588271,
                        3.470196980,
                        3.484252604,
                        4.216850947,
                        2.493555345,
                    ],
                    [
                        1.895314276,
                        3.451094504,
                        3.586500102,
                        3.970440842,
                        2.974429236,
                    ],
                    [
                        1.704901713,
                        2.770929272,
                        2.976872232,
                        3.295421499,
                        2.293567862,
                    ],
                ],
            ],
            [
                [
                    [
                        2.341738148,
                        3.218714895,
                        3.306629305,
                        2.387722676,
                        1.594560209,
                    ],
                    [
                        2.877775509,
                        4.110791767,
                        4.509485562,
                        3.714627315,
                        2.050956027,
                    ],
                    [
                        2.585403650,
                        3.693094370,
                        4.219910987,
                        3.473907137,
                        2.537436917,
                    ],
                    [
                        2.524971115,
                        3.553348249,
                        3.246122327,
                        2.602895969,
                        1.624781530,
                    ],
                    [
                        1.433171173,
                        2.469392368,
                        2.317977477,
                        1.850996417,
                        0.764438783,
                    ],
                ],
                [
                    [
                        1.557417154,
                        2.556111826,
                        2.958350329,
                        2.252276589,
                        1.928931807,
                    ],
                    [
                        2.528189505,
                        4.196171924,
                        4.897754836,
                        4.123600711,
                        2.816458856,
                    ],
                    [
                        2.345084305,
                        4.284388000,
                        4.689310706,
                        3.955014285,
                        2.590885972,
                    ],
                    [
                        2.295739686,
                        3.608707512,
                        4.343994357,
                        4.020127679,
                        2.406344089,
                    ],
                    [
                        1.728266375,
                        2.806608000,
                        2.536085562,
                        1.976999438,
                        1.053355860,
                    ],
                ],
            ],
            [
                [
                    [
                        2.177365798,
                        2.867050143,
                        2.676692925,
                        2.589579753,
                        1.621487713,
                    ],
                    [
                        2.867276244,
                        3.860664336,
                        3.342714492,
                        3.502027904,
                        2.372097181,
                    ],
                    [
                        2.223854972,
                        3.195463170,
                        3.146128689,
                        2.995503826,
                        2.028202825,
                    ],
                    [
                        2.195717997,
                        2.304792968,
                        2.358695988,
                        1.891748201,
                        1.608892624,
                    ],
                    [
                        1.302500463,
                        1.820278447,
                        1.701674084,
                        1.701619244,
                        1.136078616,
                    ],
                ],
                [
                    [
                        1.429691761,
                        2.658491139,
                        2.330681169,
                        2.074377272,
                        1.850937859,
                    ],
                    [
                        2.519388091,
                        3.844189460,
                        3.867576299,
                        3.505568722,
                        2.601530262,
                    ],
                    [
                        2.419334857,
                        3.998219232,
                        3.569506526,
                        3.486621082,
                        2.659753124,
                    ],
                    [
                        2.043669914,
                        2.818665209,
                        3.162802884,
                        2.246840350,
                        2.250522835,
                    ],
                    [
                        1.470834907,
                        1.982134885,
                        1.610279229,
                        1.706438992,
                        1.402568837,
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

    for recs in [out.records, out2.records, out3.records]:
        assert (recs[recs["DenseDim3_1"] == "0"]["lower"] == 10).all()
        assert (recs[recs["DenseDim3_1"] == "0"]["upper"] == 20).all()

        assert (recs[recs["DenseDim3_1"] == "1"]["lower"] == 1).all()
        assert (recs[recs["DenseDim3_1"] == "1"]["upper"] == 100).all()

        assert (recs[recs["DenseDim3_1"] == "2"]["lower"] == -50).all()
        assert (recs[recs["DenseDim3_1"] == "2"]["upper"] == 50).all()


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
    new_var = gp.Parameter(m, "new_var", domain=dim([10]))

    for pool in [avgpool1, minpool1, maxpool1]:
        pytest.raises(ValidationError, pool, "asd")
        pytest.raises(ValidationError, pool, 5)
        pytest.raises(ValidationError, pool, new_par)
        pytest.raises(ValidationError, pool, new_var)

    pytest.raises(ValidationError, _MPool2d, "sup", m, (2, 2))


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
