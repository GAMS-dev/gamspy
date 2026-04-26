import numpy as np
import pytest

import gamspy as gp
from gamspy import Container
from gamspy.math import dim


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

    inp2 = inp[:, :, 0, :]

    par_input = gp.Parameter(m, domain=dim(inp.shape), records=inp)
    par_input_2 = gp.Parameter(m, domain=dim(inp2.shape), records=inp2)
    ii = gp.Set(m, "ii", records=range(20))

    yield m, w1, b1, inp, par_input, ii, par_input_2

    m.close()


@pytest.fixture
def rnn_data():
    m = Container()

    w_ih = np.random.rand(3, 4)
    w_hh = np.random.rand(3, 3)
    b_ih = np.random.rand(3)
    b_hh = np.random.rand(3)

    yield m, w_ih, w_hh, b_ih, b_hh
    m.close()


@pytest.fixture
def calculate_expected_rnn_bounds():
    def _calculator(
        x_lb: np.ndarray,
        x_ub: np.ndarray,
        w_ih: np.ndarray,
        w_hh: np.ndarray,
        b_ih: np.ndarray,
        b_hh: np.ndarray,
        activation: str = "tanh",
        h0: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        batch_size, time_steps, _ = x_lb.shape
        hidden_size = w_hh.shape[0]

        w_ih_pos = np.maximum(w_ih, 0)
        w_ih_neg = np.minimum(w_ih, 0)
        w_hh_pos = np.maximum(w_hh, 0)
        w_hh_neg = np.minimum(w_hh, 0)

        pre_act_lb_seq, pre_act_ub_seq = [], []
        h_lb_seq, h_ub_seq = [], []

        if h0 is not None:
            prev_lb = h0.copy()
            prev_ub = h0.copy()
        else:
            prev_lb = np.zeros((batch_size, hidden_size))
            prev_ub = np.zeros((batch_size, hidden_size))

        for t in range(time_steps):
            x_lb_t = x_lb[:, t, :]
            x_ub_t = x_ub[:, t, :]

            in_lb = (x_lb_t @ w_ih_pos.T) + (x_ub_t @ w_ih_neg.T) + b_ih
            in_ub = (x_ub_t @ w_ih_pos.T) + (x_lb_t @ w_ih_neg.T) + b_ih

            if t == 0 and h0 is None:
                hid_lb = b_hh
                hid_ub = b_hh
            else:
                hid_lb = (prev_lb @ w_hh_pos.T) + (prev_ub @ w_hh_neg.T) + b_hh
                hid_ub = (prev_ub @ w_hh_pos.T) + (prev_lb @ w_hh_neg.T) + b_hh

            pre_lb = in_lb + hid_lb
            pre_ub = in_ub + hid_ub

            if activation == "relu":
                pre_act_lb_seq.append(pre_lb)
                pre_act_ub_seq.append(pre_ub)
                h_lb = np.maximum(0, pre_lb)
                h_ub = np.maximum(0, pre_ub)
            elif activation == "tanh":
                h_lb = np.tanh(pre_lb)
                h_ub = np.tanh(pre_ub)
            else:  # linear
                h_lb = pre_lb
                h_ub = pre_ub

            h_lb_seq.append(h_lb)
            h_ub_seq.append(h_ub)

            prev_lb = h_lb
            prev_ub = h_ub

        if activation == "relu":
            pre_act_bounds = np.stack(
                [np.stack(pre_act_lb_seq, axis=1), np.stack(pre_act_ub_seq, axis=1)],
                axis=0,
            )
        else:
            pre_act_bounds = None

        output_bounds = np.stack(
            [np.stack(h_lb_seq, axis=1), np.stack(h_ub_seq, axis=1)], axis=0
        )

        return pre_act_bounds, output_bounds

    return _calculator


@pytest.fixture
def gru_data():
    m = Container()

    w_ih = np.random.rand(3 * 3, 4)
    w_hh = np.random.rand(3 * 3, 3)
    b_ih = np.random.rand(3 * 3)
    b_hh = np.random.rand(3 * 3)

    yield m, w_ih, w_hh, b_ih, b_hh
    m.close()
