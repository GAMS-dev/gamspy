from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gamspy import Parameter

import numpy as np

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations.result import FormulationResult


class GRU:
    """
    Formulation generator for Gated Recurrent Units (GRU) in GAMSPy.It can
    be used to embed trained Gated Recurrent Units in your problem.

    Note: It currently does **NOT** support Bidirectional RNNs and Dropout layers.

    Parameters
    ----------
    container : Container
        Container that will hold the new variables and equations.
    input_size : int
        The number of expected features in the input sequence.
    hidden_size : int
        The number of features in the hidden state.
    """

    def __init__(
        self,
        container: gp.Container,
        input_size: int,
        hidden_size: int,
    ):
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValidationError("input_size must be a positive integer")

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValidationError("hidden_size must be a positive integer")

        self.container = container
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._state = 0

        self.w_ih: dict[str, Parameter] = {}
        self.w_hh: dict[str, Parameter] = {}
        self.b_ih: dict[str, Parameter] = {}
        self.b_hh: dict[str, Parameter] = {}

    def load_weights(
        self,
        weight_ih: np.ndarray,
        weight_hh: np.ndarray,
        bias_ih: np.ndarray | None = None,
        bias_hh: np.ndarray | None = None,
    ) -> None:
        """
        Mark GRU as parameter and load weights from NumPy arrays.
        Follows the standard PyTorch packing layout: (3 * hidden_size, ...),
        where the 3 chunks correspond to the reset (r), update (z), and new (n) gates.
        """
        expected_ih_shape = (3 * self.hidden_size, self.input_size)
        expected_hh_shape = (3 * self.hidden_size, self.hidden_size)

        if weight_ih.shape != expected_ih_shape:
            raise ValidationError(
                f"weight_ih shape mismatch: expected {expected_ih_shape}"
            )
        if weight_hh.shape != expected_hh_shape:
            raise ValidationError(
                f"weight_hh shape mismatch: expected {expected_hh_shape}"
            )

        if bias_ih is None:
            bias_ih = np.zeros(3 * self.hidden_size)
        elif bias_ih.shape != (3 * self.hidden_size,):
            raise ValidationError(
                f"bias_ih shape mismatch: expected {(3 * self.hidden_size,)}"
            )

        if bias_hh is None:
            bias_hh = np.zeros(3 * self.hidden_size)
        elif bias_hh.shape != (3 * self.hidden_size,):
            raise ValidationError(
                f"bias_hh shape mismatch: expected {(3 * self.hidden_size,)}"
            )

        H = self.hidden_size
        gates = ["r", "z", "n"]
        H_set, I_set = gp.math._generate_dims(
            self.container, dims=[self.hidden_size, self.input_size]
        )
        H_prev = gp.Alias(
            self.container,
            alias_with=H_set,
        )

        for i, gate in enumerate(gates):
            start, end = i * H, (i + 1) * H
            self.w_ih[gate] = gp.Parameter(
                self.container,
                domain=[H_set, I_set],
                records=weight_ih[start:end, :],
            )
            self.w_hh[gate] = gp.Parameter(
                self.container,
                domain=[H_set, H_prev],
                records=weight_hh[start:end, :],
            )
            self.b_ih[gate] = gp.Parameter(
                self.container,
                domain=H_set,
                records=bias_ih[start:end],
            )
            self.b_hh[gate] = gp.Parameter(
                self.container,
                domain=H_set,
                records=bias_hh[start:end],
            )

        self._state = 1

    def __call__(
        self,
        input_seq: gp.Parameter | gp.Variable,
        h0: gp.Parameter | None = None,
    ) -> FormulationResult:
        """
        Forward pass your input sequence, generating the output hidden states and
        equations required for calculating the gated recurrent units steps over time.

        Returns `FormulationResult` which can be unpacked as a output variable and list of equations.

        FormulationResult:
            - equations_created: ["reset_gate", "update_gate", "new_gate", "set_output"]
            - variables_created: ["r_gate", "z_gate", "n_gate", "output"]
            - parameters_created: ["w_ih_r", "w_ih_z", "w_ih_n", "w_hh_r", "w_hh_z", "w_hh_n", "b_ih_r", "b_ih_z", "b_ih_n", "b_hh_r", "b_hh_z", "b_hh_n"]

        Note:
            - The `output` variable will have the domain (batch, time_steps, hidden_size).
            - For backward compatibility, this result object can be unpacked as a tuple:
              `output, equations = rnn_layer(input_seq)`.

        Parameters
        ----------
        input_seq : gp.Parameter | gp.Variable
            Input sequence to the GRU layer. It must be a 3D symbol of the following
            shape (batch_size, time_steps, input_features).
        h0 : gp.Parameter | None
            Initial hidden state for the first time step. If None, the initial hidden
            state is assumed to be a vector of zeros. By default None.
            Shape: (batch, hidden_size)

        Returns
        -------
        FormulationResult
        """
        if self._state != 1:
            raise ValidationError("Call load_weights before generating formulation.")

        if len(input_seq.domain) != 3:
            raise ValidationError(
                f"Expected 3D input (batch, time, feature), got {len(input_seq.domain)}"
            )

        N_set, T_set, I_set = input_seq.domain
        if len(I_set) != self.input_size:
            raise ValidationError(
                f"Last dimension of Input sequence does not match. Expected {self.input_size}, got {len(I_set)}."
            )

        lin_in_r = input_seq @ self.w_ih["r"].t() + self.b_ih["r"]
        lin_in_z = input_seq @ self.w_ih["z"].t() + self.b_ih["z"]
        lin_in_n = input_seq @ self.w_ih["n"].t() + self.b_ih["n"]

        H_set = lin_in_r.domain[-1]
        _, H_prev = self.w_hh["r"].domain
        out_domain = [N_set, T_set, H_set]

        r = gp.Variable(self.container, domain=out_domain)
        z = gp.Variable(self.container, domain=out_domain)
        n = gp.Variable(self.container, domain=out_domain)
        h_out = gp.Variable(self.container, domain=out_domain)

        h_prev_H_term: Any
        if len(T_set) == 1:
            if h0 is not None:
                hid_r = (
                    gp.Sum(H_prev, h0[N_set, H_prev] * self.w_hh["r"][H_set, H_prev])
                    + self.b_hh["r"]
                )
                hid_z = (
                    gp.Sum(H_prev, h0[N_set, H_prev] * self.w_hh["z"][H_set, H_prev])
                    + self.b_hh["z"]
                )
                hid_n = (
                    gp.Sum(H_prev, h0[N_set, H_prev] * self.w_hh["n"][H_set, H_prev])
                    + self.b_hh["n"]
                )
                h_prev_H_term = h0[N_set, H_set]
            else:
                hid_r = self.b_hh["r"]
                hid_z = self.b_hh["z"]
                hid_n = self.b_hh["n"]
                h_prev_H_term = 0
        else:
            h_prev_term = h_out[N_set, T_set.lag(1), H_prev]
            h_prev_H_term = h_out[N_set, T_set.lag(1), H_set]

            if h0 is not None:
                h_prev_term = h_prev_term + h0[N_set, H_prev].where[gp.Ord(T_set) == 1]
                h_prev_H_term = (
                    h_prev_H_term + h0[N_set, H_set].where[gp.Ord(T_set) == 1]
                )

            hid_r = (
                gp.Sum(H_prev, h_prev_term * self.w_hh["r"][H_set, H_prev])
                + self.b_hh["r"]
            )
            hid_z = (
                gp.Sum(H_prev, h_prev_term * self.w_hh["z"][H_set, H_prev])
                + self.b_hh["z"]
            )
            hid_n = (
                gp.Sum(H_prev, h_prev_term * self.w_hh["n"][H_set, H_prev])
                + self.b_hh["n"]
            )

        # Equations
        eqs = {}

        # r_t
        eqs["reset_gate"] = gp.Equation(self.container, domain=out_domain)
        eqs["reset_gate"][...] = r == 1 / (1 + gp.math.exp(-(lin_in_r + hid_r)))

        # z_t
        eqs["update_gate"] = gp.Equation(self.container, domain=out_domain)
        eqs["update_gate"][...] = z == 1 / (1 + gp.math.exp(-(lin_in_z + hid_z)))

        # n_t
        eqs["new_gate"] = gp.Equation(self.container, domain=out_domain)
        eqs["new_gate"][...] = n == gp.math.tanh(lin_in_n + r * hid_n)

        # h_t
        eqs["set_output"] = gp.Equation(self.container, domain=out_domain)
        eqs["set_output"][...] = h_out == (1 - z) * n + z * h_prev_H_term

        result = FormulationResult(result=h_out, equations_created=eqs)
        result.variables_created.update(
            {"r_gate": r, "z_gate": z, "n_gate": n, "output": h_out}
        )

        result.parameters_created.update({f"w_ih_{k}": v for k, v in self.w_ih.items()})
        result.parameters_created.update({f"w_hh_{k}": v for k, v in self.w_hh.items()})
        result.parameters_created.update({f"b_ih_{k}": v for k, v in self.b_ih.items()})
        result.parameters_created.update({f"b_hh_{k}": v for k, v in self.b_hh.items()})

        return result

    def __str__(self) -> str:
        return (
            "GRU(\n"
            f"  input_size={self.input_size}\n"
            f"  hidden_size={self.hidden_size}\n"
            f"  weights_loaded={'True' if self._state == 1 else 'False'}\n)"
        )
