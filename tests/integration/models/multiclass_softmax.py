"""
## LICENSETYPE: Demo
## MODELTYPE: NLP
## KEYWORDS: nonlinear programming, classification, softmax, log_softmax

A toy model for testing log_softmax using a classification problem.
This example only tests correctness of log_softmax but does not train a
whole neural network.
"""

from __future__ import annotations

import os
import sys

import gamspy as gp
import numpy as np
from gamspy import (
    Container,
    Equation,
    Model,
    Set,
    Sum,
    Variable,
)

batch_size = 5


def simple_log_softmax(x: np.ndarray):
    return np.log(np.exp(x) / np.exp(x).sum())


def simple_softmax(x: np.ndarray):
    return np.exp(x) / np.exp(x).sum()


def verify_output(x_rec, y_recs, y2_recs):
    expected_results = simple_log_softmax(np.array([5.0, -5.0, -5.0]))
    expected_results_2 = simple_softmax(np.array([5.0, -5.0, -5.0]))

    for _, row in x_rec.iterrows():
        expected = str(int(row["DenseDim5_1"]) % 3)
        if row["DenseDim3_1"] == expected:
            assert np.isclose(row["level"], 5)
        else:
            assert np.isclose(row["level"], -5)

    for _, row in y_recs.iterrows():
        expected = str(int(row["DenseDim5_1"]) % 3)
        if row["DenseDim3_1"] == expected:
            assert np.isclose(row["level"], expected_results[0])
        else:
            assert np.isclose(row["level"], expected_results[1])

    for _, row in y2_recs.iterrows():
        expected = str(int(row["DenseDim5_1"]) % 3)
        if row["DenseDim3_1"] == expected:
            assert np.isclose(row["level"], expected_results_2[0])
        else:
            assert np.isclose(row["level"], expected_results_2[1])


def main():
    m = Container(
        system_directory=os.getenv("GAMSPY_GAMS_SYSDIR", None),
    )

    labels = Set(m, name="labels", domain=gp.math.dim([batch_size, 3]))
    # set random labels
    for i in range(batch_size):
        labels[str(i), str(i % 3)] = 1

    x = Variable(m, name="x", domain=gp.math.dim([batch_size, 3]))

    x.lo[...] = -5
    x.up[...] = 5

    y = gp.math.log_softmax(x)
    y2 = gp.math.softmax(x)

    nll = Variable(m, name="nll")  # negative log likelihood loss

    set_loss = Equation(m, name="set_loss")
    set_loss[...] = nll == Sum(labels[y.domain], -y)

    classification = Model(
        m,
        name="classification",
        equations=m.getEquations(),
        problem="nlp",
        sense="min",
        objective=nll,
    )
    classification.solve(output=sys.stdout)
    verify_output(x.records, y.records, y2.records)


if __name__ == "__main__":
    main()
