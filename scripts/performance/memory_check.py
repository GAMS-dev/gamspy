from __future__ import annotations

import os
import sys

import psutil

from gamspy import (
    Container,
    Equation,
    Model,
    Number,
    Parameter,
    Set,
    Sum,
    Variable,
)
from gamspy.math import uniform

process = psutil.Process(os.getpid())

m = Container()

density = 0.002
i = Set(m, records=range(1000))
j = Set(m, records=range(1000))
arc = Set(m, domain=[i, j])
p = Parameter(m, domain=[i, j])
v = Variable(m, domain=[i, j])
e = Equation(m, domain=[i, j])
e[i, j] = v[i, j] <= p[i, j]
model = Model(
    m,
    equations=[e],
    objective=Sum((i, j), v[i, j] * p[i, j]),
    problem="lp",
    sense="max",
)

# Warmup iterations
for _ in range(10):
    arc[...] = Number(1).where[(uniform(0, 1) < density)]
    p[arc] = uniform(0, 1)
    model.solve()
    p[...] = 0

memory_usages: list[float] = []
for _ in range(100):
    mem_tracking_overhead = sys.getsizeof(memory_usages) / float(2**20)

    arc[...] = Number(1).where[(uniform(0, 1) < density)]
    p[arc] = uniform(0, 1)
    model.solve()
    p[...] = 0

    # Append usages
    memory_usage = process.memory_info().rss / float(2**20)
    memory_usages.append(memory_usage - mem_tracking_overhead)

    print(memory_usage)

print(memory_usages[-1] / memory_usages[0])
