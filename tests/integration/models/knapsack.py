import math
import sys
from pathlib import Path

from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
    VariableType,
)

# Declaration
m = Container()
i = Set(m, "i", description="items")
p = Parameter(m, "p", description="profits", domain=i)
w = Parameter(m, "p", description="weights", domain=i)
c = Parameter(m, "c", description="capacity")
x = Variable(m, "x", domain=i, description="chosen", type=VariableType.BINARY)

cap_restr = Equation(m, name="capacity_restriction")
cap_restr[...] = Sum(i, w[i] * x[i]) <= c

utility = Sum(i, p[i] * x[i])

knapsack = Model(
    m,
    name="knapsack",
    equations=m.getEquations(),
    problem=Problem.MIP,
    sense=Sense.MAX,
    objective=utility,
)


# Instance data
def load_instance_from_file(filename):
    def ints(elems):
        return [int(elem) for elem in elems]

    global items, capacity, profits, weights
    items, capacity, profits, weights = [], [], [], []
    num_items = None
    with open(filename) as fp:
        for line in fp.readlines():
            if not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 2
            if not num_items:
                num_items, capacity = ints(parts)
                continue
            profits.append(parts[0])
            weights.append(parts[1])
    assert num_items
    items = [f"i{i + 1}" for i in range(num_items)]


# Example instance taken from
# http://artemisa.unicauca.edu.co/~johnyortega/instances_01_KP/
# and also available at https://github.com/JordiHOFC/knapsackproblemboolean
load_instance_from_file(
    str(Path(__file__).parent.absolute()) + "/f1_l-d_kp_10_269"
)
i.setRecords(items)
p.setRecords(zip(items, profits))
w.setRecords(zip(items, weights))
c.setRecords(capacity)

# Run solve and display results
knapsack.solve(output=sys.stdout)
print(f"Objective function value = {knapsack.objective_value}")
assert math.isclose(knapsack.objective_value, 269.0, rel_tol=0.001)
levels = list(x.records["level"])
print(
    f'Chosen items = {", ".join([j for ix, j in enumerate(items) if levels[ix] == 1.0])}'
)
