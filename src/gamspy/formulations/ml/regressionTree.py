import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import gamspy as gp
from gamspy.math import dim

input_data = np.array(
    [
        [2, 3],  # [l1, l2, l3] > y = 10 => 10*l1 = 1
        [3, 1],
        [1, 2],
        [5, 6],
        [6, 4],
    ]
)

output = np.array([10, 10, 10, 15, 33])

model = DecisionTreeRegressor(random_state=42)
model.fit(input_data, output)

# plt.figure(figsize=(20, 10))
# plot_tree(model, feature_names=np.array(["A", "B"]), filled=True, rounded=True, node_ids=True)
# plt.title("Decision Tree Regressor")
# plt.savefig("regression_tree.svg", format="svg")
# plt.close()
### Predict on training data (just for demo)
# predictions = model.predict(input_data)
# print("Predictions:", predictions)

# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# mse = mean_squared_error(output, predictions)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(output, predictions)
# r2 = r2_score(output, predictions)

# # Print report
# print("Regression Report:")
# print(f"MAE  = {mae:.3f}")
# print(f"MSE  = {mse:.3f}")
# print(f"RMSE = {rmse:.3f}")
# print(f"R²   = {r2:.3f}")

# exit(0)
tree_dict = {
    "children_left": model.tree_.children_left,
    "children_right": model.tree_.children_right,
    "feature": model.tree_.feature,
    "threshold": model.tree_.threshold,
    "value": model.tree_.value[:, :, 0],
    "capacity": model.tree_.capacity,
    "n_features": model.tree_.n_features,
}

# import pprint
# pprint.pprint(tree_dict)

# exit(0)

leafs = tree_dict["children_left"] < 0
leafs = leafs.nonzero()[0]
# print(f"{leafs = }")
# print(f"{tree_dict["value"][leafs, :]}")

# exit(0)


def _compute_leafs_bounds(tree, epsilon):
    """Compute the bounds that define each leaf of the tree"""
    capacity = tree["capacity"]
    n_features = tree["n_features"]
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]

    node_lb = -np.ones((n_features, capacity)) * np.inf
    node_ub = np.ones((n_features, capacity)) * np.inf

    stack = [
        0,
    ]

    # DFS for traversing through the BDT and populating the bounds
    while len(stack) > 0:
        node = stack.pop()
        left = children_left[node]
        if left < 0:
            continue
        right = children_right[node]
        assert left not in stack
        assert right not in stack
        node_ub[:, right] = node_ub[:, node]
        node_lb[:, right] = node_lb[:, node]
        node_ub[:, left] = node_ub[:, node]
        node_lb[:, left] = node_lb[:, node]

        node_ub[feature[node], left] = threshold[node]
        node_lb[feature[node], right] = threshold[node] + epsilon
        stack.append(right)
        stack.append(left)
    return (node_lb, node_ub)


node_lb, node_ub = _compute_leafs_bounds(tree_dict, 0)

# print(f"{node_lb = }\n\n {node_ub = }")


### print(f"{input_data[:,2].sum()}")  ## = 57
### exit(0)

in_data = input_data[:, 0].reshape((5, 1))  # Data for feature "A" is available.

# We introduce variable for the the missing feature: b_var

m = gp.Container(working_directory="./data")

n_features = tree_dict["n_features"]
sample_size = int(in_data.shape[0])
nleafs = len(leafs)  # nleafs = 3
leaf_vars_dim = dim((sample_size, nleafs))  # len = 15 vars but many of them will be ...

ind = gp.Parameter(m, name="ind_p", domain=dim((sample_size,)), records=in_data)
i = dim((sample_size,))
y = gp.Variable(m, name="y", domain=i, type="positive")

# y.up = float(output.sum()) ## not available

# b = gp.Variable(m, name="B_var", domain=i, type="positive")
b = gp.Variable(
    m, name="feat_var", domain=dim((sample_size, n_features)), type="positive"
)

"""
We introduce feature variables for all the features,
then we just fix the variables for which we have the data from the input. 
In the current example, we have the data for "A = 0" but not for "B=1"
"""
b.fx[..., 0] = ind[ind.domain[0]]  # not [..., 0]
b.up[..., 1] = float(input_data[:, 1].max())

obj = gp.Sum(y.domain[0], y[y.domain[0]])

e1 = gp.Equation(m, name="feature_b_contraint")
e1[...] = gp.Sum(b.domain[0], b[..., 1]) >= 3

### Now we add knowledge from the DT

ind_vars = gp.Variable(
    m,
    name=f"iv",
    type="BINARY",
    domain=leaf_vars_dim,
    description=f"indicator variable for each leaf for each sample",
)

o1 = gp.Equation(
    m,
    name="only_one_output",
    domain=dim((sample_size,)),
    description="Activate only one leaf per sample",
)

sample_domain = o1.domain[0]
feat_domain = ind_vars.domain[1]

o1[o1.domain[0]] = gp.Sum(ind_vars.domain[1], ind_vars[o1.domain[0], ind_vars.domain[1]]) == 1


out_link = gp.Parameter(
    m,
    name="predicted_value",
    domain=dim((nleafs,)),
    records=[
        (dom, val) for dom, val in zip(range(nleafs), tree_dict["value"][leafs, :])
    ],
)

out1 = gp.Equation(
    m,
    name="link_ind_out",
    domain=dim((sample_size,)),
    definition=gp.Sum(
        out_link.domain[0],
        out_link[out_link.domain[0]] * ind_vars[..., out_link.domain[0]],
    )
    == y,
    description="Link the indicator variable to the predicted value of the decision tree",
)

ub_output = gp.Equation(
    m,
    name="ub_output",
    domain=i,
    definition=y <= np.max(tree_dict["value"]),
)

lb_output = gp.Equation(
    m,
    name="lb_output",
    domain=i,
    definition=y >= np.min(tree_dict["value"]),
)


def add_binding_constraint(
    m, ind_var: gp.Variable, feat_var: gp.Variable, suffix, sense, value, M
):
    if sense == "ge":
        gp.Equation(
            m,
            name=f"iv_feat_ge_{suffix}",
            domain=dim((sample_size,)),
            definition=feat_var >= float(value) - M * (1 - ind_var),
            description="constraint to link the indicator variable with the feature",
        )
    else:
        ### Adding LE constraint
        gp.Equation(
            m,
            name=f"iv_feat_le_{suffix}",
            domain=dim((sample_size,)),
            definition=feat_var <= float(value) + M * (1 - ind_var),
            # description="constraint to link the indicator variable with the feature",
        )


for i, leaf in enumerate(leafs):
    leaf = int(leaf)
    # print(f"{leaf = }")
    for feat in range(n_features):
        mask = (b.up[:, feat] >= node_lb[:, leaf][feat]) & (
            b.lo[:, feat] <= node_ub[:, leaf][feat]
        )
        ind_vars.fx[:, i].where[
            ~mask
        ] = 0  # these indicator variables will not be reached
        # print(f"{feat = }")
        feat_ub = float(node_ub[feat, leaf])
        feat_lb = float(node_lb[feat, leaf])
        for rec in range(sample_size):
            suffix = f"l{leaf}_f{feat}_s{rec}"
            if feat_lb > -np.inf:
                # print("####Adding GE constraints")
                mask_ext = (b.lo[rec, feat] < feat_lb) & mask
                add_binding_constraint(
                    m,
                    ind_var=ind_vars[rec, i].where[mask],
                    feat_var=b[rec, feat].where[mask],
                    suffix=suffix,
                    sense="ge",
                    value=feat_lb,
                    M=1e6,
                )
            if feat_ub < np.inf:
                # print("####Adding LE constraints")
                mask_ext = (b.up[:, feat] > feat_ub) & mask
                add_binding_constraint(
                    m,
                    ind_var=ind_vars[rec, i].where[mask],
                    feat_var=b[rec, feat].where[mask],
                    suffix=suffix,
                    sense="le",
                    value=feat_ub,
                    M=1e6,
                )

dt_model = gp.Model(
    m,
    name="dt_model",
    equations=m.getEquations(),
    problem="MIP",
    sense=gp.Sense.MIN,
    objective=obj,
)

# dt_model.toGams("./GAMS")

print(dt_model.solve(options=gp.Options(equation_listing_limit=1e6)))

# print(dt_model.getEquationListing())

print(y.l.records)
print(b.records)
print(ind_vars.records)
