import numpy as np
from sklearn.tree import DecisionTreeRegressor

import gamspy as gp
from gamspy.math import dim

input_data = np.array(
    [
        [2, 3],  # [l1, l2, l3]
        [3, 1],
        [1, 2],
        [5, 6],
        [6, 4],
    ]
)

output = np.array([10, 10, 10, 15, 33])

model = DecisionTreeRegressor(random_state=42)
model.fit(input_data, output)

# # Base URL for retrieving data
# janos_data_url = "https://raw.githubusercontent.com/INFORMSJoC/2020.1023/master/data/"
# historical_data = pd.read_csv(
#     janos_data_url + "college_student_enroll-s1-1.csv", index_col=0
# )

# # classify our features between the ones that are fixed and the ones that will be
# # part of the optimization problem
# features = ["merit", "SAT", "GPA"]
# target = "enroll"


# # Run our regression
# regression = DecisionTreeRegressor(max_depth=10, max_leaf_nodes=50, random_state=1)
# regression.fit(X=historical_data.loc[:, features], y=historical_data.loc[:, target])


# input_data = pd.read_csv(janos_data_url + "college_applications6000.csv", index_col=0)
# nstudents = 100
# # Select randomly nstudents in the data
# input_data = input_data.sample(nstudents).to_numpy()
# model = regression

tree_dict = {
    "children_left": model.tree_.children_left,
    "children_right": model.tree_.children_right,
    "feature": model.tree_.feature,
    "threshold": model.tree_.threshold,
    "value": model.tree_.value[:, :, 0],
    "capacity": model.tree_.capacity,
    "n_features": model.tree_.n_features,
}

leafs = tree_dict["children_left"] < 0
leafs = leafs.nonzero()[0]


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


in_data = input_data[:, 0].reshape(
    (5, 1)
)  # Data for feature "A" is available.

# We introduce variable for the the missing feature: b_var

m = gp.Container(working_directory="./data")

n_features = tree_dict["n_features"]
sample_size = int(input_data.shape[0])
nleafs = len(leafs)

# TODO: mypy error: Argument "domain" to "Set" has incompatible type "Dim"; expected "Sequence[gamspy._symbols.set.Set | Alias | str] | gamspy._symbols.set.Set | Alias | str | None"  [arg-type]
s_set = gp.Set(m, name="s_set", domain=dim((sample_size,)))
s_set.generateRecords(1)
f_set = gp.Set(m, name="f_set", domain=dim((n_features,)))
f_set.generateRecords(1)
l_set = gp.Set(m, name="l_set", domain=dim((nleafs,)))
l_set.generateRecords(1)

ind = gp.Parameter(m, name="ind_p", domain=s_set, records=in_data)
y = gp.Variable(m, name="y", domain=s_set, type="positive")

# y.up = float(output.sum()) ## not available

# b = gp.Variable(m, name="B_var", domain=i, type="positive")
b = gp.Variable(m, name="feat_var", domain=[s_set, f_set], type="positive")

"""
We introduce feature variables for all the features,
then we just fix the variables for which we have the data from the input. 
In the current example, we have the data for "A = 0" but not for "B=1"
"""
b.fx[s_set, 0] = ind[s_set]  # not [..., 0]
# b.up[:, 0] = float(input_data[:, 0].max())
b.up[:, 1] = float(input_data[:, 1].max())

obj = gp.Sum(s_set, y[s_set])

# e1 = gp.Equation(m, name="feature_b_contraint")
# e1[...] = gp.Sum(b.domain[0], b[..., 1]) <= 30

e2 = gp.Equation(m, name="feature_b_contraint_2")
e2[...] = gp.Sum(b.domain[0], b[..., 1]) >= 25

### Now we add knowledge from the DT

ind_vars = gp.Variable(
    m,
    name="iv",
    type="BINARY",
    domain=[s_set, l_set],
    description="indicator variable for each leaf for each sample",
)

o1 = gp.Equation(
    m,
    name="only_one_output",
    domain=s_set,
    description="Activate only one leaf per sample",
)
o1[s_set] = gp.Sum(l_set, ind_vars[s_set, l_set]) == 1


out_link = gp.Parameter(
    m,
    name="predicted_value",
    domain=l_set,
    records=[
        (dom, val)
        for dom, val in zip(range(nleafs), tree_dict["value"][leafs, :])
    ],
)

out1 = gp.Equation(
    m,
    name="link_ind_out",
    domain=s_set,
    description="Link the indicator variable to the predicted value of the decision tree",
)
out1[s_set] = (
    gp.Sum(
        l_set,
        out_link[l_set] * ind_vars[s_set, l_set],
    )
    == y
)

ub_output = gp.Equation(
    m,
    name="ub_output",
    domain=s_set,
    description="Output cannot be more than the maximum of predicted value",
)
ub_output[s_set] = y <= np.max(tree_dict["value"])

lb_output = gp.Equation(
    m,
    name="lb_output",
    domain=s_set,
    description="Output cannot be less than the minimum of predicted value",
)
lb_output[s_set] = y >= np.min(tree_dict["value"])

uni_domain = [s_set, f_set, l_set]

s = gp.Set(
    m,
    name="s",
    description="Dynamic subset of possible paths",
    domain=uni_domain,  # TODO: Why we cannot just pass ss here, and GAMSPy infers the domain of ss? We get `ValueError: All linked 'domain' elements must have dimension == 1`
)

bb = gp.Set(
    m,
    name="bb",
    domain=["*"],
    records=["ge", "le"],
)

feat_thresh = gp.Parameter(
    m,
    name="feat_thres",
    description="feature splitting value",
    domain=uni_domain + [bb],
)

for i, leaf in enumerate(leafs):
    for feat in range(n_features):
        feat_ub = float(node_ub[feat, leaf])
        feat_lb = float(node_lb[feat, leaf])
        mask = (b.up[:, feat] >= feat_lb) & (b.lo[:, feat] <= feat_ub)
        # these indicator variables will not be reached
        ind_vars.fx[s_set, i].where[~mask] = 0
        s[s_set, feat, i].where[mask] = True
        if feat_lb > -np.inf:
            feat_thresh[s, "ge"] = feat_lb
        if feat_ub < np.inf:
            feat_thresh[s, "le"] = feat_ub
        s[...] = False

# TODO: mypy throws an error: Invalid index type "EllipsisType" for "gamspy._symbols.set.Set"; expected type "Sequence[Any] | str"  [index]
s[...].where[gp.Sum(bb, feat_thresh[..., bb])] = True

ge_cons = gp.Equation(
    m,
    name="iv_feat_ge",
    domain=uni_domain,
    description="Link the indicator variable with the feature which is Lower bounded using a big-M constraint",
)

le_cons = gp.Equation(
    m,
    name="iv_feat_le",
    domain=uni_domain,
    description="Link the indicator variable with the feature which is Upper bounded using a big-M constraint",
)

ge_cons[s[uni_domain]].where[feat_thresh[s, "ge"] != 0] = b[
    s_set, f_set
].where[s[uni_domain]] >= feat_thresh[s, "ge"] - 1e6 * (
    1 - ind_vars[s_set, l_set].where[s[uni_domain]]
)
le_cons[s[uni_domain]].where[feat_thresh[s, "le"] != 0] = b[
    s_set, f_set
].where[s[uni_domain]] <= feat_thresh[s, "le"] + 1e6 * (
    1 - ind_vars[s_set, l_set].where[s[uni_domain]]
)

dt_model = gp.Model(
    m,
    name="dt_model",
    equations=m.getEquations(),
    problem="MIP",
    sense=gp.Sense.MIN,
    # TODO: Argument "objective" to "Model" has incompatible type "Sum"; expected "Variable | Expression | None"  [arg-type]
    objective=obj,
)

# dt_model.toGams("./GAMS")

summary = dt_model.solve(options=gp.Options(equation_listing_limit=1e6))

if summary is not None:
    print(summary.to_string())

# print(dt_model.getEquationListing())
# print("Y.L\n\n",y.l.records)
# print("B.L\n\n",b.l[:,1].records)
# print(ind_vars.records)
