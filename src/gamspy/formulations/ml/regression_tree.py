from __future__ import annotations

import uuid

# from typing import TYPE_CHECKING
import numpy as np

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError

# from gamspy.math import dim


class RegressionTree:
    """
    Formulation generator for Regression Trees in GAMS.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    regressor: dict | DecisionTreeRegressor
        Trained decision tree. This could either be a sklearn.tree.DecisionTreeRegressor object or a dictionary of the form ...
    input_data : numpy.ndarray
        Input data ...
    name_prefix : str | None
        Prefix for generated GAMSPy symbols, by default None which means
        random prefix. Using the same name_prefix in different formulations causes name
        conflicts. Do not use the same name_prefix again.

    Examples
    --------
    >>> import gamspy as gp
    >>> import numpy as np
    >>> from gamspy.math import dim

    """

    def __init__(
        self,
        container: gp.Container,
        regressor: dict,  # | DecisionTreeRegressor,
        name_prefix: str | None = None,
    ):
        COMPLETE_TREE_DICT = {
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
            "capacity",
            "n_features",
        }
        if isinstance(regressor, dict):
            if len(check_keys := regressor.keys() - COMPLETE_TREE_DICT) != 0:
                raise ValidationError(
                    f"Dictionary containing the information from the Regression Tree is either incomplete or the keys are wrong. KEYS: {check_keys}"
                )
            self._tree_dict = regressor

        # TODO: Add sanity checks for populating the sklearn dictionary?
        else:
            self._tree_dict = {
                "children_left": regressor.tree_.children_left,
                "children_right": regressor.tree_.children_right,
                "feature": regressor.tree_.feature,
                "threshold": regressor.tree_.threshold,
                "value": regressor.tree_.value[:, :, 0],
                "capacity": regressor.tree_.capacity,
                "n_features": regressor.tree_.n_features,
            }

        self.container = container

        if name_prefix is None:
            name_prefix = str(uuid.uuid4()).split("-")[0]

        self._name_prefix = name_prefix
        self._nfeatures = self._tree_dict["n_features"]
        leafs = self._tree_dict["children_left"] < 0
        self._leafs = leafs.nonzero()[0]
        self._nleafs = len(self._leafs)
        self._node_lb, self._node_ub = self._node_bounds(tree=self._tree_dict)

    @staticmethod
    def _node_bounds(tree):
        """Traverse the tree using DFS and extract bound information for each node."""

        capacity = tree["capacity"]
        n_features = tree["n_features"]
        children_left = tree["children_left"]
        children_right = tree["children_right"]
        feature = tree["feature"]
        threshold = tree["threshold"]

        node_lb = np.empty((n_features, capacity))
        node_lb.fill(-np.inf)
        node_ub = np.empty((n_features, capacity))
        node_ub.fill(np.inf)

        stack = [0]

        while stack:
            node = stack.pop()
            left = children_left[node]
            if left < 0:
                continue
            right = children_right[node]

            node_lb[:, left] = node_lb[:, node]
            node_ub[:, left] = node_ub[:, node]
            node_lb[:, right] = node_lb[:, node]
            node_ub[:, right] = node_ub[:, node]

            node_ub[feature[node], left] = threshold[node]
            node_lb[feature[node], right] = threshold[node]

            stack.extend((right, left))

        return node_lb, node_ub

    def __call__(
        self, input: gp.Parameter | gp.Variable
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """Insert all equations here"""

        if len(input.domain) == 0:
            raise ValidationError(
                "expected an input with at least 1 dimension"
            )

        if len(input.domain[-1]) != self._nfeatures:
            raise ValidationError("number of features do not match")

        sample_size = len(input.domain[0])
        s_set, f_set, l_set = gp.math._generate_dims(
            self.container, dims=[sample_size, self._nfeatures, self._nleafs]
        )

        # TODO: Exception: Cannot generate records unless the symbol has domain objects for all dimensions (i.e., <symbol>.domain_type == 'regular')
        # Perhaps get rid of gp.Set objects?
        s_set.generateRecords(1)
        f_set.generateRecords(1)
        l_set.generateRecords(1)

        out = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "output"),
            domain=s_set,
        )

        ind_vars = gp.Variable(
            self.container,
            name="iv",
            type="BINARY",
            domain=[s_set, l_set],
            description="indicator variable for each leaf for each sample",
        )

        only_one_output = gp.Equation(
            self.container,
            name="only_one_output",
            domain=s_set,
            description="Activate only one leaf per sample",
        )
        only_one_output[s_set] = gp.Sum(l_set, ind_vars[s_set, l_set]) == 1

        out_link = gp.Parameter(
            self.container,
            name="predicted_value",
            domain=l_set,
            records=[
                (dom, val)
                for dom, val in zip(
                    range(self._nleafs),
                    self._tree_dict["value"][self._leafs, :],
                )
            ],
        )

        link_ind_out = gp.Equation(
            self.container,
            name="link_ind_out",
            domain=s_set,
            description="Link the indicator variable to the predicted value of the decision tree",
        )
        link_ind_out[s_set] = (
            gp.Sum(
                l_set,
                out_link[l_set] * ind_vars[s_set, l_set],
            )
            == out
        )

        ub_output = gp.Equation(
            self.container,
            name="ub_output",
            domain=s_set,
            description="Output cannot be more than the maximum of predicted value",
        )
        ub_output[s_set] = out <= np.max(self._tree_dict["value"])

        lb_output = gp.Equation(
            self.container,
            name="lb_output",
            domain=s_set,
            description="Output cannot be less than the minimum of predicted value",
        )
        lb_output[s_set] = out >= np.min(self._tree_dict["value"])

        uni_domain = [s_set, f_set, l_set]

        s = gp.Set(
            self.container,
            name="s",
            description="Dynamic subset of possible paths",
            domain=uni_domain,  # TODO: Why we cannot just pass ss here, and GAMSPy infers the domain of ss? We get `ValueError: All linked 'domain' elements must have dimension == 1`
        )

        bb = gp.Set(
            self.container,
            name="bb",
            domain=["*"],
            records=["ge", "le"],
        )

        feat_thresh = gp.Parameter(
            self.container,
            name="feat_thres",
            description="feature splitting value",
            domain=uni_domain + [bb],
        )

        ### TODO: Case when `out` is a gp.Parameter

        for i, leaf in enumerate(self._leafs):
            for feat in range(self._nfeatures):
                feat_ub = float(self._node_ub[feat, leaf])
                feat_lb = float(self._node_lb[feat, leaf])
                mask = (out.up[:, feat] >= feat_lb) & (
                    out.lo[:, feat] <= feat_ub
                )
                # these indicator variables will not be reached
                ind_vars.fx[s_set, i].where[~mask] = 0
                s[s_set, feat, i].where[mask] = True
                if feat_lb > -np.inf:
                    feat_thresh[s, "ge"] = feat_lb
                if feat_ub < np.inf:
                    feat_thresh[s, "le"] = feat_ub
                s[...] = False

        s[...].where[gp.Sum(bb, feat_thresh[..., bb])] = True

        ge_cons = gp.Equation(
            self.container,
            name="iv_feat_ge",
            domain=uni_domain,
            description="Link the indicator variable with the feature which is Lower bounded using a big-M constraint",
        )

        le_cons = gp.Equation(
            self.container,
            name="iv_feat_le",
            domain=uni_domain,
            description="Link the indicator variable with the feature which is Upper bounded using a big-M constraint",
        )

        return out, [
            only_one_output,
            link_ind_out,
            ub_output,
            lb_output,
            ge_cons,
            le_cons,
        ]

    # NOTE: Use gp.Set or gp.math.dim? Ans: use _generate_dims()
    # NOTE: Input flow? Initialize the class with input or later. Required for fixing the feature variables. User's worry
    # NOTE: Mapping features and input. User's worry
