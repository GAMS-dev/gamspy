from __future__ import annotations

import uuid

import numpy as np

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError


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
        self, input: gp.Parameter | gp.Variable, M: float = 1e6
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """Insert all equations here"""

        if len(input.domain) == 0:
            raise ValidationError(
                "expected an input with at least 1 dimension"
            )

        if len(input.domain[-1]) != self._nfeatures:
            raise ValidationError("number of features do not match")

        sample_size = len(input.domain[0])
        set_of_samples, set_of_features, set_of_leafs = gp.math._generate_dims(
            self.container, dims=[sample_size, self._nfeatures, self._nleafs]
        )

        out = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "output"),
            domain=set_of_samples,
        )

        _feat_vars = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "feature"),
            domain=[set_of_samples, set_of_features],
        )

        if isinstance(input, gp.Variable):
            _feat_vars = input

        elif isinstance(input, gp.Parameter):
            _feat_vars.fx[...] = input[...]

        else:
            raise ValidationError(
                "Input must be either of gp.Parameter | gp.Variable"
            )

        ind_vars = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "indicator"),
            type="BINARY",
            domain=[set_of_samples, set_of_leafs],
            description="indicator variable for each leaf for each sample",
        )

        assign_one_output = gp.Equation(
            self.container,
            name=utils._generate_name(
                "e", self._name_prefix, "assign_one_output"
            ),
            domain=set_of_samples,
            description="Activate only one leaf per sample",
        )
        assign_one_output[set_of_samples] = (
            gp.Sum(set_of_leafs, ind_vars[set_of_samples, set_of_leafs]) == 1
        )

        out_link = gp.Parameter(
            self.container,
            name=utils._generate_name(
                "p", self._name_prefix, "predicted_value"
            ),
            domain=set_of_leafs,
            records=[
                (dom, val)
                for dom, val in zip(
                    range(self._nleafs),
                    self._tree_dict["value"][self._leafs, :],
                )
            ],
        )

        link_indctr_output = gp.Equation(
            self.container,
            name=utils._generate_name(
                "e", self._name_prefix, "link_indctr_output"
            ),
            domain=set_of_samples,
            description="Link the indicator variable to the predicted value of the decision tree",
        )
        link_indctr_output[set_of_samples] = (
            gp.Sum(
                set_of_leafs,
                out_link[set_of_leafs]
                * ind_vars[set_of_samples, set_of_leafs],
            )
            == out
        )

        ub_output = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "ub_output"),
            domain=set_of_samples,
            description="Output cannot be more than the maximum of predicted value",
        )
        ub_output[set_of_samples] = out <= np.max(self._tree_dict["value"])

        lb_output = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "lb_output"),
            domain=set_of_samples,
            description="Output cannot be less than the minimum of predicted value",
        )
        lb_output[set_of_samples] = out >= np.min(self._tree_dict["value"])

        uni_domain = [set_of_samples, set_of_features, set_of_leafs]

        s = gp.Set(
            self.container,
            name=utils._generate_name(
                "s", self._name_prefix, "subset_of_paths"
            ),
            description="Dynamic subset of possible paths",
            domain=uni_domain,  # TODO: Why we cannot just pass ss here, and GAMSPy infers the domain of ss? We get `ValueError: All linked 'domain' elements must have dimension == 1`
        )

        cons_type = gp.Set(
            self.container,
            name=utils._generate_name("s", self._name_prefix, "cons_type"),
            domain=["*"],
            records=["ge", "le"],
        )

        feat_thresh = gp.Parameter(
            self.container,
            name=utils._generate_name(
                "p", self._name_prefix, "feature_threshold"
            ),
            description="feature splitting value",
            domain=uni_domain + [cons_type],
        )

        for i, leaf in enumerate(self._leafs):
            for feat in range(self._nfeatures):
                feat_ub = float(self._node_ub[feat, leaf])
                feat_lb = float(self._node_lb[feat, leaf])
                mask = (_feat_vars.up[:, feat] >= feat_lb) & (
                    _feat_vars.lo[:, feat] <= feat_ub
                )
                ind_vars.fx[set_of_samples, i].where[~mask] = 0
                if feat_lb > -np.inf:
                    mask_ext = (_feat_vars.lo[:, feat] < feat_lb) & mask
                    s[set_of_samples, feat, i].where[mask_ext] = True
                    feat_thresh[s, "ge"] = feat_lb
                if feat_ub < np.inf:
                    mask_ext = (_feat_vars.up[:, feat] > feat_ub) & mask
                    s[set_of_samples, feat, i].where[mask_ext] = True
                    feat_thresh[s, "le"] = feat_ub
                s[...] = False

        s[...].where[gp.Sum(cons_type, feat_thresh[..., cons_type])] = True

        ge_cons = gp.Equation(
            self.container,
            name=utils._generate_name(
                "e", self._name_prefix, "link_indctr_feature_ge"
            ),
            domain=uni_domain,
            description="Link the indicator variable with the feature which is Lower bounded using a big-M constraint",
        )
        ge_cons[s[uni_domain]].where[feat_thresh[s, "ge"] != 0] = _feat_vars[
            set_of_samples, set_of_features
        ].where[s[uni_domain]] >= feat_thresh[s, "ge"] - M * (
            1 - ind_vars[set_of_samples, set_of_leafs].where[s[uni_domain]]
        )

        le_cons = gp.Equation(
            self.container,
            name=utils._generate_name(
                "e", self._name_prefix, "link_indctr_feature_le"
            ),
            domain=uni_domain,
            description="Link the indicator variable with the feature which is Upper bounded using a big-M constraint",
        )
        le_cons[s[uni_domain]].where[feat_thresh[s, "le"] != 0] = _feat_vars[
            set_of_samples, set_of_features
        ].where[s[uni_domain]] <= feat_thresh[s, "le"] + M * (
            1 - ind_vars[set_of_samples, set_of_leafs].where[s[uni_domain]]
        )

        return out, [
            assign_one_output,
            link_indctr_output,
            ub_output,
            lb_output,
            ge_cons,
            le_cons,
        ]
