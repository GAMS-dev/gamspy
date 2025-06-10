from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np

import gamspy as gp
import gamspy.formulations.nn.utils as utils
from gamspy.exceptions import ValidationError
from gamspy.formulations.ml.dtStruct import DecisionTreeStruct

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeRegressor


class RegressionTree(DecisionTreeStruct):
    """
    Formulation generator for Regression Trees in GAMS.

    Parameters
    ----------
    container : Container
        Container that will contain the new variable and equations.
    regressor: DecisionTreeRegressor | None
        - A fitted `sklearn.tree.DecisionTreeRegressor` object which is processed by the
          `DecisionTreeStruct` superclass to populate the underlying tree attributes.
        - If `None`, the tree structure is NOT automatically initialized.
          In this case, the user is responsible for manually populating the tree attributes
          inherited from `DecisionTreeStruct` before the formulation can be used.
          See :meth:`DecisionTreeStruct <gamspy.formulations.DecisionTreeStruct>` for details on required attributes.

    name_prefix : str | None
        Prefix for generated GAMSPy symbols, by default None which means
        random prefix. Using the same name_prefix in different formulations causes name
        conflicts. Do not use the same name_prefix again.

    Examples
    --------
    >>> import gamspy as gp
    >>> import numpy as np
    >>> from gamspy.math import dim
    >>> np.random.seed(42)
    >>> m = gp.Container()
    >>> in_data = np.random.randint(0, 10, size=(5, 2))
    >>> out_data = np.random.randint(1, 3, size=(5, 1))
    >>> dt_model = gp.formulations.RegressionTree(m)
    >>> dt_model.capacity = 3
    >>> dt_model.children_left = np.array([1, -1, -1])
    >>> dt_model.children_right = np.array([2, -1, -1])
    >>> dt_model.feature = np.array([0, -2, -2])
    >>> dt_model.n_features = 2
    >>> dt_model.threshold = np.array([4.0, -2.0, -2.0])
    >>> dt_model.value = np.array([[1.8], [1.0], [2.0]])
    >>> x = gp.Variable(m, "x", domain=dim((5, 2)), type="positive")
    >>> x.up[:, :] = 10
    >>> y, eqns = dt_model(x)
    >>> [d.name for d in y.domain]
    ['DenseDim5_1', 'DenseDim1_1']

    """

    def __init__(
        self,
        container: gp.Container,
        regressor: DecisionTreeRegressor | None = None,
        name_prefix: str | None = None,
    ):
        super().__init__(_regressor_source=regressor)  # type: ignore

        if not isinstance(container, gp.Container):
            raise ValidationError(f"{container} is not a gp.Container.")

        self.container = container

        if name_prefix is None:
            name_prefix = str(uuid.uuid4()).split("-")[0]

        self._name_prefix = name_prefix

    def _node_bounds(self):
        """Traverse the tree using DFS and extract bound information for each node."""
        node_lb = np.empty((self.n_features, self.capacity))
        node_lb.fill(-np.inf)
        node_ub = np.empty((self.n_features, self.capacity))
        node_ub.fill(np.inf)

        stack = [0]

        while stack:
            node = stack.pop()
            left = self.children_left[node]
            if left < 0:
                continue
            right = self.children_right[node]

            node_lb[:, left] = node_lb[:, node]
            node_ub[:, left] = node_ub[:, node]
            node_lb[:, right] = node_lb[:, node]
            node_ub[:, right] = node_ub[:, node]

            node_ub[self.feature[node], left] = self.threshold[node]
            node_lb[self.feature[node], right] = self.threshold[node]

            stack.extend((right, left))

        return node_lb, node_ub

    def __call__(
        self, input: gp.Parameter | gp.Variable, M: float | None = None
    ) -> tuple[gp.Variable, list[gp.Equation]]:
        """
        Generate output variable and equations required for embedding the regression tree.

        Parameters
        ----------
        input : gp.Parameter | gp.Variable
                input for the regression tree, must be in shape (sample_size, number_of_features)
        M : float
            value for the big_M. By default, infer the value using the available bounds for variables
        """
        # TODO: Change the >input< arg name to something else?
        super()._check_tree()

        leafs = self.children_left < 0
        leafs = leafs.nonzero()[0]
        nleafs = len(leafs)
        output_dim = self.value.shape[-1]
        node_lb, node_ub = self._node_bounds()

        if len(input.domain) == 0:
            raise ValidationError(
                "expected an input with at least 1 dimension"
            )

        if len(input.domain[-1]) != self.n_features:
            raise ValidationError("number of features do not match")

        if M and not isinstance(M, (float, int)):
            raise ValidationError("M can either be of type float or int")
        set_of_samples = input.domain[0]
        set_of_features = input.domain[-1]

        find_set_of_output_dim = self.container.data.get(
            "set_of_output_dim", None
        )
        # In case of Random Forest, create this set only once as >output_dim< will not change
        if find_set_of_output_dim is None:
            set_of_output_dim = self.container.addSet(
                "set_of_output_dim", records=range(output_dim)
            )
        else:
            set_of_output_dim = find_set_of_output_dim

        set_of_leafs = self.container.addSet(
            name=utils._generate_name("s", self._name_prefix, "leafs"),
            records=range(nleafs),
        )

        _feat_par_records = []
        for i, leaf in enumerate(leafs):
            for feat in range(self.n_features):
                _feat_par_records.append((feat, i, "ub", node_ub[feat, leaf]))
                _feat_par_records.append((feat, i, "lb", node_lb[feat, leaf]))

        _feat_par = gp.Parameter(
            self.container,
            name=utils._generate_name("p", self._name_prefix, "feat_par"),
            domain=[set_of_features, set_of_leafs, "*"],
            records=_feat_par_records,
        )

        out = gp.Variable(
            self.container,
            name=utils._generate_name("v", self._name_prefix, "output"),
            domain=[set_of_samples, set_of_output_dim],
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
                "Input must be of either type gp.Parameter | gp.Variable"
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

        idx, jdx = np.indices((nleafs, output_dim), dtype=int)
        mapped_values = self.value[leafs[:, None], jdx]
        out_link = gp.Parameter(
            self.container,
            name=utils._generate_name(
                "p", self._name_prefix, "predicted_value"
            ),
            domain=[set_of_leafs, set_of_output_dim],
            records=[
                (int(i), int(j), v)
                for i, j, v in np.stack(
                    (idx, jdx, mapped_values), axis=-1
                ).reshape(-1, 3)
            ],
        )

        link_indctr_output = gp.Equation(
            self.container,
            name=utils._generate_name(
                "e", self._name_prefix, "link_indctr_output"
            ),
            domain=[set_of_samples, set_of_output_dim],
            description="Link the indicator variable to the predicted value of the decision tree",
        )
        link_indctr_output[set_of_samples, set_of_output_dim] = (
            gp.Sum(
                set_of_leafs,
                out_link[set_of_leafs, set_of_output_dim]
                * ind_vars[set_of_samples, set_of_leafs],
            )
            == out
        )

        max_out = gp.Parameter(
            self.container,
            name=utils._generate_name("p", self._name_prefix, "max_out"),
            domain=[set_of_output_dim],
            records=np.max(self.value[leafs, :], axis=0),
        )
        min_out = gp.Parameter(
            self.container,
            name=utils._generate_name("p", self._name_prefix, "min_out"),
            domain=[set_of_output_dim],
            records=np.min(self.value[leafs, :], axis=0),
        )

        ub_output = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "ub_output"),
            domain=[set_of_samples, set_of_output_dim],
            description="Output cannot be more than the maximum of predicted value",
        )
        ub_output[...] = out <= max_out

        lb_output = gp.Equation(
            self.container,
            name=utils._generate_name("e", self._name_prefix, "lb_output"),
            domain=[set_of_samples, set_of_output_dim],
            description="Output cannot be less than the minimum of predicted value",
        )
        lb_output[...] = out >= min_out

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

        _bound_big_m = gp.Parameter(
            self.container,
            name=utils._generate_name("p", self._name_prefix, "bound_big_m"),
            description="bound value for big-M",
            domain=uni_domain + [cons_type],
        )

        ### This generates the set of possible paths given the input data
        mask = (_feat_vars.up[...] >= _feat_par[..., "lb"]) & (
            _feat_vars.lo[...] <= _feat_par[..., "ub"]
        )
        mask_lo = (
            (_feat_vars.lo[...] < _feat_par[..., "lb"])
            & mask
            & (_feat_par[..., "lb"] > -np.inf)
        )
        mask_up = (
            (_feat_vars.up[...] > _feat_par[..., "ub"])
            & mask
            & (_feat_par[..., "ub"] < np.inf)
        )
        s[...].where[mask_lo | mask_up] = True
        feat_thresh[..., "ge"].where[mask_lo] = _feat_par[..., "lb"]
        feat_thresh[..., "le"].where[mask_up] = _feat_par[..., "ub"]
        ind_vars.fx[set_of_samples, set_of_leafs].where[
            gp.Sum(set_of_features, ~mask)
        ] = 0
        _bound_big_m[s[uni_domain], "ge"].where[mask_lo] = (
            M
            if M
            else (
                _feat_par[set_of_features, set_of_leafs, "lb"]
                - _feat_vars.lo[set_of_samples, set_of_features]
            )
        )
        _bound_big_m[s[uni_domain], "le"].where[mask_up] = (
            M
            if M
            else (
                _feat_vars.up[set_of_samples, set_of_features]
                - _feat_par[set_of_features, set_of_leafs, "ub"]
            )
        )

        ge_cons = gp.Equation(
            self.container,
            name=utils._generate_name(
                "e", self._name_prefix, "link_indctr_feature_ge"
            ),
            domain=uni_domain,
            description="Link the indicator variable with the feature which is Lower bounded using a big-M constraint",
        )
        ge_cons[uni_domain].where[
            (feat_thresh[..., "ge"] != 0) & s[uni_domain]
        ] = _feat_vars >= feat_thresh[..., "ge"] - _bound_big_m[..., "ge"] * (
            1 - ind_vars
        )

        le_cons = gp.Equation(
            self.container,
            name=utils._generate_name(
                "e", self._name_prefix, "link_indctr_feature_le"
            ),
            domain=uni_domain,
            description="Link the indicator variable with the feature which is Upper bounded using a big-M constraint",
        )
        le_cons[uni_domain].where[
            (feat_thresh[..., "le"] != 0) & s[uni_domain]
        ] = _feat_vars <= feat_thresh[..., "le"] + _bound_big_m[..., "le"] * (
            1 - ind_vars
        )

        return out, [
            assign_one_output,
            link_indctr_output,
            ub_output,
            lb_output,
            ge_cons,
            le_cons,
        ]
