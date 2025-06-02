from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor


@dataclass
class DecisionTreeStruct:
    """
    Represents the components of a decision tree model.

    This dataclass stores the core arrays (like children, features, thresholds,
    and values) that define the tree's architecture and decision rules.
    It can be initialized directly with these components, or it can derive them
    from a scikit-learn DecisionTreeRegressor object.

    Attributes
    ----------
    children_left: np.ndarray
        An array where `children_left[i]` is the ID
        of the left child of node `i`. Leaf nodes have -1.
        Defaults to an empty numpy array.
    children_right: np.ndarray
        An array where `children_right[i]` is the ID
        of the right child of node `i`. Leaf nodes have -1.
        Defaults to an empty numpy array.
    feature: np.ndarray
        An array where `feature[i]` is the index of the
        feature used for splitting at node `i`. Leaf nodes have -2.
        Defaults to an empty numpy array.
    threshold: np.ndarray
        An array where `threshold[i]` is the threshold
        value used for splitting at node `i` based on `feature[i]`.
        Leaf nodes have -2.0. Defaults to an empty numpy array.
    value: np.ndarray
        An array (typically 2D for scikit-learn trees,
        squeezed to 1D for single-output regressors) where `value[i]`
        contains the prediction value(s) for node `i`. For leaf nodes,
        this is the final prediction. Defaults to an empty numpy array.
    capacity : int
        The total number of nodes allocated in the underlying
        tree structure arrays. Defaults to 0.
    n_features : int
        The number of features the decision tree was trained on
        or expects as input. Defaults to 0.
    _regressor_source: DecisionTreeRegressor | None
        An optional source to initialize the tree attributes from.
        This parameter is used only during the `__post_init__` phase and is not
        stored as an instance attribute.

    Note:
            If `_regressor_source` is None, all the attributes mentioned above will need to be set externally.

    """

    children_left: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False
    )
    children_right: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False
    )
    feature: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False
    )
    threshold: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False
    )
    value: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    capacity: int = 0
    n_features: int = 0
    _regressor_source: InitVar[
        DecisionTreeRegressor | RandomForestRegressor | None
    ] = None

    def __post_init__(
        self,
        _regressor_source: DecisionTreeRegressor
        | RandomForestRegressor
        | None,
    ):
        if _regressor_source is not None:
            regressor_type = type(_regressor_source)
            if (
                regressor_type.__name__ == "DecisionTreeRegressor"
                and regressor_type.__module__.startswith("sklearn.tree")
            ) or (
                regressor_type.__name__ == "RandomForestRegressor"
                and regressor_type.__module__.startswith("sklearn.ensemble")
            ):
                self._initialize_from_sklearn_object(_regressor_source)
            else:
                raise ValidationError(
                    f"Unsupported _regressor_source type: {type(_regressor_source)}. "
                    "Expected DecisionTreeRegressor | RandomForestRegressor | None."
                )

    def _initialize_from_sklearn_object(
        self, regressor: DecisionTreeRegressor
    ):
        """
        Initializes the tree attributes from a fitted scikit-learn DecisionTreeRegressor.
        """
        _trained = hasattr(regressor, "tree_") and regressor.tree_ is not None

        if not _trained:
            raise ValidationError(
                f"{regressor} is not fitted or tree_ attribute is missing."
            )

        self.children_left = regressor.tree_.children_left
        self.children_right = regressor.tree_.children_right
        self.feature = regressor.tree_.feature
        self.threshold = regressor.tree_.threshold
        self.value = regressor.tree_.value[:, :, 0]
        self.capacity = regressor.tree_.capacity
        self.n_features = regressor.tree_.n_features

    def _check_input(self):
        """
        Validates that the core tree attributes are populated and have valid basic properties.

        Raises:
            ValidationError: If any essential attribute is empty or invalid.
        """
        if self.children_left.size == 0:
            raise ValidationError("Attribute 'children_left' is empty.")
        if self.children_right.size == 0:
            raise ValidationError("Attribute 'children_right' is empty.")
        if self.feature.size == 0:
            raise ValidationError("Attribute 'feature' is empty.")
        if self.threshold.size == 0:
            raise ValidationError("Attribute 'threshold' is empty.")
        if self.value.size == 0:
            raise ValidationError("Attribute 'value' is empty.")
        if self.n_features <= 0:
            raise ValidationError("'n_features' must be a positive integer.")
        if self.capacity <= 0:
            raise ValidationError("'capacity' must be a positive integer.")
