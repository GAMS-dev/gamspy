from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DecisionTreeStruct:
    """
    Represents the components of `sklearn.tree`.

    This dataclass stores the core arrays (like children, features, thresholds,
    and values) that define the tree's architecture and decision rules.

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

    def __repr__(self):
        def arr_info(arr):
            return (
                f"shape={arr.shape}, dtype={arr.dtype}"
                if arr.size
                else "empty"
            )

        return (
            f"{self.__class__.__name__}("
            f"children_left={arr_info(self.children_left)}, "
            f"children_right={arr_info(self.children_right)}, "
            f"feature={arr_info(self.feature)}, "
            f"threshold={arr_info(self.threshold)}, "
            f"value={arr_info(self.value)}, "
            f"capacity={self.capacity}, "
            f"n_features={self.n_features})"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}: "
            f"{self.capacity} nodes, {self.n_features} features, "
            f"arrays: "
            f"children_left({self.children_left.size}), "
            f"children_right({self.children_right.size}), "
            f"feature({self.feature.size}), "
            f"threshold({self.threshold.size}), "
            f"value({self.value.size})"
        )
