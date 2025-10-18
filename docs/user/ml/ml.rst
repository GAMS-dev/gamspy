***************************
GAMSPy and Machine Learning
***************************

.. include:: badges.rst

.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

GAMSPy heralds a new era of possibilities, bridging the gap between machine learning
and optimization that was previously difficult to cross using GAMS alone. Here's why
GAMSPy is the ultimate choice:

* Easy to understand, easy to write

  * ``eq1[...] = y == tanh(a @ W + b)``

* Versatility in Solver Selection

  * ``regression.solve(solver="your_favourite_solver")``

* It provides a robust algebraic language that allows you to experiment with how a neural network is implemented.

* Built-in flexibility:

  * You are not limited to inference; you can also :doc:`train your neural network <nn>`.

  * You can build the architecture from scratch using GAMSPy

* Development speed of Python combined with model generation speed of the GAMS execution engine

  * Equations and variables are generated in GAMS, not in Python, giving GAMSPy a speed advantage.


We are continually developing our ML-related features. If you have specific
needs or require additional information, please use our `Discourse platform <https://forum.gams.com>`_.

.. toctree::
   :maxdepth: 1

   ./introduction
   ./formulations
   ./classic_ml
   ./embed_nn


Complexity Overview
===================

This section gives an overview of the "friendly" model types used in the
formulations. For instance, if you only work with
:meth:`Linear <gamspy.formulations.Linear>`, the problem remains linear and can be solved as
an LP. On the other hand, if you combine it with
:meth:`relu_with_binary_var <gamspy.math.relu_with_binary_var>`, the formulation becomes a MIP.

Visit :doc:`NN Formulations <formulations>` for more information.

=================================================================================      =========================
NN Formulation                                                                         Type
=================================================================================      =========================
:meth:`Linear <gamspy.formulations.Linear>`                                            |lp-badge|
:meth:`Conv1d <gamspy.formulations.Conv1d>`                                            |lp-badge|
:meth:`Conv2d <gamspy.formulations.Conv2d>`                                            |lp-badge|
:meth:`MaxPool2d <gamspy.formulations.MaxPool2d>`                                      |mip-badge|
:meth:`MinPool2d <gamspy.formulations.MinPool2d>`                                      |mip-badge|
:meth:`AvgPool2d <gamspy.formulations.AvgPool2d>`                                      |lp-badge|
:meth:`relu_with_binary_var <gamspy.math.relu_with_binary_var>`                        |mip-badge|
:meth:`relu_with_complementarity_var <gamspy.math.relu_with_complementarity_var>`      |qcp-badge| (non-convex)
:meth:`relu_with_sos1_var <gamspy.math.relu_with_sos1_var>`                            |mip-badge|
:meth:`leaky_relu_with_binary_var <gamspy.math.leaky_relu_with_binary_var>`            |mip-badge|
:meth:`softmax <gamspy.math.softmax>`                                                  |nlp-badge|
:meth:`softplus <gamspy.math.softplus>`                                                |nlp-badge|
:meth:`log_softmax <gamspy.math.log_softmax>`                                          |nlp-badge|
:meth:`tanh <gamspy.math.tanh>`                                                        |nlp-badge|
:meth:`sigmoid <gamspy.math.sigmoid>`                                                  |nlp-badge|
:meth:`TorchSequential <gamspy.formulations.TorchSequential>`                          Depends\ [*]_
=================================================================================      =========================

.. [*]  The complexity of a TorchSequential model depends both on the underlying sequential structure and on how it is embedded.


While types are shown with color coding, these colors (apart from the LP case)
do not directly reflect runtime. In practice, there are situations where the
complementarity formulation for ReLU can be faster than the binary formulation,
and vice versa.


Visit :doc:`ML Formulations <classic_ml>` for more information.

=================================================================================      =========================
ML Formulation                                                                         Type
=================================================================================      =========================
:meth:`RegressionTree <gamspy.formulations.RegressionTree>`                            |mip-badge|
:meth:`RandomForest <gamspy.formulations.RandomForest>`                                |mip-badge|
:meth:`GradientBoosting <gamspy.formulations.GradientBoosting>`                        |mip-badge|
=================================================================================      =========================

