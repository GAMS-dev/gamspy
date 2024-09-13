***************************
GAMSPy and Machine Learning
***************************

.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

GAMSPy heralds a new era of possibilities, bridging the gap between machine learning 
and optimization that was previously difficult to cross using GAMS alone. Here’s why 
GAMSPy is the ultimate choice:

* Easy to understand, easy to write

  * ``eq1[...] = y == tanh(a @ W + b)``

* Versatility in Solver Selection

  * ``regression.solve(solver="your_favourite_solver")``

* It provides a robust algebraic language that allows you to experiment with how a neural network is implemented.

* Built-in flexibility:

  * You are not limited to inference; you can also train your neural network.

  * You can build the architecture from scratch using GAMSPy

* Development speed of Python combined with model generation speed of the GAMS execution engine

  * Equations and variables are generated in GAMS, not in Python, giving GAMSPy a speed advantage.


We are continually developing our ML-related features. If you have specific
needs or require additional information, please use our `Discourse platform <https://forum.gams.com/c/gamspy-help>`_.

.. toctree::
   :maxdepth: 1

   ./introduction
   ./formulations
   ./embed_nn
   ./ols
   ./logistic
   ./nn
