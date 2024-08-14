***************************
GAMSPy and Machine Learning
***************************

.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

GAMSPy heralds a new era of possibilities, offering a bridge between machine
learning and optimization that was previously challenging to traverse with GAMS
alone. Here's why GAMSPy stands out as the ultimate choice:

* Easy to understand, easy to write

  * ``eq1[...] = y == tanh(a @ W + b)``

* Versatility in Solver Selection

  * ``regression.solve(solver="your_favourite_solver")``

* It provides a strong algebraic language allows you to play with how neural network is implemented

* Built-in flexibility:

  * You are not limited with inference, you can try training your neural-network.

  * You can build the architecture from scratch in GAMSPy

* Development speed of Python combined with model generation speed of GAMS

  * Equations and variables are generated in GAMS not in Python giving GAMSPy a speed advantage


We are continuously developing our ML-related features. If you have specific
needs or require additional information, please feel free to reach out to us at
machinelearning@gams.com.

.. toctree::
   :maxdepth: 1

   ./introduction
   ./embed_nn
   ./ols
   ./logistic
   ./nn
