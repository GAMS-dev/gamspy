********************************
Matrix Operations - Introduction
********************************

.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance


Matrix operations are introduced to accompany the people who wants to use machine
learning with GAMS. In many machine-learning applications underlying algebra is written
using matrix operations rather than using an indexed algebra. While GAMS provides indexed
algebra to meet the demands of many common optimization applications, matrix operations
are introduced to complement indexed algebra. We still think with the classical optimization
problems indexed algebra is a better choice since it is more intuitive.

However for ML experts writing this:

.. code-block:: python

   calc_mm_1[...] = a2 == w1.t() @ x1

is much easier than:

.. code-block:: python

   calc_mm_1[...] = a2[i, j] == Sum(k,  w1[k, i] * x1[k, j])


Considering common operations in ML domain we have implemented:

* matrix multiplication
* dim
* vector_norm
* next_alias
* trace
* permute

In addition to these, we have extended GAMSPy's indexing logic to
make it work without explicitly specifying the indices. Also GAMSPy
now tracks domains of expressions and symbols to determine if a matrix
multiplication is possible.

Let a, b and c be parameters (e.g. matrices) with the same domain. The following works now:

.. code-block:: python

   assign_1[...] = a == b + c

where in the old times you would have to at least:


.. code-block:: python

   assign_1[...] = a[...] == b[...] + c[...]


or

.. code-block:: python

   assign_1[...] = a[i, j] == b[i, j] + c[i, j]



Declaring a matrix
==================

Sometimes you need to generate parameters or variables as a matrix
and do not put too much meaning to its indices.
``gp.math.dim`` function is our suggested method for declaring matrices,
however parameters or variables defined without using it still can
be used in matrix operations.

See the following example for using ``dim`` function:

.. code-block:: python

   import gamspy as gp
   import numpy as np
   from gamspy.math import dim

   w1_data = np.random.rand(50, 100)
   m = gp.Container()
   w = gp.Parameter(m, name="w1", domain=dim((50, 100)), records=w1_data)
   w.records


Output:

.. code-block:: text

        DenseDim50_1 DenseDim100_1     value
   0               0             0  0.429909
   1               0             1  0.831080
   2               0             2  0.656872
   3               0             3  0.959341
   4               0             4  0.758202
   ...           ...           ...       ...
   4995           49            95  0.847640
   4996           49            96  0.870642
   4997           49            97  0.369344
   4998           49            98  0.233120
   4999           49            99  0.704139


As you can see under the hood, GAMSPy generates two sets for you called
``DenseDim50_1`` and ``DenseDim100_1``. Unsuprisingly ``DenseDim50_1``
contains elements ``0, 1, ..., 49`` whereas ``DenseDim100_1`` contains
elements ``0, 1, ..., 99``. The word ``DenseDim`` is followed by the dimension,
underscore and then the alias number where ``1`` refering the original set.

.. code-block:: python

   ...
   w2_data = np.random.rand(50, 50)
   w2 = gp.Parameter(m, name="w2", domain=dim((50, 50)), records=w2_data)
   w2.records


Output:

.. code-block:: text

        DenseDim50_1 DenseDim50_2     value
   0               0            0  0.902650
   1               0            1  0.268446
   2               0            2  0.133204
   3               0            3  0.931026
   4               0            4  0.283675
   ...           ...          ...       ...
   2495           49           45  0.931849
   2496           49           46  0.991170
   2497           49           47  0.754725
   2498           49           48  0.924075
   2499           49           49  0.437851


You can see in the output ``DenseDim50_2`` is used instead of repeating
the same set twice. ``DenseDim50_2`` is an alias of set ``DenseDim50_1``.
This is done because it is more convenient for us when doing matrix
multiplications.

In the same way you can generate variable matrices:

.. code-block:: python

   ...
   x = gp.Variable(m, name="x", domain=dim((50, 50)))


You are not limited to 2 dimensions. Many times in ML applications we need more than 2 dimensions:

.. code-block:: python

   ...
   y = gp.Variable(m, name="y", domain=dim((128, 500, 1000)))

However, you are limited to 20 dimensions as GAMS supports up to 20 dimensions:

.. code-block:: python

   ...
   # The following would not work
   z = gp.Variable(m, name="z", domain=dim(list(range(1, 100))))
