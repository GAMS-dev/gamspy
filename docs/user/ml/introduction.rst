********************************
Matrix Operations - Introduction
********************************

.. meta::
   :description: GAMSPy User Guide
   :keywords: Machine Learning, User, Guide, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance


Matrix operations are introduced to accompany the people who wants to use
machine learning with GAMS or for people who prefer using matrix notation. In
many machine-learning applications underlying algebra is written using matrix
operations rather than using an indexed algebra. While GAMS provides indexed
algebra to meet the demands of many common optimization applications, matrix
operations are introduced to complement indexed algebra. We still think with
the classical optimization problems indexed algebra is a better choice since it
is more intuitive.

However for ML practitioners writing this:

.. code-block:: python

   calc_mm_1[...] = z2 == w1.t() @ a1

is much easier than:

.. code-block:: python

   calc_mm_1[...] = z2[i, j] == Sum(k,  w1[k, i] * a1[k, j])


Considering common operations in ML domain we have implemented:

* Implicit default domains
* Easy matrix declaration
* Matrix multiplication
* Vector norms
* Trace
* Permute
* Improved domain tracking

In this introduction section, we summarize each of the features. You can find
more information about the features in their own respective pages.

Implicit default domains
========================
We have extended GAMSPy to make it work without explicitly specifying the
indices. Let a, b, c be variables (e.g. matrices) with the same domain. When
symbols are accessed without specific domains, the domains specified when
declaring the symbol is used implicitly.

.. code-block:: python

   import gamspy as gp
   import numpy as np
   m = gp.Container()
   i = gp.Set(m, "i")
   j = gp.Set(m, "j")
   k = gp.Set(m, "k")

   a = gp.Variable(m, name="a", domain=[i, j, k])
   b = gp.Variable(m, name="b", domain=[j, k])
   c = gp.Variable(m, name="c", domain=[i, j])
   assign_1 = gp.Equation(m, name="assign_1", domain=[i, j, k])

The following works now:

.. code-block:: python

   assign_1[...] = a == b + c

where in the old times you would have to at least:


.. code-block:: python

   assign_1[...] = a[...] == b[...] + c[...]


or

.. code-block:: python

   assign_1[...] = a[i, j, k] == b[j, k] + c[i, j]



Easy matrix declaration
=======================

Sometimes you need to generate parameters or variables as a matrix and do not
put too much meaning to its indices. ``gp.math.dim`` function is our suggested
method for declaring matrices, however parameters or variables defined without
using it still can be used in matrix operations.

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
``DenseDim50_1`` and ``DenseDim100_1``. ``DenseDim50_1`` contains elements
``0, 1, ..., 49`` whereas ``DenseDim100_1`` contains elements
``0, 1, ..., 99``. The word ``DenseDim`` is followed by the dimension,
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


Matrix Multiplication
=====================

We tried to follow matrix multiplication rules of PyTorch, ``torch.matmul``,
therefore, you are not limited to only rank-2 tensor multiplications. GAMSPy
symbols and expressions support matrix multiplication by overriding ``@``
operator.

.. admonition:: Information


   When performing matrix multiplication, the actual computation is not carried
   out immediately. Instead, an expression is generated. This approach is taken
   because matrix multiplication is computationally intensive, and since the
   elements involved include variables in addition to numbers, certain libraries
   and optimization techniques cannot be used to accelerate the process. By
   delegating this task to GAMS rather than handling it directly in Python, we
   achieve a faster model generation experience.



Validation of dimensions and shape of the output is determined by
dimensions of the tensors as follows:

* If both tensors are vectors, the dot product is returned.
* If both tensors are matrices, matrix multiplication is returned.
* If the first tensor is a vector and the second tensor is a matrix
  then 1 is prepended to the vector to make it a matrix multiplication.
  After the operation, the prepended dimension is removed.
* If the first tensor is a matrix, and the second tensor is a vector,
  matrix-vector product is returned.
* If the first tensor is a vector, and the second tensor has a rank larger
  than 2, the first tensor is prepended with 1 and then batched matrix
  multiplication is returned. After the operation, the prepended dimension is
  removed.
* If the first tensor has a rank larger than 2, and the second tensor is
  a vector, then batched matrix-vector product is returned.
* If both tensors have ranks larger than 2, then they must have same ranks.
  We currently do not support broadcasting. Batch dimensions must match.


You can see every case in the following example:

.. code-block:: python

   import gamspy as gp
   import numpy as np
   from gamspy.math import dim

   # since we will use this a lot
   rand = np.random.rand

   m = gp.Container()
   # inputs
   vec = gp.Parameter(m, name="vec", domain=dim((25,)), records=rand(25))
   mat = gp.Parameter(m, name="mat", domain=dim((25,25)), records=rand(25, 25))
   batched_mat = gp.Parameter(m, name="bmat",
        domain=dim((128, 25, 25)), records=rand(128, 25, 25)
   )

   # results
   f = gp.Parameter(m, name="f")
   res_mat = gp.Parameter(m, name="res_mat", domain=dim((25,25)), records=rand(25, 25))


   f[...] = vec @ vec # dot product
   print(f"{f.records=}")
   # 0  9.181418

   res_mat[...] = mat @ mat
   print(f"{res_mat.records}")
   #     DenseDim25_1 DenseDim25_2     value
   # 0              0            0  7.740069
   # 1              0            1  6.019976
   # 2              0            2  7.597765
   # 3              0            3  8.177436
   # 4              0            4  6.574309
   # ..           ...          ...       ...
   # 620           24           20  5.318084
   # 621           24           21  5.558328
   # 622           24           22  5.586886
   # 623           24           23  6.160951
   # 624           24           24  5.827358
