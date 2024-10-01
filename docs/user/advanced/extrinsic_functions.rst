.. _extrinsic_functions:

*******************
Extrinsic Functions
*******************

Mathematical functions play an important role especially for nonlinear models. 
Like other programming languages, GAMSPy provides a number of :meth:`intrinsic functions <gamspy.math>`. 
GAMSPy is used in an extremely diverse set of application areas and this creates frequent requests for 
the addition of new and often sophisticated and specialized functions. There is a trade-off between 
satisfying these requests and avoiding complexity not needed by most users. Extrinsic libraries allow 
users to import extrinsic functions from an external library into the GAMSPy model. However, these 
external libraries can only provide functionality for evaluating functions (including their first 
and second derivatives) at specific points.


.. admonition:: Information

   This documentation is a shortened version of the GAMS documentation on
   `Extrinsic Functions <https://gams.com/latest/docs/UG_ExtrinsicFunctions.html>`_.
   Since we skip many parts, we suggest reading the original documentation
   after reading this one.

Solvers that need to analyze the algebraic structure of the model instance are therefore 
not able to work with extrinsic functions. This includes the class of deterministic global solvers, 
see column "Global" in `this table <https://gams.com/latest/docs/S_MAIN.html#SOLVERS_MODEL_TYPES>`_.


Building Your Own Extrinsic library
-----------------------------------
.. warning::

   This feature requires a solid understanding of programming in C/C++ or Fortran,
   compilation, and linking processes.

An extrinsic function library consists of a specification part and a number of callbacks to 
evaluate the defined functions at an input point. 

Making the Library Available in GAMSPy
--------------------------------------

Extrinsic libraries are made available to GAMSPy with :meth:`gamspy.Container.importExtrinsicLibrary`.
This function takes two arguments: ``lib_path`` and ``functions``. ``lib_path`` is the path to the 
so, dylib, or dll file depending on the platform. ``functions`` argument allows user to import only 
the specified functions. It is a dictionary key is the desired function name in GAMSPy and value is 
the function name in the extrinsic library.


Here is an example, how an extrinsic function can be imported to be used in GAMSPy:

.. code-block:: python

    import gamspy as gp

    m = gp.Container()
    
    # This is a library which contains a function called 
    # Cosine that calculates the cosine of the given number 
    trilib = m.importExtrinsicLibrary("<path/to/my_library.so>", functions={"myCos": "Cosine"})

This example imports an extrinsic function `Cosine` from a shared object and 
exposes that function as `myCos` to GAMSPy. From now on, trilib.myCos can be
used in GAMSPy as an extrinsic function as follows:

.. code-block:: python

    p = Parameter(m, "p")
    p[...] = trilib.myCos(90)
    print(p.toValue()) # The result is 0 since cos(90) is zero.

    p[...] = trilib.myCos(0) * 3 # The result is 3 since cos(0) is 1 and 1 * 3 = 3

.. note::

    Extrinsic functions are limited to 20 scalar arguments and return a scalar value.

Extrinsic Functions vs External Equations
------------------------------------------
In addition to extrinsic functions, GAMSPy offers another facility to include additional mathematical functions: external equations. 
These equations are denoted by ``type=EquationType.EXTERNAL`` or ``type="external"``. A feasible solution for a model instance must satisfy all internal and 
external equations. External equations are introduced and discussed in :ref:`external_equations`. Similar to extrinsic functions, 
it is the users responsibility to provide routines that evaluates the external equation. Further, both facilities are especially 
pertinent to nonlinear models. An overview of some characteristics of extrinsic functions and external equations is given in 
the following table:

.. list-table:: Extrinsic Function vs External Equation
   :widths: 50 25 25
   :header-rows: 1

   * - Characteristic
     - Extrinsic Function
     - External Equation
   * - Maximum number of arguments
     - 20
     - No limit
   * - Available in statements
     - Yes
     - No
   * - Debugging support
     - Yes
     - Limited (via solver's derivative debugger)
   * - Returns Hessian to solver
     - Yes
     - Hessian Vector Product :math:`\nabla^2f(x)v` may be provided

