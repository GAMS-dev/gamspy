.. _performance:

***********
Performance
***********

This section describes practical workflows in GAMSPy for improving the performance 
of your model.

Set-Based Assignments Instead of Python Loops
=============================================
One of the most important performance considerations when using GAMSPy is to
**prefer set-based (vectorized) assignments over explicit Python for loops**.
Set-based assignments are translated into native GAMS statements and executed
inside the GAMS execution engine, which is optimized for bulk operations.
In contrast, Python loops generate many small assignments that incur
Python overhead and repeated communication with the execution engine.

Simple Example
--------------

Consider computing a parameter over a set **i**.

Using a Python loop:

.. code-block:: python

    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, "i", records=range(1000))
    a = gp.Parameter(m, domain=i)
    b = gp.Parameter(m, domain=i)
    b.generateRecords()

    for ii in i.toList():
        a[ii] = 3 * b[ii]

This loop executes 1,000 individual assignments.

The same operation using a set-based assignment:

.. code-block:: python
    
    a[i] = 3 * b[i]

This generates a single GAMSPy assignment and is significantly faster and more
memory-efficient.

Assignments with Conditions
---------------------------
Set-based assignments can include conditional logic using where.

Using a Python loop:

.. code-block:: python

    for ii in i.toList():
        if int(ii) % 2 == 0:
            a[ii] = b[ii]

Set-based equivalent:

.. code-block:: python
    
    a[i].where[i.val % 2 == 0] = b[i]

The conditional expression is evaluated inside the GAMS engine and avoids
branching in Python.

Lagged Assignments
------------------
Some patterns that are often written as loops can still be expressed in a
set-based way.

For example, assigning values based on a predecessor element:

Python loop version:

.. code-block:: python
    
    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, records=range(1, 10))
    a = gp.Parameter(m, domain=i)
    a.generateRecords()
    f = gp.Parameter(m, domain=i)

    for n in i.toList():
        if int(n) > 1:
            f[n] = a[int(n) - 1] + 5

Set-based version:

.. code-block:: python
    
    f[i].where[gp.Ord(i) > 1] = a[i - 1] + 5

This avoids iteration in Python and lets GAMS handle the indexed operation.

When Loops Are Still Necessary
------------------------------
Not all logic can or should be converted into set-based assignments. Python
loops may still be appropriate when:

- Building sets dynamically
- Creating symbols conditionally
- Executing model solves inside a loop
- Performing logic that cannot be expressed algebraically

However, **numerical assignments over large index sets should almost always be
set-based**.

External Numerical Computations with setRecords
===============================================
In some cases, neither Python loops nor set-based GAMSPy assignments are ideal.
This is especially true for pure numerical computations such as simulations,
filters, or cumulative calculations that:

- are inherently sequential,
- do not require optimization or symbolic manipulation, and
- can be efficiently computed using vectorized numerical libraries.

In such cases, it can be advantageous to compute the data using other 
vectorized libraries such as NumPy and then transfer the results into 
GAMSPy using **setRecords**.

**Example: Autoregressive Time Series**

Consider the recursive process:

.. math::

    y_t = \alpha y_{t-1} + x_t

for **t = 1..T** with a given initial value.

**Step 1**: Compute the Time Series with NumPy

.. code-block:: python

    import numpy as np

    T = 1000
    alpha = 0.8

    # Time index
    t_values = np.arange(1, T + 1)

    # Exogenous input
    x = np.ones(T)

    # Allocate result array
    y = np.zeros(T)

    # Initial condition
    y[0] = 1.0

    # Fast numerical recursion (highly optimized C code)
    for i in range(1, T):
        y[i] = alpha * y[i - 1] + x[i]


Even though a Python loop is shown here, the heavy lifting is done on contiguous
NumPy arrays, which is substantially faster than assigning GAMSPy symbols
element by element.

(If desired, this can also be implemented using **scipy.signal.lfilter**
or **numba** for even higher performance.)

**Step 2**: Load the Results into GAMSPy

.. code-block:: python
    
    import gamspy as gp

    m = gp.Container()

    t = gp.Set(m, "t", records=t_values)
    y_param = gp.Parameter(m, "y", domain=t)

    # Convert to GAMSPy records format
    records = list(zip(t_values, y))

    y_param.setRecords(records)

At this point, the entire time series is transferred into the GAMS execution
engine in one operation.

**When to Use This Approach**

This pattern is especially useful when:

- The computation is purely numerical
- The logic is sequential or recursive
- Results are used as parameters in a model
- No symbolic relationships are needed inside GAMS

Typical examples include:

- Time-series simulations
- Demand or price forecasts
- Pre-computed trajectories
- Scenario generation
- Monte Carlo paths

Selective Synchronization
=========================

The state synchronization of symbols GAMSPy and the GAMS execution engine can be manipulated to improve performance in certain cases.
GAMSPy by default synchronizes the data of all symbols with the GAMS execution engine. This synchronization while in most cases done
efficiently can cause some performance degradation in some extreme cases and can be temporarily paused by setting ``.synchronize``
property to False. In this state GAMSPy assignment statements will only update the symbols in the GAMS execution enine but not update
the ``records`` attribute of the GAMSPy symbol. Similarly, the ``setRecords`` method will update the ``records`` attribute of the GAMSPy symbols
but will not update the data of the symbol in the GAMS execution engine. For example, while calculating Fibonacci numbers with the GAMS execution engine, it is not 
necessary to synchronize the records of symbol ``f`` with the GAMSPy ``records`` attribute in every iteration. 

.. code-block:: python

    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, 'i', records=range(1000))
    f = gp.Parameter(m, domain=i)
    f['0'] = 0
    f['1'] = 1

    f.synchronize = False
    for n in range(2,1000):
        f[str(n)] = f[str(n-2)] + f[str(n-1)]
    f.synchronize = True
    print(f.records)

By disabling the synchronization of ``f``, the state of ``f`` is synchronized with GAMS only at the end of the loop instead
of 999 times.

Disabling symbol synchronization should be done with caution because it can cause unexpected results. Here is an example:

.. code-block:: python

    import gamspy as gp
    
    m = gp.Container()
    f = gp.Parameter(m, records=1)
    g = gp.Parameter(m, records=10)
    f.synchronize = False
    f.setRecords(2)
    g[...] = f * g 
    print(g.records) # this will print g=10 not 20
    f[...] = 5
    print(f.records) # this will print f=2 not 5
    f.synchronize = True
    print(f.records) # this will print f=5 since the assignment was the last statement performed on f


Selective Loading on Solve
==========================
One can pick and choose the symbols that will be updated with new records on a solve statement. For certain models where you 
are only interested in the objective value or some key variables, loading the records of all symbols is unnecessary. In order to increase the performance 
by avoiding the load of symbols, one can specify ``load_symbols`` which are list of symbol objects. For example, in order to not load
any of the symbol's records, you can do the following:

.. code-block:: python

    model.solve(load_symbols=[])

This would prevent GAMSPy loading the records of symbols. If ``load_symbols`` is not specified, records of all symbols will be loaded.
Particular symbols can be loaded as follows:

.. code-block:: python

    x = Variable(m)
    ...  # specify your model here
    model.solve(load_symbols=[x])

This example would only load the records of ``x``.

Profiling and Execution Engine Memory Consumption
=================================================

GAMSPy has several options to allow profiling the instructions performed by the GAMS exeuction engine. The ``profile`` option controls whether an execution 
profile will be generated in the listing file. Alternatively, profiling information can be directred to a file of your choice
by using ``profile_file``. 

``monitor_process_tree_memory`` option allows GAMSPy to record the high-memory mark for the GAMS execution engine 
process tree excluding the Python process itself. ``memory_tick_interval`` can be used to set the wait interval in milliseconds between checks of the GAMSPy process 
tree memory usage. 

.. code-block:: python

    from gamspy import Container, Options
    m = Container(options=Options(profile=1, profile_file="<file_path>", monitor_process_tree_memory=True))

Setting GAMSPy Configurations
=============================
GAMSPy allows setting package wide options via :meth:`gp.set_options <gamspy.set_options>`. For example, 
one can skip the domain validation by setting ``DOMAIN_VALIDATION`` to 0. By default, GAMSPy performs 
domain validation which is helpful while writing mathematical models but might add a small overhead 
to the execution time.

.. code-block:: python

    import gamspy as gp
    gp.set_options({"DOMAIN_VALIDATION": 0})


.. note::
    One can also set the system directory via ``GAMSPY_GAMS_SYSDIR`` option. Beware that if a system directory 
    is given in the constructor of the ``Container``, it overrides this option. Package wide options can also 
    be set via environment variables. Environment variable names are always in the format of ``GAMSPY_<option_name>``.

.. code-block:: bash

    GAMSPY_DOMAIN_VALIDATION=0 python test.py

.. note::
    Note that package wide options are different than :ref:`model options <solve_options>`. While package wide options 
    affect the behavior of the whole package, model options affect the behavior of the solve process.

Here is a list of package wide options:

+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| Option Name                  | Type  | Description                                                                                                                              |
+==============================+=======+==========================================================================================================================================+
| VALIDATION                   | int   | Whether to enable all validations of GAMSPy. Set to 1 by default.                                                                        |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| DROP_DOMAIN_VIOLATIONS       | int   | Whether to drop domain violations in the records of a symbol. Set to 0 by default.                                                       |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| DOMAIN_VALIDATION            | int   | Whether to check for domain validation. Set to 1 by default.                                                                             |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| SOLVER_VALIDATION            | int   | Whether to validate the given solver name. Set to 1 by default.                                                                          |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| SOLVER_OPTION_VALIDATION     | int   | Whether to validate solver options. Set to 1 by default.                                                                                 |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| GAMS_SYSDIR                  | str   | Path to the GAMS system directory. Set to gamspy_base directory by default.                                                              |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| MAP_SPECIAL_VALUES           | int   | Map special values. Can be disabled for performance if there are no special values in the records. Set to 1 by default.                  |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| ASSUME_VARIABLE_SUFFIX       | int   | Activates or deactivates the automatic addition of .l or .scale attribute to variables on the right-hand side of assignments. Set to 1   |
|                              |       | by default. 0: deactivate, 1: use .l attribute, 2: use .scale attribute.                                                                 |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| USE_PY_VAR_NAME              | str   | "no": Do not try to use the Python variable name as the GAMSPy symbol name.                                                              |
|                              |       | "yes": Try using the variable name as the symbol name. If the variable name is not a valid GAMSPy symbol name, raise ValidationError.    |
|                              |       | "yes-or-autogenerate": Try using the variable name as the symbol name. If the name is not a valid symbol name or if it already exists in |
|                              |       | the container autogenerate a new name. (default value for this option).                                                                  |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| ALLOW_AMBIGUOUS_EQUATIONS    | str   | "auto": Do not allow ambiguous equations in MCP, EMP, MPEC, and RMPEC models but allow them in other model types.                        |
|                              |       | "no": Do not allow ambiguous equations in any model types.                                                                               |
|                              |       | "yes": Allow ambiguous equations in all model types.                                                                                     |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+

.. warning::
    GAMSPy validations are essential during development. Setting `VALIDATION` to 0 should only be done to improve the performance by skipping the validation steps after you are 
    convinced that your model works as intended. 