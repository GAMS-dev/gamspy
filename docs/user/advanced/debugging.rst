.. _debugging:

*************************
Debugging and Performance
*************************

Specifying a Debugging Level
----------------------------
By default, GAMSPy will delete all temporary files generated if there are no errors in the execution.
In order to keep the temporary files, the ``debugging_level`` parameter can be specified as ``keep``. ``keep_on_error`` 
is the default behaviour. This debug level keeps the temporary files only if there are errors in the execution. ``delete``
can be used to delete all temporary files even if there are errors.

.. code-block:: python

    from gamspy import Container

    m = Container(debugging_level="keep")
    print(m.working_directory)

In this example, you keep your working directory in the temporary directory in your
operating system. The temporary directories for Linux, Darwin, and Windows are usually 
``/tmp``, ``/var/tmp``, and ``C:\\Users\\<username>\\AppData\\Local\\Temp`` respectively. You can see the path for your 
model's temporary files by printing ``<container>.working_directory``.

``keep`` value for ``debugging_level`` also generates files with instructions (``.gms``) for each chunk send to the execution engine.
The results of the execution are also kept in files (``.gdx``, ``.lst`` and ``.log``) for easier debugging of your model.

.. note::
    If one specifies a working directory, setting debugging_level to ``delete`` or ``keep_on_errors`` has no effect.
    This behaviour is intentional to avoid potential problems with deleting the working_directory. For example, if the user 
    specifies working_directory="/" on Linux or "C:\" and debugging level as "delete", we don't want to nuke the whole system. 

Specifying the Working Directory
--------------------------------
Alternatively, you can specify a ``working_directory`` to keep the temporary files
generated by GAMSPy.

.. code-block:: python

    from gamspy import Container

    m = Container(working_directory=".")

In this example, specifying the working directory as the current directory causes temporary GAMSPy files 
to be saved in the current directory. Be aware that unless the debugging_level
is set to ``keep``, each chunk of instructions send by GAMSPy to the execution engine will override the previous files. 

Generating a Listing File
-------------------------
If one is only interested in the listing file that is generated after the solve statement, they can specify
the path for a listing file through :meth:`gamspy.Options`.

.. code-block:: python

    model.solve(options=Options(listing_file="<path_to_the_listing_file>.lst"))

Generating a Solution Point GDX File
------------------------------------
In order to have a persistent copy of a solution (for use as an inital point in a subsequent solve), this
solution can be saved to a GDX file which contains all primal and dual solution records of the solved model.
This savepoint facility can be enabled through :meth:`gamspy.Options`:

.. code-block:: python

    model = Model(m, name="my_model", ...)
    model.solve(options=Options(savepoint=1))

``savepoint=1`` creates a GDX file with ``my_model_p.gdx``, ``savepoint=2`` create a GDX file with ``my_model_p<n>.gdx``.
Where ``n`` is the sequence number of the model ``model.sequence`` available after the solve. Since the GDX file contains the model
name, it is necessary to provide the ``name`` argument in the model constructor.

The savepoint goes together with the corresponding loadpoint option:

.. code-block:: python

    model.solve(options=Options(loadpoint="my_model_p.gdx"))

This loads the variable and equation levels and marginal from the GDX file provided prior to the solve as an initial point.

Redirecting Output and Generating a Log File
--------------------------------------------

The output of a solve (mostly the solver log to monitor solution progress) can be redirected to the standard output or to 
a file by specifying the handle for the destination. For example:

.. code-block:: python

    import sys

    model.solve(output=sys.stdout)

One can also redirect the output of all GAMS executions by specifying ``output`` argument of the :meth:`Container <gamspy.Container>`.

.. code-block:: python

    from gamspy import Container, Set
    import sys

    m = Container(output=sys.stdout)
    i = Set(m)

The code snippet above redirects the GAMS execution output to your console by specifying the output as standard output.
You can also redirect the output to a file:

.. code-block:: python

    with open("my_output.txt", "w") as log:
        model.solve(output=log)

This code snippet redirects the output of the execution to a file named "mylog.txt".

If you want GAMSPy to redirect logs to a file, the ``log_file`` option can be provided:

.. code-block:: python

    model.solve(options=Options(log_file="my_log_file.txt"))

This code snippet would generate a log file in the specified working directory. The log can also be 
redirected to both a file and the console simultaneously.


.. code-block:: python

    import sys
    model.solve(output=sys.stdout, options=Options(log_file="my_log_file.txt"))

This code snippet would redirect the log to standard output as well as saving a log file ``my_log_file.txt`` in your working directory.

.. _generate_gams_string:

Inspecting Generated GAMS String
--------------------------------

GAMSPy takes advantage of the high performance GAMS execution system by generating GAMS code and sending them to GAMS.
Hence, a way to debug GAMSPy is to inspect this GAMS code. Instead of inspecting temporary files in the working directory 
that contains this GAMS code, one can use the ``generateGamsString`` function. This function returns the GAMS code generated 
up to that point as a string. In order to use this function, ``debugging_level`` of the Container must be set to "keep".

.. code-block:: python

    from gamspy import Container
    m = Container(debugging_level="keep")
    ... # Definition of your model
    print(m.generateGamsString())

By default, ``generateGamsString`` returns exactly the same string that is executed, but ``show_raw`` argument
allows users to see only the raw model without any data or dollar calls or other necessary statements to make the model work.

For example, the following code snippet:

.. code-block:: python

    from gamspy import Container, Set
    m = Container()
    i = Set(m, "i")
    j = Set(m, "j")
    print(m.generateGamsString(show_raw=True))

generates: ::

    Set i(*);
    Set j(*);

Without ``show_raw`` argument, the following string would be generated: ::

    $onMultiR
    $onUNDF
    $gdxIn <gdx_in_file_name>
    Set i(*);
    $loadDC i
    $offUNDF
    $gdxIn
    $onMultiR
    $onUNDF
    $gdxIn <gdx_in_file_name>
    Set j(*);
    $loadDC j
    $offUNDF
    $gdxIn


To see the generated GAMS declaration for a certain symbol, ``getDeclaration`` function can be utilized. ::

    from gamspy import Container, Set
    m = Container()
    i = Set(m, "i", records=['i1', 'i2'])
    print(i.getDeclaration())


The code snippet above prints the GAMS statement for the symbol ``i``::

    'Set i(*);'

To see the generated GAMS definition for a certain symbol, ``getDefinition`` function can be utilized. ::

    from gamspy import Sum, Container, Set, Parameter, Variable, Equation
        
    m = Container()
    i = Set(m, name="i")
    a = Parameter(m, name="a", domain=i)
    z = Variable(m, name="z")

    eq = Equation(m, name="eq")
    eq[...] = Sum(i, a[i]) <= z
    print(eq.getDefinition())


The code snippet above prints the GAMS statement for the symbol ``i``::

    'eq .. sum(i,a(i)) =l= z;'

Inspecting the Generated Equations and Variables
------------------------------------------------
The user may determine whether the model generated is the the model that the user has intended by studying the
equation listing and variable listing. For more information about how this can be done, see 
:ref:`inspecting_generated_equations` and :ref:`inspecting_generated_variables`.

Inspecting Misbehaving (Infeasible) Models
------------------------------------------

Infeasibility is always a possible outcome when solving models. Infeasibilities in a model can be calculated by using
:meth:`gamspy.Model.computeInfeasibilities`. This would list the infeasibilities in all equations and variables of the model.
Infeasibilities in a single equation as well as infeasibilities in a single variable can be computed with
:meth:`gamspy.Equation.computeInfeasibilities`, :meth:`gamspy.Variable.computeInfeasibilities` respectively.
The infeasibilities are computed by finding the outside distance of level to the nearest bound (i.e. lower bound or upper bound).
While the computeInfeasibilities function of a model returns a dictionary where keys are the names of the equations and
values are the infeasibilities as Pandas DataFrames, computeInfeasibilities function of a variable or an equation, returns
a Pandas dataframe with infeasibilities.

.. code-block:: python

    model.solve()
    print(model.computeInfeasibilities())

Causes of infeasibility are not always easily identified. Solvers may report a particular equation as infeasible in cases 
where an entirely different equation is the cause. There are solver-dependent methods for dealing with infeasibilities that can be used by providing solver_options. For example, you can turn on the 
conflict refiner of CPLEX solver also known as IIS finder if the problem is infeasible by providing a solver option. The results of such an
analysis are often written to the log and/or the listing file.

.. code-block:: python

    model.solve(options=Options(solver="cplex", solver_options={"iis": 1}))

Some solvers offer an automated approach to find the smallest feasible relaxation of constraints to make the model feasible.
For example, in GAMS/Cplex this is known as ``FeasOpt`` (for Feasible Optimization). It can 
be turned on by providing the ``feasopt`` argument in ``solver_options``, which turns feasible relaxation on.

.. code-block:: python

    model.solve(options=Options(solver="cplex", solver_options={"feasopt": 1}))

The relaxation is available through :meth:`gamspy.Model.computeInfeasibilities`. There are similar facilities available with other solvers
such as BARON, COPT, Gurobi, Lindo etc. which can be enabled in a similar way.
To see all facilities, refer to the `solver manuals <https://gams.com/latest/docs/S_MAIN.html>`_.


Selective Synchronization
-------------------------

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
--------------------------
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
-------------------------------------------------

GAMSPy has several options to allow profiling the instructions performed by the GAMS exeuction engine. The ``profile`` option controls whether an execution 
profile will be generated in the listing file. Alternatively, profiling information can be directred to a file of your choice
by using ``profile_file``. 

``monitor_process_tree_memory`` option allows GAMSPy to record the high-memory mark for the GAMS execution engine 
process tree excluding the Python process itself. ``memory_tick_interval`` can be used to set the wait interval in milliseconds between checks of the GAMSPy process 
tree memory usage. 

.. code-block:: python

    from gamspy import Container, Options
    m = Container(options=Options(profile_file="<file_path>", monitor_process_tree_memory=True))

Setting GAMSPy Configurations
-----------------------------
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
| LAZY_EVALUATION              | int   | Whether to evaluate expressions lazily. Lazy evaluation might cause recursion depth errors for very long expression. Set to 0 by default |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| ASSUME_VARIABLE_SUFFIX       | int   | Activates or deactivates the automatic addition of .l or .scale attribute to variables on the right-hand side of assignments. Set to 1   |
|                              |       | by default. 0: deactivate, 1: use .l attribute, 2: use .scale attribute.                                                                 |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+
| USE_PY_VAR_NAME              | str   | "no": Do not try to use the Python variable name as the GAMSPy symbol name. (default value for this option)                              |
|                              |       | "yes": Try using the variable name as the symbol name. If the variable name is not a valid GAMSPy symbol name, raise ValidationError.    |
|                              |       | "yes-or-autogenerate": Try using the variable name as the symbol name. If the name is not a valid symbol name, autogenerate a new name.  |
+------------------------------+-------+------------------------------------------------------------------------------------------------------------------------------------------+


.. warning::
    GAMSPy validations are essential during development. Setting `VALIDATION` to 0 should only be done to improve the performance by skipping the validation steps after you are 
    convinced that your model works as intended. 
