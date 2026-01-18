.. _debugging:

*********
Debugging
*********

This section describes practical workflows in GAMSPy for debugging models, 
inspecting generated GAMS code, and diagnosing infeasibilities.

Temporary Files and Debug Levels
================================
By default, GAMSPy removes all temporary files after a successful execution. 
You can control this behavior with the ``debugging_level`` argument of :meth:`Container <gamspy.Container>`.

Debugging Levels
----------------
- **keep_on_error (default)**: Keep temporary files only if an error occurs.
- **keep**: Always keep temporary files.
- **delete**: Always delete temporary files, even if errors occur.

.. code-block:: python

    from gamspy import Container

    m = Container(debugging_level="keep")
    print(m.working_directory)

When the debugging level is set to **keep**, GAMSPy preserves:

- Generated GAMS files (.gms files).
- Execution results (e.g. .gdx, .lst files).

This makes it easier to reproduce and debug execution issues.
Temporary directories typically reside in:

- Linux: **/tmp** or **/var/tmp**
- macOS (Darwin): **/var/tmp**
- Windows: **C:\Users\<username>\AppData\Local\Temp**

.. note::
    If a custom ``working_directory`` is provided, ``debugging_level`` is ignored. 
    This safeguard prevents accidental deletion of user-specified directories (e.g., / or C:\).

Specifying the Working Directory
--------------------------------
Instead of relying on system temporary directories, you may explicitly set a working directory.

.. code-block:: python

    from gamspy import Container

    m = Container(working_directory=".")

In this example, specifying the working directory as the current directory causes temporary GAMSPy files 
to be saved in the current directory. 

.. note::
    Be aware that unless the debugging_level is set to ``keep``, each chunk of instructions 
    send by GAMSPy to the execution engine will override the previous files. 

Generating a Listing File
-------------------------
If you only need the solver listing file produced after a solve, specify a path via :meth:`gamspy.Options`.

.. code-block:: python

    model.solve(options=Options(listing_file="<path_to_the_listing_file>.lst"))

Generating a Solution Point GDX File
====================================
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
============================================

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
================================

GAMSPy takes advantage of the high performance GAMS execution system by generating GAMS code and sending them to GAMS.
Hence, a way to debug GAMSPy is to inspect this GAMS code. Instead of inspecting temporary files in the working directory 
that contains this GAMS code, one can use the ``generateGamsString`` function. This function returns the GAMS code generated 
up to that point as a string. In order to use this function, ``debugging_level`` of the Container must be set to "keep".

.. code-block:: python

    from gamspy import Container
    m = Container(debugging_level="keep")
    ... # Definition of your model
    print(m.generateGamsString())

One can also redirect the executed GAMS code into a file by providing ``path`` argument:

.. code-block:: python

    from gamspy import Container
    m = Container(debugging_level="keep")
    ... # Definition of your model
    print(m.generateGamsString(path="executed_code.gms"))

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
================================================
The user may determine whether the model generated is the the model that the user has intended by studying the
equation listing and variable listing. For more information about how this can be done, see 
:ref:`inspecting_generated_equations` and :ref:`inspecting_generated_variables`.

Inspecting Misbehaving (Infeasible) Models
==========================================

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
