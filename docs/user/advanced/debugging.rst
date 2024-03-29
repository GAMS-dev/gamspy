.. _debugging:

*********
Debugging
*********

Specifying a Debugging Level
----------------------------
By default, GAMSPy will delete all temporary files generated by GAMS.
In order to keep the temporary files, ``debugging_level`` parameter can be
specified as ``keep`` or ``keep_on_error``.

.. code-block:: python

    from gamspy import Container
    m = Container(debugging_level="keep")
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve()
    print(m.working_directory)

In this example, you keep your working directory in the temp directory in your
operating system. The temp directories for Linux, Darwin, and Windows are usually 
``/tmp``, ``/var/tmp``, and ``C:\\Users\\username\\AppData\\Local\\Temp`` respectively. You can see the path for your 
model's temporary files by printing ``container.working_directory``.

Specifying the Working Directory
--------------------------------
Alternatively, you can specify a ``working_directory`` to keep the temporary files
generated by GAMS. If the working directory is specified, ``debugging_level`` will
not have any effect.

.. code-block:: python

    from gamspy import Container
    m = Container(working_directory=".")
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve()

In this example, specifying the working directory as the current directory causes temporary GAMS files 
(.gms, .lst, .g00. ,gdx files etc.) to be saved in the current directory.

Generating a Listing File
-------------------------
If one is only interested in the listing file that is generated after the solve statement, they can specify
the path for the lst file through :meth:`gamspy.Options`.

.. code-block:: python

    from gamspy import Container, Options
    m = Container()
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve(options=Options(listing_file="<path_to_the_listing_file>.lst"))

Generating a GDX File
---------------------
Sometimes, it might be useful to generate a GDX file which contains the records of the solved model.
The savepoint can be enabled through :meth:`gamspy.Options`. Check `GAMS Options <https://gams.com/latest/docs/UG_GamsCall.html#GAMSAOsavepoint>`_ 
to learn about the meaning of all savepoint values. 

.. code-block:: python

    from gamspy import Container, Options
    m = Container()
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve(options=Options(savepoint=1))


Generating a Log File
---------------------

The output of GAMS can be redirected to standard output or to a file by specifying the handle for the destination.
For example:

.. code-block:: python

    import sys
    from gamspy import Container
    m = Container(working_directory=".")
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve(output=sys.stdout)

The code snippet above redirects the GAMS execution output to your console by specifying the output as standard output.
You can also redirect the output to a file:

.. code-block:: python

    import sys
    from gamspy import Container
    m = Container(working_directory=".")
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    with open("mylog.txt", "w") as log:
        model.solve(output=log)

This code snippets redirects the output of the execution to a file named "mylog.txt".

If you want to have your log file generated in the working directory, ``create_log_file`` argument can be provided. 

.. code-block:: python

    import sys
    from gamspy import Container
    m = Container(working_directory=".")
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve(create_log_file=True)

This code snippet would generate a log file in the specified working directory. This argument is also useful for both
redirecting the output to standard output and generating the log file at the same time.


.. code-block:: python

    import sys
    from gamspy import Container
    m = Container(working_directory=".")
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve(output=sys.stdout, create_log_file=True)

This code snippet would redirect the output to your console as well as saving the log file in your working directory.


Inspecting Generated GAMS String
--------------------------------

Another alternative is to use the ``generateGamsString`` function. This function returns the GAMS code 
generated up to that point as a string.

.. code-block:: python

    from gamspy import Container
    m = Container()
    ....
    ....
    ....
    print(m.generateGamsString())
    ....
    ....
    ....

To see the generated GAMS statement for a certain symbol, ``getStatement`` function can be utilized. ::

    from gamspy import Container, Set
    m = Container()
    i = Set(m, "i", records=['i1', 'i2'])
    print(i.getStatement())


The code snippet above prints the GAMS statement for the symbol ``i``::

    'Set i(*);'

Inspecting Misbehaving (Infeasible) Models
------------------------------------------

Infeasibility is always a possible outcome when solving models. Infeasibilities in a model can be calculated by using
:meth:`gamspy.Model.compute_infeasibilities()`. This would list the infeasibilities in all equations of the model.
Infeasibilities in a single equation as well as infeasibilities in a single variable can be computed with
:meth:`gamspy.Equation.compute_infeasibilities()`, :meth:`gamspy.Variable.compute_infeasibilities()` respectively.
The infeasibilities are computed by finding the distance of level to the nearest bound (i.e. lower bound or upper bound).
While the compute_infeasibilities function of a model returns a dictionary where keys are the names of the equations and
values are the infeasibilities as Pandas DataFrames, compute_infeasibilities function of a variable or an equation, returns
a Pandas dataframe with infeasibilities.

.. code-block:: python

    from gamspy import Container
    m = Container()
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve()
    print(model.compute_infeasibilities())



Causes of infeasibility are not always easily identified. Solvers may report a particular equation as infeasible in cases 
where an entirely different equation is the cause. In these kind of complicated cases, one can dump all variables and equations 
in the listing file and inspect it. In the worst case, the solver returns no solution (model status 19: Infeasible - No Solution), 
leaving the variable levels untouched after a solve.


.. code-block:: python

    from gamspy import Container, Options
    m = Container()
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve(options=Options(equation_listing_limit=100, variable_listing_limit=100, listing_file="<path_to_the_listing_file>.lst"))

The level attribute of the variables used in the model determine the equation level and the Equation Listing in the listing file show 
potential infeasibilities using the INFES marker.

The solver dependent methods for dealing with infeasibilities can be used by providing solver_options. For example, you can turn on the 
conflict refiner of CPLEX solver also known as IIS finder if the problem is infeasible by providing a solver option.

.. code-block:: python

    from gamspy import Container, Options
    m = Container()
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve(options=Options(solver="CPLEX", solver_options={"iis": 1}))


An automated approach offered in GAMS/Cplex is known as FeasOpt (for Feasible Optimization) and it can be turned on by providing FeasOpt 
argument in solver_options  which turns feasible relaxation on.

.. code-block:: python

    from gamspy import Container, Options
    m = Container()
    ....
    ....
    ....
    specify your model here
    ....
    ....
    ....
    model.solve(options=Options(solver="CPLEX", solver_options={"FeasOpt": 1}))

There are also facilities of other solvers such as BARON, COPT, Gurobi, Lindo etc. which can be enabled in the same way.
To see all facilities, refer to the `Solver Manuals <https://gams.com/latest/docs/S_MAIN.html>`_.
