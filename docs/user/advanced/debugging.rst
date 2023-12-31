.. _debugging:

*****************************
Execution Modes and Debugging
*****************************

===============
Execution Modes
===============

GAMSPy supports two execution modes through the ``Container``.

Normal Execution
----------------
By default, normal execution is enabled. In this mode, each assignment or record reading attempt triggers 
the execution of generated code, and the results are saved. This mode is suitable for debugging, although 
it may be slower than delayed execution.

.. code-block:: python

    from gamspy import Container, Set, Parameter
    m = Container()
    i = Set(m, "i", records=["i1", "i2"])
    a = Parameter(m, "a", domain=[i], records=[("i1", 1), ("i2", 2)])
    a[i] = a[i] * 90

In normal execution, the last line executes the actual computation in GAMS, as soon as it's specified.

Delayed Execution
-----------------
Delayed execution is a mode designed for better performance. Assignments are not executed until the 
``solve`` function of a model is called or an attempt is made to read the records of a dirty symbol.
A dirty symbol is a symbol that was assigned a new value in previous lines.

.. code-block:: python

    from gamspy import Container, Set, Parameter
    m = Container(delayed_execution=True)
    i = Set(m, "i", records=["i1", "i2"])
    a = Parameter(m, "a", domain=[i], records=[("i1", 1), ("i2", 2)])
    a[i] = a[i] * 90 # This line is not executed yet. a is dirty now.
    print(a.records) # An attempt to read a dirty symbol cause a GAMS run. a is not dirty anymore.

This behaviour allows ``GAMSPy`` to minimize the number of actual runs in the backend. The performance
improvement compared to normal execution mode might differ depending the model significantly.

If you are familiar with ``GAMS`` language, and want to see the generated .gms file or .lst file,
you can specify the working directory of the ``Container``.

=========
Debugging
=========


Specifying the Working Directory
--------------------------------
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

If you want to have your log file generated in the working directory, `create_log_file` argument can be provided. 

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
generated up to that point as a string. This function must be used with delayed_execution mode, otherwise
you will almost always get an empty string since the previous statements already ran with GAMS and the
results were loaded into the container.

.. code-block:: python

    from gamspy import Container
    m = Container(delayed_execution=True, working_directory=".")
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


The code snippet above prints the GAMS statement for the symbol `i`::

    'Set i(*);'