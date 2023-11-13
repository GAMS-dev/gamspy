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

This behaviour allows ``GAMSPy`` to minimize the number of actual runs in the backend.

If you are familiar with ``GAMS`` language, and want to see the generated .gms file or .lst file,
you can specify the working directory of the ``Container``.

=========
Debugging
=========

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
to be saved in the current directory.

Another alternative is to use the ``generateGamsString`` function. This function returns the GAMS code 
generated up to that point as a string. This function should be used with delayed_execution mode, otherwise
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