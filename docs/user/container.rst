.. _container:

*********
Container
*********

.. code-block:: python
    
    from gamspy import Container, Set
    
    m = Container()
    i = Set(m, "i", records = ["seattle", "san-diego"])
    j = Set(m, "j", records = ['new-york', 'chicago', 'topeka'])

The ``Container`` class in GAMSPy serves as a central hub for managing essential data, sets, parameters, variables, 
and constraints, providing a structured approach for optimization problems. Every symbol in your optimization problem 
should belong to a ``Container``.

===========================
Reading and Writing Symbols
===========================

The ``Container`` class offers I/O functions for reading and writing symbols.

Writing
-------
Symbols created within a specific ``Container`` can be saved to a GDX file using the ``write`` function.

.. code-block:: python
    
    from gamspy import Set, Container
    
    m = Container()
    i = Set(m, "i", records=["seattle", "san-diego"])
    m.write("data.gdx")

Reading
-------
Symbol records can be read from a GDX file by either specifying the `load_from` argument during the ``Container`` construction or by using the ``read`` function.

To create a ``Container`` with symbols from a GDX file, use the `load_from` argument:

.. code-block:: python

    from gamspy import Container
    m = Container(load_from="data.gdx")
    print(m.listSymbols())

We can verify that symbol ``i`` is in the container ``m``.

Alternatively, you can use the ``read`` function to populate the container.

.. code-block:: python

    from gamspy import Container
    m = Container()
    m.read("data.gdx")
    print(m.listSymbols())

===============
Execution Types
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
    i = Set(m, "i", records=['i1', 'i2'])
    a = Parameter(m, "a", domain=[i], records=[('i1', 1), ('i2', 2)])
    a[i] = a[i] * 90

In normal execution, the last line executes the actual computation in GAMS, as soon as it's specified.

Delayed Execution
-----------------
Delayed execution is a mode designed for better performance. Assignments are not executed until the 
`solve` function of a model is called or an attempt is made to read the records of a dirty symbol.
A dirty symbol is a symbol that was assigned a new value in previous lines.

.. code-block:: python

    from gamspy import Container, Set, Parameter
    m = Container(delayed_execution=True)
    i = Set(m, "i", records=['i1', 'i2'])
    a = Parameter(m, "a", domain=[i], records=[('i1', 1), ('i2', 2)])
    a[i] = a[i] * 90 # This line is not executed yet. a is dirty now.
    print(a.records) # An attempt to read a dirty symbol cause a GAMS run. a is not dirty anymore.

This behaviour allows ``GAMSPy`` to minimize the number of actual runs in the backend.

=========
Debugging
=========

If you are familiar with ``GAMS`` language, and want to see the generated .gms file or .lst file,
you can specify the working directory of the ``Container``

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
generated up to that point as a string.

.. code-block:: python

    from gamspy import Container
    m = Container(working_directory=".")
    ....
    ....
    ....
    print(m.generateGamsString())
    ....
    ....
    ....