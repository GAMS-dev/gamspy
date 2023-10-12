.. _container:

*********
Container
*********

.. code-block:: python
    
    from gamspy import Container, Set
    
    m = Container()
    i = Set(m, "i", records = ["seattle", "san-diego"])
    j = Set(m, "j", records = ['new-york', 'chicago', 'topeka'])

``Container`` acts as a centralized hub, gathering essential data, sets, parameters, variables, and constraints, 
providing a clear structure for our optimization problem. Every symbol must belong to a ``Container``. 

===========================
Reading and Writing Symbols
===========================

``Container`` provides I/O functions to read and write symbols.

Writing
-------
Symbols that were created with a specific ``Container`` can be written to a GDX file with ``write`` function.

.. code-block:: python
    
    from gamspy import Set, Container
    
    m = Container()
    i = Set(m, "i", records=["seattle", "san-diego"])
    m.write("data.gdx")

Reading
-------
Symbol records can be read by either providing ``load_from`` argument to the Container at construction or by using ``read`` function.
Let's say we have a GDX file that contains the records of our symbols as in the example above.

In order to create a ``Container`` with the symbols in our GDX file, we can create a ``Container`` with ``load_from`` argument which points to the GDX file:

.. code-block:: python

    from gamspy import Container
    m = Container(load_from="data.gdx")
    print(m.listSymbols())

We can verify that symbol ``i`` is in the container ``m``.

Alternatively, ``read`` function of ``Container`` can be used to populate it.

.. code-block:: python

    from gamspy import Container
    m = Container()
    m.read("data.gdx")
    print(m.listSymbols())

===============
Execution Types
===============

``GAMSPy`` supports two form of execution through ``Container``.

Normal Execution
----------------
By default, this execution type is enabled. In normal execution, every record assignment or record reading attempt
causes ``GAMSPy`` to execute the generated code and save the results. This type of execution is better for debugging
even though it is slower compared to delayed execution.

.. code-block:: python

    from gamspy import Container, Set, Parameter
    m = Container()
    i = Set(m, "i", records=['i1', 'i2'])
    a = Parameter(m, "a", domain=[i], records=[('i1', 1), ('i2', 2)])
    a[i] = a[i] * 90

In normal execution, the last line executes the actual computation in GAMS, as soon as it's specified.

Delayed Execution
-----------------
Delayed execution is an execution mode for better performance. With this mode, the assignments are not run until
the solve function of a model is called or an attempt to read the records of a dirty symbol. Dirty symbol is a 
symbol that was assigned a new value in previous lines. 

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

In this example, since the working directory specified as the current directory, temporary GAMS files
will be dumped to the current directory. 

Another alternative is to use ``generateGamsString`` function. This function returns the GAMS code that
will be generated with up until that point as a string.

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