.. _container:

.. meta::
   :description: Documentation of GAMSPy Container (gamspy.Container)
   :keywords: Container, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

*********
Container
*********

A :meth:`Container <gamspy.Container>` object in GAMSPy serves as a central hub for managing essential data structures such as sets, parameters, variables, 
and constraints, providing a structured approach for optimization problems. It is also responsible for creating the connection between with GAMS and GAMSPy.

===============
Symbol Creation
===============

Every symbol in your optimization problem must belong to a :meth:`Container <gamspy.Container>`.

All symbols added to a :meth:`Container <gamspy.Container>` can be accessed by indexing into the :meth:`Container <gamspy.Container>`::
    
    from gamspy import Container, Set

    m = Container()
    i = Set(m, "i", records=["seattle", "san-diego"])
    print(m["i"])  # returns a reference to i variable


Each symbol is added to the container as soon as it is created. If the symbol already exists in the container, the existing symbol is returned. ::

    from gamspy import Container, Set

    m = Container()
    i1 = Set(m, "i", records=["seattle", "san-diego"])
    i2 = Set(m, "i", records=["seattle", "san-diego"])
    print(id(i1) == id(i2))  # True


Creating a symbol with the same name but different records overwrites the records of the existing symbol. ::

    from gamspy import Container, Set

    m = Container()
    i1 = Set(m, "i", records=["seattle", "san-diego"])
    i2 = Set(m, "i", records=["seattle", "san-diego", "topeka"])
    print(id(i1) == id(i2))  # True
    print(i2.records)  # ['seattle', 'san-diego', 'topeka']

An alternative way to create a symbol in GAMSPy and add it to the container is as follows ::

    from gamspy import Container

    m = Container()
    i = m.addSet("i", records=["seattle", "san-diego"])
    print(i.records)

Symbols can be created without any data. Data can be provided later ::

    from gamspy import Container

    m = Container()
    i = m.addSet("i")
    i.setRecords(["seattle", "san-diego"])
    print(i.records)

Explicit symbols names are useful when interacting with parts of the module where symbols need to be recognized by name, e.g. :meth:`toLatex <gamspy.Model.toLatex>` and GDX imports or exports (see below). If no name is provided, GAMSPy will autogenerate a name ::

    from gamspy import Container
    
    m = Container()
    i = m.addSet()
    print(i.name) # something like 's795f053a_7d21_4a17_a6c3_5a947e051930'

.. warning::
    ``.records`` attribute of a symbol contains a Pandas DataFrame which holds the symbol's records and 
    should be treated as a read-only attribute. If you want to change the records of a symbol, use 
    ``setRecords`` function. ``setRecords`` ensures that the GAMSPy state is synchronized with GAMS 
    execution system.

A container can also be used as a context manager. When a container is used as a context manager, there 
is no need to specify the container when creating symbols since the context manager container will automatically 
be used as the container for the symbols.

.. tabs:: 
    .. group-tab:: With context manager
        .. code-block:: python

            import gamspy as gp

            with gp.Container() as m:
                i = gp.Set()
                a = gp.Alias(alias_with=i)
                p = gp.Parameter()
                v = gp.Variable()
                e = gp.Equation()

    .. group-tab:: Without context manager
        .. code-block:: python

            import gamspy as gp

            m = gp.Container()

            i = gp.Set(m)
            a = gp.Alias(m, alias_with=i)
            p = gp.Parameter(m)
            v = gp.Variable(m)
            e = gp.Equation(m)

===========================
Reading and Writing Symbols
===========================

The :meth:`Container <gamspy.Container>` class provides I/O functions for reading and writing symbols to `GAMS Data eXchange (GDX) <https://www.gams.com/latest/docs/UG_GDX.html>`_ files.

Writing
-------
Symbols created within a specific :meth:`Container <gamspy.Container>` can be saved to a GDX file using the :meth:`write <gamspy.Container.write>` function.

.. code-block:: python
    
    from gamspy import Container, Set
    
    m = Container()
    i = Set(m, "i", records=["seattle", "san-diego"])
    m.write("data.gdx")

Reading
-------
Symbol records can be read from a GDX file by either specifying the ``load_from`` argument during the :meth:`Container <gamspy.Container>` construction or by using the :meth:`read <gamspy.Container.read>` function.

To create a :meth:`Container <gamspy.Container>` with symbols from a GDX file, use the ``load_from`` argument:

.. code-block:: python

    from gamspy import Container

    m = Container(load_from="data.gdx")
    print(m.listSymbols())

We can verify that symbol ``i`` is in the container ``m``.

Alternatively, you can use the :meth:`read <gamspy.Container.read>` function to populate the container:

.. code-block:: python

    from gamspy import Container

    m = Container()
    m.read("data.gdx")
    print(m.listSymbols())

One can also read from another :meth:`Container <gamspy.Container>` instead of reading the records from a gdx file:

.. code-block:: python

    from gamspy import Container, Set

    m1 = Container()
    i = Set(m1, "i", records=range(3))
    
    m2 = Container()
    m2.read(m1)
    print(m2.listSymbols())

Loading Records to Existing Symbols
-----------------------------------

You can load the records of a symbol from a GDX file if the symbol is already declared by using :meth:`loadRecordsFromGdx <gamspy.Container.loadRecordsFromGdx>`.

.. code-block:: python

    from gamspy import Container

    m = Container()
    i = Set(m, name="i")
    m.loadRecordsFromGdx("data.gdx")
    print(i.records)

By default, :meth:`loadRecordsFromGdx <gamspy.Container.loadRecordsFromGdx>` loads the records of all symbols in the gdx file. 
Alternatively, one can provide ``symbol_names`` argument to limit which symbol records to be loaded. For example: ::

    .. code-block:: python

    from gamspy import Container

    m = Container()
    i = Set(m, name="i", records=range(5))
    j = Set(m, name="j", records=range(5, 10))
    m.write("data.gdx")

    m2 = Container()
    i = Set(m, name="i")
    j = Set(m, name="j")
    m2.loadRecordsFromGdx("data.gdx", symbol_names=['j'])
    print(j.records)  # prints the records of j which is retrieved from data.gdx
    print(i.records)  # prints None

In this code snippet, we write ``write.gdx`` which contains the records of ``i`` and ``j``. Then, we load the records of ``j`` only.

If you want to map the symbol names in the GDX file to GAMSPy container symbols with different names, you can also provide ``symbol_names`` as a dictionary. 
For example: ::

    m = Container()
    i = Set(m, name="i", records=range(5))
    j = Set(m, name="j", records=range(5, 10))
    m.write("data.gdx")

    m2 = Container()
    k = Set(m, name="k")
    l = Set(m, name="l")
    m2.loadRecordsFromGdx("data.gdx", symbol_names={'i': 'k', 'j': 'l'})
    print(k.records)  # prints the records of i in data.gdx
    print(l.records)  # prints the records of j in data.gdx

Here the keys of the ``symbol_names`` argument are the names in the GDX file and values are the names in the GAMSPy container.

Serialization and Deserialization
---------------------------------

Serialization is a process to convert Container objects into a zip file which 
can be easily stored, transmitted, and reconstructed. Deserialization is the 
opposite process to reconstruct a Container from a zip file. GAMSPy provides 
:meth:`gp.serialize <gamspy.serialize>` and :meth:`gp.deserialize <gamspy.deserialize>` 
functions to perform these processes. One can serialize a container as follows:

.. code-block:: python

    import gamspy as gp
    m = gp.Container()
    i = gp.Set(m, "i", records=range(3))
    gp.serialize(m, "path_to_the_zip_file.zip")

This would create a zip file with the needed information to reconstruct the Container later.
One can reconstruct a Container from this zip file later as follows:

.. code-block:: python

    import gamspy as gp
    m = gp.deserialize("path_to_the_zip_file.zip")
    i = m["i"]

This creates a new container with the information from the zip file. The symbol `i` in the 
returned Container will be identical to the symbol `i` in the first container that was used 
to generate the zip file.


=================================
Generating the Executed GAMS Code
=================================

GAMSPy utilizes the GAMS execution system and instructs it to perform certain operations. You can check these executed 
operations by inspecting the corresponding GAMS code at any point in the program by calling :meth:`generateGamsString <gamspy.Container.generateGamsString>`.
This feature is available for avid GAMS users who want to see what’s being executed behind the scenes. For more details, 
see the :ref:`generate_gams_string` section of the :doc:`/user/advanced/debugging` page. 
