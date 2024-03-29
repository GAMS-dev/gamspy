.. _container:

.. meta::
   :description: Documentation of GAMSPy Container (gamspy.Container)
   :keywords: Container, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

*********
Container
*********

.. code-block:: python
    
    from gamspy import Container, Set
    
    m = Container()
    i = Set(m, "i", records = ["seattle", "san-diego"])
    j = Set(m, "j", records = ["new-york", "chicago", "topeka"])

===============
Symbol Creation
===============

The ``Container`` class in GAMSPy serves as a central hub for managing essential data structures such as sets, parameters, variables, 
and constraints, providing a structured approach for optimization problems. Every symbol in your optimization problem 
should belong to a ``Container``.

All added symbols to a ``Container`` can be accessed by indexing into the ``Container``::
    
    from gamspy import Container, Set
    m = Container()
    i = Set(m, "i", records = ["seattle", "san-diego"])
    print(m['i'])  # returns a reference to i variable

Each symbol is added to the container as soon as it is created. If the symbol already exists in the container, the existing symbol is returned. ::

    from gamspy import Container, Set
    m = Container()
    i1 = Set(m, "i", records = ["seattle", "san-diego"])
    i2 = Set(m, "i", records = ["seattle", "san-diego"])
    print(id(i1) == id(i2))  # True

Creating a symbol with the same name but different records overwrite the records of the existing symbol. ::

    from gamspy import Container, Set
    m = Container()
    i1 = Set(m, "i", records = ["seattle", "san-diego"])
    i2 = Set(m, "i", records = ["seattle", "san-diego", "topeka"])
    print(id(i1) == id(i2))  # True
    print(i2.records)  # ['seattle', 'san-diego', 'topeka']

An alternative way to create a symbol in GAMSPy and adding it to the container is the following ::

    from gamspy import Container
    m = Container()
    i = m.addSet("i", records = ["seattle", "san-diego"])
    print(i.records)

===========================
Reading and Writing Symbols
===========================

The ``Container`` class offers I/O functions for reading and writing symbols.

Writing
-------
Symbols created within a specific ``Container`` can be saved to a GDX file using the ``write`` function.

.. code-block:: python
    
    from gamspy import Container, Set
    
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
