.. _container:

.. meta::
   :description: Documentation of GAMSPy Container (gamspy.Container)
   :keywords: Container, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

*********
Container
*********

The :class:`Container <gamspy.Container>` is the central hub of any GAMSPy session. 
It acts as a workspace that manages your optimization problem's data, structure, and execution.

Functionally, the Container serves two primary purposes:

1. **Symbol Management:** It stores and organizes all sets, parameters, variables, and equations.
2. **Execution Interface:** It manages the communication bridge between Python and the GAMS execution engine.

Initialization
==============
You can initialize a :class:`Container <gamspy.Container>` as a fresh workspace or 
load it from existing data sources.

**Basic Initialization**

.. code-block:: python

    import gamspy as gp
    
    m = gp.Container()

**Initialization with Configuration**

You can set the working directory of the container explicitly. This is useful for debugging or managing generated files.

.. code-block:: python

    import gamspy as gp

    # Keeps generated files in the specified directory
    m = gp.Container(
        working_directory="./debug_workspace",
        debugging_level="keep"
    )

**Loading from GDX**

You can initialize a container directly from a `GAMS Data eXchange (GDX)` file using the ``load_from`` argument.

.. code-block:: python

    import gamspy as gp

    # Load all symbols and data immediately upon creation from data.gdx
    m = gp.Container(load_from="data.gdx")

**Loading from another Container**

You can also initialize a container from another Container.

.. code-block:: python

    import gamspy as gp

    m1 = gp.Container()
    i = gp.Set(m1, records=range(10))

    m2 = gp.Container(load_from=m1)
    print(m2["i"])

Using Context Managers
======================

The :class:`Container <gamspy.Container>` supports Python's context manager protocol 
(`with` statement). This approach is recommended as it automatically associates new 
symbols with the active container, reducing code verbosity.

.. tab-set:: 
    .. tab-item:: With Context Manager (Recommended)
        .. code-block:: python

            import gamspy as gp

            # Symbols are automatically added to 'm'
            with gp.Container() as m:
                i = gp.Set(description="Sets do not need explicit container arg")
                p = gp.Parameter(domain=i, description="Parameters find 'm' automatically")

    .. tab-item:: Without Context Manager
        .. code-block:: python

            import gamspy as gp

            m = gp.Container()

            # You must pass 'm' to every constructor
            i = gp.Set(m, description="Must explicitly pass container")
            p = gp.Parameter(m, domain=i)

Symbol Management
=================

Every optimization symbol must belong to a :class:`Container <gamspy.Container>`. 
The :class:`Container <gamspy.Container>` acts like a dictionary; you can access 
symbols using their names as keys.

.. code-block:: python

    import gamspy as gp

    m = gp.Container()
    i = gp.Set(m, name="cities", records=["seattle", "san-diego"])
    
    # Access the object via dictionary syntax
    print(m["cities"].records)

Adding Symbols
--------------

There are two ways to add symbols to a container:

1.  **Direct Instantiation:** Create the object and pass the container.
2.  **Helper Methods:** Use methods like ``addSet``, ``addParameter``, etc..

.. code-block:: python

    import gamspy as gp

    m = gp.Container()

    # Method 1: Direct Instantiation
    j = gp.Set(m, "j", records=["p1", "p2"])

    # Method 2: Helper Methods
    k = m.addSet("k", records=["p1", "p2"])

.. note::
    If you create a symbol with a name that already exists in the container, 
    GAMSPy will overwrite the records of the existing symbol rather than creating 
    a duplicate.

.. warning::
    If you do not provide a name for a symbol, GAMSPy tries to get the name of the symbol from the stack of Python. 
    This context might not be available in all environments. If GAMSPy cannot get the name from the stack, it 
    autogenerates a unique identifier (e.g., `s795f0...`). While convenient for intermediate calculations, explicit 
    naming is recommended for symbols you intend to inspect or export.

Data Management
===============

The ``.records`` attribute of a symbol returns a Pandas DataFrame containing the data. 
This attribute is **read-only**. To modify data, you must use specific setter methods.

Setting Records
---------------

To update the data of a symbol, use the ``setRecords`` method. This ensures the data is 
synchronized with the GAMS execution engine.

.. code-block:: python

    import gamspy as gp

    m = gp.Container()

    i = gp.Set(m)
    
    # Correct way to set data
    i.setRecords(["New", "York"])

Bulk Data Operations
--------------------
Calling ``setRecords`` on individual symbols triggers a synchronization with the GAMS engine 
for every call, which can be slow for models with large number of symbols. 

To improve performance, use :meth:`Container.setRecords <gamspy.Container.setRecords>` to 
update multiple symbols in a single transaction.

.. tab-set::

    .. tab-item:: One by one

        .. code-block:: python

            import gamspy as gp
            m = gp.Container()
            i = gp.Set(m)
            k = gp.Set(m)
            i.setRecords(range(10))
            k.setRecords(range(5))

    .. tab-item:: Bulk setRecords

         .. code-block:: python

            import gamspy as gp
            m = gp.Container()
            i = gp.Set(m)
            k = gp.Set(m)
            m.setRecords({i: range(10), k: range(5)})

Input / Output (GDX)
====================

The Container provides robust tools for interacting with GDX files.

Writing to GDX
--------------
Use :meth:`write <gamspy.Container.write>` to save the current state of symbols to a file.

.. code-block:: python
    
    m.write("model_data.gdx")

Reading from GDX
----------------
There are multiple ways to read data, depending on whether you want to import symbol definitions or just the data records.

**Read Symbol and Data:** :meth:`container.read <gamspy.Container.read>` imports symbol definitions *and* data from a GDX file or another Container.
   
.. code-block:: python

    import gamspy as gp

    m = gp.Container()

    # Reads symbols 'i' and 'j' from the file into container 'm'
    m.read("input.gdx", symbol_names=["i", "j"])

**Load Data Only:** :meth:`container.loadRecordsFromGdx <gamspy.Container.loadRecordsFromGdx>` loads data into symbols that *already exist* in your container. 
This is useful when the model structure is defined in Python, but data comes from an external source.

.. code-block:: python

    import gamspy as gp

    m = gp.Container()

    # 'i' exists in Python, but is empty
    i = gp.Set(m, "i")
    
    # Populates 'i' with data from the GDX file
    m.loadRecordsFromGdx("data.gdx", symbol_names=["i"])

You can also rename symbols during loading by providing a dictionary:

.. code-block:: python

    # Loads data from symbol 'i_remote' in GDX into symbol 'i_local' in Container
    m.loadRecordsFromGdx("data.gdx", symbol_names={"i_remote": "i_local"})

Advanced Features
=================

Serialization
-------------
You can archive the entire state of a Container (structure and data) into a zip file using :meth:`gamspy.serialize`. 
It can be reconstructed later using :meth:`gamspy.deserialize`.

.. code-block:: python

    import gamspy as gp

    # Save state
    gp.serialize(m, "checkpoint.zip")

    # Restore state
    m_new = gp.deserialize("checkpoint.zip")

Debugging: Generated GAMS Code
------------------------------
GAMSPy generates GAMS code in the background. To inspect this code for debugging or educational purposes, 
use :meth:`generateGamsString <gamspy.Container.generateGamsString>`.

.. code-block:: python

    import gamspy as gp

    m = gp.Container(debugging_level="keep")

    i = gp.Set(m, records=range(5))
    j = gp.Set(m, records=range(3))

    print(m.generateGamsString())

For more details, see the :ref:`generate_gams_string` section of the :doc:`/user/advanced/debugging` page. 

Injecting Raw GAMS Code
-----------------------
For advanced users, specific GAMS statements can be injected directly into the execution stream using :meth:`addGamsCode <gamspy.Container.addGamsCode>`.

.. warning::
    This bypasses GAMSPy's safety checks. Use with caution.

.. code-block:: python

    # Adds a scalar directly via GAMS syntax
    m.addGamsCode("scalar piHalf / [pi/2] /;")

Solver Options File
-------------------
If you need to pass specific configuration files to a solver (e.g., a ``conopt.opt`` file), 
you can generate them using :meth:`writeSolverOptions <gamspy.Container.writeSolverOptions>`.

.. code-block:: python

    # Creates a 'conopt.opt' file in the working directory
    m.writeSolverOptions("conopt", solver_options={"rtmaxv": "1.e12"})
