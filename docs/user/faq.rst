.. _examples:

**************************
Frequently Asked Questions
**************************

Which solvers does GAMSPy support?
----------------------------------
At the moment, GAMSPy supports 29 solvers:

- CONOPT3
- CONOPT4
- CONVERT
- CPLEX
- IPOPT
- IPOPTH
- KESTREL
- NLPEC
- PATH
- SHOT
- BARON
- CBC
- COPT
- DICOPT
- EXAMINER
- EXAMINER2
- GUROBI
- HIGHS
- KNITRO
- MILES
- MINOS
- MOSEK
- MPSGE
- PATHNLP
- SBB
- SCIP
- SNOPT
- SOPLEX
- XPRESS

The list can also be accessed from commandline by executing: ::

    gamspy list solvers -a

Or it can be accesed by using the utility function getAvailableSolvers: ::

    import gamspy.utils as utils
    print(utils.getAvailableSolvers())

What is the default solver if I don't specify one?
--------------------------------------------------

The listing of default solver for each problem type is below:

+---------+----------------+
| Problem | Default Solver |
+---------+----------------+
| LP      | CPLEX          |
+---------+----------------+
| MIP     | CPLEX          |
+---------+----------------+
| RMIP    | CPLEX          |
+---------+----------------+
| NLP     | IPOPTH         |
+---------+----------------+
| MCP     | PATH           |
+---------+----------------+
| MPEC    | NLPEC          |
+---------+----------------+
| CNS     | PATH           |
+---------+----------------+
| DNLP    | IPOPTH         |
+---------+----------------+
| RMINLP  | IPOPTH         |
+---------+----------------+
| MINLP   | SHOT           |
+---------+----------------+
| QCP     | IPOPTH         |
+---------+----------------+
| MIQCP   | SHOT           |
+---------+----------------+
| RMIQCP  | IPOPTH         |
+---------+----------------+
| EMP     | CONVERT        |
+---------+----------------+

The full list of default solvers that comes with GAMSPy can be listed with: ::

    gamspy list solvers

The current default solvers list is: CONOPT, CONVERT, CPLEX, IPOPT, IPOPTH, KESTREL, NLPEC, PATH, and SHOT. 
But be aware that this list might be subject to change in the future.

Why can't I redefine a GAMSPy symbol with the same name?
--------------------------------------------------------

Trying to run the following lines of code will raise an error.

.. code-block:: 

    from gamspy import Container, Set, Parameter

    m = Container()
    p = Set(container=m, name="p", description="products")
    price = Parameter(container=m, name="p", domain=p, description="price for product p")

The problem with the above code is that the ``Set`` statement creates a symbol in the GAMSPy database
with name "p". Consequently, the namespace "p" is now exclusively reserved for a ``Set``. The following
``Parameter`` statement attempts to create a GAMSPy ``Parameter`` within the same namespace "p", which is 
already reserved for the ``Set`` ``p``. Thus, you want to keep in mind that the type for a GAMSPy symbol 
is fixed once it was declared.

GAMSPy symbols do not need a name. Nevertheless, a name can be very useful when debugging, inspecting the
listing file, or interfacing with other systems, e.g. ``model.toLatex()`` which uses the names provided in
the symbol constructor.


Why do I need a GAMSPy ``Alias``?
---------------------------------

Consider the following example code::

    from gamspy import Container, Set, Parameter

    m = Container()
    i = j = Set(container=m, name="i", records=range(3))
    p = Parameter(container=m, domain=[i, j])

    p[i, j] = 1

You would probably expect that the value for :math:`p_{i,j}` is equal to one for each combination of :math:`(i,j)`

::

    In [1]: p.pivot()
    Out[1]:
         0    1    2
    0  1.0  1.0  1.0
    1  1.0  1.0  1.0
    2  1.0  1.0  1.0

However, the above lines of code give you::

    In [1]: p.pivot()
    Out[1]:
         0    1    2
    0  1.0  0.0  0.0
    1  0.0  1.0  0.0
    2  0.0  0.0  1.0

Only by declaring ``j`` an ``Alias`` of ``i`` you will get the desired outcome::

    from gamspy import Alias, Container, Set, Parameter

    m = Container()
    i = Set(container=m, name="i", records=range(3))
    j = Alias(container=m, name='j', alias_with=i)
    p = Parameter(container=m, domain=[i, j])

    p[i, j] = 1

::

    In [1]: p.pivot()
    Out[1]:
         0    1    2
    0  1.0  1.0  1.0
    1  1.0  1.0  1.0
    2  1.0  1.0  1.0


Do I use a ``Parameter`` or a Python variable to represent scalar parameters?
-----------------------------------------------------------------------------

.. code-block::

    from gamspy import Container, Parameter, Equation, Sum

    m = Container()
    p_python = 40
    p_parameter = Parameter(container=m, records=40)


In most of the cases it does not matter whether a scalar ``Parameter`` or a 
Python variable is used. It is more a matter of taste and convenience as::
    
    eq = Equation(container=m, domain=i)
    eq[i] = Sum(j, x[i, j]) <= p_python

is equivalent to::

    eq = Equation(container=m, domain=i)
    eq[i] = Sum(j, x[i, j]) <= p_parameter

as both equation definitions generate :math:`\sum_{j \in \mathcal{J}} x_{i,j} \le 40`.    

However, if you want to change the value of your scalar parameter in between two solve 
statements like::

    from gamspy import Container, Parameter, Equation, Sum

    m = Container()
    p_python = 40
    p_parameter = Parameter(container=m, records=40)
    ...
    model.solve()
    p_python = 50
    p_parameter.setRecords(50)
    model.solve()
    
you want to use the GAMSPy ``Parameter``, as changes to a Python variable are not 
reflected in the generated GAMSPy model. Changes to a GAMSPy symbol, however, will
be evaluated by the second solve statement.

Which functionalities available in GAMS are not (yet) accessible in GAMSPy?
---------------------------------------------------------------------------

While GAMSPy provides a powerful interface between Python and the GAMS execution engione, there are some 
features from the original GAMS language that are not (yet) fully accessible in GAMSPy. 

Some of the features that have not been fully implemented in GAMSPy include:

1. MPSGE, EMP, EMP-SP:
    Certain specialized GAMS features corresponding to MPSGE, EMP, and EMP-SP are currently 
    not available in GAMSPy. However, efforts are underway to incorporate these features in 
    future updates.
2. Solver-specific features communicated via option files:
    Certain solver-specific features, like indicator constraints, are available in GAMSPy. 
    However, due to the absence of the put facility, generating these constructs may be more 
    challenging. Efforts are being made to provide better ways to communicate such constructs 
    to the solver for enhanced compatibility.
3. GAMS has powerful sparse looping and other program control. These are, for obvious reasons,
    not available in GAMSPy and native Python constructs need to be utilized.
4. Arbitrary traditional GAMS code can be injected via the ``Container.addGamsCode('...')`` method. 
    This might require an extended GAMSPy++ license.

It's important to emphasize that the GAMSPy team is actively working on expanding the feature 
set to bridge the gap between GAMS and GAMSPy. If there are specific features or functionalities 
you would like to see in GAMSPy, please share your feedback with us.

How are GAMS and GAMSPy related?
--------------------------------

**Dependency**

GAMSPy relies on the gamspy_base package, which essentially represents a modularized GAMS 
installation. When creating a GAMSPy ``Container``, you have the option to specify a GAMS 
installation independently via the ``system_directory`` argument. This enables flexibility 
in choosing the GAMS version that best suits your needs.

**Execution**

GAMSPy utilizes the GAMS machinery for critical operations, including the execution of 
indexed assignment statements, equation definitions, and the solve method. While the typical 
GAMSPy user does not need to delve into the intricacies of this connection, it's worth noting 
that these details may evolve for performance reasons.

**Debugging and GAMS Listing File**

Although regular Python debugging facilities are usually sufficient, there may be scenarios 
where additional insights from GAMS prove valuable. If needed, GAMS can provide useful information 
via the GAMS listing file. For more details on debugging with GAMS, refer to the :ref:`GAMSPy debugging 
documentation<debugging>` or the `GAMS debugging documentation <https://www.gams.com/latest/docs/UG_ExecErrPerformance.html#INDEX_error_22_debugging>`_.

**Solver Options**

The options for solvers used by GAMSPy are described in the `Solver Manuals <https://www.gams.com/latest/docs/S_MAIN.html>`_, which is part of 
the GAMS Documentation. It's important to note that examples in the solver manual are based on 
GAMS syntax, not GAMSPy syntax. When configuring solvers in GAMSPy, users can refer to the 
relevant sections in the `GAMS Documentation <https://www.gams.com/latest/docs/S_MAIN.html>`_ for detailed information.


Why does Windows Defender block the gamspy.exe executable?
----------------------------------------------------------

When you execute ``pip install gamspy``, it creates an executable on your machine (e.g. ``gamspy.exe`` on Windows) 
which acts like a regular commandline script. This means that it cannot be signed by us. Therefore, Windows Defender 
sometimes thinks that it is probably a malware. Because of this issue, when you run commands such as ``gamspy install license <access code>``, 
Windows Defender blocks the executable. A workaround is to run ``python -m gamspy install license <access code>``. Another way
is to whitelist ``gamspy.exe`` executable on your machine. Since GAMSPy is open source, to make sure about the safety of the executable, 
one can check the following script which GAMSPy uses: `script <https://github.com/GAMS-dev/gamspy/blob/develop/src/gamspy/_cli/cli.py>`_.

Why can I not run GAMSPy with the Python interpreter from the Microsoft Store
-----------------------------------------------------------------------------

Due to compatibility issues, the GAMS Python API (which is a dependency of GAMSPy) does not work with the Python interpreter from
the Microsoft Store.
