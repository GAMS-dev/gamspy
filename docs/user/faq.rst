.. _examples:

**************************
Frequently Asked Questions
**************************

Which solvers does GAMSPy support?
----------------------------------
At the moment, GAMSPy supports 30 solvers:

- BARON
- CBC
- CONOPT3
- CONOPT4
- CONVERT
- COPT
- CPLEX
- DICOPT
- EXAMINER
- EXAMINER2
- GUROBI
- HIGHS
- IPOPT
- IPOPTH
- KESTREL
- KNITRO
- MILES
- MINOS
- MOSEK
- MPSGE
- NLPEC
- PATH
- PATHNLP
- RESHOP
- SBB
- SCIP
- SHOT
- SNOPT
- SOPLEX
- XPRESS

This list can also be accessed from the command line by executing: ::

    gamspy list solvers -a

Or it can be accessed by using the utility function ``getAvailableSolvers``: ::

    import gamspy.utils as utils
    print(utils.getAvailableSolvers())

As soon as you install GAMSPy, you get the default solvers and you can install 
other solvers later. The solvers that are already installed on your machine can 
be queried as follows: ::

    import gamspy_base
    import gamspy.utils as utils
    print(utils.getInstalledSolvers(gamspy_base.directory))

All installable solvers can be listed with: ::

    import gamspy_base
    import gamspy.utils as utils
    print(utils.getInstallableSolvers(gamspy_base.directory))

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

The list of solvers that comes with a base GAMSPy install can be obtained with: ::

    gamspy list solvers

The current default solvers list is: CONOPT4, CONVERT, CPLEX, IPOPT, IPOPTH, KESTREL, NLPEC, PATH, and SHOT. 
Beaware that this list might be subject to change in the future.

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

Only by declaring ``j`` as an ``Alias`` of ``i`` you will get the desired outcome::

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


Should I use a ``Parameter`` or a Python variable to represent scalar parameters?
---------------------------------------------------------------------------------

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

However, if you want to change the value of your scalar parameter between two solve 
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
    
you need to use the GAMSPy ``Parameter``, as changes to a Python variable are not 
reflected in the generated GAMSPy model. Changes to a GAMSPy symbol, however, will
be taken into account by the second solve statement.

Which functionalities available in GAMS are not (yet) accessible in GAMSPy?
---------------------------------------------------------------------------

While GAMSPy provides a powerful interface between Python and the GAMS execution engine, there are some 
features from the original GAMS language that are not (yet) fully accessible in GAMSPy. 

Some of the features that have not been fully implemented in GAMSPy include:

1. MPSGE, EMP, EMP-SP:
    Some advanced GAMS features corresponding to MPSGE, EMP, and EMP-SP are currently 
    not available in GAMSPy. However, efforts are underway to incorporate these features in 
    future updates.
2. GAMS has powerful sparse looping and other program control.
      These are, for obvious reasons, not available in GAMSPy and native Python constructs need to be utilized.

.. note::
   Arbitrary traditional GAMS code can be injected via the ``Container.addGamsCode('...')`` method. 
   This might require an extended GAMSPy++ license.

The GAMSPy team is actively working on bridging the gap between GAMS and GAMSPy.
If there are specific features or functionalities you would like to see in GAMSPy,
please share your feedback with us either via the support channel or the
`GAMS Forum <https://forum.gams.com>`_.

How are GAMS and GAMSPy related?
--------------------------------

GAMSPy is built on top of the GAMS execution engine. Historically, the latter was
only accessible from the GAMS domain specific language (DSL). Now, GAMSPy
s an additional entry point into the execution engine.

Here are a few aspects of the links between GAMSPy and the GAMS system.

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
which acts like a regular command line script. This means that it cannot be signed by us. Therefore, Windows Defender 
sometimes thinks that it is probably a malware. Because of this issue, when you run commands such as ``gamspy install license <access code>``, 
Windows Defender blocks the executable. A workaround is to run ``python -m gamspy install license <access code>``. Another way
is to whitelist ``gamspy.exe`` executable on your machine. Since GAMSPy is open source, to make sure about the safety of the executable, 
one can check the following script which GAMSPy uses: `script <https://github.com/GAMS-dev/gamspy/blob/develop/src/gamspy/_cli/cli.py>`_.

Why can I not run GAMSPy with the Python interpreter from the Microsoft Store
-----------------------------------------------------------------------------

Due to compatibility issues, the GAMS Python API (which is a dependency of GAMSPy) does not work with the Python interpreter from
the Microsoft Store.

Do network licenses incur extra overhead?
-----------------------------------------

While highly flexible, network licenses come with a fixed start-up cost: every GAMSPy job must 
create a session on the license server before execution can begin. The time to establish this session 
is governed by network latency and typically takes under a second on a fast connection, but can take 
longer on slower or distant networks. For long-running jobs this overhead is usually negligible, but 
workflows that launch many short-lived GAMSPy jobs can accumulate noticeable delays. If running a large 
number of such jobs is part of your optimization pipeline, consider checking out the network license 
for a suitable period. While the license is checked out it behaves like a local license, eliminating 
per-job connection delays.

I am on a restricted network. How can I use a proxy to install a GAMSPy license?
--------------------------------------------------------------------------------

If you need to install a license on a restricted network, you can declare ``HTTPS_PROXY`` environment variable 
that specifies the proxy server: ::

    HTTPS_PROXY=<proxy_server> gamspy install license <access_code>

If you have a network license, you should declare ``CURL_PROXY`` environment variable to perform 
communication via proxy with the license server: ::

    CURL_PROXY=<proxy_server> python <your_script>.py 
