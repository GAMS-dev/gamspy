.. _model:

*****
Model
*****

The model class is used to collect equations into groups and to label them so that they can be solved.
It also allows specifying the problem type (e.g. LP, MIP etc.), sense of the problem (MIN, MAX, FEASIBILITY)
and the objective of the problem at hand.

The overall syntax of a model is as follows: ::

    from gamspy import Container, Variable, Equation, Model, Sense, Problem

    m = Container()
    
    z = Variable(m, "z") # objective variable
    e1 = Equation(m, "e1")
    e1[...] = <definition_of_the_equation>
    e2 = Equation(m, "e2")
    e2[...] = <definition_of_the_equation>
    
    example_model = Model(m, "dummy", equations=[e1,e2], problem=Problem.LP, sense=Sense.Max, objective=z)

Classification of Models
========================
Various types of problems can be solved with GAMS. Note that the type of the model must be known before it 
may be solved. The model types are briefly discussed in this section. GAMS checks that the model is in fact 
the type the user thinks it is, and issues explanatory error messages if it discovers a mismatch - for instance, 
that a supposedly linear model contains nonlinear terms. Some problems may be solved in more than one way, and 
the user has to choose which way to use. For instance, if there are binary or integer variables in the model, 
it can be solved either as a MIP or as a RMIP.

========== ==========================================================
Model Type Model Type Description
========== ==========================================================
  LP       Linear Program   
 NLP       Nonlinear Program
 QCP       Quadratically Constrained Program
DNLP       Discontinuous Nonlinear Program
 MIP       Mixed Integer Program
RMIP       Relaxed Mixed Integer Program
MINLP      Mixed Integer Nonlinear Program
RMINLP     Relaxed Mixed Integer Nonlinear Program
MIQCP      Mixed Integer Quadratically Constrained Program
RMIQCP     Relaxed Mixed Integer Quadratically Constrained Program
MCP        Mixed Complementarity Problem
CNS        Constrained Nonlinear System
MPEC       Mathematical Programs with Equilibrium Constraints	
RMPEC      Relaxed Mathematical Program with Equilibrium Constraints
EMP        Extended Mathematical Program
MPSGE      General Equilibrium
========== ==========================================================

All model types are exposed with ``Problem`` enum but the problem type
can be specified as a string as well.

Also the direction types of the optimization (MIN, MAX, or FEASIBILITY) are
exposed with ``Sense`` enum but it can be specified as a string similarly.

Model Attributes
================

Models have attributes that hold a variety of information, including

* information about the results of a solve performed, a solve statement, the solution of a model,
* information about certain features to be used by GAMS or the solver,
* information passed to GAMS or the solver specifying various settings that are also subject to option statements.

====================== ===========================
Model Attribute        Description
====================== ===========================
num_domain_violations  Number of domain violations
algorithm_time         Solver dependent timing information
total_solve_time       Elapsed time it took to execute a solve statement in total
total_solver_time      Elapsed time taken by the solver only
num_iterations         Number of iterations used
marginals              Indicator for marginals present
max_infeasibility      Maximum of infeasibilities
mean_infeasibility     Mean of infeasibilities
status                 Integer number that indicates the model status
num_nodes_used         Number of nodes used by the MIP solver
num_dependencies       Number of dependencies in a CNS model
num_discrete_variables Number of discrete variables
num_infeasibilities    Number of infeasibilities
num_nonlinear_insts    Number of nonlinear instructions
num_nonlinear_zeros    Number of nonlinear nonzeros
num_nonoptimalities    Number of nonoptimalities
num_nonzeros           Number of nonzero entries in the model coefficient matrix
num_mcp_redefinitions  Number of MCP redefinitions
num_variables          Number of variables
num_bound_projections  Number of bound projections during model generation
objective_estimation   Estimate of the best possible solution for a mixed-integer model
objective_value        Objective function value
used_model_type        Integer number that indicates the used model type
model_generation_time  Time GAMS took to generate the model in wall-clock seconds
solve_model_time       Time the solver used to solve the model in seconds
sum_infeasibilities    Sum of infeasibilities
solver_status          Indicates the solver termination condition
solver_version         Solver version
====================== ===========================

Solving a Model
===============

Model has a function named ``solve`` that allows user to solve the specified model. ::

    from gamspy import Container, Variable, Equation, Model, Sense, Problem, Options

    m = Container()
    
    z = Variable(m, "z") # objective variable
    e1 = Equation(m, "e1")
    e1[...] = <definition_of_the_equation>
    e2 = Equation(m, "e2")
    e2[...] = <definition_of_the_equation>
    
    model = Model(m, "dummy", equations=[e1,e2], problem=Problem.LP, sense=Sense.Max, objective=z)
    model.solve(solver="CONOPT", options=Options(iteration_limit=2), solver_options={"rtmaxv": "1.e12"})

In most cases, calling the ``solve`` function of your model without any parameters is sufficient. 
In this scenario, the default solver depending on the problem type, default options will be used. But for users
who requires a higher level of control can set the ``solver`` to be used, general options and solver
specific options. All installed solvers on your system can be queried by running the following command: ::

    gamspy list solvers

If you want to get all available solvers that you can install and use, the following command would give you
the list of solvers that are available.::

    gamspy list solvers -a

Solve Options
-------------

Solve options can be specified as an :meth:`gamspy.Options` class. For example: ::

    from gamspy import Container, Variable, Equation, Model, Sense, Problem, Options

    m = Container()
    
    ...
    ...
    Definition of your model
    ...
    ...

    model = Model(m, "my_model", equations=m.getEquations(), problem=Problem.LP, sense=Sense.Max, objective=z)
    model.solve(options=Options(iteration_limit=2))



Here is the list of options and their descriptions:

+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| Option                            | Description                                                                       | Possible Values                                          |
+===================================+===================================================================================+==========================================================+
| cns                               | Default cns solver                                                                | Any solver installed in your system that can solve cns   |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| dnlp                              | Default dnlp solver                                                               | Any solver installed in your system that can solve dnlp  |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| emp                               | Default emp solver                                                                | Any solver installed in your system that can solve emp   |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| lp                                | Default lp solver                                                                 | Any solver installed in your system that can solve lp    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| mcp                               | Default mcp solver                                                                | Any solver installed in your system that can solve mcp   |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| minlp                             | Default minlp solver                                                              | Any solver installed in your system that can solve minlp |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| mip                               | Default mip solver                                                                | Any solver installed in your system that can solve mip   |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| miqcp                             | Default miqcp solver                                                              | Any solver installed in your system that can solve miqcp |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| mpec                              | Default mpec solver                                                               | Any solver installed in your system that can solve mpec  |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| nlp                               | Default nlp solver                                                                | Any solver installed in your system that can solve nlp   |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| qcp                               | Default qcp solver                                                                | Any solver installed in your system that can solve qcp   |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| rminlp                            | Default rminlp solver                                                             |                                                          |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| rmip                              | Default rmip solver                                                               | Any solver installed in your system that can solve rmip  |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| rmiqcp                            | Default rmiqcp solver                                                             |                                                          |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| rmpec                             | Default rmpec solver                                                              | Any solver installed in your system that can solve rmpec |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| allow_suffix_in_equation          | Allow variables with suffixes in model algebra                                    | bool                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| allow_suffix_in_limited_variables | Allow domain limited variables with suffixes in model                             | bool                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| basis_detection_threshold         | Basis detection threshold                                                         | float                                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| compile_error_limit               | Compile time error limiy                                                          | int                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| domain_violation_limit            | Domain violation limit solver default                                             | int                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| job_time_limit                    | Elapsed time limit in seconds                                                     | float                                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| job_heap_limit                    | Maximum Heap size allowed in MB                                                   | float                                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| hold_fixed_variables              | Treat fixed variables as constants                                                | bool                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| integer_variable_upper_bound      | Set mode for default upper bounds on integer variables                            | 0: Set to +INF                                           |
|                                   |                                                                                   |                                                          |          
|                                   |                                                                                   | 1: Set to 100.                                           |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 2: Set to 100 and write to the log if the level > 100    |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 3: Same as 2 but issues an error if the level > 100      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| iteration_limit                   | Iteration limit of solver                                                         | int                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| keep_temporary_files              | Controls keeping or deletion of process directory and scratch files               | bool                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| license                           | Use alternative license file                                                      | Path to the alternative license                          |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| variable_listing_limit            | Maximum number of columns listed in one variable block                            | int                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| equation_listing_limit            | Maximum number of rows listed in one equation block                               | int                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| node_limit                        | Node limit in branch and bound tree                                               | int                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| absolute_optimality_gap           | Absolute Optimality criterion solver default                                      | float                                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| relative_optimality_gap           | Relative Optimality criterion solver default                                      | float                                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| profile                           | Execution profiling                                                               | 0: No profiling                                          |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 1: Minimum profiling                                     |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 2: Profiling depth for nested control structures         |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| profile_tolerance                 | Minimum time a statement must use to appear in profile generated output           | float                                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| time_limit                        | Wall-clock time limit for solver                                                  | float                                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| savepoint                         | Save solver point in GDX file                                                     | 0: No point GDX file is to be saved                      |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 1: A point GDX file from the last solve is to be saved   |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 2: A point GDX file from every solve is to be saved      |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 3: A point GDX file from the last solve is to be saved   |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 4: A point GDX file from every solve is to be saved      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| seed                              | Random number seed                                                                | int                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| report_solution                   | Solution report print option                                                      | 0: Remove solution listings following solves             |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 1: Include solution listings following solves            |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 2: Suppress all solution information                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| show_os_memory                    |                                                                                   | 0: Show memory reported by internal accounting           |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 1: Show resident set size reported by operating system   |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 2: Show virtual set size reported by operating system    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| solver_link_type                  | Solver link option                                                                | https://gams.com/45/docs/UG_GamsCall.html#GAMSAOsolvelink|
|                                   |                                                                                   |                                                          |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| multi_solve_strategy              | Multiple solve management                                                         | "replace" | "merge" | "clear"                            |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| step_summary                      | Summary of computing resources used by job steps                                  | bool                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| suppress_compiler_listing         | Compiler listing option                                                           | bool                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| report_solver_status              | Solver Status file reporting option                                               | bool                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| threads                           | Number of threads to be used by a solver                                          | int                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| trace_file                        | Trace file name                                                                   | Name of the trace file                                   |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| trace_level                       | Modelstat/Solvestat threshold used in conjunction with action=GT                  | int                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| trace_file_format                 | Trace file format option                                                          | 0: Solver and GAMS step trace                            |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 1: Solver and GAMS exit trace                            |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 2: Solver trace only                                     |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 3: Trace only in format used for GAMS performance world  |
|                                   |                                                                                   |                                                          |
|                                   |                                                                                   | 5: Gams exit trace with all available trace fields       |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| write_listing_file                | Controls listing file creation                                                    | bool                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| zero_rounding_threshold           | The results of certain operations will be set to zero if abs(result) LE ZeroRes   | float                                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+
| report_underflow                  | Report underflow as a warning when abs(results) LE ZeroRes and result set to zero | bool                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+----------------------------------------------------------+

Solver Options
--------------

In addition to solve options, user can specify solver options to be used by the solver as a dictionary. For all possible
solver options, please check the corresponding `solver manual <https://www.gams.com/latest/docs/S_MAIN.html>`_

Solve Locally
---------------

Models are solved locally (on your machine) by default. 

Solve Using GAMS Engine
-----------------------

In order to send your model to be solved to GAMS Engine, you need to define the configuration of GAMS Engine.
This can be done by importing ``EngineConfig`` and creating an instance. Then, the user can pass it to the 
``solve`` method and specify the backend as ``engine``. ::

    from gamspy import Container, Variable, Equation, Model, Sense, Problem, EngineConfig

    m = Container()
    
    z = Variable(m, "z") # objective variable
    e1 = Equation(m, "e1")
    e1[...] = <definition_of_the_equation>
    e2 = Equation(m, "e2")
    e2[...] = <definition_of_the_equation>
    
    model = Model(m, "dummy", equations=[e1,e2], problem=Problem.LP, sense=Sense.Max, objective=z)

    config = EngineConfig(
        host=os.environ["ENGINE_URL"],
        username=os.environ["ENGINE_USER"],
        password=os.environ["ENGINE_PASSWORD"],
        namespace=os.environ["ENGINE_NAMESPACE"],
    )
    model.solve(solver="CONOPT", backend="engine", engine_config=config)

Solve Using NEOS Server
-----------------------

In order to send your model to be solved to NEOS Server, you need to create a NeosClient.
This can be done by importing ``NeosClient`` and creating an instance. Then, the user can pass it to the 
``solve`` method and specify the backend as ``neos``. ::

    from gamspy import Container, Variable, Equation, Model, Sense, Problem, NeosClient

    m = Container()
    
    z = Variable(m, "z") # objective variable
    e1 = Equation(m, "e1")
    e1[...] = <definition_of_the_equation>
    e2 = Equation(m, "e2")
    e2[...] = <definition_of_the_equation>
    
    model = Model(m, "dummy", equations=[e1,e2], problem=Problem.LP, sense=Sense.Max, objective=z)

    client = NeosClient(
        email=os.environ["NEOS_EMAIL"],
        username=os.environ["NEOS_USER"],
        password=os.environ["NEOS_PASSWORD"],
    )
    model.solve(backend="neos", neos_client=client)

Defining your username and password is optional for NEOS Server backend but it is recommended since
it allows you to investigate your models on `NEOS web client <https://neos-server.org/neos/>`_. The
environment variables can be set in a .env file or with export statements in command line. Example to
run your model on NEOS Server without authentication: ::

    NEOS_EMAIL=<your_email> python <your_script>

If one wants to investigate the results later on NEOS Server web client, they can provide the username
and password in the same way: ::

    NEOS_EMAIL=<your_email> NEOS_USER=<your_username> NEOS_PASSWORD=<your_password> python <your_script>

If you just want to send your jobs to NEOS server without blocking until the results are received,
`is_blocking` parameter can be set to `False` in `NeosClient`.

All submitted jobs are stored in `client.jobs` in case you want to reach to the job numbers and job passwords
you already sent to the server. ::

    from gamspy import Container, Variable, Equation, Model, Sense, Problem, NeosClient
    m = Container()
    ...
    ...
    <define_your_model>
    ...
    ...
    client = NeosClient(
        email=os.environ["NEOS_EMAIL"],
        username=os.environ["NEOS_USER"],
        password=os.environ["NEOS_PASSWORD"],
    )

    for _ in range(3):
        ...
        ...
        <changes_in_your_model>
        ...
        ...
        model.solve(backend="neos", neos_client=client)

    print(client.jobs) # This prints all job numbers and jon passwords as a list of tuples


Terms of use for NEOS can be found here: `Terms of Use <https://neos-server.org/neos/termofuse.html> _`.

Redirecting Output
------------------

The output of GAMS after solving the model can be redirected to a file or to standard input by
specifying the output parameter of the ``solve``.::
    
    from gamspy import Container, Variable, Equation, Model, Sense, Problem
    import sys

    m = Container()
    
    z = Variable(m, "z") # objective variable
    e1 = Equation(m, "e1")
    e1[...] = <definition_of_the_equation>
    e2 = Equation(m, "e2")
    e2[...] = <definition_of_the_equation>
    
    model = Model(m, "dummy", equations=[e1,e2], problem=Problem.LP, sense=Sense.Max, objective=z)
    
    # redirect output to stdout
    model.solve(output=sys.stdout)

    # redirect output to a file
    with open("my_out_file", "w") as file:
        model.solve(output=file)


