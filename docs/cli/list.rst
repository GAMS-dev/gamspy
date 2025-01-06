gamspy list
===========

Lists available or installed solvers in the GAMSPy installation.

Usage
-----

::

  gamspy list solvers [OPTIONS]

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-all 
     - -a
     - False
     - Shows all available solvers that can be installed.
   * - -\-defaults
     - -d
     - False
     - Shows default solvers for each problem type.

Examples
--------

List installed solvers::

  $ gamspy list solvers
  Installed Solvers
  =================
  CONOPT, CONVERT, CPLEX, IPOPT, IPOPTH, KESTREL, NLPEC, PATH, SHOT

  Model types that can be solved with the installed solvers
  =======================================================
  ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ Solver  ┃ Problem Types                                                              ┃
  ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ CONOPT  │ LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP                              │
  │ CONVERT │ LP, MIP, RMIP, NLP, MCP, MPEC, RMPEC, CNS, DNLP, RMINLP, MINLP, QCP, MIQCP │
  │ CPLEX   │ LP, MIP, RMIP, QCP, MIQCP, RMIQCP                                          │
  │ IPOPT   │ LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP                              │
  │ IPOPTH  │ LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP                              │
  │ KESTREL │ LP, MIP, RMIP, NLP, MCP, MPEC, RMPEC, CNS, DNLP, RMINLP, MINLP, QCP, MIQCP │
  │ NLPEC   │ MCP, MPEC, RMPEC                                                           │
  │ PATH    │ MCP, CNS                                                                   │
  │ SHOT    │ MINLP, MIQCP                                                               │
  └─────────┴────────────────────────────────────────────────────────────────────────────┘

List all available solvers::

  $ gamspy list solvers --all
  Available Solvers
  =================
  BARON, CBC, CONOPT, CONOPT3, CONVERT, COPT, CPLEX, DICOPT, EXAMINER, EXAMINER2, GUROBI, HIGHS, IPOPT, IPOPTH,
  KESTREL, KNITRO, MILES, MINOS, MOSEK, MPSGE, NLPEC, PATH, PATHNLP, SBB, SCIP, SHOT, SNOPT, SOPLEX, XPRESS

  Model types that can be solved with the installed solvers
  =======================================================
  ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ Solver  ┃ Problem Types                                                             ┃
  ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ CONOPT  │ LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP                             │
  │ CONVERT │ LP, MIP, RMIP, NLP, MCP, MPEC, RMPEC, CNS, DNLP, RMINLP, MINLP, QCP, MIQCP│
  │ CPLEX   │ LP, MIP, RMIP, QCP, MIQCP, RMIQCP                                         │
  │ IPOPT   │ LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP                             │
  │ IPOPTH  │ LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP                             │
  │ KESTREL │ LP, MIP, RMIP, NLP, MCP, MPEC, RMPEC, CNS, DNLP, RMINLP, MINLP, QCP, MIQCP│
  │ NLPEC   │ MCP, MPEC, RMPEC                                                          │
  │ PATH    │ MCP, CNS                                                                  │
  │ SHOT    │ MINLP, MIQCP                                                              │
  └─────────┴───────────────────────────────────────────────────────────────────────────┘

List default solvers for each problem type::

  $ gamspy list solvers --defaults
  ┏━━━━━━━━━┳━━━━━━━━┓
  ┃ Problem ┃ Solver ┃
  ┡━━━━━━━━━╇━━━━━━━━┩
  │ LP      │ CPLEX  │
  │ MIP     │ CPLEX  │
  │ RMIP    │ CPLEX  │
  │ NLP     │ CONOPT │
  │ MCP     │ PATH   │
  │ MPEC    │ NLPEC  │
  │ CNS     │ PATH   │
  │ DNLP    │ CONOPT │
  │ RMINLP  │ CONOPT │
  │ MINLP   │ SHOT   │
  │ QCP     │ CPLEX  │
  │ MIQCP   │ CPLEX  │
  │ RMIQCP  │ CPLEX  │
  └─────────┴────────┘

.. note::
    The possible model types for a solver become available after the solver has been installed.
    For a complete list of solvers and their capabilities, visit: https://www.gams.com/latest/docs/S_MAIN.html#SOLVERS_MODEL_TYPES