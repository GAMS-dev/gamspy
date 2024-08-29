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
     - 
     - Shows all available solvers that can be installed.

Example: ::

  $ gamspy list solvers
  Installed Solvers
  =================
  CONOPT, CONVERT, CPLEX, IPOPT, IPOPTH, KESTREL, NLPEC, PATH, SHOT

  Model types that can be solved with the installed solvers
  =========================================================
  CONOPT    : LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP
  CONVERT   : LP, MIP, RMIP, NLP, MCP, MPEC, RMPEC, CNS, DNLP, RMINLP, MINLP, QCP, MIQCP, RMIQCP, EMP
  CPLEX     : LP, MIP, RMIP, QCP, MIQCP, RMIQCP
  IPOPT     : LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP
  IPOPTH    : LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP
  KESTREL   : LP, MIP, RMIP, NLP, MCP, MPEC, RMPEC, CNS, DNLP, RMINLP, MINLP, QCP, MIQCP, RMIQCP, EMP
  NLPEC     : MCP, MPEC, RMPEC
  PATH      : MCP, CNS
  SHOT      : MINLP, MIQCP

::

  $ gamspy list solvers -a
  Available Solvers
  =================
  BARON, CBC, CONOPT, CONOPT3, CONVERT, COPT, CPLEX, DICOPT, EXAMINER, EXAMINER2, GUROBI, HIGHS, IPOPT, IPOPTH,
  KESTREL, KNITRO, MILES, MINOS, MOSEK, MPSGE, NLPEC, PATH, PATHNLP, SBB, SCIP, SHOT, SNOPT, SOPLEX, XPRESS

  Model types that can be solved with the installed solvers:

  CONOPT    : LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP
  CONVERT   : LP, MIP, RMIP, NLP, MCP, MPEC, RMPEC, CNS, DNLP, RMINLP, MINLP, QCP, MIQCP, RMIQCP, EMP
  CPLEX     : LP, MIP, RMIP, QCP, MIQCP, RMIQCP
  IPOPT     : LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP
  IPOPTH    : LP, RMIP, NLP, CNS, DNLP, RMINLP, QCP, RMIQCP
  KESTREL   : LP, MIP, RMIP, NLP, MCP, MPEC, RMPEC, CNS, DNLP, RMINLP, MINLP, QCP, MIQCP, RMIQCP, EMP
  NLPEC     : MCP, MPEC, RMPEC
  PATH      : MCP, CNS
  SHOT      : MINLP, MIQCP

.. note::
    The possible model types for a solver become available after the solver has been installed.