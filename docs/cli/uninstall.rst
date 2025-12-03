gamspy uninstall
================

Uninstalls solvers or the current license from the GAMSPy installation.

Uninstall License
-----------------

Uninstalls the current license.

Usage
~~~~~

::

  gamspy uninstall license

Example::

  $ gamspy uninstall license

Uninstall Solver
----------------

Uninstalls one or more solvers from the GAMSPy installation.

Usage
~~~~~

::

  gamspy uninstall solver [solver_name(s)] [OPTIONS]

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-skip-pip-install
     - -s
     - False
     - If you already have the solver uninstalled, skip pip uninstall and update gamspy installed solver list.
   * - -\-uninstall-all-solvers
     - -a
     - False
     - Uninstalls all add-on solvers.
   * - -\-use-uv 
     - 
     - False
     - Use uv instead of pip to uninstall solvers.

Examples
~~~~~~~~

Uninstall specific solvers::

  $ gamspy uninstall solver mosek conopt

Uninstall all add-on solvers::

  $ gamspy uninstall solver --uninstall-all-solvers

Skip pip uninstallation::

  $ gamspy uninstall solver mosek -s

