gamspy uninstall
================

Uninstalls an existing solver or a license from the GAMSPy installation.

Usage
-----

::

  gamspy uninstall solver <solver_name> [OPTIONS]  

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-skip-pip-uninstall 
     - -u
     - 
     - Skips the pip uninstall command in case the package was manually deleted.
   * - -\-uninstall-all-solvers
     - -u
     - 
     - Uninstalls all addon solvers.

Example: ::

  gamspy uninstall solver mosek

.. note::
    Default solvers cannot be uninstalled.

Usage
-----

::

  gamspy uninstall license

This uninstalls a previously installed license and reinstates the GAMSPy demo license that comes with the GAMSPy installation.

