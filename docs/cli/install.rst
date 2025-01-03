.. _gamspy_install:

gamspy install
==============

Installs add-on solvers or a license to the GAMSPy installation.

Usage
-----

::

  gamspy install solver <solver_name(s)> [OPTIONS]  

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-skip-pip-install 
     - -s
     - 
     - Skips the pip install command in case the package was manually installed.
   * - -\-install-all-solvers
     - 
     - 
     - Installs all add-on solvers.
   * - -\-existing-solvers
     - 
     - 
     - Installs add-on solvers that were previously installed with an older version of gamspy.

Example 1: ::

  $ gamspy install solver mosek conopt xpress

Example 2: ::

  $ gamspy install solver --install-all-solvers

Usage
-----

::

  gamspy install license <access_code>|<path/to/license/file> [OPTIONS]  

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-uses-port 
     - -u
     - 
     - Interprocess communication starting port. Only relevant for local licenses that restrict concurrent use of GAMSPy.


Example: ::

  $ gamspy install license 876e5812-1222-4aba-819d-e1e91b7e2f52

::  

  $ gamspy install license /home/joe/gamslice.txt
