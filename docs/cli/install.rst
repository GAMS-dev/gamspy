.. _gamspy_install:

gamspy install
==============

Installs a new solver or a license to the GAMSPy installation.

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

Example: ::

  $ gamspy install solver mosek conopt xpress

Usage
-----

::

  gamspy install license <license_id>|<path/to/license/file> [OPTIONS]  

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-uses-port 
     - -u
     - 33333
     - Interprocess communication starting port. Only relevant for local licenses that restrict concurrent use of GAMSPy.


Example: ::

  $ gamspy install license 876e5812-1222-4aba-819d-e1e91b7e2f52

::  

  $ gamspy install license /home/joe/gamslice.txt
