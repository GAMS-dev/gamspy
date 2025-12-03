.. _gamspy_install:

gamspy install
==============

Installs add-on solvers or a license to the GAMSPy installation.

Install License
---------------

Installs a new license using either an access code or a license file.

Usage
~~~~~

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
     - 
     - None
     - Interprocess communication starting port. Only relevant for local licenses that restrict concurrent use of GAMSPy.
   * - -\-server
     - -s
     - https://license.gams.com
     - License server adress.
   * - -\-port 
     - -p
     - None
     - Port for the license server connection.
   * - -\-checkout-duration 
     - -c
     - None
     - Specify a duration in hours to checkout a session.

Examples
~~~~~~~~

Install using access code::

  $ gamspy install license 876e5812-1222-4aba-819d-e1e91b7e2f52

Install using license file::

  $ gamspy install license /home/joe/gamslice.txt

.. note::
  If one needs to use a proxy server, the proxy server can be specified via "HTTPS_PROXY" environment variable.

Install Solver
--------------

Installs one or more solvers to the GAMSPy installation.

Usage
~~~~~

::

  gamspy install solver [solver_name(s)] [OPTIONS]  

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
     - If you already have the solver installed, skip pip install and update gamspy installed solver list.
   * - -\-install-all-solvers
     - -a
     - False
     - Installs all available add-on solvers.
   * - -\-existing-solvers
     - 
     - False
     - Reinstalls previously installed add-on solvers.
   * - -\-use-uv 
     - 
     - False
     - Use uv instead of pip to install solvers.

Examples
~~~~~~~~

Install specific solvers::

  $ gamspy install solver mosek conopt xpress

Install all available solvers::

  $ gamspy install solver --install-all-solvers

Reinstall previously installed solvers::

  $ gamspy install solver --existing-solvers

Skip pip installation::

  $ gamspy install solver mosek -s