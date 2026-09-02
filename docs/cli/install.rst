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
     - License server address.
   * - -\-port 
     - -p
     - None
     - Port for the license server connection.
   * - -\-checkout-duration 
     - -c
     - None
     - Specify a duration in hours to checkout a session.
   * - -\-output 
     - -o
     - None
     - Specify a file path to write the license file.
   * - -\-uses-port
     - 
     - None
     - Interprocess communication starting port.
   * - -\-use-uv 
     - 
     - False
     - Use uv instead of pip to install solvers.

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

The installation of the ``cuopt`` solver can be adjusted with the following environment variables:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Environment Variable
     - Default
     - Description
   * - ``GAMSPY_CUOPT_VERSION``
     - ``0.0.8b``
     - Version of the GAMS ``cuopt`` link to install, ``0.0.8b`` or newer, e.g. ``0.0.9``.
   * - ``GAMSPY_CUDA_VERSION``
     - autodetected
     - Major version of the CUDA runtime (``12`` or ``13``) to install ``cuopt`` for.
   * - ``GAMSPY_CUDA_RUNTIME``
     - ``1``
     - Whether to install the CUDA runtime libraries that ``cuopt`` requires along with it. Either ``1`` or ``0``.

Examples
~~~~~~~~

Install specific solvers::

  $ gamspy install solver mosek conopt xpress cuopt

Install all available solvers::

  $ gamspy install solver --install-all-solvers

Reinstall previously installed solvers::

  $ gamspy install solver --existing-solvers

Skip pip installation::

  $ gamspy install solver mosek -s

Install a specific ``cuopt`` version for CUDA 12 without the CUDA runtime libraries::

  $ GAMSPY_CUOPT_VERSION=0.0.8b GAMSPY_CUDA_VERSION=12 GAMSPY_CUDA_RUNTIME=0 gamspy install solver cuopt