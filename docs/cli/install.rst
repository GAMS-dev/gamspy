gamspy install
==============

Installs a new solver or a license to the GAMSPy installation.

Usage
-----

::

  gamspy install solver <solver_name> [OPTIONS]  

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-skip-pip-install 
     - -a
     - 
     - Skips the pip install command in case the package was manually installed.

::

  gamspy install license <license_id> [OPTIONS]  

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-port 
     - 
     - 
     - Sets the port to communicate with GAMS license server.