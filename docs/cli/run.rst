gamspy run
==========

Runs the GAMS MIRO application.

Usage
-----

::

  gamspy run miro [OPTIONS]  


.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-path <path>
     - -p <path>
     - 
     - Path to your GAMS MIRO installation.
   * - -\-model <model>
     - -g <model>
     - 
     - Path to your model.
   * - -\-mode <config|base|deploy>
     - -m <config|base|deploy>
     - base
     - Execution mode of MIRO
   * - -\-skip-execution
     -
     -
     - Whether to skip model execution

Example: ::

  $ gamspy run miro -m config -p "/Applications/GAMS MIRO.app/Contents/MacOS/GAMS MIRO" -g ~/miro_apps/myapp.py