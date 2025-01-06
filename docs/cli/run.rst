gamspy run
=========

Runs GAMSPy models with GAMS MIRO.

Run with MIRO
------------

Runs a GAMSPy model with GAMS MIRO application.

Usage
~~~~~

::

  gamspy run miro [OPTIONS]

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-model
     - -g
     - None
     - Path to the GAMSPy model.
   * - -\-mode
     - -m
     - base
     - Execution mode of MIRO (config, base, or deploy).
   * - -\-path
     - -p
     - None
     - Path to the MIRO executable (.exe on Windows, .app on macOS or .AppImage on Linux).
   * - -\-skip-execution
     - 
     - False
     - Whether to skip model execution.

Examples
~~~~~~~~

Run a model with MIRO::

  $ gamspy run miro --model transport.py

Run a model with MIRO in configuration mode::

  $ gamspy run miro --model transport.py --mode config

Run a model with MIRO using a specific MIRO executable::

  $ gamspy run miro --model transport.py --path /path/to/miro.exe

Run a model with MIRO skipping model execution::

  $ gamspy run miro --model transport.py --skip-execution