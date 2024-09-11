:html_theme.sidebar_secondary.remove:

.. _cli:

gamspy
======

GAMSPy comes with a command-line interface (CLI) to allow users to 
easily install solvers, licenses and much more.

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-help 
     - -h
     - 
     - Shows the help message
   * - -\-version 
     - -v
     - 
     - Shows the version of GAMSPy, GAMS and gamspy_base

Example: ::

    $ gamspy --help
    usage: gamspy [-h] [-v]
           gamspy install license <access_code> or <path/to/license/file> [--uses-port <port>]
           gamspy uninstall license
           gamspy install solver <solver_name> [--skip-pip-install] [--existing-solvers] [--install-all-solvers]
           gamspy uninstall solver <solver_name> [--skip-pip-uninstall] [--uninstall-all-solvers]
           gamspy list solvers [--all]
           gamspy show license
           gamspy show base
           gamspy probe [-j <json_output_path>]
           gamspy retrieve license <access_code> [-i <json_file_path>] [-o <output_path>]
           gamspy run miro [--path <path_to_miro>] [--model <path_to_model>]

    GAMSPy CLI

    options:
      -h, --help            show this help message and exit
      -v, --version         Shows the version of GAMSPy, GAMS and gamspy_base

    gamspy install license <access_code> or <path/to/license/file>:
      Options for installing a license.

      --uses-port USES_PORT
                            Interprocess communication starting port.

    gamspy uninstall license:
      Command to uninstall user license.

    gamspy install solver <solver_name>:
      Options for installing solvers

      --skip-pip-install, -s
                            If you already have the solver installed, skip pip install and update gamspy installed solver list.

    gamspy uninstall solver <solver_name>:
      Options for uninstalling solvers

      --skip-pip-uninstall, -u
                            If you don't want to uninstall the package of the solver, skip uninstall and update gamspy installed solver list.

    gamspy list solvers:
      `gamspy list solvers` options

      -a, --all             Shows all available solvers.

    gamspy probe:
      `gamspy probe` options

      --json-out JSON_OUT, -j JSON_OUT
                            Output path for the json file.

    gamspy retrieve license <access_code>:
      `gamspy retrieve license` options

      --output OUTPUT, -o OUTPUT
                            Output path for the license file.
      --input INPUT, -i INPUT
                            json file path to retrieve a license based on node information.

    gamspy run miro:
      `gamspy run miro` options

      -g MODEL, --model MODEL
                            Path to the gamspy model
      -m {config,base,deploy}, --mode {config,base,deploy}
                            Execution mode of MIRO
      -p PATH, --path PATH  Path to the MIRO executable (.exe on Windows, .app on macOS or .AppImage on Linux)
      --skip-execution      Whether to skip model execution

::

    $ gamspy --version
    GAMSPy version: 0.14.6
    GAMS version: 47.4.1
    gamspy_base version: 47.4.1                              

List of Commands
----------------

.. toctree::
    :maxdepth: 1

    install
    list
    probe
    retrieve
    run
    show
    uninstall